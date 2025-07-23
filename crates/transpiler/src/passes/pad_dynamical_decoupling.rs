// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::TranspilerError;
use hashbrown::{HashMap, HashSet};
use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::dag_node::{DAGInNode, DAGOpNode};
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::{
    DelayUnit, OperationRef, Param, StandardGate, StandardInstruction,
};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::Qubit;
use qiskit_synthesis::euler_one_qubit_decomposer::{angles_from_unitary, EulerBasis};
use smallvec::{smallvec, SmallVec};
use std::cmp;
use std::f64::consts::PI;

#[pyfunction]
#[pyo3(name = "pad_dynamical_decoupling")]
pub fn run_pad_dynamical_decoupling(
    py: Python,
    t_end: usize, // When t_end is passed as a float, it leads to errors & panics (YOU MUST USE A f64...)
    t_start: usize,
    alignment: usize,
    prev_node: &Bound<PyAny>, // Can't I use rust-native struct(s) for this? (No, because you don't know what sub-class of DAGNode it is...)
    next_node: &Bound<PyAny>,
    _no_dd_qubits: &Bound<PySet>,
    _qubits: Option<Vec<usize>>,
    qubit: ShareableQubit,
    dag: &mut DAGCircuit, // I don't think a &Bound<PyAny> is required since even Python references a rust-native DAGCircuit
    property_set: &Bound<PyAny>,
    skip_reset_qubits: bool,
    _dd_sequence_lengths: HashMap<ShareableQubit, Vec<usize>>,
    _sequence_phase: &Bound<PyAny>,
    _dd_sequence: Option<Vec<OperationFromPython>>,
    prev_node_op_is_reset: bool,
    spacing: Vec<f64>, // I think an Option<Vec<f64>> isn't required because _pre_runhook sets default values
    extra_slack_distribution: &str,
) -> PyResult<()> {
    // This routine takes care of the pulse alignment constraint for the DD sequence.
    // Note that the alignment constraint acts on the t0 of the DAGOpNode.
    // Now this constrained scheduling problem is simplified to the problem of
    // finding a delay amount which is a multiple of the constraint value by assuming
    // that the duration of every DAGOpNode is also a multiple of the constraint value.

    // For example, given the constraint value of 16 and XY4 with 160 dt gates.
    // Here we assume current interval is 992 dt.

    // relative spacing := [0.125, 0.25, 0.25, 0.25, 0.125]
    // slack = 992 dt - 4 x 160 dt = 352 dt

    // unconstraind sequence: 44dt-X1-88dt-Y2-88dt-X3-88dt-Y4-44dt
    // constrained sequence  : 32dt-X1-80dt-Y2-80dt-X3-80dt-Y4-32dt + extra slack 48 dt

    // Now we evenly split extra slack into start and end of the sequence.
    // The distributed slack should be multiple of 16.
    // Start = +16, End += 32

    // final sequence       : 48dt-X1-80dt-Y2-80dt-X3-80dt-Y4-64dt / in total 992 dt

    // Now we verify t0 of every node starts from multiple of 16 dt.

    // X1:  48 dt (3 x 16 dt)
    // Y2:  48 dt + 160 dt + 80 dt = 288 dt (18 x 16 dt)
    // Y3: 288 dt + 160 dt + 80 dt = 528 dt (33 x 16 dt)
    // Y4: 368 dt + 160 dt + 80 dt = 768 dt (48 x 16 dt)

    // As you can see, constraints on t0 are all satisfied without explicit scheduling.

    let time_interval = t_end - t_start; //Leads to underflow during conversion
    if time_interval % alignment != 0 {
        return Err(TranspilerError::new_err(
            format!(
                "Time interval {time_interval} is not divisible by alignment {alignment} between prev_node and next_node."
            )
        ));
    }

    // Extract _no_dd_qubits into a HashSet
    let _no_dd_qubits: HashSet<usize> = _no_dd_qubits
        .iter()
        .map(|item| item.extract::<usize>())
        .collect::<PyResult<_>>()?;

    let qubit_index = dag
        .qubits()
        .objects()
        .iter()
        .position(|q| q == &qubit)
        .ok_or_else(|| TranspilerError::new_err("Qubit not found in dag.qubits".to_string()))?;

    if !__is_dd_qubit(qubit_index, &_no_dd_qubits, &_qubits) {
        // Target physical qubit is not the target of this DD sequence.
        apply_scheduled_delay_op(
            py,
            dag,
            &(t_start as f64),
            &(time_interval as f64),
            &qubit,
            property_set,
        )?;
        return Ok(());
    }

    if skip_reset_qubits && ((prev_node.is_instance_of::<DAGInNode>()) || prev_node_op_is_reset) {
        // Previous node is the start edge or reset, i.e. qubit is ground state
        apply_scheduled_delay_op(
            py,
            dag,
            &(t_start as f64),
            &(time_interval as f64),
            &qubit,
            property_set,
        )?;
        return Ok(());
    }

    let slack = time_interval
        - _dd_sequence_lengths
            .get(&qubit)
            .map_or(0, |lengths| lengths.iter().sum::<usize>());

    let _sequence_phase: f64 = _sequence_phase.extract()?;
    let mut sequence_gphase = _sequence_phase;

    if slack <= 0 {
        // Interval too short
        apply_scheduled_delay_op(
            py,
            dag,
            &(t_start as f64),
            &(time_interval as f64),
            &qubit,
            property_set,
        )?;
        return Ok(());
    }

    let dd_sequence: Vec<PackedOperation> = match &_dd_sequence {
        Some(seq) => seq.iter().map(|op| op.operation.clone()).collect(),
        None => vec![],
    };

    if dd_sequence.len() == 1 {
        // Special case of using a single gate for DD

        let gate_params: SmallVec<[Param; 3]> = match &_dd_sequence {
            Some(gate_seq) => gate_seq[0].params.clone(),
            None => smallvec![],
        };

        // Obtain the the inverse:
        let u_inv: Array2<Complex64> = match dd_sequence[0].view() {
            OperationRef::StandardGate(gate) => {
                if let Some((inv_gate, _inv_params)) = gate.inverse(&gate_params) {
                    let inv_packed_op = PackedOperation::from_standard_gate(inv_gate);
                    inv_packed_op
                        .matrix(&[])
                        .expect("No matrix representation for inverse gate.") // This is throwing a panic exception in some cases.... (why??; test_insert_midmeas_hahn_asap)
                } else {
                    panic!("No inverse for this standard gate!");
                }
            }
            OperationRef::Gate(py_gate) => {
                // Extract the inverse from the Python gate
                let inv_gate: PyObject = py_gate.gate.call_method0(py, "inverse")?;
                let inv_gate_rust: PackedOperation = inv_gate.extract::<OperationFromPython>(py).unwrap().operation.clone();
                let inv_matrix: Array2<Complex64> = inv_gate_rust
                    .matrix(&[])
                    .expect("No matrix representation for inverse gate.");
                inv_matrix
            }
            _ => unreachable!("Only Python Gate types (standard/non-standard) should be passed into this function."),
        };

        let [theta, phi, lam, phase] = angles_from_unitary(u_inv.view(), EulerBasis::U3);

        if next_node.is_instance_of::<DAGOpNode>() {
            let mut next_node_ref: PyRefMut<DAGOpNode> = next_node.extract()?;
            let next_node_ind = next_node_ref.as_ref().node.unwrap();
            let next_node_inst: &mut CircuitInstruction = &mut next_node_ref.instruction; //Replace the node in the DAG instead (or else, just call the python to update it manually...)
            let next_node_op: &PackedOperation = &next_node_inst.operation;

            // Check if the next node corresponds to a U/U3 gate
            if let Some(gate) = next_node_op.try_standard_gate() {
                match gate {
                    StandardGate::U | StandardGate::U3 => {
                        // Absorb the inverse into the successor (from right in circuit)
                        let [theta_r_param, phi_r_param, lam_r_param]: [Param; 3] =
                            next_node_inst.params.clone().into_inner().ok().unwrap();
                        let mut theta_r: f64 = 0.;
                        let mut phi_r: f64 = 0.;
                        let mut lam_r: f64 = 0.;
                        if let Param::Float(theta) = theta_r_param {
                            theta_r = theta;
                        }
                        if let Param::Float(phi) = phi_r_param {
                            phi_r = phi;
                        }
                        if let Param::Float(lam) = lam_r_param {
                            lam_r = lam;
                        }

                        let [theta_new, phi_new, lam_new] =
                            compose_u3_rust(theta_r, phi_r, lam_r, theta, phi, lam);

                        next_node_inst.params = smallvec![
                            Param::Float(theta_new),
                            Param::Float(phi_new),
                            Param::Float(lam_new),
                        ];

                        // Replace the node in the DAG
                        let op_obj = next_node_inst.get_operation(py).unwrap();
                        let op: &Bound<PyAny> = op_obj.bind(py);
                        dag.substitute_node_with_py_op(next_node_ind, op)?;
                        sequence_gphase += phase;
                    }
                    _ => {}
                }
            }
        } else if prev_node.is_instance_of::<DAGOpNode>() {
            let mut prev_node_ref: PyRefMut<DAGOpNode> = prev_node.extract()?;
            let prev_node_ind = prev_node_ref.as_ref().node.unwrap();
            let prev_node_inst: &mut CircuitInstruction = &mut prev_node_ref.instruction;
            let prev_node_op: &PackedOperation = &prev_node_inst.operation;

            // Check if the next node corresponds to a U/U3 gate
            if let Some(gate) = prev_node_op.try_standard_gate() {
                match gate {
                    StandardGate::U | StandardGate::U3 => {
                        // Absorb the inverse into the predecessor (from left in circuit)
                        let [theta_l_param, phi_l_param, lam_l_param]: [Param; 3] =
                            prev_node_inst.params.clone().into_inner().ok().unwrap();
                        let mut theta_l: f64 = 0.;
                        let mut phi_l: f64 = 0.;
                        let mut lam_l: f64 = 0.;
                        if let Param::Float(theta) = theta_l_param {
                            theta_l = theta;
                        }
                        if let Param::Float(phi) = phi_l_param {
                            phi_l = phi;
                        }
                        if let Param::Float(lam) = lam_l_param {
                            lam_l = lam;
                        }
                        let [theta_new, phi_new, lam_new] =
                            compose_u3_rust(theta, phi, lam, theta_l, phi_l, lam_l);

                        // Update the circuit instruction params:
                        prev_node_inst.params = smallvec![
                            Param::Float(theta_new),
                            Param::Float(phi_new),
                            Param::Float(lam_new),
                        ];

                        let op_obj = prev_node_inst.get_operation(py).unwrap();
                        let op: &Bound<PyAny> = op_obj.bind(py);
                        dag.substitute_node_with_py_op(prev_node_ind, op)?;

                        let node_start_time_obj = property_set.get_item("node_start_time")?;
                        let node_start_time_dict = node_start_time_obj.downcast::<PyDict>()?;
                        let start_time_opt = node_start_time_dict.get_item(prev_node).ok();
                        node_start_time_dict.del_item(prev_node).ok();
                        if let Some(start_time) = start_time_opt {
                            node_start_time_dict.set_item(prev_node, start_time)?;
                        }

                        sequence_gphase += phase;
                    }
                    _ => {}
                }
            }
        } else {
            // Don't do anything if there's no single-qubit gate to absorb the inverse
            apply_scheduled_delay_op(
                py,
                dag,
                &(t_start as f64),
                &(time_interval as f64),
                &qubit,
                property_set,
            )?;
            return Ok(());
        }
    }

    // (1) Compute DD intervals satisfying the constraint
    let mut taus: Array1<f64> = constrained_length(
        alignment as f64,
        &ArrayView1::from(&spacing).mapv(|v| slack as f64 * v).view(),
    );
    let extra_slack: f64 = slack as f64 - taus.sum();
    let taus_len = taus.len();

    // (2) Distribute extra slack
    if extra_slack_distribution == "middle" {
        let mid_ind: usize = (taus_len - 1) / 2;
        let to_middle: f64 = constrained_length_scalar(alignment as f64, extra_slack);
        taus[mid_ind] += to_middle;
        if (extra_slack - to_middle) != 0.0 {
            // If to_middle is not a multiple value of the pulse alignment,
            // it is truncated to the nearlest multiple value and
            // the rest of slack is added to the end.
            taus[taus_len - 1] += extra_slack - to_middle;
        }
    } else if extra_slack_distribution == "edges" {
        let to_begin_edge: f64 = constrained_length_scalar(alignment as f64, extra_slack / 2.0);
        taus[0] += to_begin_edge;
        taus[taus_len - 1] += extra_slack - to_begin_edge;
    } else {
        return Err(TranspilerError::new_err(format!(
            "Option extra_slack_distribution = {extra_slack_distribution} is invalid."
        )));
    }

    // (3) Construct DD sequence with delays
    let num_elements: usize = cmp::max(dd_sequence.len(), taus_len);
    let mut idle_after = t_start as f64;

    for dd_ind in 0..num_elements {
        if dd_ind < taus_len {
            let tau: f64 = taus[dd_ind];
            if tau > 0.0 {
                apply_scheduled_delay_op(py, dag, &idle_after, &tau, &qubit, property_set)?;
                idle_after += tau;
            }
        }
        if dd_ind < dd_sequence.len() {
            // Apply the DD gate at this position
            let gate = &dd_sequence[dd_ind];
            let gate_length = _dd_sequence_lengths
                .get(&qubit)
                .and_then(|v| v.get(dd_ind))
                .copied()
                .unwrap_or(0) as f64;

            let new_node = dag.apply_operation_back(
                gate.clone().into(),                     // PackedOperation -> Operation
                &[Qubit(qubit.index().unwrap() as u32)], //This is panicking and returing None sometimes (why??)
                &[],
                None,
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
            // Set the node start time in the property set
            let py_new_node = dag.get_node(py, new_node)?;
            let node_start_time_obj = property_set.get_item("node_start_time")?;
            let node_start_time_dict = node_start_time_obj.downcast::<PyDict>()?;
            node_start_time_dict.set_item(py_new_node, idle_after)?;
            idle_after += gate_length;
        }
    }

    if let Param::Float(curr_global_phase) = dag.get_global_phase() {
        dag.set_global_phase(Param::Float(curr_global_phase + sequence_gphase))?;
    }

    Ok(())
}

/// Checks if the qubit at ``qubit_index`` is a dynamical decoupling qubit.
fn __is_dd_qubit(
    qubit_index: usize,
    no_dd_qubits: &HashSet<usize>,
    _qubits: &Option<Vec<usize>>,
) -> bool {
    !no_dd_qubits.contains(&qubit_index)
        && _qubits.clone().map_or(true, |q| q.contains(&qubit_index))
}

/// May be better to add this as an ``impl`` for the ``DelayUnit``
fn map_delay_str_to_enum(delay_str: &str) -> DelayUnit {
    match delay_str {
        "ns" => DelayUnit::NS,
        "ps" => DelayUnit::PS,
        "us" => DelayUnit::US,
        "ms" => DelayUnit::MS,
        "s" => DelayUnit::S,
        "dt" => DelayUnit::DT,
        "expr" => DelayUnit::EXPR,
        _ => unreachable!(),
    }
}

/// Applies a delay operation to the DAGCircuit at the specified time interval.
/// It would make sense to replace this by a generic rust-native ``apply_scheduled_op``, once BOTH the ``PadDynamicalDecoupling`` and ``PadDelay`` passes (which inherit from ``BasePadding``)
/// are oxidized, so it can be called for both passes...
fn apply_scheduled_delay_op(
    py: Python, //If we want to avoid using the Python GIL (fully porting to Rust), we first need to port ``PropertySet`` to a rust-native type...
    dag: &mut DAGCircuit,
    t_start: &f64,
    time_interval: &f64,
    qubit: &ShareableQubit,
    property_set: &Bound<PyAny>,
) -> PyResult<()> {
    let delay_instr = StandardInstruction::Delay(map_delay_str_to_enum(
        &dag.get_internal_unit()
            .unwrap_or("dt".to_string())
            .to_string(),
    ));
    let params = Some(smallvec![Param::Float(*time_interval)]);
    let new_node = dag.apply_operation_back(
        delay_instr.into(),
        &[Qubit(qubit.index().unwrap() as u32)],
        &[],
        params,
        None,
        #[cfg(feature = "cache_pygates")]
        None,
    )?;

    let py_new_node = dag.get_node(py, new_node)?;
    let node_start_time_obj = property_set.get_item("node_start_time")?;
    let node_start_time_dict = node_start_time_obj.downcast::<PyDict>()?;
    node_start_time_dict.set_item(&py_new_node, t_start)?;

    Ok(())
}
// Maybe we could make these functions ``pub`` and move them to a ``pad_utils`` file to keep this file lean.

/// Function from ``qiskit_accelerate/src/optimize_1q_gates.rs`` and isn't imported due to circular import with this crate.
/// This has been copied to this file for temporary usage & testing, and is intended to be imported after being moved out of ``qiskit_accelerate``
fn compose_u3_rust(
    theta1: f64,
    phi1: f64,
    lambda1: f64,
    theta2: f64,
    phi2: f64,
    lambda2: f64,
) -> [f64; 3] {
    let q = [(theta1 / 2.0).cos(), 0., (theta1 / 2.0).sin(), 0.];
    let r = [
        ((lambda1 + phi2) / 2.0).cos(),
        0.,
        0.,
        ((lambda1 + phi2) / 2.0).sin(),
    ];
    let s = [(theta2 / 2.0).cos(), 0., (theta2 / 2.0).sin(), 0.];

    // Compute YZY decomp (q.r.s in variable names)
    let temp: [f64; 4] = [
        r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
        r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
        r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
        r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0],
    ];

    let out: [f64; 4] = [
        s[0] * temp[0] - s[1] * temp[1] - s[2] * temp[2] - s[3] * temp[3],
        s[0] * temp[1] + s[1] * temp[0] - s[2] * temp[3] + s[3] * temp[2],
        s[0] * temp[2] + s[1] * temp[3] + s[2] * temp[0] - s[3] * temp[1],
        s[0] * temp[3] - s[1] * temp[2] + s[2] * temp[1] + s[3] * temp[0],
    ];

    // out is now in YZY decomp, make into ZYZ
    let mat: [f64; 9] = [
        1. - 2. * out[2] * out[2] - 2. * out[3] * out[3],
        2. * out[1] * out[2] - 2. * out[3] * out[0],
        2. * out[1] * out[3] + 2. * out[2] * out[0],
        2. * out[1] * out[2] + 2. * out[3] * out[0],
        1. - 2. * out[1] * out[1] - 2. * out[3] * out[3],
        2. * out[2] * out[3] - 2. * out[1] * out[0],
        2. * out[1] * out[3] - 2. * out[2] * out[0],
        2. * out[2] * out[3] + 2. * out[1] * out[0],
        1. - 2. * out[1] * out[1] - 2. * out[2] * out[2],
    ];

    // Grab the euler angles
    let mut euler: [f64; 3] = if mat[8] < 1.0 {
        if mat[8] > -1.0 {
            [mat[5].atan2(mat[2]), (mat[8]).acos(), mat[7].atan2(-mat[6])]
        } else {
            [-1. * (mat[3].atan2(mat[4])), PI, 0.]
        }
    } else {
        [mat[3].atan2(mat[4]), 0., 0.]
    };
    euler
        .iter_mut()
        .filter(|k| k.abs() < 1e-15)
        .for_each(|k| *k = 0.0);

    let out_angles: [f64; 3] = [euler[1], phi1 + euler[0], lambda2 + euler[2]];
    out_angles
}

/// Though the following 2 functions can be implemented as a single function, it would lead to
/// a mild inefficiency for scalar values by converting back-and-forth between a ``f64`` and
/// ``ArrayView1<f64>`` or ``Array1<f64>`` when used above
fn constrained_length(alignment: f64, values: &ArrayView1<f64>) -> Array1<f64> {
    values.mapv(|v| alignment * (v / alignment).floor())
}

fn constrained_length_scalar(alignment: f64, value: f64) -> f64 {
    alignment * (value / alignment).floor()
}

pub fn pad_dynamical_decoupling_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_pad_dynamical_decoupling))?;
    Ok(())
}
