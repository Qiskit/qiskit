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
use qiskit_circuit::Qubit;
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
use qiskit_circuit::parameter::parameter_expression::{ParameterExpression, ParameterValueType};
use qiskit_synthesis::euler_one_qubit_decomposer::{EulerBasis, angles_from_unitary};
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{SmallVec, smallvec};
use std::cmp;
use std::f64::consts::PI;

#[pyfunction]
#[pyo3(name = "pad_dynamical_decoupling")]
pub fn run_pad_dynamical_decoupling(
    py: Python,
    t_end: f64,
    t_start: f64,
    alignment: f64,
    prev_node: &Bound<PyAny>,
    next_node: &Bound<PyAny>,
    _no_dd_qubits: &Bound<PySet>,
    _qubits: Option<Vec<usize>>,
    qubit: ShareableQubit,
    dag: &mut DAGCircuit,
    property_set: &Bound<PyAny>,
    skip_reset_qubits: bool,
    _dd_sequence_lengths: HashMap<ShareableQubit, Vec<usize>>,
    _sequence_phase: &Bound<PyAny>,
    _dd_sequence: Option<Vec<OperationFromPython>>,
    prev_node_op_is_reset: bool,
    spacing: Vec<f64>,
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

    let time_interval = t_end - t_start;
    if time_interval % alignment != 0.0 {
        return Err(TranspilerError::new_err(format!(
            "Time interval {} is not divisible by alignment {}",
            time_interval, alignment
        )));
    }

    let _no_dd_qubits: HashSet<usize> = _no_dd_qubits
        .iter()
        .map(|item| item.extract())
        .collect::<PyResult<_>>()?;
    let qubit_index = qubit
        .index()
        .ok_or_else(|| TranspilerError::new_err("Qubit not found in dag.qubits".to_string()))?;

    if !__is_dd_qubit(qubit_index, &_no_dd_qubits, &_qubits) {
        apply_scheduled_delay_op(py, dag, &t_start, &time_interval, &qubit, property_set)?;
        return Ok(());
    }

    if skip_reset_qubits && (prev_node.is_instance_of::<DAGInNode>() || prev_node_op_is_reset) {
        apply_scheduled_delay_op(py, dag, &t_start, &time_interval, &qubit, property_set)?;
        return Ok(());
    }

    let slack = time_interval
        - (_dd_sequence_lengths
            .get(&qubit)
            .map_or(0.0, |lengths| lengths.iter().sum::<usize>() as f64));
    let mut sequence_gphase: f64 = _sequence_phase.extract()?;

    if slack <= 0.0 {
        apply_scheduled_delay_op(py, dag, &t_start, &time_interval, &qubit, property_set)?;
        return Ok(());
    }

    // Process the DD sequence only if it exists
    if let Some(dd_instructions) = &_dd_sequence {
        // --- SINGLE-GATE DD BLOCK ---
        if dd_instructions.len() == 1 {
            let instruction = &dd_instructions[0];
            let gate_op = &instruction.operation;
            let gate_params = &instruction.params;

            //cTry to get the inverse operation.
            let inv_op_res: PyResult<Option<(PackedOperation, SmallVec<[Param; 3]>)>> =
                match gate_op.view() {
                    OperationRef::StandardGate(gate) => {
                        Ok(gate.inverse(gate_params).map(|(inv_gate, inv_params)| {
                            (PackedOperation::from_standard_gate(inv_gate), inv_params)
                        }))
                    }
                    OperationRef::Gate(py_gate) => {
                        let inv_gate_py: Py<PyAny> =
                            py_gate.gate.call_method(py, "inverse", (), None)?;
                        inv_gate_py
                            .extract::<OperationFromPython>(py)
                            .map(|inv_op_from_py| {
                                Some((inv_op_from_py.operation, inv_op_from_py.params))
                            })
                    }
                    _ => Ok(None), // Not a gate
                };

            // Safely handle potential error during inverse calculation (e.g. Python exception)
            let inv_op_tuple_opt = inv_op_res.unwrap_or(None); // Default to None on PyErr

            // If we got an inverse, try to get its matrix using correct parameters.
            if let Some((inv_op, inv_params)) = inv_op_tuple_opt {
                if let Some(u_inv) = inv_op.matrix(&inv_params) {
                    // --- SUCCESS: We have the matrix, proceed with absorption. ---
                    let [theta, phi, lam, phase] =
                        angles_from_unitary(u_inv.view(), EulerBasis::U3);
                    let inverse_phase_shift = phase;
                    let mut absorbed = false;

                    let next_node_ind_opt: Option<NodeIndex> =
                        if next_node.is_instance_of::<DAGOpNode>() {
                            next_node
                                .extract::<PyRef<DAGOpNode>>()
                                .ok()
                                .and_then(|node_ref| node_ref.as_ref().node)
                                .filter(|&node_ind| dag.dag().contains_node(node_ind))
                        } else {
                            None
                        };

                    // Extract node index from prev_node, verifying it exists in the DAG
                    let prev_node_ind_opt: Option<NodeIndex> =
                        if prev_node.is_instance_of::<DAGOpNode>() {
                            prev_node
                                .extract::<PyRef<DAGOpNode>>()
                                .ok()
                                .and_then(|node_ref| node_ref.as_ref().node)
                                .filter(|&node_ind| dag.dag().contains_node(node_ind))
                        } else {
                            None
                        };

                    // --- Check next_node ---
                    if let Some(next_node_ind) = next_node_ind_opt {
                        if let Ok(next_node_ref) = next_node.extract::<PyRef<DAGOpNode>>() {
                            let next_node_inst = &next_node_ref.instruction;
                            if let Some(gate) = next_node_inst.operation.try_standard_gate() {
                                if matches!(gate, StandardGate::U | StandardGate::U3) {
                                    if let Ok(params) = next_node_inst.params.clone().into_inner() {
                                        let theta_r = if let Param::Float(val) = params[0] {
                                            val
                                        } else {
                                            0.0
                                        };
                                        let phi_r = if let Param::Float(val) = params[1] {
                                            val
                                        } else {
                                            0.0
                                        };
                                        let lam_r = if let Param::Float(val) = params[2] {
                                            val
                                        } else {
                                            0.0
                                        };
                                        let [theta_new, phi_new, lam_new] =
                                            compose_u3_rust(theta_r, phi_r, lam_r, theta, phi, lam);

                                        let new_instruction = CircuitInstruction {
                                            operation: next_node_inst.operation.clone(),
                                            qubits: next_node_inst.qubits.clone_ref(py),
                                            clbits: next_node_inst.clbits.clone_ref(py),
                                            label: next_node_inst.label.clone(),
                                            params: smallvec![
                                                Param::Float(theta_new),
                                                Param::Float(phi_new),
                                                Param::Float(lam_new)
                                            ],
                                            #[cfg(feature = "cache_pygates")]
                                            py_op: std::sync::OnceLock::new(),
                                        };
                                        let new_op_obj = new_instruction.get_operation(py)?;
                                        // Mutate the instruction in-place instead of substituting the node.
                                        // This preserves the node object as a key in the property_set.
                                        dag.substitute_node_with_py_op(
                                            next_node_ind,
                                            new_op_obj.bind(py),
                                        )?;
                                        sequence_gphase += inverse_phase_shift;
                                        absorbed = true;
                                    }
                                }
                            }
                        }
                    }
                    // --- Check prev_node ---
                    if !absorbed {
                        if let Some(prev_node_ind) = prev_node_ind_opt {
                            if let Ok(prev_node_ref) = prev_node.extract::<PyRef<DAGOpNode>>() {
                                let prev_node_inst = &prev_node_ref.instruction;
                                if let Some(gate) = prev_node_inst.operation.try_standard_gate() {
                                    if matches!(gate, StandardGate::U | StandardGate::U3) {
                                        if let Ok(params) =
                                            prev_node_inst.params.clone().into_inner()
                                        {
                                            let theta_l = if let Param::Float(val) = params[0] { val } else { 0.0 };
                                            let phi_l = if let Param::Float(val) = params[1] { val } else { 0.0 };
                                            let lam_l = if let Param::Float(val) = params[2] { val } else { 0.0 };
                                            
                                            let [theta_new, phi_new, lam_new] = compose_u3_rust(
                                                theta, phi, lam, theta_l, phi_l, lam_l,
                                            );

                                            let new_instruction = CircuitInstruction {
                                                operation: prev_node_inst.operation.clone(),
                                                qubits: prev_node_inst.qubits.clone_ref(py),
                                                clbits: prev_node_inst.clbits.clone_ref(py),
                                                label: prev_node_inst.label.clone(),
                                                params: smallvec![
                                                    Param::Float(theta_new),
                                                    Param::Float(phi_new),
                                                    Param::Float(lam_new),
                                                ],
                                                #[cfg(feature = "cache_pygates")]
                                                py_op: std::sync::OnceLock::new(),
                                            };
                                            let new_op_obj = new_instruction.get_operation(py)?;

                                            // Substitute the node (inplace=False logic)
                                            dag.substitute_node_with_py_op(
                                                prev_node_ind,
                                                new_op_obj.bind(py),
                                            )?;

                                            // Update property_set
                                            if let Ok(start_times) =
                                                property_set.get_item("node_start_time")
                                            {
                                                if let Ok(start_times_dict) =
                                                    start_times.downcast::<PyDict>()
                                                {
                                                    if let Some(start_time) =
                                                        start_times_dict.get_item(prev_node)?
                                                    {
                                                        let old_node_key = prev_node;
                                                        let new_node_obj =
                                                            dag.get_node(py, prev_node_ind)?;
                                                        start_times_dict.del_item(old_node_key)?;
                                                        start_times_dict
                                                            .set_item(new_node_obj, start_time)?;
                                                    }
                                                }
                                            }

                                            sequence_gphase += inverse_phase_shift;
                                            absorbed = true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // --- Fallback if absorption didn't happen ---
                    if !absorbed {
                        apply_scheduled_delay_op(
                            py,
                            dag,
                            &t_start,
                            &time_interval,
                            &qubit,
                            property_set,
                        )?;
                        return Ok(());
                    }
                } else {
                    // Matrix calculation failed
                    apply_scheduled_delay_op(
                        py,
                        dag,
                        &t_start,
                        &time_interval,
                        &qubit,
                        property_set,
                    )?;
                    return Ok(());
                }
            } else {
                // Inverse calculation failed
                apply_scheduled_delay_op(py, dag, &t_start, &time_interval, &qubit, property_set)?;
                return Ok(());
            }
        }
    }

    // --- If _dd_sequence was None OR len > 1, proceed with multi-gate logic or simple padding ---
    let mut taus: Array1<f64> = constrained_length(
        alignment,
        &ArrayView1::from(&spacing).mapv(|v| slack * v).view(),
    );
    let extra_slack: f64 = slack - taus.sum();
    let taus_len = taus.len();

    // Distribute extra slack
    if extra_slack_distribution == "middle" {
        let mid_ind: usize = (taus_len - 1) / 2;
        let to_middle: f64 = constrained_length_scalar(alignment, extra_slack);
        taus[mid_ind] += to_middle;
        if (extra_slack - to_middle) != 0.0 {
            taus[taus_len - 1] += extra_slack - to_middle;
        }
    } else if extra_slack_distribution == "edges" {
        let to_begin_edge: f64 = constrained_length_scalar(alignment, extra_slack / 2.0);
        taus[0] += to_begin_edge;
        taus[taus_len - 1] += extra_slack - to_begin_edge;
    } else {
        return Err(TranspilerError::new_err(format!(
            "Option extra_slack_distribution = {extra_slack_distribution} is invalid."
        )));
    }

    // Construct DD sequence with delays
    let num_dd_gates = _dd_sequence.as_ref().map_or(0, |s| s.len());
    let num_elements = cmp::max(num_dd_gates, taus_len);
    let mut idle_after = t_start;

    for dd_ind in 0..num_elements {
        if dd_ind < taus_len {
            let tau: f64 = taus[dd_ind];
            if tau > 0.0 {
                apply_scheduled_delay_op(py, dag, &idle_after, &tau, &qubit, property_set)?;
                idle_after += tau;
            }
        }
        if dd_ind < num_dd_gates {
            let full_instruction = &_dd_sequence.as_ref().unwrap()[dd_ind];
            let gate = &full_instruction.operation;
            let params = &full_instruction.params;

            let gate_length = _dd_sequence_lengths
                .get(&qubit)
                .and_then(|v| v.get(dd_ind))
                .copied()
                .unwrap_or(0) as f64;

            let qubit_idx = qubit
                .index()
                .ok_or_else(|| TranspilerError::new_err("DD qubit has no index."))?;

            let new_node = dag.apply_operation_back(
                gate.clone().into(),
                &[Qubit(qubit_idx as u32)],
                &[],
                Some(params.clone()),
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

    // --- Global Phase Update (Using Native Rust Add) ---
    let current_phase = dag.get_global_phase();
    let new_phase = match current_phase {
        Param::Float(val) => Param::Float(val + sequence_gphase),
        Param::ParameterExpression(py_obj) => {
            // py_obj is Arc<ParameterExpression>
            let rhs_value = ParameterValueType::Float(sequence_gphase);
            let rhs_expr = ParameterExpression::from(rhs_value);
            let new_expr_arc = py_obj.add(&rhs_expr)?; // add returns Result<Arc<...>>
            Param::ParameterExpression(new_expr_arc.into())
        }
        Param::Obj(_) => {
            return Err(TranspilerError::new_err(
                "Cannot add to global phase: phase has an invalid type 'Obj'.",
            ));
        }
    };
    dag.set_global_phase(new_phase)?;

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
        &[Qubit(
            qubit
                .index()
                .ok_or_else(|| TranspilerError::new_err("Delay qubit has no index."))?
                as u32,
        )],
        &[],
        params,
        None,
        #[cfg(feature = "cache_pygates")]
        None,
    )?;

    let py_new_node = dag.get_node(py, new_node)?;
    let node_start_time_obj = property_set.get_item("node_start_time")?;
    let node_start_time_dict = node_start_time_obj.downcast::<PyDict>()?;
    node_start_time_dict.set_item(py_new_node, t_start)?;

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
