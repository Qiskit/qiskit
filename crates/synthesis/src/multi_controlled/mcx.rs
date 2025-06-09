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

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::{PyResult, Python};
use qiskit_circuit::circuit_data::{CircuitData, CircuitError};
use qiskit_circuit::imports;
use qiskit_circuit::operations::{
    multiply_param, Operation, OperationRef, Param, PyGate, StandardGate,
};
use qiskit_circuit::{Clbit, Qubit};
use smallvec::SmallVec;

use std::f64::consts::PI;
const PI2: f64 = PI / 2.0;

/// Definition circuit for CCX.
pub fn ccx() -> PyResult<CircuitData> {
    StandardGate::CCX
        .definition(&[])
        .ok_or(CircuitError::new_err(
            "Error extracting the definition of CCX",
        ))
}

/// Definition circuit for C3X.
#[pyfunction]
pub fn c3x() -> PyResult<CircuitData> {
    StandardGate::C3X
        .definition(&[])
        .ok_or(CircuitError::new_err(
            "Error extracting the definition of C3X",
        ))
}

/// Definition circuit for RCCX.
fn rccx() -> PyResult<CircuitData> {
    StandardGate::RCCX
        .definition(&[])
        .ok_or(CircuitError::new_err(
            "Error extracting the definition of RCCX",
        ))
}

/// Definition circuit for RC3X.
fn rc3x() -> PyResult<CircuitData> {
    StandardGate::RC3X
        .definition(&[])
        .ok_or(CircuitError::new_err(
            "Error extracting the definition of RC3X",
        ))
}

/// Definition circuit for C3SX.
fn c3sx() -> PyResult<CircuitData> {
    StandardGate::C3SX
        .definition(&[])
        .ok_or(CircuitError::new_err(
            "Error extracting the definition of C3SX",
        ))
}

/// Convenience methods to add gates to the circuit.
///
/// This trait is **not** intended to be user-facing. It defines utility functions
/// that make the code easier to read and that are used only for synthesis.
trait CircuitDataForSynthesis {
    /// Appends H to the circuit.
    fn h(&mut self, q: u32);

    /// Appends X to the circuit.
    #[allow(dead_code)]
    fn x(&mut self, q: u32);

    /// Appends T to the circuit.
    fn t(&mut self, q: u32);

    /// Appends Tdg to the circuit.
    fn tdg(&mut self, q: u32);

    /// Appends Phase to the circuit.
    #[allow(dead_code)]
    fn p(&mut self, theta: f64, q: u32);

    /// Appends CX to the circuit.
    fn cx(&mut self, q1: u32, q2: u32);

    /// Appends CPhase to the circuit.
    fn cp(&mut self, theta: f64, q1: u32, q2: u32);

    /// Appends CCX to the circuit.
    fn ccx(&mut self, q1: u32, q2: u32, q3: u32) -> PyResult<()>;

    /// Appends RCCX to the circuit.
    fn rccx(&mut self, q1: u32, q2: u32, q3: u32) -> PyResult<()>;

    /// Compose ``other`` into ``self``, while remapping the qubits
    /// over which ``other`` is defined. The operations are added in-place.
    fn compose(&mut self, other: &Self, qargs_map: &[Qubit], cargs_map: &[Clbit]) -> PyResult<()>;

    /// Construct the inverse circuit
    fn inverse(&self) -> PyResult<CircuitData>;
}

impl CircuitDataForSynthesis for CircuitData {
    /// Appends H to the circuit.
    #[inline]
    fn h(&mut self, q: u32) {
        self.push_standard_gate(StandardGate::H, &[], &[Qubit(q)]);
    }

    /// Appends X to the circuit.
    #[inline]
    fn x(&mut self, q: u32) {
        self.push_standard_gate(StandardGate::X, &[], &[Qubit(q)]);
    }

    /// Appends T to the circuit.
    #[inline]
    fn t(&mut self, q: u32) {
        self.push_standard_gate(StandardGate::T, &[], &[Qubit(q)]);
    }

    /// Appends Tdg to the circuit.
    #[inline]
    fn tdg(&mut self, q: u32) {
        self.push_standard_gate(StandardGate::Tdg, &[], &[Qubit(q)]);
    }

    /// Appends Phase to the circuit.
    #[inline]
    fn p(&mut self, theta: f64, q: u32) {
        self.push_standard_gate(StandardGate::Phase, &[Param::Float(theta)], &[Qubit(q)]);
    }

    /// Appends CX to the circuit.
    #[inline]
    fn cx(&mut self, q1: u32, q2: u32) {
        self.push_standard_gate(StandardGate::CX, &[], &[Qubit(q1), Qubit(q2)]);
    }

    /// Appends CPhase to the circuit.
    #[inline]
    fn cp(&mut self, theta: f64, q1: u32, q2: u32) {
        self.push_standard_gate(
            StandardGate::CPhase,
            &[Param::Float(theta)],
            &[Qubit(q1), Qubit(q2)],
        );
    }

    /// Appends the decomposition of the CCX to the circuit.
    fn ccx(&mut self, q1: u32, q2: u32, q3: u32) -> PyResult<()> {
        self.compose(&ccx()?, &[Qubit(q1), Qubit(q2), Qubit(q3)], &[])
    }

    /// Appends RCCX to the circuit.
    fn rccx(&mut self, q1: u32, q2: u32, q3: u32) -> PyResult<()> {
        self.compose(&rccx()?, &[Qubit(q1), Qubit(q2), Qubit(q3)], &[])
    }

    /// Compose ``other`` into ``self``, while remapping the qubits over which ``other`` is defined.
    /// The operations are added in-place.
    fn compose(&mut self, other: &Self, qargs_map: &[Qubit], cargs_map: &[Clbit]) -> PyResult<()> {
        for inst in other.data() {
            let remapped_qubits: Vec<Qubit> = other
                .get_qargs(inst.qubits)
                .iter()
                .map(|q| qargs_map[q.index()])
                .collect();
            let remapped_clbits: Vec<Clbit> = other
                .get_cargs(inst.clbits)
                .iter()
                .map(|c| cargs_map[c.index()])
                .collect();

            self.push_packed_operation(
                inst.op.clone(),
                inst.params_view(),
                &remapped_qubits,
                &remapped_clbits,
            );
        }

        self.add_global_phase(other.global_phase())?;
        Ok(())
    }

    /// Construct the inverse circuit
    fn inverse(&self) -> PyResult<CircuitData> {
        let inverse_global_phase =
            Python::with_gil(|py| -> Param { multiply_param(self.global_phase(), -1.0, py) });

        let mut inverse_circuit = CircuitData::clone_empty_like(self, None)?;
        inverse_circuit.set_global_phase(inverse_global_phase)?;

        let data = self.data();

        for i in 0..data.len() {
            let inst = &data[data.len() - 1 - i];

            let inverse_inst: Option<(StandardGate, SmallVec<[Param; 3]>)> = match &inst.op.view() {
                OperationRef::StandardGate(gate) => gate.inverse(inst.params_view()),
                _ => None,
            };

            if inverse_inst.is_none() {
                return Err(CircuitError::new_err(format!(
                    "The circuit cannot be inverted: {} is not a standard gate.",
                    inst.op.name()
                )));
            }

            let (inverse_op, inverse_op_params) = inverse_inst.unwrap();

            inverse_circuit.push_packed_operation(
                inverse_op.into(),
                &inverse_op_params,
                self.get_qargs(inst.qubits),
                self.get_cargs(inst.clbits),
            );
        }
        Ok(inverse_circuit)
    }
}

/// Efficient synthesis for 4-controlled X-gate.
#[pyfunction]
pub fn c4x() -> PyResult<CircuitData> {
    let mut circuit = CircuitData::with_capacity(5, 0, 0, Param::Float(0.0))?;
    circuit.h(4);
    circuit.cp(PI2, 3, 4);
    circuit.h(4);
    circuit.compose(&rc3x()?, &[Qubit(0), Qubit(1), Qubit(2), Qubit(3)], &[])?;
    circuit.h(4);
    circuit.cp(-PI2, 3, 4);
    circuit.h(4);
    circuit.compose(
        &rc3x()?.inverse()?,
        &[Qubit(0), Qubit(1), Qubit(2), Qubit(3)],
        &[],
    )?;
    circuit.compose(&c3sx()?, &[Qubit(0), Qubit(1), Qubit(2), Qubit(4)], &[])?;
    Ok(circuit)
}

/// Adds gates of the "action gadget" to the circuit
fn add_action_gadget(circuit: &mut CircuitData, q0: u32, q1: u32, q2: u32) {
    circuit.h(q2);
    circuit.t(q2);
    circuit.cx(q0, q2);
    circuit.tdg(q2);
    circuit.cx(q1, q2);
}

/// Adds gates of the "reset gadget" to the circuit
fn add_reset_gadget(circuit: &mut CircuitData, q0: u32, q1: u32, q2: u32) {
    circuit.cx(q1, q2);
    circuit.t(q2);
    circuit.cx(q0, q2);
    circuit.tdg(q2);
    circuit.h(q2);
}

/// Synthesize a multi-controlled X gate with :math:`k` controls based on the paper
/// by Iten et al. [1].
///
/// For :math:`k\ge 4` the method uses :math:`k - 2` dirty ancillary qubits, producing a circuit
/// with :math:`2 * k - 1` qubits and at most :math:`8 * k - 6` CX gates. For :math:`k\le 3`
/// explicit efficient circuits are used instead.
///
/// # Arguments
/// - num_controls: the number of control qubits.
/// - relative_phase: when set to `true`, the method applies the optimized multi-controlled
///   X gate up to a relative phase, in a way that the relative phases of the `action part`
///   cancel out with the relative phases of the `reset part`.
/// - action_only: when set to `true`, the methods applies only the `action part`.
///
/// # References
///
/// 1. Iten et al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
///    [arXiv:1501.06911] (http://arxiv.org/abs/1501.06911).
pub fn synth_mcx_n_dirty_i15(
    num_controls: usize,
    relative_phase: bool,
    action_only: bool,
) -> PyResult<CircuitData> {
    if num_controls == 1 {
        let mut circuit = CircuitData::with_capacity(2, 0, 1, Param::Float(0.0))?;
        circuit.cx(0, 1);
        Ok(circuit)
    } else if num_controls == 2 {
        ccx()
    } else if num_controls == 3 && !relative_phase {
        c3x()
    } else {
        let num_ancillas = num_controls - 2;
        let num_qubits = num_controls + 1 + num_ancillas;
        let mut circuit = CircuitData::with_capacity(num_qubits as u32, 0, 0, Param::Float(0.0))?;

        let controls: Vec<u32> = (0..num_controls).map(|q| q as u32).collect();
        let target = num_controls as u32;
        let ancillas: Vec<u32> = ((num_controls + 1)..num_qubits).map(|q| q as u32).collect();

        for j in 0..2 {
            if !relative_phase {
                circuit.ccx(
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
            } else if j == 0 {
                add_action_gadget(
                    &mut circuit,
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                );
            } else if j == 1 {
                add_reset_gadget(
                    &mut circuit,
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                );
            }

            // action part
            for i in (0..num_controls - 3).rev() {
                add_action_gadget(&mut circuit, controls[i + 2], ancillas[i], ancillas[i + 1]);
            }

            circuit.rccx(controls[0], controls[1], ancillas[0])?;

            // reset part
            for i in 0..num_controls - 3 {
                add_reset_gadget(&mut circuit, controls[i + 2], ancillas[i], ancillas[i + 1]);
            }

            if action_only {
                circuit.ccx(
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
                break;
            }
        }
        Ok(circuit)
    }
}

/// Synthesize a multi-controlled X gate with :math:`k` controls based on
/// the implementation for `MCPhaseGate`.
///
/// In turn, the MCPhase gate uses the decomposition for multi-controlled
/// special unitaries described in [1].
///
/// # Arguments
/// - num_controls: the number of control qubits.
///
/// # Returns
///
/// A quantum circuit with :math:`k + 1` qubits. The number of CX-gates is
/// quadratic in :math:`k`.
///
/// # References
///
/// 1. Vale et. al., *Circuit Decomposition of Multicontrolled Special Unitary
///    Single-Qubit Gates*, IEEE TCAD 43(3) (2024),
///    [arXiv:2302.06377] (https://arxiv.org/abs/2302.06377).
pub fn synth_mcx_noaux_v24(py: Python, num_controls: usize) -> PyResult<CircuitData> {
    if num_controls == 3 {
        c3x()
    } else if num_controls == 4 {
        c4x()
    } else {
        let num_qubits = (num_controls + 1) as u32;
        let target = num_controls as u32;

        let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;
        circuit.h(target);

        let mcphase_cls = imports::MCPHASE_GATE.get_bound(py);
        let mcphase_gate = mcphase_cls.call1((PI, num_controls))?;

        let as_py_gate = PyGate {
            qubits: num_qubits,
            clbits: 0,
            params: 1,
            op_name: "mcphase".to_string(),
            gate: mcphase_gate.into(),
        };

        circuit.push_packed_operation(
            as_py_gate.into(),
            &[],
            &(0..num_qubits).map(Qubit).collect::<Vec<Qubit>>(),
            &[],
        );

        circuit.h(target);

        Ok(circuit)
    }
}
