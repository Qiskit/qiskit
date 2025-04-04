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

use pyo3::types::PyAnyMethods;
use pyo3::{PyResult, Python};
use qiskit_circuit::imports;
use qiskit_circuit::operations::PyGate;
use qiskit_circuit::{circuit_data::CircuitData, operations::Param, Qubit};

use std::f64::consts::PI;
// const PI2: f64 = PI / 2.;
// const PI4: f64 = PI / 4.;
const PI8: f64 = PI / 8.;

pub fn ccx() -> PyResult<CircuitData> {
    let mut circuit = CircuitData::with_capacity(3, 0, 15, Param::Float(0.0))?;
    circuit.h(2);
    circuit.cx(1, 2);
    circuit.tdg(2);
    circuit.cx(0, 2);
    circuit.t(2);
    circuit.cx(1, 2);
    circuit.tdg(2);
    circuit.cx(0, 2);
    circuit.t(1);
    circuit.t(2);
    circuit.h(2);
    circuit.cx(0, 1);
    circuit.t(0);
    circuit.tdg(1);
    circuit.cx(0, 1);
    Ok(circuit)
}

/// Implements an optimized toffoli operation up to a diagonal gate,
/// akin to lemma 6 of [arXiv:1501.06911] (https://arxiv.org/abs/1501.06911).
fn rccx() -> PyResult<CircuitData> {
    let mut circuit = CircuitData::with_capacity(3, 0, 9, Param::Float(0.0))?;
    circuit.h(2);
    circuit.t(2);
    circuit.cx(1, 2);
    circuit.tdg(2);
    circuit.cx(0, 2);
    circuit.t(2);
    circuit.cx(1, 2);
    circuit.tdg(2);
    circuit.h(2);
    Ok(circuit)
}

/// Efficient synthesis for 3-controlled X-gate.
pub fn c3x() -> PyResult<CircuitData> {
    let mut circuit = CircuitData::with_capacity(4, 0, 31, Param::Float(0.0))?;
    circuit.h(3);
    circuit.p(PI8, 0);
    circuit.p(PI8, 1);
    circuit.p(PI8, 2);
    circuit.p(PI8, 3);
    circuit.cx(0, 1);
    circuit.p(-PI8, 1);
    circuit.cx(0, 1);
    circuit.cx(1, 2);
    circuit.p(-PI8, 2);
    circuit.cx(0, 2);
    circuit.p(PI8, 2);
    circuit.cx(1, 2);
    circuit.p(-PI8, 2);
    circuit.cx(0, 2);
    circuit.cx(2, 3);
    circuit.p(-PI8, 3);
    circuit.cx(1, 3);
    circuit.p(PI8, 3);
    circuit.cx(2, 3);
    circuit.p(-PI8, 3);
    circuit.cx(0, 3);
    circuit.p(PI8, 3);
    circuit.cx(2, 3);
    circuit.p(-PI8, 3);
    circuit.cx(1, 3);
    circuit.p(PI8, 3);
    circuit.cx(2, 3);
    circuit.p(-PI8, 3);
    circuit.cx(0, 3);
    circuit.h(3);
    Ok(circuit)
}

// /// Standard definition for RC3XGate
// // ToDo: maybe we should instead introduce from_definition?
// rules = [
//             (U2Gate(0, pi), [q[3]], []),  # H gate
//             (U1Gate(pi / 4), [q[3]], []),  # T gate
//             (CXGate(), [q[2], q[3]], []),
//             (U1Gate(-pi / 4), [q[3]], []),  # inverse T gate
//             (U2Gate(0, pi), [q[3]], []),
//             (CXGate(), [q[0], q[3]], []),
//             (U1Gate(pi / 4), [q[3]], []),
//             (CXGate(), [q[1], q[3]], []),
//             (U1Gate(-pi / 4), [q[3]], []),
//             (CXGate(), [q[0], q[3]], []),
//             (U1Gate(pi / 4), [q[3]], []),
//             (CXGate(), [q[1], q[3]], []),
//             (U1Gate(-pi / 4), [q[3]], []),
//             (U2Gate(0, pi), [q[3]], []),
//             (U1Gate(pi / 4), [q[3]], []),
//             (CXGate(), [q[2], q[3]], []),
//             (U1Gate(-pi / 4), [q[3]], []),
//             (U2Gate(0, pi), [q[3]], []),
//         ]

// /// Efficient synthesis for 4-controlled X-gate.
// pub fn c4x<'a>() -> CircuitData<'a> {
//     let mut circuit = CircuitData::new(5);
//     circuit.h(4);
//     circuit.cu1(PI2, 3, 4);
//     circuit.h(4);
// }

//     rules = [
//         (RC3XGate(), [q[0], q[1], q[2], q[3]], []),
//         (HGate(), [q[4]], []),
//         (CU1Gate(-np.pi / 2), [q[3], q[4]], []),
//         (HGate(), [q[4]], []),
//         (RC3XGate().inverse(), [q[0], q[1], q[2], q[3]], []),
//         (C3SXGate(), [q[0], q[1], q[2], q[4]], []),
//     ]
//     for instr, qargs, cargs in rules:
//         qc._append(instr, qargs, cargs)

//     return qc

/// A block in the `action part`, see Iten et al.
fn action_gadget() -> PyResult<CircuitData> {
    let mut circuit = CircuitData::with_capacity(3, 0, 5, Param::Float(0.0))?;
    circuit.h(2);
    circuit.t(2);
    circuit.cx(0, 2);
    circuit.tdg(2);
    circuit.cx(1, 2);
    Ok(circuit)
}

/// A block in the `reset part`, see Iten et al.
fn reset_gadget() -> PyResult<CircuitData> {
    let mut circuit = CircuitData::with_capacity(3, 0, 5, Param::Float(0.0))?;
    circuit.cx(1, 2);
    circuit.t(2);
    circuit.cx(0, 2);
    circuit.tdg(2);
    circuit.h(2);
    Ok(circuit)
}

/// Synthesize a multi-controlled X gate with :math:`k` controls using :math:`k - 2`
/// dirty ancillary qubits, producing a circuit with :math:`2 * k - 1` qubits and at most
/// :math:`8 * k - 6` CX gates. For details, see lemma 8 in [1].
///
/// # Arguments
/// - num_ctrl_qubits: the number of control qubits.
/// - relative_phase: when set to `true`, the method applies the optimized multi-controlled
///   X gate up to a relative phase, in a way that the relative phases of the `action part`
///   cancel out with the relative phases of the `reset part`.
/// - action_only: when set to `true`, the methods applies only the `action part`.
///
/// # References
///
/// 1. Iten et al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
/// [arXiv:1501.06911] (http://arxiv.org/abs/1501.06911).
pub fn synth_mcx_n_dirty_i15(
    num_controls: usize,
    relative_phase: bool,
    action_only: bool,
) -> PyResult<CircuitData> {
    println!("I AM IN RUST 2!!");
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
                circuit.compose(
                    &ccx()?,
                    Some(&[
                        Qubit(controls[num_controls - 1]),
                        Qubit(ancillas[num_controls - 3]),
                        Qubit(target),
                    ]),
                )?;
            } else if j == 0 {
                circuit.compose(
                    &action_gadget()?,
                    Some(&[
                        Qubit(controls[num_controls - 1]),
                        Qubit(ancillas[num_controls - 3]),
                        Qubit(target),
                    ]),
                )?;
            } else if j == 1 {
                circuit.compose(
                    &reset_gadget()?,
                    Some(&[
                        Qubit(controls[num_controls - 1]),
                        Qubit(ancillas[num_controls - 3]),
                        Qubit(target),
                    ]),
                )?;
            }

            // action part
            for i in (0..num_controls - 3).rev() {
                circuit.compose(
                    &action_gadget()?,
                    Some(&[
                        Qubit(controls[i + 2]),
                        Qubit(ancillas[i]),
                        Qubit(ancillas[i + 1]),
                    ]),
                )?;
            }

            circuit.compose(
                &rccx()?,
                Some(&[Qubit(controls[0]), Qubit(controls[1]), Qubit(ancillas[0])]),
            )?;

            // reset part
            for i in 0..num_controls - 3 {
                circuit.compose(
                    &reset_gadget()?,
                    Some(&[
                        Qubit(controls[i + 2]),
                        Qubit(ancillas[i]),
                        Qubit(ancillas[i + 1]),
                    ]),
                )?;
            }

            if action_only {
                circuit.compose(
                    &ccx()?,
                    Some(&[
                        Qubit(controls[num_controls - 1]),
                        Qubit(ancillas[num_controls - 3]),
                        Qubit(target),
                    ]),
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
/// - num_ctrl_qubits: the number of control qubits.
///
/// # Returns
///
/// A quantum circuit with :math:`k + 1` qubits. The number of CX-gates is
/// quadratic in :math:`k`.
///
/// # References
///
/// 1. Vale et. al., *Circuit Decomposition of Multicontrolled Special Unitary
/// Single-Qubit Gates*, IEEE TCAD 43(3) (2024),
/// [arXiv:2302.06377] (https://arxiv.org/abs/2302.06377).
pub fn synth_mcx_noaux_v24(py: Python, num_controls: usize) -> PyResult<CircuitData> {
    println!("I AM IN RUST FOR V24 2");
    // ToDo: should we return Result?
    if num_controls == 3 {
        c3x()
    }
    // restore me!
    // else if num_controls == 4 {
    //     c4x()
    // }
    else {
        let num_qubits = (num_controls + 1) as u32;
        let target = num_controls as u32;

        let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;
        circuit.h(target);

        let mcphase_cls = imports::MCPHASE_GATE.get_bound(py);
        let mcphase_gate = mcphase_cls
            .call1((PI, num_controls))
            .expect("Could not create MCPhaseGate Python-side");

        let as_py_gate = PyGate {
            qubits: num_qubits,
            clbits: 0,
            params: 1,
            op_name: "mcphase".to_string(),
            gate: mcphase_gate.into(),
        };

        circuit.push_py_gate(
            as_py_gate,
            &[],
            &(0..num_qubits).map(Qubit).collect::<Vec<Qubit>>(),
            &[],
        )?;

        circuit.h(target);

        Ok(circuit)
    }
}
