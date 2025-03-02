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

use crate::synthesis::common::SynthesisData;
use pyo3::Python;
// use qiskit_circuit::imports;
// use qiskit_circuit::operations::{OperationRef, PyGate};
// use smallvec::smallvec;

use std::f64::consts::PI;
const PI2: f64 = PI / 2.;
// const PI4: f64 = PI / 4.;
const PI8: f64 = PI / 8.;

pub fn ccx<'a>() -> SynthesisData<'a> {
    let mut circuit = SynthesisData::new(3);
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
    circuit
}

/// Implements an optimized toffoli operation up to a diagonal gate,
/// akin to lemma 6 of [arXiv:1501.06911] (https://arxiv.org/abs/1501.06911).
fn rccx<'a>() -> SynthesisData<'a> {
    let mut circuit = SynthesisData::new(3);
    circuit.h(2);
    circuit.t(2);
    circuit.cx(1, 2);
    circuit.tdg(2);
    circuit.cx(0, 2);
    circuit.t(2);
    circuit.cx(1, 2);
    circuit.tdg(2);
    circuit.h(2);
    circuit
}

/// Efficient synthesis for 3-controlled X-gate.
pub fn c3x<'a>() -> SynthesisData<'a> {
    let mut circuit = SynthesisData::new(4);
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
    circuit
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
// pub fn c4x<'a>() -> SynthesisData<'a> {
//     let mut circuit = SynthesisData::new(5);
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
fn action_gadget<'a>() -> SynthesisData<'a> {
    let mut circuit = SynthesisData::new(3);
    circuit.h(2);
    circuit.t(2);
    circuit.cx(0, 2);
    circuit.tdg(2);
    circuit.cx(1, 2);
    circuit
}

/// A block in the `reset part`, see Iten et al.
fn reset_gadget<'a>() -> SynthesisData<'a> {
    let mut circuit = SynthesisData::new(3);
    circuit.cx(1, 2);
    circuit.t(2);
    circuit.cx(0, 2);
    circuit.tdg(2);
    circuit.h(2);
    circuit
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
pub fn synth_mcx_n_dirty_i15<'a>(
    num_controls: usize,
    relative_phase: bool,
    action_only: bool,
) -> SynthesisData<'a> {
    if num_controls == 1 {
        let mut circuit = SynthesisData::new(2);
        circuit.cx(0, 1);
        circuit
    } else if num_controls == 2 {
        ccx()
    } else if num_controls == 3 && !relative_phase {
        c3x()
    } else {
        let num_ancillas = num_controls - 2;
        let num_qubits = num_controls + 1 + num_ancillas;
        let mut circuit = SynthesisData::new(num_qubits as u32);

        let controls: Vec<u32> = (0..num_controls).map(|q| q as u32).collect();
        let target = num_controls as u32;
        let ancillas: Vec<u32> = ((num_controls + 1)..num_qubits).map(|q| q as u32).collect();

        for j in 0..2 {
            if !relative_phase {
                circuit.compose(
                    &ccx(),
                    Some(&[
                        controls[num_controls - 1],
                        ancillas[num_controls - 3],
                        target,
                    ]),
                );
            } else if j == 0 {
                circuit.compose(
                    &action_gadget(),
                    Some(&[
                        controls[num_controls - 1],
                        ancillas[num_controls - 3],
                        target,
                    ]),
                );
            } else if j == 1 {
                circuit.compose(
                    &reset_gadget(),
                    Some(&[
                        controls[num_controls - 1],
                        ancillas[num_controls - 3],
                        target,
                    ]),
                );
            }

            // action part
            for i in (0..num_controls - 3).rev() {
                circuit.compose(
                    &action_gadget(),
                    Some(&[controls[i + 2], ancillas[i], ancillas[i + 1]]),
                );
            }

            circuit.compose(&rccx(), Some(&[controls[0], controls[1], ancillas[0]]));

            // reset part
            for i in 0..num_controls - 3 {
                circuit.compose(
                    &reset_gadget(),
                    Some(&[controls[i + 2], ancillas[i], ancillas[i + 1]]),
                );
            }

            if action_only {
                circuit.compose(
                    &ccx(),
                    Some(&[
                        controls[num_controls - 1],
                        ancillas[num_controls - 3],
                        target,
                    ]),
                );
                break;
            }
        }
        circuit
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
pub fn synth_mcx_noaux_v24<'a>(_py: Python, num_controls: usize) -> SynthesisData<'a> {
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

        let mut circuit = SynthesisData::new(num_qubits);
        circuit.h(target);

        // let mcphase_cls = imports::MCPHASE_GATE.get_bound(py);
        // let mcphase_gate = mcphase_cls.call1((PI, num_controls)).expect("Could not create MCPhaseGate Python-side");

        // let py_obj = Box::new(mcphase_gate.into());

        // let as_py_gate = PyGate {
        //     qubits: num_qubits,
        //     clbits: 0,
        //     params: 1, // check me!
        //     op_name: "mcphase".to_string(),
        //     gate: *py_obj,
        // };
        // circuit.data.push(
        //     (
        //     OperationRef::Gate(&as_py_gate),
        //     smallvec![],
        //     (0..num_qubits).map(|q| q as u32).collect::<Vec<u32>>().into()
        //     )
        // );

        circuit.h(target);

        circuit
    }
}
