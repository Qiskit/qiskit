// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::{ArrayView1, ArrayView2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_quantum_info::clifford::Clifford;
use smallvec::smallvec;

use crate::QiskitError;
use crate::clifford::utils::CliffordGatesVec;

/// Tracks the circuit built so far both as a gate sequence and as a Clifford
/// tableau, so that stabilizers can be evolved through it.
struct TrackedCircuit {
    gates: CliffordGatesVec,
    clifford: Clifford,
}

impl TrackedCircuit {
    fn push1(&mut self, gate: StandardGate, qubit: usize) {
        match gate {
            StandardGate::H => self.clifford.append_h(qubit),
            StandardGate::S => self.clifford.append_s(qubit),
            StandardGate::X => self.clifford.append_x(qubit),
            _ => unreachable!(),
        }
        self.gates
            .push((gate, smallvec![], smallvec![Qubit::new(qubit)]));
    }

    fn push2(&mut self, gate: StandardGate, qubit0: usize, qubit1: usize) {
        match gate {
            StandardGate::CX => self.clifford.append_cx(qubit0, qubit1),
            StandardGate::Swap => self.clifford.append_swap(qubit0, qubit1),
            _ => unreachable!(),
        }
        self.gates.push((
            gate,
            smallvec![],
            smallvec![Qubit::new(qubit0), Qubit::new(qubit1)],
        ));
    }

    /// Conjugate stabilizer ``(z, x)`` by the circuit built so far, returning
    /// the dense evolved Pauli and whether its sign is negative.
    fn evolve(&self, z: ArrayView1<bool>, x: ArrayView1<bool>) -> (bool, Vec<bool>, Vec<bool>) {
        let in_z: Vec<bool> = z.to_vec();
        let in_x: Vec<bool> = x.to_vec();
        self.clifford.conjugate_pauli(&in_z, &in_x)
    }
}

/// Synthesize a circuit that generates a state stabilized by the given
/// stabilizers, using Gaussian elimination with Clifford gates.  Based on the
/// stim implementation; see the Python-space
/// ``qiskit.synthesis.synth_circuit_from_stabilizers`` for details.
///
/// Each stabilizer is given by a row of the boolean ``z`` and ``x`` matrices
/// together with its group phase in ``phase`` (0 for `+1`, 2 for `-1`);
/// ``labels`` are the string representations used in error messages.
pub fn synth_circuit_from_stabilizers_inner(
    z: ArrayView2<bool>,
    x: ArrayView2<bool>,
    phase: ArrayView1<u8>,
    labels: &[String],
    allow_redundant: bool,
    allow_underconstrained: bool,
    invert: bool,
) -> Result<(usize, CliffordGatesVec), String> {
    let num_qubits = z.ncols();
    let mut circuit = TrackedCircuit {
        gates: CliffordGatesVec::new(),
        clifford: Clifford::identity(num_qubits),
    };

    let mut used = 0;
    for i in 0..z.nrows() {
        let negative = phase[i] == 2;
        let (evolved_sign, curr_z, curr_x) = circuit.evolve(z.row(i), x.row(i));

        // Find pivot.
        let mut pivot = used;
        while pivot < num_qubits && !(curr_x[pivot] || curr_z[pivot]) {
            pivot += 1;
        }

        if pivot == num_qubits {
            if curr_x.iter().any(|&bit| bit) {
                return Err(format!(
                    "Stabilizer {} ({}) anti-commutes with some of the previous stabilizers.",
                    i, labels[i]
                ));
            }
            if evolved_sign ^ negative {
                return Err(format!(
                    "Stabilizer {} ({}) contradicts some of the previous stabilizers.",
                    i, labels[i]
                ));
            }
            if curr_z.iter().any(|&bit| bit) && !allow_redundant {
                return Err(format!(
                    "Stabilizer {} ({}) is a product of the others and allow_redundant is False. \
                     Add allow_redundant=True to the function call if you want to allow redundant \
                     stabilizers.",
                    i, labels[i]
                ));
            }
            continue;
        }

        // Change pivot basis to the Z axis.
        if curr_x[pivot] {
            if curr_z[pivot] {
                circuit.push1(StandardGate::H, pivot);
                circuit.push1(StandardGate::S, pivot);
                circuit.push1(StandardGate::H, pivot);
                circuit.push1(StandardGate::S, pivot);
                circuit.push1(StandardGate::S, pivot);
            } else {
                circuit.push1(StandardGate::H, pivot);
            }
        }

        // Cancel other terms in Pauli string.
        for j in 0..num_qubits {
            if j == pivot || !(curr_x[j] || curr_z[j]) {
                continue;
            }
            match (curr_x[j], curr_z[j]) {
                // X
                (true, false) => {
                    circuit.push1(StandardGate::H, pivot);
                    circuit.push2(StandardGate::CX, pivot, j);
                    circuit.push1(StandardGate::H, pivot);
                }
                // Z
                (false, true) => {
                    circuit.push2(StandardGate::CX, j, pivot);
                }
                // Y
                (true, true) => {
                    circuit.push1(StandardGate::H, pivot);
                    circuit.push1(StandardGate::S, j);
                    circuit.push1(StandardGate::S, j);
                    circuit.push1(StandardGate::S, j);
                    circuit.push2(StandardGate::CX, pivot, j);
                    circuit.push1(StandardGate::H, pivot);
                    circuit.push1(StandardGate::S, j);
                }
                (false, false) => unreachable!(),
            }
        }

        // Move pivot to diagonal.
        if pivot != used {
            circuit.push2(StandardGate::Swap, pivot, used);
        }

        // Fix sign.
        let (evolved_sign, _, _) = circuit.evolve(z.row(i), x.row(i));
        if evolved_sign ^ negative {
            circuit.push1(StandardGate::X, used);
        }
        used += 1;
    }

    if used < num_qubits && !allow_underconstrained {
        return Err(
            "Stabilizers are underconstrained and allow_underconstrained is False. \
             Add allow_underconstrained=True  to the function call \
             if you want to allow underconstrained stabilizers."
                .to_string(),
        );
    }

    let gates = if invert {
        circuit.gates
    } else {
        // Invert the circuit: all emitted gates except S are self-inverse.
        circuit
            .gates
            .into_iter()
            .rev()
            .map(|(gate, params, qubits)| {
                let gate = match gate {
                    StandardGate::S => StandardGate::Sdg,
                    other => other,
                };
                (gate, params, qubits)
            })
            .collect()
    };
    Ok((num_qubits, gates))
}

/// Synthesize a circuit that generates a state stabilized by the given
/// stabilizers, each passed as a row of the boolean ``z`` and ``x`` matrices
/// together with its group phase (0 or 2) in ``phase``.  ``labels`` are the
/// stabilizer string representations, used only in error messages.
#[pyfunction]
#[pyo3(signature = (z, x, phase, labels, allow_redundant, allow_underconstrained, invert))]
#[allow(clippy::too_many_arguments)]
fn synth_circuit_from_stabilizers(
    z: PyReadonlyArray2<bool>,
    x: PyReadonlyArray2<bool>,
    phase: PyReadonlyArray1<u8>,
    labels: Vec<String>,
    allow_redundant: bool,
    allow_underconstrained: bool,
    invert: bool,
) -> PyResult<PyCircuitData> {
    let (num_qubits, gates) = synth_circuit_from_stabilizers_inner(
        z.as_array(),
        x.as_array(),
        phase.as_array(),
        &labels,
        allow_redundant,
        allow_underconstrained,
        invert,
    )
    .map_err(QiskitError::new_err)?;
    Ok(CircuitData::from_standard_gates(num_qubits as u32, gates, Param::Float(0.0))?.into())
}

pub fn stabilizer(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_circuit_from_stabilizers, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Parse simple stabilizer labels like "-XIZ" into (z, x, phase) rows.
    fn parse(labels: &[&str]) -> (Array2<bool>, Array2<bool>, Array1<u8>, Vec<String>) {
        let num_qubits = labels[0].trim_start_matches('-').len();
        let mut z = Array2::from_elem((labels.len(), num_qubits), false);
        let mut x = Array2::from_elem((labels.len(), num_qubits), false);
        let mut phase = Array1::zeros(labels.len());
        for (row, label) in labels.iter().enumerate() {
            let body = label.strip_prefix('-').map_or(*label, |rest| {
                phase[row] = 2;
                rest
            });
            // Labels are little-endian: the leftmost character is the highest qubit.
            for (col, ch) in body.chars().rev().enumerate() {
                match ch {
                    'I' => (),
                    'X' => x[[row, col]] = true,
                    'Z' => z[[row, col]] = true,
                    'Y' => {
                        x[[row, col]] = true;
                        z[[row, col]] = true;
                    }
                    _ => panic!("bad label"),
                }
            }
        }
        let labels = labels.iter().map(|label| label.to_string()).collect();
        (z, x, phase, labels)
    }

    /// Check the algorithm's invariant: the ``invert=true`` circuit conjugates
    /// stabilizer ``i`` to ``+Z`` on qubit ``i``.
    fn assert_diagonalizes(labels: &[&str]) {
        let (z, x, phase, label_strings) = parse(labels);
        let num_qubits = z.ncols();
        let (_, gates) = synth_circuit_from_stabilizers_inner(
            z.view(),
            x.view(),
            phase.view(),
            &label_strings,
            false,
            false,
            true,
        )
        .unwrap();
        let mut circuit = TrackedCircuit {
            gates: CliffordGatesVec::new(),
            clifford: Clifford::identity(num_qubits),
        };
        for (gate, _, qubits) in gates {
            match qubits.len() {
                1 => circuit.push1(gate, qubits[0].index()),
                _ => circuit.push2(gate, qubits[0].index(), qubits[1].index()),
            }
        }
        for (row, negative) in (0..z.nrows()).zip(phase.iter().map(|&p| p == 2)) {
            let (sign, out_z, out_x) = circuit.evolve(z.row(row), x.row(row));
            assert!(
                !(sign ^ negative),
                "stabilizer {row} evolved to a negative Pauli"
            );
            assert!(!out_x.iter().any(|&bit| bit));
            let expected: Vec<bool> = (0..num_qubits).map(|qubit| qubit == row).collect();
            assert_eq!(out_z, expected, "stabilizer {row} did not diagonalize");
        }
    }

    #[test]
    fn test_diagonalization() {
        assert_diagonalizes(&["ZZ", "XX"]);
        assert_diagonalizes(&["-ZZ", "XX"]);
        assert_diagonalizes(&["XZI", "-ZXZ", "IZX"]);
        assert_diagonalizes(&["ZIZ", "XXX", "-IZZ"]);
    }

    #[test]
    fn test_errors() {
        let check = |labels: &[&str], redundant: bool, underconstrained: bool| {
            let (z, x, phase, label_strings) = parse(labels);
            synth_circuit_from_stabilizers_inner(
                z.view(),
                x.view(),
                phase.view(),
                &label_strings,
                redundant,
                underconstrained,
                false,
            )
        };
        // Redundant stabilizer, rejected then allowed.
        assert!(
            check(&["ZZ", "XX", "-YY"], false, false)
                .unwrap_err()
                .contains("product")
        );
        assert!(check(&["ZZ", "XX", "-YY"], true, false).is_ok());
        // Contradicting stabilizer.
        assert!(
            check(&["ZZ", "XX", "YY"], true, false)
                .unwrap_err()
                .contains("contradicts")
        );
        // Anti-commuting stabilizer.  (The public Python API rejects anti-commuting
        // inputs up-front; this in-loop error only catches evolved Paulis whose
        // support lies entirely in the already-diagonalized block.)
        assert!(
            check(&["Z", "X"], false, false)
                .unwrap_err()
                .contains("anti-commutes")
        );
        // Underconstrained set, rejected then allowed.
        assert!(
            check(&["ZZ"], false, false)
                .unwrap_err()
                .contains("underconstrained")
        );
        assert!(check(&["ZZ"], false, true).is_ok());
    }
}
