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

use crate::clifford::clifford_circuit::{CliffordCircuit, CliffordGate};

/// This trait should be implemented by any struct that can be conjugated by a Clifford gate/circuit
pub trait PauliLike {
    /// Conjugate the PauliLike object via a H gate
    fn h(&mut self, i: usize);
    /// Conjugate the PauliLike object via a S gate
    fn s(&mut self, i: usize);
    /// Conjugate the PauliLike object via a S dagger gate
    fn sd(&mut self, i: usize);
    /// Conjugate the PauliLike object via a SQRT_X gate
    fn sqrt_x(&mut self, i: usize);
    /// Conjugate the PauliLike object via a SQRT_X dagger gate
    fn sqrt_xd(&mut self, i: usize);
    /// Conjugate the PauliLike object via a CNOT gate
    fn cnot(&mut self, i: usize, j: usize);
    // Conjugate the PauliLike object via a CZ gate
    fn cz(&mut self, i: usize, j: usize) {
        self.h(j);
        self.cnot(i, j);
        self.h(j);
    }
    /// Conjugate the PauliLike object via a Gate
    fn conjugate_with_gate(&mut self, gate: &CliffordGate) {
        match gate {
            CliffordGate::CNOT(i, j) => self.cnot(*i, *j),
            CliffordGate::CZ(i, j) => self.cz(*i, *j),
            CliffordGate::H(i) => self.h(*i),
            CliffordGate::S(i) => self.s(*i),
            CliffordGate::Sd(i) => self.sd(*i),
            CliffordGate::SqrtX(i) => self.sqrt_x(*i),
            CliffordGate::SqrtXd(i) => self.sqrt_xd(*i),
        }
    }
    /// Conjugate the PauliLike object via a Circuit
    fn conjugate_with_circuit(&mut self, circuit: &CliffordCircuit) {
        for gate in circuit.gates.iter() {
            self.conjugate_with_gate(gate);
        }
    }
}
