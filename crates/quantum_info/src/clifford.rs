// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::{azip, s, Array2};

/// Symplectic matrix.
/// Currently this class is internal to the synthesis library.
pub struct SymplecticMatrix {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits).
    pub smat: Array2<bool>,
}

/// Clifford.
/// Currently this class has a very different functionality from Qiskit's
/// python-based Clifford class.
#[derive(Clone)]
pub struct Clifford {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits + 1).
    pub tableau: Array2<bool>,
}

impl Clifford {
    /// Modifies the tableau in-place by appending S-gate
    pub fn append_s(&mut self, qubit: usize) {
        let (x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the tableau in-place by appending Sdg-gate
    #[allow(dead_code)]
    pub fn append_sdg(&mut self, qubit: usize) {
        let (x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & !z);
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the tableau in-place by appending SX-gate
    pub fn append_sx(&mut self, qubit: usize) {
        let (mut x, z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= !x & z);
        azip!((&z in &z, x in &mut x) *x ^= z);
    }

    /// Modifies the tableau in-place by appending SXDG-gate
    pub fn append_sxdg(&mut self, qubit: usize) {
        let (mut x, z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((&z in &z, x in &mut x) *x ^= z);
    }

    /// Modifies the tableau in-place by appending H-gate
    pub fn append_h(&mut self, qubit: usize) {
        let (mut x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    /// Modifies the tableau in-place by appending SWAP-gate
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.tableau.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    /// Modifies the tableau in-place by appending CX-gate
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
            s![.., 2 * self.num_qubits],
        ));
        azip!((p in &mut p, &x0 in &x0, &z0 in &z0, &x1 in &x1, &z1 in &z1) *p ^= (x1 ^ z0 ^ true) & z1 & x0);
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }

    /// Modifies the tableau in-place by appending W-gate.
    /// This is equivalent to an Sdg gate followed by an H gate.
    pub fn append_v(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));

        azip!((x in &mut x, z in &mut z) (*x, *z) = (*x ^ *z, *x));
    }

    /// Modifies the tableau in-place by appending V-gate.
    /// This is equivalent to two V gates.
    pub fn append_w(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));

        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x ^ *z));
    }
}
