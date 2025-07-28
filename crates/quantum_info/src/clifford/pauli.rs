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

use crate::clifford::pauli_like::PauliLike;
use std::ops;

pub struct Pauli {
    pub n: usize,
    pub data: Vec<bool>,
    pub phase: u8,
}

impl Pauli {
    pub fn new(n: usize) -> Self {
        Pauli {
            n,
            data: vec![true; 2 * n],
            phase: 0,
        }
    }
    pub fn from_vec_bool(data: Vec<bool>, phase: u8) -> Self {
        Pauli {
            n: data.len() / 2,
            data,
            phase,
        }
    }
}

impl ops::Mul<Pauli> for Pauli {
    type Output = Pauli;

    fn mul(self, _rhs: Pauli) -> Pauli {
        assert_eq!(self.n, _rhs.n);
        let mut output = Pauli::new(self.n);
        output.phase = self.phase + _rhs.phase;
        for i in 0..self.n {
            if self.data[i] && _rhs.data[i + self.n] {
                output.phase += 2;
            }
            if self.data[i + self.n] && _rhs.data[i] {
                output.phase += 2;
            }
        }
        for i in 0..2 * self.n {
            output.data[i] = self.data[i] ^ _rhs.data[i];
        }
        output.phase %= 4;
        output
    }
}

impl PauliLike for Pauli {
    fn h(&mut self, i: usize) {
        self.data.swap(i, i + self.n);
    }

    fn s(&mut self, i: usize) {
        self.data[i + self.n] ^= self.data[i];
    }

    fn sd(&mut self, i: usize) {
        self.data[i + self.n] ^= self.data[i];
    }

    fn sqrt_x(&mut self, i: usize) {
        self.data[i] ^= self.data[i + self.n];
    }

    fn sqrt_xd(&mut self, i: usize) {
        self.data[i] ^= self.data[i + self.n];
    }

    fn cnot(&mut self, i: usize, j: usize) {
        self.data[i + self.n] ^= self.data[j + self.n];
        self.data[j] ^= self.data[i];
    }
}
