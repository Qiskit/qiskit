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

use crate::clifford::pauli::Pauli;
use crate::clifford::pauli_like::PauliLike;
use itertools::izip;
use std::cmp::max;

const WIDTH: usize = 64;

fn get_stride(index: usize) -> usize {
    index / WIDTH
}

fn get_offset(index: usize) -> usize {
    index % WIDTH
}

/// A set of Pauli operators (module global phase)
/// Conjugation by Clifford gates are vectorized
#[derive(Clone, Debug, PartialEq)]
pub struct PauliSet {
    pub n: usize,
    nstrides: usize,
    noperators: usize,
    start_offset: usize,
    /// The X and Z parts of the Pauli operators (in row major)
    /// The X part spans the first `n` rows and the Z part spans the last `n` rows
    data_array: Vec<Vec<u64>>,
    phases: Vec<u64>,
}

impl PauliSet {
    /// Allocate an empty set of n-qubit Pauli operators
    pub fn new(n: usize) -> Self {
        Self {
            n,
            nstrides: 0,
            noperators: 0,
            start_offset: 0,
            data_array: vec![Vec::new(); 2 * n],
            phases: Vec::new(),
        }
    }
    /// Allocate a set of m n-qubit Pauli operators set to the identity
    pub fn new_empty(n: usize, m: usize) -> Self {
        let nstrides = get_stride(m) + 1;
        Self {
            n,
            nstrides,
            noperators: m,
            start_offset: 0,
            data_array: vec![vec![0; nstrides]; 2 * n],
            phases: vec![0; nstrides],
        }
    }
    // Construction from a list of operators
    pub fn from_slice(data: &[String]) -> Self {
        if data.is_empty() {
            return Self::new(0);
        }
        let n = data.first().unwrap().len();
        let mut pset = Self::new(n);
        for piece in data {
            pset.insert(piece, false);
        }
        pset
    }
    /// Returns the number of operators stored in the set
    pub fn len(&self) -> usize {
        self.noperators
    }

    pub fn is_empty(&self) -> bool {
        self.noperators == 0
    }

    /// Inserts a new Pauli operator in the set and returns its index
    pub fn insert(&mut self, axis: &str, phase: bool) -> usize {
        let stride = get_stride(self.noperators + self.start_offset);
        let offset = get_offset(self.noperators + self.start_offset);
        if stride == self.nstrides {
            self.nstrides += 1;
            self.data_array.iter_mut().for_each(|row| row.push(0));
            self.phases.push(0);
        }
        // Setting the phase
        if phase {
            self.phases[stride] |= 1 << offset;
        }
        // Setting the operator
        for (index, pauli) in axis.chars().enumerate() {
            match pauli {
                'Z' => self.data_array[index + self.n][stride] |= 1 << offset,
                'X' => self.data_array[index][stride] |= 1 << offset,
                'Y' => {
                    self.data_array[index][stride] |= 1 << offset;
                    self.data_array[index + self.n][stride] |= 1 << offset
                }
                _ => {}
            }
        }
        self.noperators += 1;
        self.noperators - 1
    }

    /// Inserts a new Pauli operator described as a vector of bool in the set and returns its index
    pub fn insert_vec_bool(&mut self, axis: &[bool], phase: bool) -> usize {
        let stride = get_stride(self.noperators + self.start_offset);
        let offset = get_offset(self.noperators + self.start_offset);
        if stride == self.nstrides {
            self.nstrides += 1;
            self.data_array.iter_mut().for_each(|row| row.push(0));
            self.phases.push(0);
        }
        if phase {
            self.phases[stride] |= 1 << offset;
        }
        for (index, value) in axis.iter().enumerate() {
            if *value {
                self.data_array[index][stride] |= 1 << offset;
            }
        }
        self.noperators += 1;
        self.noperators - 1
    }
    pub fn insert_pauli(&mut self, pauli: &Pauli) -> usize {
        self.insert_vec_bool(&pauli.data, pauli.phase == 2)
    }
    pub fn set_phase(&mut self, col: usize, phase: bool) {
        let stride = get_stride(col);
        let offset = get_offset(col);
        if phase != ((self.phases[stride] >> offset & 1) != 0) {
            self.phases[stride] ^= 1 << offset;
        }
    }

    pub fn set_entry(&mut self, operator_index: usize, qbit: usize, x_part: bool, z_part: bool) {
        let stride = get_stride(operator_index + self.start_offset);
        let offset = get_offset(operator_index + self.start_offset);
        if x_part != (1 == (self.data_array[qbit][stride] >> offset) & 1) {
            self.data_array[qbit][stride] ^= 1 << offset;
        }
        if z_part != (1 == (self.data_array[qbit + self.n][stride] >> offset) & 1) {
            self.data_array[qbit + self.n][stride] ^= 1 << offset;
        }
    }
    pub fn set_raw_entry(&mut self, row: usize, col: usize, value: bool) {
        let stride = get_stride(col);
        let offset = get_offset(col);
        if value != (1 == (self.data_array[row][stride] >> offset) & 1) {
            self.data_array[row][stride] ^= 1 << offset;
        }
    }

    /// Clears the data of the Pauli set
    pub fn clear(&mut self) {
        for j in 0..self.nstrides {
            for i in 0..2 * self.n {
                self.data_array[i][j] = 0;
            }
            self.phases[j] = 0;
        }
        self.noperators = 0;
        self.start_offset = 0;
    }
    /// Pops the first rotation in the set
    pub fn pop(&mut self) {
        let stride = get_stride(self.start_offset);
        let offset = get_offset(self.start_offset);
        for i in 0..2 * self.n {
            self.data_array[i][stride] &= !(1 << offset);
        }
        self.phases[stride] &= !(1 << offset);
        self.start_offset += 1;
        self.noperators -= 1;
    }
    /// Pops the last rotation in the set
    pub fn pop_last(&mut self) {
        let stride = get_stride(self.start_offset + self.noperators - 1);
        let offset = get_offset(self.start_offset + self.noperators - 1);
        for i in 0..2 * self.n {
            self.data_array[i][stride] &= !(1 << offset);
        }
        self.phases[stride] &= !(1 << offset);
        self.noperators -= 1;
    }
    /// Set some operator to identity (because popping in the middle is expensive :O)
    pub fn set_to_identity(&mut self, operator_index: usize) {
        // set_entry(&mut self, operator_index: usize, qbit: usize, x_part: bool, z_part: bool)
        for i in 0..self.n {
            self.set_entry(operator_index, i, false, false);
        }
    }
    /// Get the operator at index `operator_index` as a pair (phase, string)
    pub fn get(&self, operator_index: usize) -> (bool, String) {
        let operator_index = operator_index + self.start_offset;
        let mut output = String::new();
        let stride = get_stride(operator_index);
        let offset = get_offset(operator_index);
        for i in 0..self.n {
            match (
                (self.data_array[i][stride] >> offset) & 1,
                (self.data_array[i + self.n][stride] >> offset) & 1,
            ) {
                (1, 0) => {
                    output += "X";
                }
                (0, 1) => {
                    output += "Z";
                }
                (1, 1) => {
                    output += "Y";
                }
                _ => {
                    output += "I";
                }
            }
        }
        (((self.phases[stride] >> offset) & 1 != 0), output)
    }

    /// Get the operator at index `operator_index` as a pair `(bool, Vec<bool>)`
    pub fn get_as_vec_bool(&self, operator_index: usize) -> (bool, Vec<bool>) {
        let operator_index = operator_index + self.start_offset;
        let mut output = Vec::new();
        let stride = get_stride(operator_index);
        let offset = get_offset(operator_index);
        for i in 0..2 * self.n {
            output.push(((self.data_array[i][stride] >> offset) & 1) != 0);
        }
        (((self.phases[stride] >> offset) & 1 != 0), output)
    }

    /// Get the operator at index `operator_index` as a `Pauli` object
    pub fn get_as_pauli(&self, operator_index: usize) -> Pauli {
        let (phase, data) = self.get_as_vec_bool(operator_index);
        Pauli::from_vec_bool(data, if phase { 2 } else { 0 })
    }
    /// Get a single entry of the PauliSet
    pub fn get_entry(&self, row: usize, col: usize) -> bool {
        let col = col + self.start_offset;
        let stride = get_stride(col);
        let offset = get_offset(col);
        ((self.data_array[row][stride] >> offset) & 1) != 0
    }
    pub fn get_phase(&self, col: usize) -> bool {
        let col = col + self.start_offset;
        let stride = get_stride(col);
        let offset = get_offset(col);
        ((self.phases[stride] >> offset) & 1) != 0
    }

    pub fn get_i_factors(&self) -> Vec<u8> {
        let mut output = Vec::new();
        for i in 0..self.len() {
            let mut ifact: u8 = 0;
            for j in 0..self.n {
                if self.get_entry(j, i) & self.get_entry(j + self.n, i) {
                    ifact += 1;
                }
            }
            output.push(ifact % 4);
        }
        output
    }
    pub fn get_i_factors_single_col(&self, col: usize) -> u8 {
        let mut ifact: u8 = 0;
        for j in 0..self.n {
            if self.get_entry(j, col) & self.get_entry(j + self.n, col) {
                ifact += 1;
            }
        }
        ifact % 4
    }
    /// Get the inverse Z output of the tableau (assuming the PauliSet is a Tableau, i.e. has exactly 2n operators storing X1...Xn Z1...Zn images)
    pub fn get_inverse_z(&self, qbit: usize) -> (bool, String) {
        let mut pstring = String::new();
        for i in 0..self.n {
            let x_bit = self.get_entry(qbit, i + self.n);
            let z_bit = self.get_entry(qbit, i);
            match (x_bit, z_bit) {
                (false, false) => {
                    pstring.push('I');
                }
                (true, false) => {
                    pstring.push('X');
                }
                (false, true) => {
                    pstring.push('Z');
                }
                (true, true) => {
                    pstring.push('Y');
                }
            }
        }
        (self.get_phase(qbit + self.n), pstring)
    }
    /// Get the inverse X output of the tableau (assuming the PauliSet is a Tableau, i.e. has exactly 2n operators storing X1...Xn Z1...Zn images)
    pub fn get_inverse_x(&self, qbit: usize) -> (bool, String) {
        let mut pstring = String::new();
        let mut cy = 0;
        for i in 0..self.n {
            let x_bit = self.get_entry(qbit + self.n, i + self.n);
            let z_bit = self.get_entry(qbit + self.n, i);
            match (x_bit, z_bit) {
                (false, false) => {
                    pstring.push('I');
                }
                (true, false) => {
                    pstring.push('X');
                }
                (false, true) => {
                    pstring.push('Z');
                }
                (true, true) => {
                    pstring.push('Y');
                    cy += 1;
                }
            }
        }
        ((cy % 2 != 0), pstring)
    }

    /// Returns the sum mod 2 of the logical AND of a row with an external vector of booleans
    pub fn and_row_acc(&self, row: usize, vec: &[bool]) -> bool {
        let mut output = false;
        for (i, item) in vec.iter().enumerate().take(2 * self.n) {
            output ^= self.get_entry(row, i) & item;
        }
        output
    }

    /// Check equality between two operators
    pub fn equals(&self, i: usize, j: usize) -> bool {
        let (_, vec1) = self.get_as_vec_bool(i);
        let (_, vec2) = self.get_as_vec_bool(j);
        vec1 == vec2
    }

    /*
           Internal methods
    */

    /// XORs row `i` into row `j`
    fn row_op(&mut self, i: usize, j: usize) {
        let (left, right) = self.data_array.split_at_mut(max(i, j));
        let (target_row, source_row) = if i < j {
            (right.get_mut(0).unwrap(), left.get(i).unwrap())
        } else {
            (left.get_mut(j).unwrap(), right.first().unwrap())
        };

        for (v1, v2) in source_row.iter().zip(target_row.iter_mut()) {
            *v2 ^= *v1;
        }
    }

    pub fn swap_qbits(&mut self, i: usize, j: usize) {
        self.data_array.swap(i, j);
        self.data_array.swap(self.n + i, self.n + j);
    }

    /// Offset the phases by the logical bitwise and of two target rows
    fn update_phase_and(&mut self, i: usize, j: usize) {
        for (v1, v2, phase) in izip!(
            self.data_array[i].iter(),
            self.data_array[j].iter(),
            self.phases.iter_mut()
        ) {
            *phase ^= *v1 & *v2;
        }
    }

    /// Same thing as `update_phase_and` but computes the and of 4 rows instead
    fn update_phase_and_many(&mut self, i: usize, j: usize, k: usize, l: usize) {
        for (v1, v2, v3, v4, phase) in izip!(
            self.data_array[i].iter(),
            self.data_array[j].iter(),
            self.data_array[k].iter(),
            self.data_array[l].iter(),
            self.phases.iter_mut()
        ) {
            *phase ^= *v1 & *v2 & *v3 & *v4;
        }
    }
}

impl PauliLike for PauliSet {
    /// Conjugate the set of rotations via a H gate
    fn h(&mut self, i: usize) {
        self.data_array.swap(i, i + self.n);
        self.update_phase_and(i, i + self.n);
    }
    /// Conjugate the set of rotations via a S gate
    fn s(&mut self, i: usize) {
        self.update_phase_and(i, i + self.n);
        self.row_op(i, i + self.n);
    }
    /// Conjugate the set of rotations via a S dagger gate
    fn sd(&mut self, i: usize) {
        self.row_op(i, i + self.n);
        self.update_phase_and(i, i + self.n);
    }
    /// Conjugate the set of rotations via a SQRT_X gate
    fn sqrt_x(&mut self, i: usize) {
        self.row_op(i + self.n, i);
        self.update_phase_and(i, i + self.n);
    }
    /// Conjugate the set of rotations via a SQRT_X dagger gate
    fn sqrt_xd(&mut self, i: usize) {
        self.update_phase_and(i, i + self.n);
        self.row_op(i + self.n, i);
    }
    /// Conjugate the set of rotations via a CNOT gate
    fn cnot(&mut self, i: usize, j: usize) {
        self.update_phase_and_many(i, j, i + self.n, j + self.n);
        self.row_op(j + self.n, i + self.n);
        self.row_op(i, j);
        self.update_phase_and_many(i, j, i + self.n, j + self.n);
    }
}
