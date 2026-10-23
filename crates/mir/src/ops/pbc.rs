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

//! This module contains operations that are used in Pauli-Based Computations.

use std::fmt;

///  Basic Pauli enum
///
/// TODO: Decide whether we want to use existing from Qiskit or define one
/// specifically for MIR. Currently I'm inclined to create a new enum here,
/// specifically for use in MIR, as we can add some helper functions here that
/// will be needed throughout MIR (e.g., determine commuting/anti-commuting
/// properties, and maybe some conversion functions to/from string). That said,
/// I have not yet looked at the existing implementation pwdin Qiskit to know
/// whether the same could be done with them.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub enum Pauli {
    #[default]
    I,
    X,
    Z,
    Y,
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// New type for a list of Paulis.
///
/// This will allow us to change the underlying backing store over time (if
/// necessary) and allow the compiler to catch any type errors. This list will
/// be used in several of the PBC ops we need to implement.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliList(Vec<Pauli>);

impl PauliList {
    pub fn new(paulis: Vec<Pauli>) -> Self {
        PauliList(paulis)
    }

    pub fn inner(&self) -> &Vec<Pauli> {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// AngleType for Mir
///
/// These will store angles in units of rad / π (or some variation thereof).
/// This causes angles corresponding to Clifford rotations are stored without
/// round-off from inexact representations of π .
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AngleType(f64);

impl AngleType {
    pub fn new(angle: f64) -> Self {
        AngleType(angle)
    }

    pub fn inner(&self) -> f64 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PauliRotation {
    paulis: PauliList,
    angle: AngleType,
}

impl PauliRotation {
    pub fn new(paulis: PauliList, angle: AngleType) -> Self {
        Self { paulis, angle }
    }

    pub fn paulis(&self) -> &PauliList {
        &self.paulis
    }

    pub fn angle(&self) -> &AngleType {
        &self.angle
    }
}

impl fmt::Display for PauliRotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pauli_rotation({:?}, {:?}", self.paulis, self.angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_pauli_rotation() {
        let pauli_rotation = PauliRotation {
            paulis: PauliList(vec![Pauli::I; 8]),
            angle: AngleType(0.5),
        };

        assert_eq!(*pauli_rotation.paulis(), PauliList(vec![Pauli::I; 8]));
        assert_eq!(*pauli_rotation.angle(), AngleType(0.5));
    }
}
