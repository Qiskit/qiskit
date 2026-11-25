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

//! The algebraic type of a virtual gate.

use std::ops::BitOr;

use super::flags::{BitFlag, BitFlags};
use crate::tensor::{DType, TensorType, dim_shape};

/// The group a virtual gate belongs to.
///
/// A set of virtual types is a [`BitFlags<VirtualType>`](BitFlags).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VirtualType {
    Pauli = 1 << 0,
    C1 = 1 << 1,
    U2 = 1 << 2,
    Z2 = 1 << 3,
}

impl BitFlag for VirtualType {
    const FLAGS: &'static [Self] = &[Self::Pauli, Self::C1, Self::U2, Self::Z2];

    fn bits(self) -> u32 {
        self as u32
    }
}

impl BitOr for VirtualType {
    type Output = BitFlags<Self>;

    fn bitor(self, other: Self) -> BitFlags<Self> {
        BitFlags::from_flag(self) | other
    }
}

impl VirtualType {
    /// The dtype a virtual gate of this type is stored as.
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Pauli | Self::C1 => DType::U8, // TODO: Pauli likely won't stay as u8
            Self::U2 => DType::C128,
            Self::Z2 => DType::Bit,
        }
    }

    /// Return the type of one stored virtual gate of this type on `num_qubits` qubits, or
    /// `None` when the type cannot act on `num_qubits` (e.g. a Pauli must act on 1 qubit).
    pub fn tensor_type(&self, num_qubits: usize) -> Option<TensorType> {
        let shape: Vec<usize> = match self {
            Self::Pauli | Self::C1 | Self::Z2 => (num_qubits == 1).then(Vec::new),
            Self::U2 => (num_qubits == 1).then(|| vec![2, 2]),
        }?;
        Some(TensorType {
            dtype: self.dtype(),
            shape: dim_shape(shape),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_type_on_one_qubit() {
        for (virtual_type, expected) in [
            (VirtualType::Pauli, "U8[]"),
            (VirtualType::C1, "U8[]"),
            (VirtualType::U2, "C128[2, 2]"),
            (VirtualType::Z2, "Bit[]"),
        ] {
            let gate_type = virtual_type.tensor_type(1).unwrap();
            assert_eq!(gate_type.to_string(), expected, "for {virtual_type:?}");
        }
    }

    #[test]
    fn test_gate_type_rejects_other_sizes() {
        assert_eq!(VirtualType::Pauli.tensor_type(0), None);
        assert_eq!(VirtualType::Pauli.tensor_type(2), None);
        assert_eq!(VirtualType::C1.tensor_type(2), None);
        assert_eq!(VirtualType::U2.tensor_type(2), None);
        assert_eq!(VirtualType::Z2.tensor_type(2), None);
    }

    #[test]
    fn test_flags_are_distinct_bits_in_order() {
        let bits: Vec<u32> = VirtualType::FLAGS.iter().map(|flag| flag.bits()).collect();
        assert!(bits.iter().all(|bit| bit.is_power_of_two()));
        assert!(bits.windows(2).all(|pair| pair[0] < pair[1]));
    }
}

impl VirtualType {
    /// Every virtual type.
    pub const ALL: [VirtualType; 4] = [Self::Pauli, Self::C1, Self::U2, Self::Z2];
}
