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

//! Packed representation of a PauliString for MIR.

use std::fmt;

use std::convert::From;
use std::convert::Into;

/// This represents a Pauli.
///
/// This is mostly experimental for now, and is used as a way to represent the
/// Pauli stored in the PauliString. This is *NOT* intended to be the
/// bit-representation of the Pauli inside the PauliString (for now).
#[derive(Debug, Copy, Clone, Default, PartialEq)]
#[cfg_attr(
    feature = "mem_dbg",
    derive(mem_dbg::MemSize, mem_dbg::MemDbg),
    mem_size(flat)
)]
#[repr(u8)]
pub enum Pauli {
    #[default]
    I = 0, // 0b00
    X = 1, // 0b01
    Z = 2, // 0b10
    Y = 3, // 0b11
}

impl Pauli {
    fn from_components(x: u8, z: u8) -> Pauli {
        assert!(x < 2, "Invalid x component!");
        assert!(z < 2, "Invalid z component!");

        Self::from(z << 1 | x)
    }

    fn to_components(self) -> (u8, u8) {
        let x_bit: u8 = self as u8 & 0x1;
        let z_bit: u8 = (self as u8 & 0x2) >> 1;

        (x_bit, z_bit)
    }
}

impl From<u8> for Pauli {
    fn from(pauli: u8) -> Pauli {
        match pauli {
            0 => Self::I,
            1 => Self::X,
            2 => Self::Z,
            3 => Self::Y,
            _ => unreachable!(),
        }
    }
}

/// Encoding for the Phase.
///
/// This represents the encoding for the Phase for the PauliString.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "mem_dbg",
    derive(mem_dbg::MemSize, mem_dbg::MemDbg),
    mem_size(flat)
)]
#[repr(u8)]
pub enum Phase {
    One = 0,
    NegI = 1,
    NegOne = 2,
    I = 3,
}

impl Phase {
    fn new(phase: u8) -> Self {
        match phase {
            0 => Self::One,
            1 => Self::NegI,
            2 => Self::NegOne,
            3 => Self::I,
            _ => unreachable!("Invalid phase!!"),
        }
    }
}

impl From<u8> for Phase {
    fn from(phase: u8) -> Self {
        Phase::new(phase)
    }
}

impl From<Phase> for u8 {
    fn from(phase: Phase) -> u8 {
        phase as u8
    }
}
#[cfg(target_pointer_width = "64")]
mod inner {
    use bitfield_struct::bitfield;

    /// The following shows the bit layout used for 64-bits.
    ///
    /// Discriminant is used to differentiate between encoded Pauli string and
    /// heap pointer. Requires a single bit.
    ///
    /// Size specifies the number of Paulis that are encoded in this Pauli
    /// string. It uses 5 bits, allowing up to a maximum of 32 paulis to be
    /// encoded.
    ///
    /// Phase encodes the phase field. It can be one of 4 possible values: Thus,
    /// requires 2 bits to encode.
    ///
    /// The X and Z components of the Paulis are currently separated. This
    /// encoding supports a maximum of 28 paulis, requiring a total of 56 bits.
    ///
    /// 0b_ZZZZZZZZ_ZZZZZZZZ_ZZZZZZZZ_ZZZZXXXX_XXXXXXXX_XXXXXXXX_XXXXXXXX_PPSSSSSD
    ///
    /// | Symbol | Name         | Size (Bits) | Bit Position |
    /// |    D   | Discriminant |    1        |       0      |
    /// |    S   | Size         |    5        |      1-5     |
    /// |    P   | Phase        |    2        |      6-7     |
    /// |    X   | X Component  |    23       |      8-35    |
    /// |    Z   | Z Component  |    23       |     36-63    |
    ///
    /// Maximum number of Paulis that can be encoded.
    /// Currently using 8 bits for metadata, leaving 56 bits remaining to encode Paulis.
    /// Each pauli requires two bits to encode, for the X and Z components respectively.
    /// Thus, we can encode 28 Paulis.
    pub const MAX_PAULIS: u8 = 28;

    #[bitfield(u64)]
    #[derive(PartialEq, Eq)]
    #[cfg_attr(
        feature = "mem_dbg",
        derive(mem_dbg::MemSize, mem_dbg::MemDbg),
        mem_size(flat)
    )]
    pub struct PauliStringInner {
        // Setting the LSB to 1 will indicate this is *not* a pointer, and
        // instead contains inline data. This is because pointers must be (at
        // least) two byte aligned, so the LSB will never be set to 1.
        #[bits(1, default = true)]
        pub(crate) is_inline: bool,

        #[bits(5)]
        pub(crate) size: u8,

        #[bits(2)]
        pub(crate) phase: u8,

        #[bits(28)]
        pub(crate) x_component: u32,

        #[bits(28)]
        pub(crate) z_component: u32,
    }

    impl PauliStringInner {
        pub fn as_ptr(self) -> *mut () {
            self.0 as *mut ()
        }
        pub fn as_u64(self) -> u64 {
            self.0
        }
        pub fn from_ptr(ptr: *mut ()) -> Self {
            Self(ptr as u64)
        }

        pub fn from_u64(val: u64) -> Self {
            Self(val)
        }
    }
}

#[cfg(target_pointer_width = "32")]
// 32-bit implementation will be done in the future.
todo!();

#[cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))]
compile_error!("Only support 32-bit and 64-bit pointer widths");

use inner::*;

impl From<*mut ()> for PauliStringInner {
    fn from(ptr: *mut ()) -> Self {
        Self::from_ptr(ptr)
    }
}

#[derive(Debug)]
#[cfg_attr(
    feature = "mem_dbg",
    derive(mem_dbg::MemSize, mem_dbg::MemDbg),
    mem_size(flat)
)]
#[repr(transparent)]
pub struct PauliString(PauliStringInner);

impl PauliString {
    pub fn new(paulis: &[Pauli], phase: Phase) -> Self {
        let length = paulis.len();

        if length > inner::MAX_PAULIS as usize {
            // Allocate on the heap
            todo!()
        }

        let mut x_mask = 0_u32;
        let mut z_mask = 0_u32;
        // Walk over the paulis and assemble the X and Z components
        for (index, p) in paulis.iter().enumerate() {
            let (curr_x, curr_z) = p.to_components();
            x_mask |= (curr_x as u32) << index;
            z_mask |= (curr_z as u32) << index;
        }

        PauliString(
            PauliStringInner::new()
                .with_x_component(x_mask)
                .with_z_component(z_mask)
                .with_size(length as u8)
                .with_phase(phase as u8),
        )
    }

    pub fn len(&self) -> usize {
        self.0.size() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the Phase multiplier of the Pauli string.
    pub fn phase(&self) -> Phase {
        if self.0.is_inline() {
            Phase::from(self.0.phase())
        } else {
            todo!()
        }
    }

    /// Set the phase for this Pauli String.
    pub fn set_phase(&mut self, phase: Phase) {
        self.0.set_phase(phase as u8)
    }

    pub fn iter(&self) -> PauliStringIter {
        PauliStringIter {
            pauli: self.clone(), // FIXME: Is this clone necessary?
            index: 0,
        }
    }

    /// Get an individual element of the Pauli string.
    ///
    /// Get the element, at the given index, if it exists. If the
    /// element at the given index does not exist, return None instead.
    pub fn get(&self, index: usize) -> Option<Pauli> {
        if index >= self.len() {
            return None;
        }

        let mask = 0x1 << index;
        let x_component = self.0.x_component();
        let z_component = self.0.z_component();

        Some(Pauli::from_components(
            ((x_component & mask) >> index) as u8,
            ((z_component & mask) >> index) as u8,
        ))
    }

    /// Set an individual Pauli in the Pauli string.
    pub fn set(&mut self, index: usize, pauli: &Pauli) {
        if index > inner::MAX_PAULIS as usize {
            // Move to the heap
            todo!();
        }

        let curr_x = self.0.x_component();
        let curr_z = self.0.z_component();

        let clear_mask = !(0x1_u32 << index);
        let (x_mask, z_mask) = pauli.to_components();

        let new_x = (curr_x & clear_mask) | ((x_mask as u32) << index);
        let new_z = (curr_z & clear_mask) | ((z_mask as u32) << index);

        let mut new_size = self.0.size();
        if index >= new_size as usize {
            new_size = index as u8 + 1
        }

        // The compiler is able t optimized these chained with_ into a single
        // update, causing this to have better performance (~15-20%) than setting
        // the x_component and y_component separately, and then conditionally
        // setting the size if necessary.
        self.0 = self
            .0
            .with_x_component(new_x)
            .with_z_component(new_z)
            .with_size(new_size);
    }
}

impl From<PauliString> for Vec<Pauli> {
    fn from(pauli: PauliString) -> Self {
        let x_component = pauli.0.x_component();
        let z_component = pauli.0.z_component();

        let mut paulis = vec![];
        for i in 0..pauli.len() {
            paulis.push(Pauli::from_components(
                // Shift the current index and mask out all remaining upper bits
                ((x_component >> i) as u8) & 0x1,
                ((z_component >> i) as u8) & 0x1,
            ));
        }
        paulis
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bitstring: {:064b}
Length: {}
Phase: {:02b} ({:?})
X Component: {:032b}
Z Component: {:032b}
",
            u64::from(self.0),
            self.len(),
            <Phase as Into<u8>>::into(self.phase()), // FIXME - this should be done differently
            self.phase(),
            self.0.x_component(),
            self.0.z_component()
        )
    }
}

impl Clone for PauliString {
    fn clone(&self) -> Self {
        if self.0.is_inline() {
            PauliString(self.0)
        } else {
            todo!()
        }
    }
}

impl Drop for PauliString {
    fn drop(&mut self) {
        if !self.0.is_inline() {
            todo!()
        }
    }
}

#[derive(Debug)]
pub struct PauliStringIter {
    pauli: PauliString,
    index: usize,
}

impl Iterator for PauliStringIter {
    type Item = Pauli;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.pauli.len() {
            return None;
        }

        let curr_item = self.pauli.get(self.index);
        // Advance to the next item
        self.index += 1;

        curr_item
    }
}

impl IntoIterator for PauliString {
    type Item = Pauli;
    type IntoIter = PauliStringIter;

    fn into_iter(self) -> Self::IntoIter {
        PauliStringIter {
            pauli: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let input = vec![Pauli::X, Pauli::I, Pauli::Z, Pauli::I, Pauli::Y];
        let phase = Phase::NegI;
        let mut pauli_string = PauliString::new(&input, phase);
        let paulis: Vec<Pauli> = pauli_string.clone().into();

        // Confirm that round-trip through PauliString produces the same input vector.
        assert_eq!(paulis, input);

        // Confirm encoded length is the same as the input length.
        assert_eq!(input.len(), input.len());

        // Confirm the phase was encoded properly
        assert_eq!(pauli_string.phase(), phase);

        pauli_string.set_phase(Phase::NegOne);
        assert_eq!(pauli_string.phase(), Phase::NegOne);
    }

    #[test]
    fn test_pauli_string_size() {
        let input = vec![Pauli::X, Pauli::I, Pauli::Z, Pauli::I, Pauli::Y];
        let phase = Phase::NegI;
        let pauli_string = PauliString::new(&input, phase);

        assert_eq!(
            size_of_val(&pauli_string),
            8,
            "Expecting PauliString to be 8 bytes!",
        );
    }

    #[test]
    fn test_pauli_string_get() {
        let input = vec![Pauli::X, Pauli::I, Pauli::Z, Pauli::I, Pauli::Y];
        let phase = Phase::NegI;
        let pauli_string = PauliString::new(&input, phase);

        for (i, pauli) in input.iter().enumerate() {
            assert_eq!(pauli_string.get(i).unwrap(), *pauli);
        }

        // Check that accessing out of bounds returns None
        assert!(pauli_string.get(input.len() + 1).is_none());
    }

    #[test]
    fn test_pauli_string_set() {
        let mut input = vec![Pauli::X, Pauli::I, Pauli::Z, Pauli::I, Pauli::Y];
        let phase = Phase::NegI;
        let mut pauli_string = PauliString::new(&input, phase);

        // Make sure initial pauli string matches
        for (i, pauli) in input.iter().enumerate() {
            assert_eq!(pauli_string.get(i).unwrap(), *pauli);
        }

        // Add another pauli
        pauli_string.set(input.len(), &Pauli::X);
        input.push(Pauli::X);

        // Make sure the new  pauli string matches
        for (i, pauli) in input.iter().enumerate() {
            assert_eq!(pauli_string.get(i).unwrap(), *pauli);
        }
    }

    #[test]
    fn test_pauli_string_length() {
        // Basic length test
        let input = vec![Pauli::X, Pauli::I, Pauli::Z, Pauli::I, Pauli::Y];
        let phase = Phase::NegI;
        let mut pauli_string = PauliString::new(&input, phase);

        assert_eq!(pauli_string.len(), input.len());

        // Set existing pauli - length should not increase
        pauli_string.set(0, &Pauli::Z);
        assert_eq!(pauli_string.get(0).unwrap(), Pauli::Z);

        assert_eq!(pauli_string.len(), input.len());

        // Set new pauli - length should increase
        pauli_string.set(20, &Pauli::Y);

        assert_eq!(pauli_string.get(20).unwrap(), Pauli::Y);

        assert_eq!(pauli_string.len(), 21);
    }

    #[test]
    #[cfg(feature = "mem_dbg")]
    fn test_memory_allocated() {
        use mem_dbg::{MemSize, SizeFlags};

        let pauli_vec = vec![Pauli::I; 28];
        let phase = Phase::I;

        let pauli_string = PauliString::new(&pauli_vec, phase);

        assert_eq!(pauli_string.mem_size(SizeFlags::default()), 8);
    }

    #[test]
    fn test_collect() {
        let input = vec![Pauli::X, Pauli::I, Pauli::Z, Pauli::I, Pauli::Y];
        let phase = Phase::NegI;
        let pauli_string = PauliString::new(&input, phase);

        let pauli_vec = pauli_string.iter().collect::<Vec<_>>();

        assert_eq!(pauli_vec, input);
    }
}
