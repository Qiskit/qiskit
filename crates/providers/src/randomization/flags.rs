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

//! Sets of flags, where each flag is a variant of a fieldless enum.
//!
//! TODO: migrate to using `enumflags2`.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{BitAnd, BitOr};

/// A fieldless enum whose variants are distinct single bits of a `u32`.
pub trait BitFlag: Copy + 'static {
    /// Every variant in increasing bit order.
    const FLAGS: &'static [Self];

    /// Return the bit of this variant.
    fn bits(self) -> u32;
}

/// A set of `T` flags.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitFlags<T> {
    bits: u32,
    marker: PhantomData<T>,
}

impl<T: BitFlag> BitFlags<T> {
    /// Return the set with only `flag`.
    pub fn from_flag(flag: T) -> Self {
        Self {
            bits: flag.bits(),
            marker: PhantomData,
        }
    }

    /// Return the set with no flags.
    pub fn empty() -> Self {
        Self {
            bits: 0,
            marker: PhantomData,
        }
    }

    /// Return the set with every flag.
    pub fn all() -> Self {
        T::FLAGS.iter().copied().collect()
    }

    /// Whether the set has no flags.
    pub fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// The number of flags in the set.
    pub fn len(self) -> usize {
        self.bits.count_ones() as usize
    }

    /// The bits of every flag in the set.
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Whether every flag in `other` is in this set.
    pub fn contains<B: Into<BitFlags<T>>>(self, other: B) -> bool {
        let other = other.into();
        self.bits & other.bits == other.bits
    }

    /// Whether some flag in `other` is in this set.
    pub fn intersects<B: Into<BitFlags<T>>>(self, other: B) -> bool {
        self.bits & other.into().bits != 0
    }

    /// Add every flag in `other` to this set.
    pub fn insert<B: Into<BitFlags<T>>>(&mut self, other: B) {
        self.bits |= other.into().bits;
    }

    /// Remove every flag in `other` from this set.
    pub fn remove<B: Into<BitFlags<T>>>(&mut self, other: B) {
        self.bits &= !other.into().bits;
    }

    /// Return an iterator over the flags in the set, in increasing bit order.
    pub fn iter(self) -> Iter<T> {
        Iter { rest: self }
    }
}

impl<T: BitFlag> Default for BitFlags<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: BitFlag> From<T> for BitFlags<T> {
    fn from(flag: T) -> Self {
        Self::from_flag(flag)
    }
}

impl<T: BitFlag, B: Into<BitFlags<T>>> BitOr<B> for BitFlags<T> {
    type Output = Self;

    fn bitor(self, other: B) -> Self {
        Self {
            bits: self.bits | other.into().bits,
            marker: PhantomData,
        }
    }
}

impl<T: BitFlag, B: Into<BitFlags<T>>> BitAnd<B> for BitFlags<T> {
    type Output = Self;

    fn bitand(self, other: B) -> Self {
        Self {
            bits: self.bits & other.into().bits,
            marker: PhantomData,
        }
    }
}

impl<T: BitFlag, B: Into<BitFlags<T>>> FromIterator<B> for BitFlags<T> {
    fn from_iter<I: IntoIterator<Item = B>>(iter: I) -> Self {
        iter.into_iter()
            .fold(Self::empty(), |set, flags| set | flags)
    }
}

impl<T: BitFlag> IntoIterator for BitFlags<T> {
    type Item = T;
    type IntoIter = Iter<T>;

    fn into_iter(self) -> Iter<T> {
        self.iter()
    }
}

/// Render enum variants by name separated by `|`, or as `<empty>` for the empty set.
impl<T: BitFlag + fmt::Debug> fmt::Display for BitFlags<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut flags = self.iter();
        let Some(first) = flags.next() else {
            return f.write_str("<empty>");
        };
        write!(f, "{first:?}")?;
        for flag in flags {
            write!(f, " | {flag:?}")?;
        }
        Ok(())
    }
}

/// Render as `BitFlags(0b11, Pauli | C1)`, or as `BitFlags(0b0)` for the empty set.
impl<T: BitFlag + fmt::Debug> fmt::Debug for BitFlags<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tuple = f.debug_tuple("BitFlags");
        tuple.field(&format_args!("{:#b}", self.bits));
        if !self.is_empty() {
            tuple.field(&format_args!("{self}"));
        }
        tuple.finish()
    }
}

/// An iterator over the flags in a [`BitFlags`], in increasing bit order.
#[derive(Clone, Debug)]
pub struct Iter<T: BitFlag> {
    rest: BitFlags<T>,
}

impl<T: BitFlag> Iterator for Iter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let flag = T::FLAGS
            .iter()
            .copied()
            .find(|&flag| self.rest.contains(flag))?;
        self.rest.remove(flag);
        Some(flag)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.rest.len();
        (len, Some(len))
    }
}

impl<T: BitFlag> ExactSizeIterator for Iter<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Flag {
        A = 1 << 0,
        B = 1 << 1,
        C = 1 << 2,
    }

    impl BitFlag for Flag {
        const FLAGS: &'static [Self] = &[Self::A, Self::B, Self::C];

        fn bits(self) -> u32 {
            self as u32
        }
    }

    impl BitOr for Flag {
        type Output = BitFlags<Self>;

        fn bitor(self, other: Self) -> BitFlags<Self> {
            BitFlags::from_flag(self) | other
        }
    }

    #[test]
    fn test_empty_and_all() {
        assert!(BitFlags::<Flag>::empty().is_empty());
        assert_eq!(BitFlags::<Flag>::default(), BitFlags::empty());
        assert_eq!(BitFlags::<Flag>::all(), Flag::A | Flag::B | Flag::C);
        assert_eq!(BitFlags::<Flag>::all().len(), 3);
        assert_eq!(BitFlags::<Flag>::all().bits(), 0b111);
    }

    #[test]
    fn test_contains_and_intersects() {
        let set = Flag::A | Flag::C;
        assert!(set.contains(Flag::A));
        assert!(!set.contains(Flag::B));
        assert!(set.contains(Flag::A | Flag::C));
        assert!(!set.contains(Flag::A | Flag::B));
        assert!(set.intersects(Flag::A | Flag::B));
        assert!(!set.intersects(Flag::B));
        assert!(set.contains(BitFlags::empty()));
        assert!(!set.intersects(BitFlags::empty()));
    }

    #[test]
    fn test_insert_and_remove() {
        let mut set = BitFlags::from_flag(Flag::B);
        set.insert(Flag::A | Flag::C);
        assert_eq!(set, BitFlags::all());
        set.remove(Flag::B | Flag::C);
        assert_eq!(set, BitFlags::from_flag(Flag::A));
        set.remove(Flag::B);
        assert_eq!(set, BitFlags::from_flag(Flag::A));
    }

    #[test]
    fn test_or_and_and() {
        let set = Flag::A | Flag::B;
        assert_eq!(set | Flag::C, BitFlags::all());
        assert_eq!(set & (Flag::B | Flag::C), BitFlags::from_flag(Flag::B));
        assert_eq!(set & Flag::C, BitFlags::empty());
    }

    #[test]
    fn test_iter_is_in_bit_order() {
        let set = Flag::C | Flag::A;
        assert_eq!(set.iter().len(), 2);
        assert_eq!(set.iter().collect::<Vec<_>>(), [Flag::A, Flag::C]);
        assert_eq!(set.into_iter().collect::<BitFlags<_>>(), set);
        assert_eq!(BitFlags::<Flag>::empty().iter().next(), None);
    }

    #[test]
    fn test_display_and_debug() {
        let set = Flag::C | Flag::A;
        assert_eq!(set.to_string(), "A | C");
        assert_eq!(format!("{set:?}"), "BitFlags(0b101, A | C)");
        assert_eq!(BitFlags::<Flag>::empty().to_string(), "<empty>");
        assert_eq!(format!("{:?}", BitFlags::<Flag>::empty()), "BitFlags(0b0)");
    }
}
