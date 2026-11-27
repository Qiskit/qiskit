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

//! Small Pauli representations and actions.

use super::{bit, gf2};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum SinglePauli {
    I = 0,
    X = 1,
    Y = 3,
    Z = 2,
}
impl SinglePauli {
    pub const IN_ORDER: [Self; 4] = [Self::I, Self::X, Self::Y, Self::Z];

    pub fn from_u8(v: u8) -> Self {
        match v & 0x3 {
            0 => Self::I,
            1 => Self::X,
            2 => Self::Z,
            3 => Self::Y,
            _ => unreachable!(),
        }
    }
}
impl fmt::Display for SinglePauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::I => f.write_str("I"),
            Self::X => f.write_str("X"),
            Self::Y => f.write_str("Y"),
            Self::Z => f.write_str("Z"),
        }
    }
}

/// Simple phaseless Pauli in the XZ convention.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pauli<const N: usize> {
    pub x: gf2::Str<N>,
    pub z: gf2::Str<N>,
}
impl<const N: usize> Pauli<N> {
    /// The identity string.
    pub const I: Self = Self {
        x: gf2::Str::ZERO,
        z: gf2::Str::ZERO,
    };

    #[inline]
    pub const fn new(x: u8, z: u8) -> Self {
        Self {
            x: gf2::Str::new(x),
            z: gf2::Str::new(z),
        }
    }

    /// Get a single-bit Pauli X.
    #[inline]
    pub const fn x(bit: u8) -> Self {
        Self {
            x: gf2::Str::new(1 << bit),
            z: gf2::Str::ZERO,
        }
    }
    /// Get a single-bit Pauli Y.
    #[inline]
    pub const fn y(bit: u8) -> Self {
        Self {
            x: gf2::Str::new(1 << bit),
            z: gf2::Str::new(1 << bit),
        }
    }
    /// Get a single-bit Pauli Z.
    #[inline]
    pub const fn z(bit: u8) -> Self {
        Self {
            x: gf2::Str::ZERO,
            z: gf2::Str::new(1 << bit),
        }
    }

    /// Return a new Pauli whose support is the intersection of `self` and `mask`.
    #[inline]
    pub const fn mask_support(self, mask: u8) -> Self {
        Self {
            x: gf2::Str::new(self.x.get() & mask),
            z: gf2::Str::new(self.z.get() & mask),
        }
    }

    /// Get the single Pauli at a given index.
    #[inline]
    pub const fn at(self, idx: u8) -> SinglePauli {
        match (((self.z.get() >> idx) << 1) | ((self.x.get() >> idx) & 0x1)) & 0x3 {
            0 => SinglePauli::I,
            1 => SinglePauli::X,
            3 => SinglePauli::Y,
            2 => SinglePauli::Z,
            _ => unreachable!(),
        }
    }

    /// True iff the two Paulis commute.
    #[inline]
    pub const fn commutes(self, other: Pauli<N>) -> bool {
        ((self.x.get() & other.z.get()) ^ (self.z.get() & other.x.get()))
            .count_ones()
            .is_multiple_of(2)
    }

    /// True iff the two Paulis anticommute.
    #[inline]
    pub const fn anticommutes(self, other: Pauli<N>) -> bool {
        !self.commutes(other)
    }

    /// Evolve the Pauli by a CSS-code invertible binary matrix.
    #[inline]
    pub const fn evolve_css(self, mat: gf2::Mat<N>, inv: gf2::Mat<N>) -> Self {
        Self {
            x: mat.dot(self.x),
            z: inv.transpose().dot(self.z),
        }
    }

    /// Iterate over all the Paulis.
    pub fn all() -> impl Iterator<Item = Self> {
        (0..=gf2::Str::<N>::MASK).flat_map(|x| {
            (0..=gf2::Str::<N>::MASK).map(move |z| Self {
                x: gf2::Str::new(x),
                z: gf2::Str::new(z),
            })
        })
    }

    #[inline]
    pub const fn matmul(self, rhs: Self) -> Self {
        // Write (X_l Z_l . X_r Z_r), then Z_l X_r = X_r Z_l up to phase, and we don't track
        // phase so we can ignore it.
        Self {
            x: self.x.add(rhs.x),
            z: self.z.add(rhs.z),
        }
    }
}
impl<const N: usize> fmt::Display for Pauli<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Map the Pauli to `I:0, X:1, Y:2, Z:3` in the bytes of a `u64`.
        let x = self.x.get();
        let z = self.z.get();
        let offset = (bit::spread(z) << 1) | bit::spread(z ^ x);
        let i_mask = !(bit::spread(x | z) * 0xFF);
        let ascii = (bit::splat(b'I') & i_mask) | ((bit::splat(b'W') + offset) & !i_mask);
        // We use `to_be_bytes` because we need them in most-significant on the left order.
        f.write_str(str::from_utf8(&ascii.to_be_bytes()[8 - N..]).unwrap())
    }
}
