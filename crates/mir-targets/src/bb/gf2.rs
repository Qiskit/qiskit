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

//! Types and operations on the Galois field GF(2) and associated bitstrings of up to length 8.
use super::bit;
use std::ops;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Debug)]
pub struct Bit(bool);

/// A bitstring of elements in GF(2).
///
/// Stored in standard Lsb0 ordering.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Debug)]
#[repr(transparent)]
pub struct Str<const N: usize>(u8);
impl<const N: usize> Str<N> {
    pub const MASK: u8 = u8::MAX >> (8 - N);
    pub const ZERO: Self = Self(0);

    /// Create a new member of the field from a value.
    ///
    /// Stores the value modulo the field size.
    #[inline]
    pub const fn new(val: u8) -> Self {
        const { assert!(N > 0 && N <= 8, "Str<N> is only defined up to N = 8.") };
        Self(val & Self::MASK)
    }
    /// Get the internal value.
    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Take the dot product of this string with another.
    #[inline]
    pub const fn dot(self, other: Self) -> Bit {
        Bit(!(self.0 & other.0).count_ones().is_multiple_of(2))
    }

    /// Bitwise `and` over all entries.
    #[inline]
    pub const fn bitand(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
    /// Per-entry addition in GF(2).
    #[inline]
    pub const fn add(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }
}
impl<const N: usize> ops::BitAnd for Str<N> {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.bitand(other)
    }
}
/// Addition distributed over the GF(2) elements.  Effectively this is bitwise xor.
impl<const N: usize> ops::Add for Str<N> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        self.add(other)
    }
}
impl<const N: usize> ops::AddAssign for Str<N> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        // Makes no difference since we're very cheaply `Copy` anyway.
        *self = *self + other;
    }
}

/// A binary matrix defined over the GF(2) addition and multiplication.
///
/// This is stored in row-major order where the row at index 0 corresponds to bit 0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[repr(align(8))]
pub struct Mat<const N: usize>([Str<N>; N]);
impl<const N: usize> Mat<N> {
    /// Valid entries in a `u64` representation of the matrix for size `N`.
    const MASK: u64 = {
        let mut mask = Str::<N>::MASK as u64;
        let mut i = 1;
        while i < N {
            mask |= mask << 8;
            i += 1;
        }
        mask
    };
    /// The `N`-bit identity matrix.
    pub const ID: Self = Self::from_u64(0x8040_2010_0804_0201 & Self::MASK);

    pub const fn new(rows: [Str<N>; N]) -> Self {
        const { assert!(N > 0 && N <= 8, "Mat<N> is only defined up to N = 8.") };
        Self(rows)
    }

    /// Load up the matrix into a single u64 with zero padding.
    ///
    /// If you imagine a string has binary representation `(x7 x6 x5 x4 x3 x2 x1 x0)` then we
    /// have the matrix
    ///
    /// ```text
    ///    a0 a1 a2 a3 a4 a5 a6 a7
    ///    b0 b1 b2 b3 b4 b5 b6 b7
    ///    // ...
    ///    h0 h1 h2 h3 h4 h5 h6 h7
    /// ```
    ///
    /// If we're not a full 8x8, then the high bits (right-hand side of the matrix) are zero,
    /// as are the high-indexed rows (bottom).
    ///
    /// Most operations don't care about this zero padding; they shake out identically.
    #[inline]
    pub const fn to_u64(self) -> u64 {
        let mut bytes = [0u8; 8];
        let mut i = 0;
        while i < N {
            bytes[i] = self.0[i].0;
            i += 1;
        }
        u64::from_le_bytes(bytes)
    }

    /// Convert from a `u64` in the little-endian-row convention of [`to_u64`](Self::to_u64).
    #[inline]
    pub const fn from_u64(val: u64) -> Self {
        const { assert!(N > 0 && N <= 8, "Mat<N> is only defined up to N = 8.") };
        let bytes = (val & Self::MASK).to_le_bytes();
        // SAFETY: the mask above puts each individual `Str<N>` in range.
        let strs = unsafe { std::mem::transmute::<[u8; 8], [Str<N>; 8]>(bytes) };
        Self(*strs.first_chunk::<N>().unwrap())
    }

    /// Transpose the matrix.
    pub const fn transpose(self) -> Self {
        // Algorithm from Hacker's Delight, via Claude Sonnet 5.5.  Implementation and comments
        // by hand.

        // First read self as a `u64`.  See the comment on the method for the matrix layout.  In
        // practice, LLVM will optimise this out entirely thanks to the forced u64 alignemnt.
        let mut x = self.to_u64();
        // For the rest of the algorithm, consider that for a transpose, a bit that appears in
        // the zero-padded 8x8 matrix in (row, column) `(r, c)` (zero-indexed, where row `a` is
        // zero) is at bit position `8*r + c`.  If we write the position as a binary number,
        // it's `r2 r1 r0 c2 c1 c0`.  The transpose needs to swap that with the bit at position
        // `c2 c1 c0 r2 r1 r0` - i.e. swap the row and the column - and the same for all bits.
        //
        // We now iterate; first we swap all pairs of entries with locations
        //      (r2 r1 r0 c2 c1 c0) and (r2 r1 c0 c2 c1 r0)
        // (so positions that swap `r0` and `c0`) and have `r0 != c0`.  Then we repeat the
        // process for `r1, c1` and again for `r2, c2`.  If the row/column only require 1 or 2
        // bits, we can stop after the respective step (because we'd just be exchanging zeros).
        //
        // Each block includes a hidden "xor-swap trick", but with an extra mask in the middle
        // because only some of the bits need swapping with their shifted versions.
        if N > 0b001 {
            // Swap pairs with locations
            //      (r2 r1 r0 c2 c1 c0) and (r2 r1 c0 c2 c1 r0)
            // where `r0 != c0`.  Such locations are all 7 spaces apart (either `r0, c0 = 1, 0`
            // in which case `(r2 r1 #1 c2 c1 #0) - (r2 r1 #0 c2 c1 #1) = (0 0 0 1 1 1) = 7`.
            // We shift everything over by the 7, do the first xor on everything, then mask out
            // to keep only `r0 < c0` before shifting back to do the other two xors.
            let t = (x >> 7 ^ x) & 0x00AA_00AA_00AA_00AA;
            x ^= t << 7 | t; // `t << 7` and `t` have disjoint bits because of the mask.
        }
        if N > 0b010 {
            // Same, but for `r1, c1` and the mask `r1 < c1`.  These differences are `7 << 1` =
            // 14 because of the increased place value.
            let t = (x >> 14 ^ x) & 0x0000_CCCC_0000_CCCC;
            x ^= t << 14 | t;
        }
        if N > 0b100 {
            // Same, but for `r2, c2` and the mask `r2 < c2`.  These differences are `7 << 2` =
            // 28 because of the increased place value.
            let t = (x >> 28 ^ x) & 0x0000_0000_F0F0_F0F0;
            x ^= t << 28 | t;
        }
        Self::from_u64(x)
    }

    /// Add two matrices elementwise in GF(2).
    pub const fn add(self, rhs: Self) -> Self {
        Self::from_u64(self.to_u64() ^ rhs.to_u64())
    }

    /// Matrix multiplication with `self` on the left.
    pub const fn matmul(self, rhs: Self) -> Self {
        // This is a formulation of the multiplication as an outer product; at each iteration of
        // the loop, we do the outer product the `i`th column in `lhs` with the `i`th row in
        // `rhs`, and combine them with the sum (`xor`).
        let lhs = self.to_u64();
        let mut out = 0u64;
        let rhs_rows = rhs.to_u64().to_le_bytes();
        let mut i = 0;
        while i < N {
            // Masking out the Lsbs and multiplying by the row is the same as "splatting" `row`
            // across the bytes (output rows) where the multiplication is 1.
            out ^= (lhs >> i & bit::U64_LSBS) * (rhs_rows[i] as u64);
            i += 1;
        }
        Self::from_u64(out)
    }

    /// Matrix multiplication of a vector.
    pub const fn dot(self, vec: Str<N>) -> Str<N> {
        let mut out = 0;
        let mut i = 0;
        while i < N {
            out ^= (self.0[i].dot(vec).0 as u8) << i;
            i += 1;
        }
        Str(out)
    }
}
impl<const N: usize> ops::Add<Mat<N>> for Mat<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}
impl<const N: usize> ops::Mul<Str<N>> for Mat<N> {
    type Output = Str<N>;
    #[inline]
    fn mul(self, vec: Str<N>) -> Str<N> {
        self.dot(vec)
    }
}
impl<const N: usize> ops::Mul<Mat<N>> for Mat<N> {
    type Output = Mat<N>;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.matmul(rhs)
    }
}
impl<const N: usize> ops::MulAssign<Mat<N>> for Mat<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        // We're cheaply `Copy`.
        *self = *self * rhs;
    }
}
