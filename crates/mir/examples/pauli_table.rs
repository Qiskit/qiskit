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

/// Simple bitwise manipulations.
mod bit {
    /// The least significant bits in each byte of a `u64`.
    pub const U64_LSBS: u64 = 0x0101_0101_0101_0101;

    /// Spread each bit of a byte into the Lsb of the corresponding byte of a `u64`.
    pub const fn spread(x: u8) -> u64 {
        let x = x as u64;
        // Move the top-four bits up by four bytes (minus four bits, so they're at the bottom).
        let x = (x | x << 28) & 0x0000_000F_0000_000F;
        // Move both sets of top-two bits up two bytes.
        let x = (x | x << 14) & 0x0003_0003_0003_0003;
        // ... and the top bits.
        (x | x << 7) & U64_LSBS
    }

    /// Copy the byte into each byte of a `u64`.
    pub const fn splat(x: u8) -> u64 {
        x as u64 * U64_LSBS
    }
}

/// Types and operations on the Galois field GF(2) and associated bitstrings of up to length 8.
pub mod gf2 {
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
        pub const fn t(self) -> Self {
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
}

/// Small Pauli representations and actions.
pub mod pauli {
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
                z: inv.t().dot(self.z),
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
}

/// Data and functions related to bivariate bicycle (BB) codes.
pub mod bb {
    use fixedbitset::FixedBitSet;
    use std::{fmt, mem};

    use super::gf2;
    use super::pauli::{Pauli, SinglePauli};

    // The bit patterns "read" right-to-left, bottom-to-top compared to the TdG paper because we
    // store col 0 in the Lsb of each byte, and row 0 in the least-significant byte.  The two
    // most-significant bits in each byte are 0 to pad the matrix.
    pub const AX_G: gf2::Mat<6> =
        gf2::Mat::from_u64(0b00101111_00010010_00011011_00001100_00100010_00001010);
    pub const AY_G: gf2::Mat<6> =
        gf2::Mat::from_u64(0b00101100_00100110_00000010_00010000_00100111_00100001);
    pub const AX_2G: gf2::Mat<6> =
        gf2::Mat::from_u64(0b00011001_00111110_00111101_00010101_00110101_00101110);
    pub const AY_2G: gf2::Mat<6> =
        gf2::Mat::from_u64(0b00100001_00111001_00010001_00000110_00111011_00011111);

    /// Get all relevant powers of a single-shift automorphism matrix.
    pub const fn powers(m: gf2::Mat<6>) -> [gf2::Mat<6>; 6] {
        let mut out = [gf2::Mat::ID; 6];
        let mut i = 0;
        while i < 5 {
            out[i + 1] = m.matmul(out[i]);
            i += 1;
        }
        out
    }

    /// Get the full binary matrix and its inverse for each shift automorphism.
    ///
    /// Includes the identity matrix as the first element.
    const fn shift_automorphisms(ax: gf2::Mat<6>, ay: gf2::Mat<6>) -> [[gf2::Mat<6>; 2]; 36] {
        let ax_power = powers(ax);
        let ay_power = powers(ay);
        let mut out = [[gf2::Mat::ID; 2]; 36];
        let mut x = 0;
        while x < 6 {
            let x_inv = if x == 0 { 0 } else { 6 - x };
            let mut y = 0;
            while y < 6 {
                let y_inv = if y == 0 { 0 } else { 6 - y };
                out[x * 6 + y] = [
                    ax_power[x].matmul(ay_power[y]),
                    ax_power[x_inv].matmul(ay_power[y_inv]),
                ];
                y += 1;
            }
            x += 1;
        }
        out
    }

    /// The shift automorphisms of the gross code, as binary matrices and their inverses.
    pub static SHIFT_AUTOMORPHISMS_G: [[gf2::Mat<6>; 2]; 36] = shift_automorphisms(AX_G, AY_G);
    /// The shift automorphisms of the two-gross code, as binary matrices and their inverses.
    pub static SHIFT_AUTOMORPHISMS_2G: [[gf2::Mat<6>; 2]; 36] = shift_automorphisms(AX_2G, AY_2G);

    /// The built-in hardware measurements as pairs of (low index), (high index) Paulis.
    pub static BUILTIN_MEASURES: [[Pauli<6>; 2]; 15] = [
        [Pauli::I, Pauli::x(0)],
        [Pauli::I, Pauli::y(0)],
        [Pauli::I, Pauli::z(0)],
        [Pauli::x(0), Pauli::I],
        [Pauli::y(0), Pauli::I],
        [Pauli::z(0), Pauli::I],
        [Pauli::x(0), Pauli::x(0)],
        [Pauli::x(0), Pauli::y(0)],
        [Pauli::x(0), Pauli::z(0)],
        [Pauli::y(0), Pauli::x(0)],
        [Pauli::y(0), Pauli::y(0)],
        [Pauli::y(0), Pauli::z(0)],
        [Pauli::z(0), Pauli::x(0)],
        [Pauli::z(0), Pauli::y(0)],
        [Pauli::z(0), Pauli::z(0)],
    ];

    /// Iterate over all native measurements of a bicycle code.
    pub fn native_measures(
        automorphisms: &[[gf2::Mat<6>; 2]],
    ) -> impl Iterator<Item = [Pauli<6>; 2]> {
        automorphisms.iter().flat_map(|&[a, a_inv]| {
            let [a_t, a_inv_t] = [a.t(), a_inv.t()];
            BUILTIN_MEASURES
                .into_iter()
                .map(move |[low, high]| [low.evolve_css(a, a_inv), high.evolve_css(a_t, a_inv_t)])
        })
    }

    /// Iterate over all native rotations of a bicycle code.
    ///
    /// Each element of the output is guaranteed to have Pauli support on qubit index 0; this
    /// indicates the pivot measurement.  There will be no full-Pauli-string duplicates in the
    /// output, but there may be multiple instances of the same native rotation on the data qubits,
    /// if multiple pivot single Paulis are valid for that rotation.
    pub fn native_rotations(
        automorphisms: &[[gf2::Mat<6>; 2]],
    ) -> impl Iterator<Item = [Pauli<6>; 2]> {
        native_measures(automorphisms).filter_map(|[low, high]| {
            let data = low.mask_support(0xFE);
            (low != data && [data, high] != [Pauli::I; 2]).then_some([low, high])
        })
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
    #[repr(transparent)]
    pub struct Successor([u8; 3]);
    impl Successor {
        // The all-identity Pauli is a no-op, so it can't even be anything's parent.  It's safe to
        // use as the marker of "native" measurements.
        const NATIVE: Self = Self([0x0, 0x0, 0x0]);
        const INDEX_MASK: u32 = 0x003f_ffff;

        #[inline]
        const fn from_packed_data(p: PackedData) -> Self {
            let b = p.0.to_le_bytes();
            Self([b[0], b[1], b[2]])
        }
        #[inline]
        const fn to_packed_data(self) -> PackedData {
            PackedData::from_index_unmasked(self.index())
        }

        #[inline]
        const fn to_u32(self) -> u32 {
            u32::from_le_bytes([self.0[0], self.0[1], self.0[2], 0])
        }

        /// The index of the next value.
        #[inline]
        pub const fn index(self) -> usize {
            (self.to_u32() & Self::INDEX_MASK) as usize
        }

        #[inline]
        pub const fn unpack(self) -> [Pauli<6>; 2] {
            let v = self.to_u32();
            [
                Pauli::new(((v & 0x1f) << 1) as u8, ((v >> 10) & 0x3e) as u8),
                Pauli::new(((v >> 5) & 0x3f) as u8, (v >> 16) as u8),
            ]
        }
    }

    /// A bitpacked representation of the 11 data qubits.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
    pub struct PackedData(u32);
    impl PackedData {
        pub const MASK: u32 = (1 << 22) - 1;

        /// Iterate over all valid Paulis.
        #[inline]
        pub fn all() -> impl ExactSizeIterator<Item = Self> {
            (0..Self::MASK + 1).map(Self)
        }

        /// Pack the low and high components into one Pauli string, storing only the data qubits.
        #[inline]
        pub const fn pack([low, high]: [Pauli<6>; 2]) -> Self {
            Self(
                ((low.x.get() as u32) >> 1)
                    | (((high.x.get()) as u32) << 5)
                    | (((low.z.get() & 0xfe) as u32) << 10)
                    | (((high.z.get()) as u32) << 16),
            )
        }
        /// Unpack the data qubits into the two `[low, high]` components.  This cannot fully
        /// reconstruct the output of [`pack`](Self::pack) beacuse that throws away the pivot Pauli.
        #[inline]
        pub const fn unpack(self) -> [Pauli<6>; 2] {
            [
                Pauli::new((self.0 << 1) as u8 & 0x3e, (self.0 >> 10) as u8 & 0x3e),
                Pauli::new((self.0 >> 5) as u8 & 0x3f, (self.0 >> 16) as u8 & 0x3f),
            ]
        }

        /// Get the (phaseless) Pauli evolution of `self` by the pi/2 rotation `rotation`.
        #[inline]
        pub const fn evolve_pi_2(self, rotation: Self) -> Self {
            let s = self.0;
            let r = rotation.0;
            let anticommute_mask = u32::MAX * self.anticommutes(rotation) as u32;
            Self(s ^ (r & anticommute_mask))
        }

        /// True iff `self` anticommutes (in total) with `other`.
        #[inline]
        pub const fn anticommutes(self, other: Self) -> bool {
            let a = self.0;
            let b = other.0;
            //(|-- x0 --| & |-- z1 -|) ^ (|-- z0 -| & |-- x1 ---|)
            (((a & 0x7ff) & (b >> 11)) ^ ((a >> 11) & (b & 0x7ff))).count_ones() & 1 != 0
        }

        /// Convert to an index.
        #[inline]
        pub const fn index(self) -> usize {
            self.0 as usize
        }

        /// Read in from an index.  Masks to only the valid bits.
        #[inline]
        pub const fn from_index(val: usize) -> Self {
            Self::from_index_unmasked(val & Self::MASK as usize)
        }
        /// Read in from an index.  Does not mask; this can result in nonsense if the `index` is out
        /// of range.
        #[inline]
        pub const fn from_index_unmasked(val: usize) -> Self {
            Self(val as u32)
        }
    }
    impl fmt::Display for PackedData {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let [low, high] = self.unpack();
            write!(f, "{high}{low}")
        }
    }

    /// A measurement- and rotation-nativization table for the 11q in-module PBC operations for some
    /// gross/two-gross code.
    ///
    /// This is stored as a set of prefix trees, one per pivot Pauli.  To read the chain, you first
    /// select the table you care about (indexed by [`SinglePauli`]), then follow chain of
    /// [`Successor::index`] until you reach [`Successor::NATIVE`] (which is 0: the identity), which
    /// indicates the measurement is native.
    #[derive(Clone, Debug)]
    pub struct MeasurementTable {
        /// One prefix tree per pivot Pauli in the resulting native measurement (so index 0 all have
        /// `I` in the pivot Pauli, then `123->XZY`).  You start from the result of
        /// [`PackedData::index`], then follow the chain of [`Successor::index`] until you reach the
        /// identity (0), which marks a native measurement.
        table: [Box<[Successor]>; 4],
        /// Bitpacked lengths of the chains in `table`.  The low 4 bits correspond to the length of
        /// the `I` chain, the next 4 to `X`, the next to `Z` and the most-significant four to `Y`.
        lens: Box<[u16]>,
    }
    impl MeasurementTable {
        /// Generate a measurement-nativization table for the 11q in-module PBC operations for the
        /// gross and two-gross codes.
        pub fn from_automorphisms(automorphisms: &[[gf2::Mat<6>; 2]]) -> Self {
            let rotations = native_rotations(automorphisms)
                .map(PackedData::pack)
                .collect::<qiskit_util::IndexSet<_>>();

            // State for the BFS search.
            let num_paulis = 1 << 22;
            let mut first = true;
            // Primary tracker for if we've handled a Pauli.
            let mut visited = FixedBitSet::with_capacity(num_paulis);
            // During the reverse BFS step, we need to update a generation all at once, or we risk
            // finding chains out of order, making them longer.
            let mut backwards_visited = FixedBitSet::with_capacity(num_paulis);
            let mut cur_generation = Vec::new();
            let mut next_generation = Vec::new();
            let mut lens = vec![0u16; num_paulis].into_boxed_slice();
            let make_table = |p: SinglePauli| {
                if !first {
                    visited.clear();
                    backwards_visited.clear();
                }
                first = false;
                let mut table = vec![Successor::NATIVE; num_paulis].into_boxed_slice();
                cur_generation.clear();
                for meas in native_measures(automorphisms)
                    .filter(|m| m[0].at(0) == p)
                    .map(PackedData::pack)
                {
                    next_generation.push(meas);
                    visited.insert(meas.index());
                }
                // The all-identity rotation is trivially "native" (it's a no-op).
                visited.insert(0);

                let mut num_generation = 0u16;
                let mut tot_visited = 0;
                // Forwards BFS: for each Pauli that was discovered in the prior generation, see
                // where we can reach from it.
                while !next_generation.is_empty() {
                    tot_visited += next_generation.len();
                    mem::swap(&mut cur_generation, &mut next_generation);
                    if cur_generation.len() > num_paulis - tot_visited {
                        // Swap from forwards BFS to reverse BFS.
                        break;
                    }
                    num_generation += 1;
                    for s in cur_generation.drain(..) {
                        for &rotation in rotations.iter() {
                            let meas = s.evolve_pi_2(rotation);
                            if !visited.contains(meas.index()) {
                                table[meas.index()] = Successor::from_packed_data(s);
                                visited.insert(meas.index());
                                lens[meas.index()] |= (num_generation & 0xF) << (4 * (p as u32));
                                next_generation.push(meas)
                            }
                        }
                    }
                }
                // Reverse BFS: the forwards BFS typically will get to a point where the next
                // generation is much larger than the number of Paulis left to find.  That means
                // it's much more efficient to search "backwards" through all the Paulis we haven't
                // yet found.  The rotations are self-inverse in terms of Pauli mapping, so finding
                // a rotation that goes from "unvisited -> visited" is equivalent to the reverse.
                next_generation.extend(visited.zeroes().map(PackedData::from_index_unmasked));
                while !next_generation.is_empty() {
                    num_generation += 1;
                    next_generation.retain(|&s| {
                        for &rotation in rotations.iter() {
                            let meas = s.evolve_pi_2(rotation);
                            if visited.contains(meas.index()) {
                                table[s.index()] = Successor::from_packed_data(meas);
                                backwards_visited.insert(s.index());
                                lens[s.index()] |= (num_generation & 0xF) << (4 * (p as u32));
                                return false;
                            }
                        }
                        true
                    });
                    visited |= &backwards_visited;
                }
                table
            };
            Self {
                table: SinglePauli::IN_ORDER.map(make_table),
                lens,
            }
        }

        fn chain_lengths(&self, pauli: PackedData) -> [(u8, SinglePauli); 4] {
            let p = self.lens[pauli.index()];
            [
                ((p & 0x0f) as u8, SinglePauli::I),
                (((p >> 4) & 0x0f) as u8, SinglePauli::X),
                (((p >> 8) & 0x0f) as u8, SinglePauli::Z),
                (((p >> 12) & 0x0f) as u8, SinglePauli::Y),
            ]
        }

        /// Get the `(length, pivot)` of the best chain for which `pivot` is anything.  Will
        /// preferentially return an identity pivot, if that's the fewest rotations.
        #[inline]
        pub fn best_chain_any(&self, pauli: PackedData) -> (u8, SinglePauli) {
            self.chain_lengths(pauli)
                .into_iter()
                .min()
                .expect("the iterator has 4 elements")
        }

        /// Get the `(length, pivot)` of the best chain for which `pivot` is not the identity.
        #[inline]
        pub fn best_chain_pauli(&self, pauli: PackedData) -> (u8, SinglePauli) {
            self.chain_lengths(pauli)
                .into_iter()
                .skip(1)
                .min()
                .expect("the iterator has 3 elements")
        }

        /// Get the length of the chain for the given `pauli`, if the `pivot` in the native
        /// measurement uses the given Pauli.
        #[inline]
        pub fn chain_len(&self, pauli: PackedData, pivot: SinglePauli) -> u8 {
            (self.lens[pauli.index()] >> (4 * pivot as u32)) as u8 & 0x0F
        }

        /// Get the next item in the chain, if any.
        #[inline]
        pub fn next(&self, cur: PackedData, pivot: SinglePauli) -> Option<PackedData> {
            let next = self.table[pivot as usize][cur.index()];
            (next != Successor::NATIVE).then_some(next.to_packed_data())
        }
    }
}

#[allow(clippy::print_stdout)]
pub fn main() {
    let table = bb::MeasurementTable::from_automorphisms(&bb::SHIFT_AUTOMORPHISMS_G);
    for pivot in pauli::SinglePauli::IN_ORDER {
        let mut histogram = [0; 5];
        for p in bb::PackedData::all() {
            histogram[table.chain_len(p, pivot) as usize] += 1;
        }
        println!("Rotations for {}: {:?}", pivot, histogram);
    }
}
