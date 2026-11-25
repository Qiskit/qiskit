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

use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use std::{array, fmt, mem};

use super::gf2;
use super::pauli::{Pauli, SinglePauli};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[repr(transparent)]
struct Successor([u8; 3]);
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

struct MeasurementTableBuilder<'a> {
    automorphisms: &'a [[gf2::Mat<6>; 2]],
    rotations: qiskit_util::IndexSet<PackedData>,
    /// Primary tracker for if we've handled a Pauli.
    visited: FixedBitSet,
    /// During the reverse BFS step, we need to update a generation all at once, or we
    /// risk finding chains out of order, making them longer.
    backwards_visited: FixedBitSet,
    cur_generation: Vec<PackedData>,
    next_generation: Vec<PackedData>,
    first_use: bool,
}
impl<'a> MeasurementTableBuilder<'a> {
    #[inline]
    fn new(automorphisms: &'a [[gf2::Mat<6>; 2]]) -> Self {
        Self {
            automorphisms,
            rotations: super::native_rotations(automorphisms)
                .map(PackedData::pack)
                .collect(),
            visited: FixedBitSet::with_capacity(MeasurementTable::NUM_DATA_PAULIS),
            backwards_visited: FixedBitSet::with_capacity(MeasurementTable::NUM_DATA_PAULIS),
            cur_generation: Vec::new(),
            next_generation: Vec::new(),
            first_use: true,
        }
    }
}
impl MeasurementTableBuilder<'_> {
    #[inline]
    fn reset(&mut self) {
        if !self.first_use {
            self.visited.clear();
            self.backwards_visited.clear();
            self.cur_generation.clear();
            self.next_generation.clear();
        }
        self.first_use = false;
        // The all-identity rotation is trivially "native" (it's a no-op).
        self.visited.insert(0);
    }

    fn make_table(&mut self, pivot: SinglePauli, lens: &mut [u16]) -> Box<[Successor]> {
        self.reset();
        let mut table =
            vec![Successor::NATIVE; MeasurementTable::NUM_DATA_PAULIS].into_boxed_slice();
        for meas in super::native_measures(self.automorphisms)
            .filter(|m| m[0].at(0) == pivot)
            .map(PackedData::pack)
        {
            self.next_generation.push(meas);
            self.visited.insert(meas.index());
        }

        let mut num_generation = 0u16;
        let mut tot_visited = 0;
        // Forwards BFS: for each Pauli that was discovered in the prior generation, see
        // where we can reach from it.
        while !self.next_generation.is_empty() {
            tot_visited += self.next_generation.len();
            mem::swap(&mut self.cur_generation, &mut self.next_generation);
            if self.cur_generation.len() > MeasurementTable::NUM_DATA_PAULIS - tot_visited {
                // Swap from forwards BFS to reverse BFS.
                break;
            }
            num_generation += 1;
            for s in self.cur_generation.drain(..) {
                for &rotation in self.rotations.iter() {
                    let meas = s.evolve_pi_2(rotation);
                    if !self.visited.contains(meas.index()) {
                        table[meas.index()] = Successor::from_packed_data(s);
                        self.visited.insert(meas.index());
                        lens[meas.index()] |= (num_generation & 0xF) << (4 * (pivot as u32));
                        self.next_generation.push(meas)
                    }
                }
            }
        }
        // Reverse BFS: the forwards BFS typically will get to a point where the next
        // generation is much larger than the number of Paulis left to find.  That means
        // it's much more efficient to search "backwards" through all the Paulis we
        // haven't yet found.  The rotations are self-inverse in terms of Pauli mapping,
        // so finding a rotation that goes from "unvisited -> visited" is equivalent to
        // the reverse.
        self.next_generation
            .extend(self.visited.zeroes().map(PackedData::from_index_unmasked));
        while !self.next_generation.is_empty() {
            num_generation += 1;
            self.next_generation.retain(|&s| {
                for &rotation in self.rotations.iter() {
                    let meas = s.evolve_pi_2(rotation);
                    if self.visited.contains(meas.index()) {
                        table[s.index()] = Successor::from_packed_data(meas);
                        self.backwards_visited.insert(s.index());
                        lens[s.index()] |= (num_generation & 0xF) << (4 * (pivot as u32));
                        return false;
                    }
                }
                true
            });
            self.visited |= &self.backwards_visited;
        }
        table
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
    const NUM_DATA_PAULIS: usize = 1 << 22;

    /// Generate a measurement-nativization table for the 11q in-module PBC operations for the
    /// gross and two-gross codes.
    pub fn from_automorphisms(automorphisms: &[[gf2::Mat<6>; 2]]) -> Self {
        if rayon::current_num_threads() == 1 {
            let mut lens = vec![0u16; Self::NUM_DATA_PAULIS].into_boxed_slice();
            let mut state = MeasurementTableBuilder::new(automorphisms);
            Self {
                table: SinglePauli::IN_ORDER.map(|pivot| state.make_table(pivot, &mut lens)),
                lens,
            }
        } else {
            let mut lens: [_; 4] =
                array::from_fn(|_| vec![0u16; Self::NUM_DATA_PAULIS].into_boxed_slice());
            let table: [Box<[Successor]>; 4] = lens
                .par_iter_mut()
                .zip(SinglePauli::IN_ORDER)
                .map(|(lens, pivot)| {
                    MeasurementTableBuilder::new(automorphisms).make_table(pivot, lens)
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("iterator is a 4-length array");
            // This is a Clippy false positive, and rewriting it into zipped iterators to
            // satisfy it properly is illegibly ugly.
            #[allow(clippy::needless_range_loop)]
            for i in 0..Self::NUM_DATA_PAULIS {
                // Each table already does the shifts.
                lens[0][i] |= lens[1][i] | lens[2][i] | lens[3][i];
            }
            let [lens, ..] = lens;
            Self { table, lens }
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
