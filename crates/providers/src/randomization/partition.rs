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

//! A partition of elements.

use std::fmt;
use std::hash::Hash;

use thiserror::Error;

/// Why a partition could not be built.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PartitionError {
    #[error("a partition must contain at least one part")]
    NoParts,
    #[error("a part must contain at least one element")]
    EmptyPart,
    #[error("an element appears in more than one part, or twice in one")]
    DuplicateElement,
}

/// A grouping of elements into non-overlapping subsets.
///
/// Order within a given part is meaningful. For instance, when sampling from a distribution over
/// multiple elements, it may not be symmetric under permutations of those elements.
///
/// Order between parts is not meaningful, so parts are sorted by their first element. That is what
/// makes [`PartialEq`] and [`Hash`] mean the same grouping rather than built in the same order.
///
/// A partition has at least one part, and every part at least one element.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Partition<T> {
    /// Every element covered, in part order then slot order.
    elements: Box<[T]>,
    /// Where each part after the first begins.
    splits: Box<[u32]>,
}

impl<T: Copy + Ord + Hash> Partition<T> {
    /// One part per element.
    pub fn singletons(elements: impl IntoIterator<Item = T>) -> Result<Self, PartitionError> {
        Self::new(elements.into_iter().map(|element| vec![element]))
    }

    /// Every element in one part.
    pub fn whole(elements: impl IntoIterator<Item = T>) -> Result<Self, PartitionError> {
        let elements: Vec<T> = elements.into_iter().collect();
        if elements.is_empty() {
            return Err(PartitionError::NoParts);
        }
        Self::new([elements])
    }

    /// Group elements explicitly.
    ///
    /// Each part's own order is kept.
    pub fn new(
        parts: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    ) -> Result<Self, PartitionError> {
        let mut parts: Vec<Vec<T>> = parts
            .into_iter()
            .map(|part| part.into_iter().collect())
            .collect();
        if parts.is_empty() {
            return Err(PartitionError::NoParts);
        }
        if parts.iter().any(|part| part.is_empty()) {
            return Err(PartitionError::EmptyPart);
        }
        // By first element, which is unique once the duplicate check below has passed, so this is a
        // total order and the result does not depend on the order the parts arrived in.
        parts.sort_unstable_by(|left, right| left[0].cmp(&right[0]));

        let mut splits = Vec::with_capacity(parts.len().saturating_sub(1));
        let mut elements = Vec::with_capacity(parts.iter().map(Vec::len).sum());
        for part in &parts {
            if !elements.is_empty() {
                splits.push(elements.len() as u32);
            }
            elements.extend_from_slice(part);
        }
        // An element may occupy only one slot of one part, so every element is distinct across the whole
        // partition. The stored order is the caller's, so this needs a sorted copy rather than a
        // window over `elements` itself.
        let mut seen = elements.clone();
        seen.sort_unstable();
        if seen.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(PartitionError::DuplicateElement);
        }

        Ok(Partition {
            elements: elements.into_boxed_slice(),
            splits: splits.into_boxed_slice(),
        })
    }

    /// How many parts there are, which is at least one.
    pub fn num_parts(&self) -> usize {
        self.splits.len() + 1
    }

    /// How many elements the partition covers, which may be fewer than its carrier has.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Every element covered in part order then slot order.
    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    /// The parts in ascending order of their first element.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[T]> {
        let starts: Vec<usize> = std::iter::once(0)
            .chain(self.splits.iter().map(|&at| at as usize))
            .chain(std::iter::once(self.elements.len()))
            .collect();
        (0..self.num_parts()).map(move |part| &self.elements[starts[part]..starts[part + 1]])
    }

    /// Whether every part is a single element.
    pub fn is_singletons(&self) -> bool {
        self.num_parts() == self.elements.len()
    }

    /// Whether this partition contains `element`.
    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }
}

impl<T: fmt::Debug> fmt::Debug for Partition<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        let mut start = 0;
        for &at in self.splits.iter() {
            list.entry(&&self.elements[start..at as usize]);
            start = at as usize;
        }
        list.entry(&&self.elements[start..]);
        list.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::hash::{BuildHasher, RandomState};

    /// A partition of `usize` elements from explicit parts, which must be valid.
    fn parts(parts: &[&[usize]]) -> Partition<usize> {
        Partition::new(parts.iter().map(|part| part.to_vec())).expect("a valid partition")
    }

    #[test]
    fn test_singletons() {
        let partition = Partition::singletons([3, 5, 4]).unwrap();
        assert_eq!(partition.num_parts(), 3);
        assert_eq!(partition.num_elements(), 3);
        assert!(partition.is_singletons());
        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            vec![[3].as_slice(), [4].as_slice(), [5].as_slice()]
        );
    }

    #[test]
    fn test_whole() {
        let partition = Partition::whole([5, 3]).unwrap();
        assert_eq!(partition.num_parts(), 1);
        assert_eq!(partition.num_elements(), 2);
        assert!(!partition.is_singletons());
        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            vec![[5, 3].as_slice()]
        );
        assert!(Partition::whole([7]).unwrap().is_singletons());
    }

    #[test]
    fn test_a_parts_own_order_is_kept_and_is_meaningful() {
        let forward = parts(&[&[0, 1]]);
        let reversed = parts(&[&[1, 0]]);
        assert_eq!(forward.iter().collect::<Vec<_>>(), vec![[0, 1].as_slice()]);
        assert_eq!(reversed.iter().collect::<Vec<_>>(), vec![[1, 0].as_slice()]);
        assert_ne!(forward, reversed);

        assert_eq!(
            Partition::new([vec![0usize, 1], vec![1, 0]]).unwrap_err(),
            PartitionError::DuplicateElement
        );
    }

    #[test]
    fn test_a_partition_with_arbitrary_elements() {
        let partition = parts(&[&[1], &[3]]);
        assert_eq!(partition.num_elements(), 2);
        assert_eq!(partition.elements(), [1, 3]);
        assert!(partition.contains(&1) && partition.contains(&3));
        assert!(!partition.contains(&0) && !partition.contains(&2));
    }

    #[test]
    fn test_the_representation_is_canonical() {
        let canonical = parts(&[&[0], &[1, 2]]);
        let hasher = RandomState::new();
        let equivalent = parts(&[&[1, 2], &[0]]);
        assert_eq!(canonical, equivalent);
        assert_eq!(hasher.hash_one(&canonical), hasher.hash_one(&equivalent));

        for different in [parts(&[&[0, 1], &[2]]), parts(&[&[0], &[2, 1]])] {
            assert_ne!(canonical, different);
            assert_ne!(hasher.hash_one(&canonical), hasher.hash_one(&different));
        }
    }

    #[test]
    fn test_iter_round_trips_the_parts_it_was_built_from() {
        let partition = parts(&[&[0], &[7, 5, 6], &[2, 1]]);
        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            vec![[0].as_slice(), [2, 1].as_slice(), [7, 5, 6].as_slice()]
        );
        assert_eq!(partition.num_parts(), 3);
        assert_eq!(partition.iter().len(), 3);
        assert_eq!(partition.num_elements(), 6);
    }

    #[test]
    fn test_a_partition_must_have_at_least_one_part() {
        // A grouping of nothing is not a grouping, and refusing it is what lets `len` be one more
        // than the split count with no special case. Every way in reports the same thing.
        assert_eq!(
            Partition::<usize>::new(Vec::<Vec<usize>>::new()).unwrap_err(),
            PartitionError::NoParts
        );
        assert_eq!(
            Partition::<usize>::whole([]).unwrap_err(),
            PartitionError::NoParts
        );
        assert_eq!(
            Partition::<usize>::singletons([]).unwrap_err(),
            PartitionError::NoParts
        );
    }

    #[test]
    fn test_new_rejects_what_is_not_a_partition() {
        assert_eq!(
            Partition::new([vec![0usize], vec![]]).unwrap_err(),
            PartitionError::EmptyPart
        );
        assert_eq!(
            Partition::new([vec![1usize, 1]]).unwrap_err(),
            PartitionError::DuplicateElement
        );
        assert_eq!(
            Partition::new([vec![0usize, 1], vec![1, 2]]).unwrap_err(),
            PartitionError::DuplicateElement
        );
    }

    #[test]
    fn test_debug_shows_the_parts() {
        assert_eq!(format!("{:?}", parts(&[&[0], &[2, 1]])), "[[0], [2, 1]]");
        assert_eq!(format!("{:?}", parts(&[&[7]])), "[[7]]");
    }
}
