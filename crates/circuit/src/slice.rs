// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use thiserror::Error;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PySlice;

use self::sealed::{Descending, SequenceIndexIter};

/// A Python-space indexer for the standard `PySequence` type; a single integer or a slice.
///
/// These come in as `isize`s from Python space, since Python typically allows negative indices.
/// Use `with_len` to specialize the index to a valid Rust-space indexer into a collection of the
/// given length.
pub enum PySequenceIndex<'py> {
    Int(isize),
    Slice(Bound<'py, PySlice>),
}

impl<'py> FromPyObject<'py> for PySequenceIndex<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // `slice` can't be subclassed in Python, so it's safe (and faster) to check for it exactly.
        // The `downcast_exact` check is just a pointer comparison, so while `slice` is the less
        // common input, doing that first has little-to-no impact on the speed of the `isize` path,
        // while the reverse makes `slice` inputs significantly slower.
        if let Ok(slice) = ob.downcast_exact::<PySlice>() {
            return Ok(Self::Slice(slice.clone()));
        }
        Ok(Self::Int(ob.extract()?))
    }
}

impl<'py> PySequenceIndex<'py> {
    /// Specialize this index to a collection of the given `len`, returning a Rust-native type.
    pub fn with_len(&self, len: usize) -> Result<SequenceIndex, PySequenceIndexError> {
        match self {
            PySequenceIndex::Int(index) => {
                let index = if *index >= 0 {
                    let index = *index as usize;
                    if index >= len {
                        return Err(PySequenceIndexError::OutOfRange);
                    }
                    index
                } else {
                    len.checked_sub(index.unsigned_abs())
                        .ok_or(PySequenceIndexError::OutOfRange)?
                };
                Ok(SequenceIndex::Int(index))
            }
            PySequenceIndex::Slice(slice) => {
                let indices = slice
                    .indices(len as ::std::os::raw::c_long)
                    .map_err(PySequenceIndexError::from)?;
                if indices.step > 0 {
                    Ok(SequenceIndex::PosRange {
                        start: indices.start as usize,
                        stop: indices.stop as usize,
                        step: indices.step as usize,
                    })
                } else {
                    Ok(SequenceIndex::NegRange {
                        // `indices.start` can be negative if the collection length is 0.
                        start: (indices.start >= 0).then_some(indices.start as usize),
                        // `indices.stop` can be negative if the 0 index should be output.
                        stop: (indices.stop >= 0).then_some(indices.stop as usize),
                        step: indices.step.unsigned_abs(),
                    })
                }
            }
        }
    }
}

/// Error type for problems encountered when calling methods on `PySequenceIndex`.
#[derive(Error, Debug)]
pub enum PySequenceIndexError {
    #[error("index out of range")]
    OutOfRange,
    #[error(transparent)]
    InnerPy(#[from] PyErr),
}
impl From<PySequenceIndexError> for PyErr {
    fn from(value: PySequenceIndexError) -> PyErr {
        match value {
            PySequenceIndexError::OutOfRange => PyIndexError::new_err("index out of range"),
            PySequenceIndexError::InnerPy(inner) => inner,
        }
    }
}

/// Rust-native version of a Python sequence-like indexer.
///
/// Typically this is constructed by a call to `PySequenceIndex::with_len`, which guarantees that
/// all the indices will be in bounds for a collection of the given length.
///
/// This splits the positive- and negative-step versions of the slice in two so it can be translated
/// more easily into static dispatch.  This type can be converted into several types of iterator.
#[derive(Clone, Copy, Debug)]
pub enum SequenceIndex {
    Int(usize),
    PosRange {
        start: usize,
        stop: usize,
        step: usize,
    },
    NegRange {
        start: Option<usize>,
        stop: Option<usize>,
        step: usize,
    },
}

impl SequenceIndex {
    /// The number of indices this refers to.
    pub fn len(&self) -> usize {
        match self {
            Self::Int(_) => 1,
            Self::PosRange { start, stop, step } => {
                let gap = stop.saturating_sub(*start);
                gap / *step + (gap % *step != 0) as usize
            }
            Self::NegRange { start, stop, step } => 'arm: {
                let Some(start) = start else { break 'arm 0 };
                let gap = stop
                    .map(|stop| start.saturating_sub(stop))
                    .unwrap_or(*start + 1);
                gap / step + (gap % step != 0) as usize
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        // This is just to keep clippy happy; the length is already fairly inexpensive to calculate.
        self.len() == 0
    }

    /// Get an iterator over the indices.  This will be a single-item iterator for the case of
    /// `Self::Int`, but you probably wanted to destructure off that case beforehand anyway.
    pub fn iter(&self) -> SequenceIndexIter {
        match self {
            Self::Int(value) => SequenceIndexIter::Int(Some(*value)),
            Self::PosRange { start, step, .. } => SequenceIndexIter::PosRange {
                lowest: *start,
                step: *step,
                indices: 0..self.len(),
            },
            Self::NegRange { start, step, .. } => SequenceIndexIter::NegRange {
                // We can unwrap `highest` to an arbitrary value if `None`, because in that case the
                // `len` is 0 and the iterator will not yield any objects.
                highest: start.unwrap_or_default(),
                step: *step,
                indices: 0..self.len(),
            },
        }
    }

    // Get an iterator over the contained indices that is guaranteed to iterate from the highest
    // index to the lowest.
    pub fn descending(&self) -> Descending {
        Descending(self.iter())
    }
}

impl IntoIterator for SequenceIndex {
    type Item = usize;
    type IntoIter = SequenceIndexIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Private module to make it impossible to construct or inspect the internals of the iterator types
// from outside this file, while still allowing them to be used.
mod sealed {
    /// Custom iterator for indices for Python sequence-likes.
    ///
    /// In the range types, the `indices ` are `Range` objects that run from 0 to the length of the
    /// iterator.  In theory, we could generate the iterators ourselves, but that ends up with a lot of
    /// boilerplate.
    #[derive(Clone, Debug)]
    pub enum SequenceIndexIter {
        Int(Option<usize>),
        PosRange {
            lowest: usize,
            step: usize,
            indices: ::std::ops::Range<usize>,
        },
        NegRange {
            highest: usize,
            // The step of the iterator, but note that this is a negative range, so the forwards method
            // steps downwards from `upper` towards `lower`.
            step: usize,
            indices: ::std::ops::Range<usize>,
        },
    }
    impl Iterator for SequenceIndexIter {
        type Item = usize;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            match self {
                Self::Int(value) => value.take(),
                Self::PosRange {
                    lowest,
                    step,
                    indices,
                } => indices.next().map(|idx| *lowest + idx * *step),
                Self::NegRange {
                    highest,
                    step,
                    indices,
                } => indices.next().map(|idx| *highest - idx * *step),
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            match self {
                Self::Int(None) => (0, Some(0)),
                Self::Int(Some(_)) => (1, Some(1)),
                Self::PosRange { indices, .. } | Self::NegRange { indices, .. } => {
                    indices.size_hint()
                }
            }
        }
    }
    impl DoubleEndedIterator for SequenceIndexIter {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            match self {
                Self::Int(value) => value.take(),
                Self::PosRange {
                    lowest,
                    step,
                    indices,
                } => indices.next_back().map(|idx| *lowest + idx * *step),
                Self::NegRange {
                    highest,
                    step,
                    indices,
                } => indices.next_back().map(|idx| *highest - idx * *step),
            }
        }
    }
    impl ExactSizeIterator for SequenceIndexIter {}

    pub struct Descending(pub SequenceIndexIter);
    impl Iterator for Descending {
        type Item = usize;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            match self.0 {
                SequenceIndexIter::Int(_) | SequenceIndexIter::NegRange { .. } => self.0.next(),
                SequenceIndexIter::PosRange { .. } => self.0.next_back(),
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }
    impl DoubleEndedIterator for Descending {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            match self.0 {
                SequenceIndexIter::Int(_) | SequenceIndexIter::NegRange { .. } => {
                    self.0.next_back()
                }
                SequenceIndexIter::PosRange { .. } => self.0.next(),
            }
        }
    }
    impl ExactSizeIterator for Descending {}
}

#[cfg(test)]
mod test {
    use super::*;

    /// Get a set of test parametrisations for iterator methods.  The second argument is the
    /// expected values from a normal forward iteration.
    fn index_iterator_cases() -> impl Iterator<Item = (SequenceIndex, Vec<usize>)> {
        let pos = |start, stop, step| SequenceIndex::PosRange { start, stop, step };
        let neg = |start, stop, step| SequenceIndex::NegRange { start, stop, step };

        [
            (SequenceIndex::Int(3), vec![3]),
            (pos(0, 5, 2), vec![0, 2, 4]),
            (pos(2, 10, 1), vec![2, 3, 4, 5, 6, 7, 8, 9]),
            (pos(1, 15, 3), vec![1, 4, 7, 10, 13]),
            (neg(Some(3), None, 1), vec![3, 2, 1, 0]),
            (neg(Some(3), None, 2), vec![3, 1]),
            (neg(Some(2), Some(0), 1), vec![2, 1]),
            (neg(Some(2), Some(0), 2), vec![2]),
            (neg(Some(2), Some(0), 3), vec![2]),
            (neg(Some(10), Some(2), 3), vec![10, 7, 4]),
            (neg(None, None, 1), vec![]),
            (neg(None, None, 3), vec![]),
        ]
        .into_iter()
    }

    /// Test that the index iterator's implementation of `ExactSizeIterator` is correct.
    #[test]
    fn index_iterator() {
        for (index, forwards) in index_iterator_cases() {
            // We're testing that all the values are the same, and the `size_hint` is correct at
            // every single point.
            let mut actual = Vec::new();
            let mut sizes = Vec::new();
            let mut iter = index.iter();
            loop {
                sizes.push(iter.size_hint().0);
                if let Some(next) = iter.next() {
                    actual.push(next);
                } else {
                    break;
                }
            }
            assert_eq!(
                actual, forwards,
                "values for {:?}\nActual  : {:?}\nExpected: {:?}",
                index, actual, forwards,
            );
            let expected_sizes = (0..=forwards.len()).rev().collect::<Vec<_>>();
            assert_eq!(
                sizes, expected_sizes,
                "sizes for {:?}\nActual  : {:?}\nExpected: {:?}",
                index, sizes, expected_sizes,
            );
        }
    }

    /// Test that the index iterator's implementation of `DoubleEndedIterator` is correct.
    #[test]
    fn reversed_index_iterator() {
        for (index, forwards) in index_iterator_cases() {
            let actual = index.iter().rev().collect::<Vec<_>>();
            let expected = forwards.into_iter().rev().collect::<Vec<_>>();
            assert_eq!(
                actual, expected,
                "reversed {:?}\nActual  : {:?}\nExpected: {:?}",
                index, actual, expected,
            );
        }
    }

    /// Test that `descending` produces its values in reverse-sorted order.
    #[test]
    fn descending() {
        for (index, mut expected) in index_iterator_cases() {
            let actual = index.descending().collect::<Vec<_>>();
            expected.sort_by(|left, right| right.cmp(left));
            assert_eq!(
                actual, expected,
                "descending {:?}\nActual  : {:?}\nExpected: {:?}",
                index, actual, expected,
            );
        }
    }
}
