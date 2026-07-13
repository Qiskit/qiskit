// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::qubit_sparse_pauli::{
    ArithmeticError, CoherenceError, LabelError, Pauli, QubitSparsePauli, QubitSparsePauliList,
    QubitSparsePauliView,
};

/// A list of Pauli operators stored in a qubit-sparse format.
///
/// See [PyQubitSparsePauliList] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct PhasedQubitSparsePauliList {
    /// The paulis.
    pub(crate) qubit_sparse_pauli_list: QubitSparsePauliList,
    /// Phases.
    phases: Vec<isize>,
}

impl PhasedQubitSparsePauliList {
    /// Create a new phased qubit-sparse Pauli list from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        qubit_sparse_pauli_list: QubitSparsePauliList,
        phases: Vec<isize>,
    ) -> Result<Self, CoherenceError> {
        if phases.len() != qubit_sparse_pauli_list.num_terms() {
            return Err(CoherenceError::MismatchedPhaseCount {
                phases: phases.len(),
                qspl: qubit_sparse_pauli_list.num_terms(),
            });
        }
        // SAFETY: we've just done the coherence checks.
        Ok(unsafe { Self::new_unchecked(qubit_sparse_pauli_list, phases) })
    }

    /// Create a new [QubitSparsePauliList] from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        qubit_sparse_pauli_list: QubitSparsePauliList,
        phases: Vec<isize>,
    ) -> Self {
        Self {
            qubit_sparse_pauli_list,
            phases,
        }
    }

    /// Create a new empty list with pre-allocated space for the given number of Paulis and
    /// single-qubit pauli terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_paulis: usize) -> Self {
        Self {
            qubit_sparse_pauli_list: QubitSparsePauliList::with_capacity(
                num_qubits, num_terms, num_paulis,
            ),
            phases: Vec::with_capacity(num_terms),
        }
    }

    /// Get an iterator over the individual elements of the list.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = PhasedQubitSparsePauliView<'_>> + '_ {
        self.phases
            .iter()
            .zip(self.qubit_sparse_pauli_list.iter())
            .map(|(phase, qspv)| PhasedQubitSparsePauliView {
                qubit_sparse_pauli_view: qspv,
                phase,
            })
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli_list.num_qubits()
    }

    /// Get the number of elements in the list.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.phases.len()
    }

    /// Create a [PhasedQubitSparsePauliList] representing the empty list on ``num_qubits`` qubits.
    pub fn empty(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Clear all the elements of the list.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction of elements in the list may not need to reallocate.
    pub fn clear(&mut self) {
        self.qubit_sparse_pauli_list.clear();
        self.phases.clear();
    }

    /// Get a view onto a representation of a single phased sparse Pauli.
    ///
    /// This is effectively an indexing operation into the [PhasedQubitSparsePauliList].
    ///
    /// # Panics
    ///
    /// If the index is out of bounds.
    pub fn term(&self, index: usize) -> PhasedQubitSparsePauliView<'_> {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        PhasedQubitSparsePauliView {
            qubit_sparse_pauli_view: self.qubit_sparse_pauli_list.term(index),
            phase: &self.phases[index],
        }
    }

    /// Add an element to the list implied by a dense string label.
    pub fn add_dense_label<L: AsRef<[u8]>>(&mut self, label: L) -> Result<(), LabelError> {
        self.qubit_sparse_pauli_list.add_dense_label(label)?;
        let num_ys = self
            .qubit_sparse_pauli_list
            .term(self.qubit_sparse_pauli_list.num_terms() - 1)
            .num_ys();
        self.phases.push(num_ys.rem_euclid(4));
        Ok(())
    }

    /// Add a single phased sparse Pauli term to the list.
    pub fn add_phased_qubit_sparse_pauli(
        &mut self,
        term: PhasedQubitSparsePauliView,
    ) -> Result<(), ArithmeticError> {
        self.qubit_sparse_pauli_list
            .add_qubit_sparse_pauli(term.qubit_sparse_pauli_view)?;
        self.phases.push(*term.phase);
        Ok(())
    }

    /// Apply a transpiler layout.
    pub fn apply_layout(
        &self,
        layout: Option<&[u32]>,
        num_qubits: u32,
    ) -> Result<Self, CoherenceError> {
        let new_qubit_sparse_pauli_list = self
            .qubit_sparse_pauli_list
            .apply_layout(layout, num_qubits)?;

        Ok(PhasedQubitSparsePauliList {
            qubit_sparse_pauli_list: new_qubit_sparse_pauli_list,
            phases: self.phases.clone(),
        })
    }
}

/// A view object onto a single term of a `QubitSparsePauliList`.
///
/// The lengths of `paulis` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PhasedQubitSparsePauliView<'a> {
    pub qubit_sparse_pauli_view: QubitSparsePauliView<'a>,
    pub phase: &'a isize,
}
impl PhasedQubitSparsePauliView<'_> {
    /// Convert this `PhasedQubitSparsePauliView` into an owning [PhasedQubitSparsePauli] of the same data.
    pub fn to_term(&self) -> PhasedQubitSparsePauli {
        PhasedQubitSparsePauli {
            qubit_sparse_pauli: self.qubit_sparse_pauli_view.to_term(),
            phase: *self.phase,
        }
    }

    pub fn to_sparse_str(self) -> String {
        let num_ys = self.qubit_sparse_pauli_view.num_ys();
        let phase_str = match (self.phase - num_ys).rem_euclid(4) {
            0 => "",
            1 => "-i",
            2 => "-",
            3 => "i",
            _ => unreachable!("`x % 4` has only four values"),
        };

        phase_str.to_owned() + &self.qubit_sparse_pauli_view.to_sparse_str()
    }
}

/// A single phased qubit-sparse Pauli operator.
#[derive(Clone, Debug, PartialEq)]
pub struct PhasedQubitSparsePauli {
    /// The qubit sparse Pauli.
    pub(crate) qubit_sparse_pauli: QubitSparsePauli,
    /// ZX phase. Note that this is different from the "group phase" the user interacts with in the
    /// python interface.
    pub(crate) phase: isize,
}

impl PhasedQubitSparsePauli {
    /// Create a new phased qubit-sparse Pauli from the raw components that make it up.
    pub fn new(qubit_sparse_pauli: QubitSparsePauli, phase: isize) -> Self {
        Self {
            qubit_sparse_pauli,
            phase,
        }
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli.num_qubits()
    }

    /// Create the identity [QubitSparsePauli] on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self {
            qubit_sparse_pauli: QubitSparsePauli::identity(num_qubits),
            phase: 0,
        }
    }

    // Composition of two pauli operators self @ other.
    pub fn compose(
        &self,
        other: &PhasedQubitSparsePauli,
    ) -> Result<PhasedQubitSparsePauli, ArithmeticError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: other.num_qubits(),
            });
        }

        // if either are proportional to the identity, return a clone of the other with corrected
        // phase
        if self.qubit_sparse_pauli.indices().is_empty() {
            return Ok(PhasedQubitSparsePauli {
                qubit_sparse_pauli: other.qubit_sparse_pauli.clone(),
                phase: self.phase + other.phase,
            });
        }

        if other.qubit_sparse_pauli.indices().is_empty() {
            return Ok(PhasedQubitSparsePauli {
                qubit_sparse_pauli: self.qubit_sparse_pauli.clone(),
                phase: self.phase + other.phase,
            });
        }

        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        let mut new_phase = self.phase + other.phase;

        let mut self_idx = 0;
        let mut other_idx = 0;

        let self_indices = self.qubit_sparse_pauli.indices();
        let self_paulis = self.qubit_sparse_pauli.paulis();
        let other_indices = other.qubit_sparse_pauli.indices();
        let other_paulis = other.qubit_sparse_pauli.paulis();

        // iterate through each entry of self and other one time, incrementing based on the ordering
        // or equality of self_idx and other_idx, until one of them runs out of entries
        while self_idx < self_indices.len() && other_idx < other_indices.len() {
            if self_indices[self_idx] < other_indices[other_idx] {
                // if the current qubit index of self is strictly less than other, append the pauli
                paulis.push(self_paulis[self_idx]);
                indices.push(self_indices[self_idx]);
                self_idx += 1;
            } else if self_indices[self_idx] == other_indices[other_idx] {
                // if the indices are the same, perform multiplication and append if non-identity
                let new_pauli = (self_paulis[self_idx] as u8) ^ (other_paulis[other_idx] as u8);
                if new_pauli != 0 {
                    paulis.push(match new_pauli {
                        0b01 => Ok(Pauli::Z),
                        0b10 => Ok(Pauli::X),
                        0b11 => Ok(Pauli::Y),
                        _ => Err(ArithmeticError::PauliMultiplication { b: new_pauli }),
                    }?);
                    indices.push(self_indices[self_idx])
                }
                if (self_paulis[self_idx] == Pauli::X || self_paulis[self_idx] == Pauli::Y)
                    && (other_paulis[other_idx] == Pauli::Z || other_paulis[other_idx] == Pauli::Y)
                {
                    new_phase += 2;
                }
                self_idx += 1;
                other_idx += 1;
            } else {
                // same as the first if block but with roles of self and other reversed
                paulis.push(other_paulis[other_idx]);
                indices.push(other_indices[other_idx]);
                other_idx += 1;
            }
        }

        // if any entries remain in either pauli, append them
        if other_idx != other_indices.len() {
            paulis.append(&mut other_paulis[other_idx..].to_vec());
            indices.append(&mut other_indices[other_idx..].to_vec());
        } else if self_idx != self_indices.len() {
            paulis.append(&mut self_paulis[self_idx..].to_vec());
            indices.append(&mut self_indices[self_idx..].to_vec());
        }

        Ok(PhasedQubitSparsePauli {
            qubit_sparse_pauli: unsafe {
                QubitSparsePauli::new_unchecked(
                    self.num_qubits(),
                    paulis.into_boxed_slice(),
                    indices.into_boxed_slice(),
                )
            },
            phase: new_phase,
        })
    }

    /// Get a view version of this object.
    pub fn view(&self) -> PhasedQubitSparsePauliView<'_> {
        PhasedQubitSparsePauliView {
            qubit_sparse_pauli_view: self.qubit_sparse_pauli.view(),
            phase: &self.phase,
        }
    }

    /// Convert this single Pauli into a :class:`PhasedQubitSparsePauliList`.
    pub fn to_phased_qubit_sparse_pauli_list(&self) -> PhasedQubitSparsePauliList {
        PhasedQubitSparsePauliList {
            qubit_sparse_pauli_list: self.qubit_sparse_pauli.to_qubit_sparse_pauli_list(),
            phases: vec![self.phase],
        }
    }

    // Check if `self` commutes with `other`
    pub fn commutes(&self, other: &PhasedQubitSparsePauli) -> Result<bool, ArithmeticError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: other.num_qubits(),
            });
        }

        self.qubit_sparse_pauli.commutes(&other.qubit_sparse_pauli)
    }
}
