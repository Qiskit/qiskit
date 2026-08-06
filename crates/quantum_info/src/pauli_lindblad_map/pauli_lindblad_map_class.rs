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

use hashbrown::HashSet;
use std::collections::btree_map;

use rand::prelude::*;
use rand::rngs::SysRng;
use rand_distr::Bernoulli;
use rand_pcg::Pcg64Mcg;

use super::qubit_sparse_pauli::{
    ArithmeticError, CoherenceError, LabelError, Pauli, QubitSparsePauli, QubitSparsePauliList,
    QubitSparsePauliView,
};

/// A Pauli Lindblad map that stores its data in a qubit-sparse format. Note that gamma,
/// probabilities, and non_negative_rates are quantities derived from rates.
///
/// See [PyPauliLindbladMap] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliLindbladMap {
    /// The rates of each abstract term in the generator sum.  This has as many elements as
    /// terms in the sum.
    pub(crate) rates: Vec<f64>,
    /// A list of qubit sparse Paulis corresponding to the rates
    pub(crate) qubit_sparse_pauli_list: QubitSparsePauliList,
    /// The gamma parameter.
    pub(crate) gamma: f64,
    /// Probability of application of a given Pauli operator in product form. Note that if the
    /// corresponding rate is less than 0, this is a quasi-probability.
    pub(crate) probabilities: Vec<f64>,
    /// List of boolean values for the statement rate >= 0 for each rate in rates.
    pub(crate) non_negative_rates: Vec<bool>,
}

impl PauliLindbladMap {
    pub fn new(
        rates: Vec<f64>,
        qubit_sparse_pauli_list: QubitSparsePauliList,
    ) -> Result<Self, CoherenceError> {
        if rates.len() + 1 != qubit_sparse_pauli_list.boundaries().len() {
            return Err(CoherenceError::MismatchedTermCount {
                rates: rates.len(),
                qspl: qubit_sparse_pauli_list.boundaries().len() - 1,
            });
        }

        let (gamma, probabilities, non_negative_rates) = derived_values_from_rates(&rates);

        Ok(Self {
            rates,
            qubit_sparse_pauli_list,
            gamma,
            probabilities,
            non_negative_rates,
        })
    }

    /// Get an iterator over the individual generator terms of the map.
    ///
    /// Recall that two [PauliLindbladMap]s that have different term orders can still represent the
    /// same object.  Use [canonicalize] to apply a canonical ordering to the terms.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = GeneratorTermView<'_>> + '_ {
        self.rates
            .iter()
            .zip(self.qubit_sparse_pauli_list.iter())
            .map(|(&rate, qubit_sparse_pauli)| GeneratorTermView {
                rate,
                qubit_sparse_pauli,
            })
    }

    /// Get the number of qubits the map is defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli_list.num_qubits()
    }

    /// Get the number of generator terms in the map.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.rates.len()
    }

    /// Get the rates of the generator terms.
    #[inline]
    pub fn rates(&self) -> &[f64] {
        &self.rates
    }

    /// Get the probabilities associated with each generator term.
    #[inline]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Get the list of booleans for which rates are non-negative.
    #[inline]
    pub fn non_negative_rates(&self) -> &[bool] {
        &self.non_negative_rates
    }

    /// Get the indices of each [Pauli].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        self.qubit_sparse_pauli_list.indices()
    }

    /// Get the boundaries of each term.
    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        self.qubit_sparse_pauli_list.boundaries()
    }

    /// Get the [Pauli]s in the map.
    #[inline]
    pub fn paulis(&self) -> &[Pauli] {
        self.qubit_sparse_pauli_list.paulis()
    }

    /// Create a [PauliLindbladMap] representing the identity map on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Add the generator term implied by a dense string label onto this map.
    pub fn add_dense_label<L: AsRef<[u8]>>(
        &mut self,
        label: L,
        rate: f64,
    ) -> Result<(), LabelError> {
        self.qubit_sparse_pauli_list.add_dense_label(label)?;

        let (g, p, pr) = derived_values_from_rate(rate);
        self.rates.push(rate);
        self.gamma *= g;
        self.probabilities.push(p);
        self.non_negative_rates.push(pr);
        Ok(())
    }

    /// Apply a transpiler layout.
    pub fn apply_layout(
        &self,
        layout: Option<&[u32]>,
        num_qubits: u32,
    ) -> Result<Self, CoherenceError> {
        let qubit_sparse_pauli_list = self
            .qubit_sparse_pauli_list
            .apply_layout(layout, num_qubits)?;
        PauliLindbladMap::new(self.rates.clone(), qubit_sparse_pauli_list)
    }

    /// Create a new identity map (with zero generator) with pre-allocated space for the given
    /// number of summands and single-qubit bit terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_paulis: usize) -> Self {
        let qubit_sparse_pauli_list =
            QubitSparsePauliList::with_capacity(num_qubits, num_terms, num_paulis);
        Self {
            rates: Vec::with_capacity(num_terms),
            qubit_sparse_pauli_list,
            gamma: 1.0,
            probabilities: Vec::with_capacity(num_terms),
            non_negative_rates: Vec::with_capacity(num_terms),
        }
    }

    /// Clear all the generator terms from this map, making it equal to the identity map again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction of generator terms may not need to reallocate.
    pub fn clear(&mut self) {
        self.rates.clear();
        self.gamma = 1.0;
        self.probabilities.clear();
        self.non_negative_rates.clear();
        self.qubit_sparse_pauli_list.clear();
    }

    /// Add a single generator term to this map.
    pub fn add_term(&mut self, term: GeneratorTermView) -> Result<(), ArithmeticError> {
        let term = term.to_term();
        if self.num_qubits() != term.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: term.num_qubits(),
            });
        }
        let (g, p, pr) = derived_values_from_rate(term.rate);
        self.rates.push(term.rate);
        self.gamma *= g;
        self.probabilities.push(p);
        self.non_negative_rates.push(pr);
        // SAFETY: at this point we already know the term needs to be valid.
        let new_pauli = unsafe {
            QubitSparsePauli::new_unchecked(
                self.num_qubits(),
                term.qubit_sparse_pauli.paulis().to_vec().into_boxed_slice(),
                term.indices().to_vec().into_boxed_slice(),
            )
        };
        self.qubit_sparse_pauli_list
            .add_qubit_sparse_pauli(new_pauli.view())?;
        Ok(())
    }

    /// Get a view onto a representation of a single sparse term.
    ///
    /// This is effectively an indexing operation into the [PauliLindbladMap].  Recall that two
    /// [PauliLindbladMap]s that have different generator term orders can still represent the same
    /// object. Use [canonicalize] to apply a canonical ordering to the terms.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds.
    pub fn term(&self, index: usize) -> GeneratorTermView<'_> {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        GeneratorTermView {
            rate: self.rates[index],
            qubit_sparse_pauli: self.qubit_sparse_pauli_list.term(index),
        }
    }

    /// Scale the rates by a set factor.
    pub fn scale_rates(self, scale_factor: f64) -> Self {
        let new_rates = self.rates.iter().map(|r| scale_factor * r).collect();
        PauliLindbladMap::new(new_rates, self.qubit_sparse_pauli_list.clone()).unwrap()
    }

    /// Invert the map.
    pub fn inverse(self) -> Self {
        self.scale_rates(-1.)
    }

    // Compose with another PauliLindbladMap
    pub fn compose(&self, other: &PauliLindbladMap) -> Result<PauliLindbladMap, ArithmeticError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: other.num_qubits(),
            });
        }

        let mut rates: Vec<f64> = Vec::with_capacity(self.num_terms() + other.num_terms());
        rates.extend_from_slice(&self.rates);
        rates.extend_from_slice(&other.rates);

        let mut paulis = Vec::with_capacity(self.num_terms() + other.num_terms());
        paulis.extend_from_slice(self.paulis());
        paulis.extend_from_slice(other.paulis());

        let mut indices: Vec<u32> = Vec::with_capacity(self.num_terms() + other.num_terms());
        indices.extend_from_slice(self.indices());
        indices.extend_from_slice(other.indices());

        let mut boundaries: Vec<usize> = Vec::with_capacity(self.num_terms() + other.num_terms());
        boundaries.extend_from_slice(self.boundaries());
        let offset = self.boundaries()[self.boundaries().len() - 1];
        boundaries.extend(
            other.boundaries()[1..]
                .iter()
                .map(|boundary| offset + boundary),
        );

        let qubit_sparse_pauli_list = unsafe {
            QubitSparsePauliList::new_unchecked(self.num_qubits(), paulis, indices, boundaries)
        };
        Ok(PauliLindbladMap::new(rates, qubit_sparse_pauli_list).unwrap())
    }

    /// Drop every Pauli on the given `indices`, effectively replacing them with an identity.
    ///
    /// It ignores all the indices that are larger than `self.num_qubits`.
    pub fn drop_paulis(&self, indices: HashSet<u32>) -> Result<Self, CoherenceError> {
        let mut new_paulis: Vec<Pauli> = Vec::with_capacity(self.paulis().len());
        let mut new_indices: Vec<u32> = Vec::with_capacity(self.indices().len());
        let mut new_boundaries: Vec<usize> = Vec::with_capacity(self.boundaries().len());

        new_boundaries.push(0);
        let mut boundaries_idx = 1;
        let mut current_boundary = self.boundaries()[boundaries_idx];

        let mut num_dropped_paulis = 0;
        for (i, (&pauli, &index)) in self.paulis().iter().zip(self.indices().iter()).enumerate() {
            if current_boundary == i {
                new_boundaries.push(current_boundary - num_dropped_paulis);

                boundaries_idx += 1;
                current_boundary = self.boundaries()[boundaries_idx]
            }

            if indices.contains(&index) {
                num_dropped_paulis += 1;
            } else {
                new_indices.push(index);
                new_paulis.push(pauli);
            }
        }
        new_boundaries.push(current_boundary - num_dropped_paulis);

        Self::new(
            self.rates().to_vec(),
            QubitSparsePauliList::new(self.num_qubits(), new_paulis, new_indices, new_boundaries)
                .unwrap(),
        )
    }

    /// Drop qubits corresponding to the given `indices`.
    ///
    /// It ignores all the indices that are larger than `self.num_qubits`.
    pub fn drop_qubits(&self, indices: HashSet<u32>) -> Result<Self, CoherenceError> {
        let mut new_paulis: Vec<Pauli> = Vec::with_capacity(self.paulis().len());
        let mut new_indices: Vec<u32> = Vec::with_capacity(self.indices().len());
        let mut new_boundaries: Vec<usize> = Vec::with_capacity(self.boundaries().len());

        new_boundaries.push(0);
        let mut boundaries_idx = 1;
        let mut current_boundary = self.boundaries()[boundaries_idx];

        let mut num_dropped_paulis = 0;
        for (i, (&pauli, &index)) in self.paulis().iter().zip(self.indices().iter()).enumerate() {
            if current_boundary == i {
                new_boundaries.push(current_boundary - num_dropped_paulis);

                boundaries_idx += 1;
                current_boundary = self.boundaries()[boundaries_idx]
            }

            if indices.contains(&index) {
                num_dropped_paulis += 1;
            } else {
                new_indices.push(index - (indices.iter().filter(|&&x| x < index).count() as u32));
                new_paulis.push(pauli);
            }
        }
        new_boundaries.push(current_boundary - num_dropped_paulis);

        let new_num_qubits = self.num_qubits() - (indices.len() as u32);
        Self::new(
            self.rates().to_vec(),
            QubitSparsePauliList::new(new_num_qubits, new_paulis, new_indices, new_boundaries)
                .unwrap(),
        )
    }

    /// Compute the fidelity of the map for a single pauli
    pub fn pauli_fidelity(
        &self,
        qubit_sparse_pauli: &QubitSparsePauli,
    ) -> Result<f64, ArithmeticError> {
        let mut log_fid = 0.0;

        for generator_term in self.iter() {
            if !qubit_sparse_pauli.commutes(&generator_term.qubit_sparse_pauli.to_term())? {
                log_fid += -2.0 * generator_term.rate;
            }
        }

        Ok(log_fid.exp())
    }

    /// Sample sign and Pauli operator pairs from the map.
    /// Note that here the "sign" bool is interpreted as the exponent of (-1)^b.
    #[allow(clippy::type_complexity)]
    pub fn parity_sample_with_history(
        &self,
        num_samples: u64,
        seed: Option<u64>,
        scale: Option<f64>,
        local_scale: Option<Vec<f64>>,
    ) -> (
        Vec<bool>,
        QubitSparsePauliList,
        Vec<Vec<bool>>,
        Vec<Vec<bool>>,
    ) {
        let mut rng = match seed {
            Some(seed) => Pcg64Mcg::seed_from_u64(seed),
            None => Pcg64Mcg::try_from_rng(&mut SysRng).unwrap(),
        };
        let modified_probabilities;
        let modified_non_negative_rates;
        let (probabilities, non_negative_rates) = if local_scale.is_some() || scale.is_some() {
            let global = scale.unwrap_or(1.);
            let locals = local_scale.as_ref();
            let rates = self
                .rates
                .iter()
                .enumerate()
                .map(|(i, rate)| *rate * locals.map(|locals| locals[i]).unwrap_or(1.) * global)
                .collect::<Vec<_>>();
            (_, modified_probabilities, modified_non_negative_rates) =
                derived_values_from_rates(&rates);
            (
                modified_probabilities.as_slice(),
                modified_non_negative_rates.as_slice(),
            )
        } else {
            (
                self.probabilities.as_slice(),
                self.non_negative_rates.as_slice(),
            )
        };
        let mut random_signs = Vec::with_capacity(num_samples as usize);
        let mut random_paulis = QubitSparsePauliList::empty(self.num_qubits());
        let mut pauli_history = Vec::with_capacity(num_samples as usize);
        let mut signs_history = Vec::with_capacity(num_samples as usize);

        for _ in 0..num_samples {
            let mut random_sign = false;
            let mut random_pauli = QubitSparsePauli::identity(self.num_qubits());
            let mut inner_pauli_history = vec![false; self.qubit_sparse_pauli_list.num_terms()];
            let mut inner_signs_history = vec![false; self.qubit_sparse_pauli_list.num_terms()];

            for (((idx, probability), generator), non_negative_rate) in probabilities
                .iter()
                .enumerate()
                .zip(self.qubit_sparse_pauli_list.iter())
                .zip(non_negative_rates.iter())
            {
                // Sample true or false with given probability. If false, apply the Pauli
                if !Bernoulli::new(*probability).unwrap().sample(&mut rng) {
                    random_pauli = random_pauli.compose(&generator.to_term()).unwrap();
                    // if rate is negative, flip random_sign
                    random_sign = random_sign == *non_negative_rate;
                    // keep track of sampled generator
                    inner_pauli_history[idx] = true;
                    inner_signs_history[idx] = *non_negative_rate;
                }
            }

            random_signs.push(random_sign);
            random_paulis
                .add_qubit_sparse_pauli(random_pauli.view())
                .unwrap();
            pauli_history.push(inner_pauli_history);
            signs_history.push(inner_signs_history);
        }

        (random_signs, random_paulis, pauli_history, signs_history)
    }

    /// Reduce the map to its canonical form.
    ///
    /// This sums like terms, removing them if the final rate's absolute value is less than or equal
    /// to the tolerance.  The terms are reordered to some canonical ordering.
    ///
    /// This function is idempotent.
    pub fn simplify(&self, tol: f64) -> PauliLindbladMap {
        let mut terms = btree_map::BTreeMap::new();
        for term in self.iter() {
            terms
                .entry((
                    term.qubit_sparse_pauli.indices,
                    term.qubit_sparse_pauli.paulis,
                ))
                .and_modify(|r| *r += term.rate)
                .or_insert(term.rate);
        }

        let mut new_rates = Vec::with_capacity(self.num_terms());
        let mut new_paulis = Vec::with_capacity(self.num_terms());
        let mut new_indices = Vec::with_capacity(self.num_terms());
        let mut new_boundaries = Vec::with_capacity(self.num_terms());
        new_boundaries.push(0);
        for ((indices, paulis), r) in terms {
            // Don't add terms with zero coefficient or are pure identity
            if r.abs() <= tol || paulis.is_empty() {
                continue;
            }
            new_rates.push(r);
            new_paulis.extend_from_slice(paulis);
            new_indices.extend_from_slice(indices);
            new_boundaries.push(new_indices.len());
        }
        Self::new(
            new_rates,
            QubitSparsePauliList::new(self.num_qubits(), new_paulis, new_indices, new_boundaries)
                .unwrap(),
        )
        .unwrap()
    }
}

/// Given a rate, return the corresponding gamma, probability, and boolean for whether the rate is
/// non-negative.
fn derived_values_from_rate(rate: f64) -> (f64, f64, bool) {
    let w: f64 = 0.5 * (1.0 + (-2.0 * rate).exp());
    let g: f64 = w.abs() + (1.0 - w).abs();
    let p: f64 = w / g;
    let nnr: bool = rate >= 0.0;
    (g, p, nnr)
}

/// Return the gamma, probabilities, and non-negative rates bools for a vector of rates.
fn derived_values_from_rates(rates: &[f64]) -> (f64, Vec<f64>, Vec<bool>) {
    let mut gamma = 1.0;
    let mut probabilities = Vec::with_capacity(rates.len());
    let mut non_negative_rates = Vec::with_capacity(rates.len());

    for rate in rates {
        let (g, p, nnr) = derived_values_from_rate(*rate);
        gamma *= g;
        probabilities.push(p);
        non_negative_rates.push(nnr);
    }
    (gamma, probabilities, non_negative_rates)
}

/// A view object onto a single generator term of a `PauliLindbladMap`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GeneratorTermView<'a> {
    pub rate: f64,
    pub qubit_sparse_pauli: QubitSparsePauliView<'a>,
}
impl GeneratorTermView<'_> {
    /// Convert this `GeneratorTermView` into an owning [GeneratorTerm] of the same data.
    pub fn to_term(&self) -> GeneratorTerm {
        GeneratorTerm {
            rate: self.rate,
            qubit_sparse_pauli: self.qubit_sparse_pauli.to_term(),
        }
    }

    pub fn to_sparse_str(self) -> String {
        let rate = format!("{}", self.rate).replace('i', "j");
        let paulis = self
            .qubit_sparse_pauli
            .indices
            .iter()
            .zip(self.qubit_sparse_pauli.paulis)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        format!("({rate})L({paulis})")
    }
}

/// A single term from a complete `PauliLindbladMap`.
///
/// These are typically created by indexing into or iterating through a `PauliLindbladMap`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorTerm {
    /// The real rate of the term.
    pub(crate) rate: f64,
    pub(crate) qubit_sparse_pauli: QubitSparsePauli,
}
impl GeneratorTerm {
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli.num_qubits()
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    pub fn indices(&self) -> &[u32] {
        self.qubit_sparse_pauli.indices()
    }

    pub fn paulis(&self) -> &[Pauli] {
        self.qubit_sparse_pauli.paulis()
    }

    pub fn view(&self) -> GeneratorTermView<'_> {
        GeneratorTermView {
            rate: self.rate,
            qubit_sparse_pauli: self.qubit_sparse_pauli.view(),
        }
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    pub fn to_pauli_lindblad_map(&self) -> Result<PauliLindbladMap, CoherenceError> {
        PauliLindbladMap::new(
            vec![self.rate],
            self.qubit_sparse_pauli.to_qubit_sparse_pauli_list(),
        )
    }
}
