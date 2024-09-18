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

use crate::QiskitError;
use hashbrown::HashMap;
use hashbrown::HashSet;
use pyo3::prelude::*;
use qiskit_circuit::Qubit;

/// Track qubits by their state
#[pyclass]
pub struct QubitTracker {
    qubits: Vec<Qubit>,
    clean: HashSet<Qubit>,
    dirty: HashSet<Qubit>,
}

#[pymethods]
impl QubitTracker {
    #[new]
    pub fn new(qubits: Vec<Qubit>, clean: HashSet<Qubit>, dirty: HashSet<Qubit>) -> Self {
        QubitTracker {
            qubits,
            clean,
            dirty,
        }
    }

    /// Return the number of clean qubits, excluding `active_qubits`
    #[pyo3(signature = (/, active_qubits=None))]
    pub fn num_clean(&self, active_qubits: Option<Vec<Qubit>>) -> usize {
        if let Some(active_qubits) = active_qubits {
            let active_qubits_as_set: HashSet<Qubit> = active_qubits.into_iter().collect();
            self.clean.difference(&active_qubits_as_set).count()
        } else {
            self.clean.len()
        }
    }

    /// Return the number of dirty qubits, excluding `active_qubits`
    #[pyo3(signature = (/, active_qubits=None))]
    pub fn num_dirty(&self, active_qubits: Option<Vec<Qubit>>) -> usize {
        if let Some(active_qubits) = active_qubits {
            let active_qubits_as_set: HashSet<Qubit> = active_qubits.into_iter().collect();
            self.dirty.difference(&active_qubits_as_set).count()
        } else {
            self.dirty.len()
        }
    }

    /// Set the state of `qubits` to used.
    /// Returns an error when a qubit in `qubits` is untracked.
    #[pyo3(signature = (qubits, /, check=false))]
    pub fn used(&mut self, qubits: Vec<Qubit>, check: bool) -> PyResult<()> {
        if check {
            for q in &qubits {
                if !self.qubits.contains(q) {
                    return Err(QiskitError::new_err(format!(
                        "Setting state of an untracked qubit: {}",
                        q.0
                    )));
                }
            }
        }

        for q in qubits {
            self.clean.remove(&q);
            self.dirty.insert(q);
        }
        Ok(())
    }

    /// Set the state of `qubits` to `0`.
    /// Returns an error when a qubit in `qubits` is untracked.
    #[pyo3(signature = (qubits, /, check=false))]
    pub fn reset(&mut self, qubits: Vec<Qubit>, check: bool) -> PyResult<()> {
        if check {
            for q in &qubits {
                if !self.qubits.contains(q) {
                    return Err(QiskitError::new_err(format!(
                        "Setting state of an untracked qubit: {}",
                        q.0
                    )));
                }
            }
        }

        for q in qubits {
            self.dirty.remove(&q);
            self.clean.insert(q);
        }
        Ok(())
    }

    /// Drops `qubits` from the tracker, making these qubits no longer available.
    /// Returns an error when a qubit in `qubits` is untracked.
    #[pyo3(signature = (qubits, /, check=false))]
    pub fn drop(&mut self, qubits: Vec<Qubit>, check: bool) -> PyResult<()> {
        if check {
            for q in &qubits {
                if !self.qubits.contains(q) {
                    return Err(QiskitError::new_err(format!(
                        "Dropping an untracked qubit: {}",
                        q.0
                    )));
                }
            }
        }

        for q in qubits {
            self.qubits.retain(|&x| x != q);
            self.dirty.remove(&q);
            self.clean.remove(&q);
        }
        Ok(())
    }

    /// Get `num_qubits` qubits, excluding `active_qubits`.
    /// `active_qubits` may include qubits that are not part of the tracker.
    #[pyo3(signature = (num_qubits, /, active_qubits=None))]
    fn borrow(&self, num_qubits: usize, active_qubits: Option<Vec<Qubit>>) -> PyResult<Vec<usize>> {
        // Compute the set of tracked active qubits
        let tracked_active_qubits: HashSet<Qubit> = if let Some(active_qubits) = active_qubits {
            active_qubits
                .iter()
                .filter(|q| self.qubits.contains(*q))
                .copied()
                .collect()
        } else {
            HashSet::new()
        };

        let num_available = self.qubits.len() - tracked_active_qubits.len();
        if num_available < num_qubits {
            return Err(QiskitError::new_err(format!(
                "Cannot borrow {} qubits, only {} available",
                num_qubits, num_available
            )));
        }

        let clean_qubits = self
            .qubits
            .iter()
            .filter(|q| !tracked_active_qubits.contains(*q))
            .filter(|q| self.clean.contains(*q));
        let dirty_qubits = self
            .qubits
            .iter()
            .filter(|q| !tracked_active_qubits.contains(*q))
            .filter(|q| self.dirty.contains(*q));

        let borrowed_qubits = clean_qubits
            .chain(dirty_qubits)
            .take(num_qubits)
            .map(|q| q.0 as usize)
            .collect();
        Ok(borrowed_qubits)
    }

    pub fn copy(
        &self,
        qubit_map: Option<HashMap<Qubit, Qubit>>,
        drop: Option<Vec<Qubit>>,
    ) -> PyResult<Self> {
        println!("IN COPY: qubit_map = {:?}, drop = {:?}", qubit_map, drop);
        let (qubits, clean, dirty) = if qubit_map.is_none() && drop.is_none() {
            // Copy everything
            (self.qubits.clone(), self.clean.clone(), self.dirty.clone())
        } else if qubit_map.is_some() {
            // Filter based on the map
            let qubit_map = qubit_map.unwrap();
            let mut qubits: Vec<Qubit> = Vec::with_capacity(self.qubits.len());
            let mut clean: HashSet<Qubit> = HashSet::with_capacity(self.clean.len());
            let mut dirty: HashSet<Qubit> = HashSet::with_capacity(self.dirty.len());
            for (old_index, new_index) in qubit_map.iter() {
                qubits.push(*new_index);
                if self.clean.contains(old_index) {
                    clean.insert(*new_index);
                } else if self.dirty.contains(old_index) {
                    dirty.insert(*new_index);
                } else {
                    return Err(QiskitError::new_err(format!(
                        "Unknown old qubit index: {}.",
                        old_index.0 as usize
                    )));
                }
            }
            (qubits, clean, dirty)
        } else {
            // Filter based on drop
            let drop = drop.unwrap();
            let mut qubits = self.qubits.clone();
            let mut clean = self.clean.clone();
            let mut dirty = self.dirty.clone();
            qubits.retain(|q| !drop.contains(q));
            clean.retain(|q| !drop.contains(q));
            dirty.retain(|q| !drop.contains(q));
            (qubits, clean, dirty)
        };

        Ok(QubitTracker {
            qubits,
            clean,
            dirty,
        })
    }

    pub fn __str__(&self) -> String {
        format!(
            "QubitTracker({}), clean: {}, dirty: {}\n\
        \tclean: {:?}\n\
        \tdirty: {:?}",
            self.qubits.len(),
            self.clean.len(),
            self.dirty.len(),
            self.clean,
            self.dirty
        )
    }
}

pub fn qubit_tracker_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<QubitTracker>()?;
    Ok(())
}
