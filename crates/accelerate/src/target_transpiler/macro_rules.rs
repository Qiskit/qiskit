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

/**
Creates an ordered set key-like collection that will be preserve insertion order in Python
while keeping the convenience of the ``set`` data structure.
 */
macro_rules! key_like_set_iterator {
    ($name:ident, $iter:ident, $keys:ident, $T:ty, $IterType:ty, $doc:literal, $pyrep:literal) => {
        #[doc = $doc]
        #[pyclass(sequence, module = "qiskit._accelerate.target")]
        #[derive(Debug, Clone)]
        pub struct $name {
            pub $keys: IndexSet<$T>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                Self::default()
            }

            fn __iter__(slf: PyRef<Self>) -> PyResult<Py<$iter>> {
                let iter = $iter {
                    iter: slf.$keys.clone().into_iter(),
                };
                Py::new(slf.py(), iter)
            }

            fn __eq__(slf: PyRef<Self>, other: Bound<PyAny>) -> PyResult<bool> {
                if let Ok(set) = other.downcast::<PySet>() {
                    for item in set.iter() {
                        let key = item.extract::<$T>()?;
                        if !(slf.$keys.contains(&key)) {
                            return Ok(false);
                        }
                    }
                } else if let Ok(self_like) = other.extract::<Self>() {
                    for item in self_like.$keys.iter() {
                        if !(slf.$keys.contains(item)) {
                            return Ok(false);
                        }
                    }
                }

                Ok(true)
            }

            fn __len__(slf: PyRef<Self>) -> usize {
                slf.$keys.len()
            }

            fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
                self.difference(other)
            }

            fn union(&self, other: &Bound<PyAny>) -> PyResult<Self> {
                if let Ok(set) = other.extract::<Self>() {
                    Ok(Self {
                        $keys: self.$keys.union(&set.$keys).cloned().collect(),
                    })
                } else if let Ok(set) = other.extract::<HashSet<$T>>() {
                    Ok(Self {
                        $keys: self
                            .$keys
                            .iter()
                            .cloned()
                            .collect::<HashSet<$T>>()
                            .union(&set)
                            .cloned()
                            .collect(),
                    })
                } else {
                    Err(PyKeyError::new_err(
                        "Could not perform union, Wrong Key Types",
                    ))
                }
            }

            fn intersection(&self, other: &Bound<PyAny>) -> PyResult<Self> {
                if let Ok(set) = other.extract::<Self>() {
                    Ok(Self {
                        $keys: self.$keys.intersection(&set.$keys).cloned().collect(),
                    })
                } else if let Ok(set) = other.extract::<HashSet<$T>>() {
                    Ok(Self {
                        $keys: self
                            .$keys
                            .iter()
                            .cloned()
                            .collect::<HashSet<$T>>()
                            .intersection(&set)
                            .cloned()
                            .collect(),
                    })
                } else {
                    Err(PyKeyError::new_err(
                        "Could not perform intersection, Wrong Key Types",
                    ))
                }
            }

            fn difference(&self, other: &Bound<PyAny>) -> PyResult<Self> {
                if let Ok(set) = other.extract::<Self>() {
                    Ok(Self {
                        $keys: self.$keys.difference(&set.$keys).cloned().collect(),
                    })
                } else if let Ok(set) = other.extract::<HashSet<$T>>() {
                    Ok(Self {
                        $keys: self
                            .$keys
                            .iter()
                            .cloned()
                            .collect::<HashSet<$T>>()
                            .difference(&set)
                            .cloned()
                            .collect(),
                    })
                } else {
                    Err(PyKeyError::new_err(
                        "Could not perform difference, Wrong Key Types",
                    ))
                }
            }

            fn __ior__(&mut self, other: &Bound<PyAny>) -> PyResult<()> {
                self.$keys = self.union(other)?.$keys;
                Ok(())
            }

            fn __iand__(&mut self, other: &Bound<PyAny>) -> PyResult<()> {
                self.$keys = self.intersection(other)?.$keys;
                Ok(())
            }

            fn __isub__(&mut self, other: &Bound<PyAny>) -> PyResult<()> {
                self.$keys = self.difference(other)?.$keys;
                Ok(())
            }
            fn __contains__(slf: PyRef<Self>, obj: $T) -> PyResult<bool> {
                Ok(slf.$keys.contains(&obj))
            }

            fn __getstate__(&self) -> (HashSet<$T>,) {
                return (self.$keys.clone().into_iter().collect::<HashSet<$T>>(),);
            }

            fn __setstate__(&mut self, state: (HashSet<$T>,)) -> PyResult<()> {
                self.$keys = state.0.into_iter().collect::<IndexSet<$T>>();
                Ok(())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    $keys: IndexSet::new(),
                }
            }
        }

        #[pyclass]
        pub struct $iter {
            pub iter: $IterType,
        }

        #[pymethods]
        impl $iter {
            fn __next__(mut slf: PyRefMut<Self>) -> Option<$T> {
                slf.iter.next()
            }

            fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
                slf
            }

            fn __length_hint__(slf: PyRef<Self>) -> usize {
                slf.iter.len()
            }
        }
    };
}

pub(crate) use key_like_set_iterator;
