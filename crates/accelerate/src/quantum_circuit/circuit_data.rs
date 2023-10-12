// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::quantum_circuit::circuit_data::SliceOrInt::{Int, Slice};
use crate::quantum_circuit::circuit_instruction::CircuitInstruction;
use crate::quantum_circuit::intern_context::{BitType, IndexType, InternContext};
use hashbrown::HashMap;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyIterator, PyList, PySlice, PyTuple};
use pyo3::{PyObject, PyResult, PyTraverseError, PyVisit};
use std::cmp::{max, min};
use std::iter::zip;
use std::mem::swap;

// Private type use to store instructions with interned arg lists.
#[derive(Clone, Debug)]
struct InternedInstruction(Option<PyObject>, IndexType, IndexType);

#[pyclass(sequence, module = "qiskit._accelerate.quantum_circuit")]
#[derive(Clone, Debug)]
pub struct CircuitData {
    data: Vec<InternedInstruction>,
    intern_context: InternContext,
    qubits: Py<PyList>,
    clbits: Py<PyList>,
    qubit_indices: Py<PyDict>,
    clbit_indices: Py<PyDict>,
}

#[derive(FromPyObject)]
pub enum SliceOrInt<'a> {
    Slice(&'a PySlice),
    Int(isize),
}

#[pymethods]
impl CircuitData {
    #[new]
    pub fn new(
        qubits: Py<PyList>,
        clbits: Py<PyList>,
        qubit_indices: Py<PyDict>,
        clbit_indices: Py<PyDict>,
    ) -> PyResult<Self> {
        Ok(CircuitData {
            data: Vec::new(),
            intern_context: InternContext::new(),
            qubits,
            clbits,
            qubit_indices,
            clbit_indices,
        })
    }

    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(CircuitData {
            data: self.data.clone(),
            // TODO: reuse intern context once concurrency is properly
            //  handled.
            intern_context: self.intern_context.clone(),
            qubits: self.qubits.clone_ref(py),
            clbits: self.clbits.clone_ref(py),
            qubit_indices: self.qubit_indices.clone_ref(py),
            clbit_indices: self.clbit_indices.clone_ref(py),
        })
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    // Note: we also rely on this to make us iterable!
    pub fn __getitem__(&self, py: Python<'_>, index: SliceOrInt) -> PyResult<PyObject> {
        match index {
            Slice(slice) => {
                let slice = self.convert_py_slice(py, slice)?;
                let result = slice
                    .into_iter()
                    .map(|i| self.__getitem__(py, Int(i)))
                    .collect::<PyResult<Vec<PyObject>>>()?;
                Ok(result.into_py(py))
            }
            Int(index) => {
                let index = self.convert_py_index(index, IndexFor::Lookup)?;
                let extract_args =
                    |bits: &PyList, args: &Vec<BitType>| -> PyResult<Vec<PyObject>> {
                        args.iter()
                            .map(|i| bits.get_item(*i as usize).map(|x| x.into()))
                            .collect()
                    };

                if let Some(InternedInstruction(op, qargs_slot, cargs_slot)) = self.data.get(index)
                {
                    let qargs = self.intern_context.lookup(*qargs_slot);
                    let cargs = self.intern_context.lookup(*cargs_slot);
                    Py::new(
                        py,
                        CircuitInstruction {
                            operation: op.as_ref().unwrap().clone_ref(py),
                            qubits: PyTuple::new(py, extract_args(self.qubits.as_ref(py), qargs)?)
                                .into_py(py),
                            clbits: PyTuple::new(py, extract_args(self.clbits.as_ref(py), cargs)?)
                                .into_py(py),
                        },
                    )
                    .map(|i| i.into_py(py))
                } else {
                    Err(PyIndexError::new_err(format!(
                        "No element at index {:?} in circuit data",
                        index
                    )))
                }
            }
        }
    }

    pub fn __delitem__(&mut self, py: Python<'_>, index: SliceOrInt) -> PyResult<()> {
        match index {
            Slice(slice) => {
                let slice = self.convert_py_slice(py, slice)?;
                for i in slice {
                    self.__delitem__(py, Int(i))?;
                }
                Ok(())
            }
            Int(index) => {
                let index = self.convert_py_index(index, IndexFor::Lookup)?;
                if self.data.get(index).is_some() {
                    self.data.remove(index);
                    Ok(())
                } else {
                    Err(PyIndexError::new_err(format!(
                        "No element at index {:?} in circuit data",
                        index
                    )))
                }
            }
        }
    }

    pub fn __setitem__(
        &mut self,
        py: Python<'_>,
        index: SliceOrInt,
        value: &PyAny,
    ) -> PyResult<()> {
        match index {
            Slice(slice) => {
                let indices = slice.indices(self.data.len().try_into().unwrap())?;
                let slice = self.convert_py_slice(py, slice)?;
                let values = value.iter()?.collect::<PyResult<Vec<&PyAny>>>()?;
                if indices.step != 1 && slice.len() != values.len() {
                    return Err(PyValueError::new_err(format!(
                        "attempt to assign sequence of size {:?} to extended slice of size {:?}",
                        values.len(),
                        slice.len(),
                    )));
                }

                let enumerated = zip(slice.iter(), values.iter());
                for (i, v) in enumerated {
                    let v = v;
                    self.__setitem__(py, Int(*i), *v)?;
                }

                // Delete any extras.
                if slice.len() >= values.len() {
                    for _ in 0..(slice.len() - values.len()) {
                        let res = self.__delitem__(py, Int(indices.stop - 1));
                        if res.is_err() {
                            // We're empty!
                            break;
                        }
                    }
                } else {
                    // Insert any extra values.
                    for v in values.iter().skip(slice.len()).rev() {
                        let v: PyRef<CircuitInstruction> = v.extract()?;
                        self.insert(py, indices.stop, v)?;
                    }
                }

                Ok(())
            }
            Int(index) => {
                let index = self.convert_py_index(index, IndexFor::Lookup)?;
                let value: PyRef<CircuitInstruction> = value.extract()?;
                let mut cached_entry = self.get_or_cache(py, value)?;
                swap(&mut cached_entry, &mut self.data[index]);
                Ok(())
            }
        }
    }

    pub fn insert(
        &mut self,
        py: Python<'_>,
        index: isize,
        value: PyRef<CircuitInstruction>,
    ) -> PyResult<()> {
        let index = self.convert_py_index(index, IndexFor::Insertion)?;
        let cache_entry = self.get_or_cache(py, value)?;
        self.data.insert(index, cache_entry);
        Ok(())
    }

    pub fn pop(&mut self, py: Python<'_>, index: Option<isize>) -> PyResult<PyObject> {
        let index = index.unwrap_or_else(|| max(0, self.data.len() as isize - 1));
        let item = self.__getitem__(py, Int(index))?;
        self.__delitem__(py, Int(index))?;
        Ok(item)
    }

    pub fn append(&mut self, py: Python<'_>, value: PyRef<CircuitInstruction>) -> PyResult<()> {
        self.insert(py, self.data.len() as isize, value)
    }

    pub fn extend(&mut self, py: Python<'_>, itr: &PyAny) -> PyResult<()> {
        // if let Ok(len) = itr.len() {
        //     self.data.reserve(len);
        // }
        let itr: Py<PyIterator> = itr.iter()?.into_py(py);
        loop {
            // Create a new pool, so that PyO3 can clear memory at the end of the loop.
            let pool = unsafe { py.new_pool() };

            // It is recommended to *always* immediately set py to the pool's Python, to help
            // avoid creating references with invalid lifetimes.
            let py = pool.python();

            match itr.as_ref(py).next() {
                None => {
                    break;
                }
                Some(v) => {
                    self.append(py, v?.extract()?)?;
                }
            }
        }
        Ok(())
    }

    pub fn clear(&mut self, _py: Python<'_>) -> PyResult<()> {
        let mut to_drop = vec![];
        swap(&mut self.data, &mut to_drop);
        Ok(())
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __richcmp__(
        slf: &PyCell<Self>,
        other: &PyAny,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        match op {
            CompareOp::Eq => CircuitData::equals(slf, other).map(|r| r.into_py(py)),
            CompareOp::Ne => CircuitData::equals(slf, other).map(|r| (!r).into_py(py)),
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for InternedInstruction(op, _, _) in self.data.iter() {
            if let Some(op) = op {
                visit.call(op)?;
            }
        }
        visit.call(&self.qubits)?;
        visit.call(&self.clbits)?;
        visit.call(&self.qubit_indices)?;
        visit.call(&self.clbit_indices)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        // Clear anything that could have a reference cycle.
        for inst in self.data.iter_mut() {
            inst.0 = None;
        }
    }
}

enum IndexFor {
    Lookup,
    Insertion,
}

impl CircuitData {
    fn convert_py_slice(&self, py: Python<'_>, slice: &PySlice) -> PyResult<Vec<isize>> {
        let dict: HashMap<&str, PyObject> =
            HashMap::from([("s", slice.into()), ("length", self.data.len().into_py(py))]);
        py.eval(
            "list(range(*s.indices(length)))",
            None,
            Some(dict.into_py_dict(py)),
        )?
        .extract()
    }

    fn convert_py_index(&self, index: isize, kind: IndexFor) -> PyResult<usize> {
        let index = if index < 0 {
            index + self.data.len() as isize
        } else {
            index
        };

        let index = match kind {
            IndexFor::Lookup => {
                if index < 0 || index >= self.data.len() as isize {
                    return Err(PyIndexError::new_err(format!(
                        "Index {:?} is out of bounds.",
                        index,
                    )));
                }
                index
            }
            IndexFor::Insertion => min(max(0, index), self.data.len() as isize),
        };
        Ok(index as usize)
    }

    fn equals(slf: &PyAny, other: &PyAny) -> PyResult<bool> {
        let slf_len = slf.len()?;
        let other_len = other.len();
        if other_len.is_ok() && slf_len != other_len.unwrap() {
            return Ok(false);
        }
        let mut ours_itr = slf.iter()?;
        let mut theirs_itr = match other.iter() {
            Ok(i) => i,
            Err(_) => {
                return Ok(false);
            }
        };
        loop {
            match (ours_itr.next(), theirs_itr.next()) {
                (Some(ours), Some(theirs)) => {
                    if !ours?.eq(theirs?)? {
                        return Ok(false);
                    }
                }
                (None, None) => {
                    return Ok(true);
                }
                _ => {
                    return Ok(false);
                }
            }
        }
    }

    fn get_or_cache(
        &mut self,
        py: Python<'_>,
        elem: PyRef<CircuitInstruction>,
    ) -> PyResult<InternedInstruction> {
        let mut cache_args = |indices: &PyDict, bits: &PyTuple| -> PyResult<IndexType> {
            let args = bits
                .into_iter()
                .map(|b| {
                    let py_idx = indices.as_ref().get_item(b)?;
                    let bit_locations = py_idx.downcast_exact::<PyTuple>()?;
                    Ok(bit_locations.get_item(0)?.extract::<BitType>()?)
                })
                .collect::<PyResult<Vec<BitType>>>()?;
            Ok(self.intern_context.intern(args))
        };

        Ok(InternedInstruction(
            Some(elem.operation.clone_ref(py)),
            cache_args(self.qubit_indices.as_ref(py), elem.qubits.as_ref(py))?,
            cache_args(self.clbit_indices.as_ref(py), elem.clbits.as_ref(py))?,
        ))
    }
}
