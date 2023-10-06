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
use crate::quantum_circuit::intern_context::{BitType, IndexType, InternContext};
use hashbrown::HashMap;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList, PySlice, PyTuple};
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
    intern_context: Py<InternContext>,
    new_callable: Option<PyObject>,
    qubits: Py<PyList>,
    clbits: Py<PyList>,
    qubit_indices: Py<PyDict>,
    clbit_indices: Py<PyDict>,
}

#[derive(FromPyObject)]
pub struct ElementType {
    operation: PyObject,
    qubits: Py<PyTuple>,
    clbits: Py<PyTuple>,
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
        intern_context: Py<InternContext>,
        new_callable: PyObject,
        qubits: Py<PyList>,
        clbits: Py<PyList>,
        qubit_indices: Py<PyDict>,
        clbit_indices: Py<PyDict>,
    ) -> PyResult<Self> {
        Ok(CircuitData {
            data: Vec::new(),
            intern_context,
            new_callable: Some(new_callable),
            qubits,
            clbits,
            qubit_indices,
            clbit_indices,
        })
    }

    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        // TODO: reuse intern context once concurrency is properly
        //  handled.
        let intern_context = Py::new(py, self.context(py)?.clone())?;
        Ok(CircuitData {
            data: self.data.clone(),
            intern_context,
            new_callable: self.new_callable.as_ref().map(|c| c.clone_ref(py)),
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
                    let context = self.context(py)?;
                    let qargs = context.lookup(*qargs_slot);
                    let cargs = context.lookup(*cargs_slot);
                    self.new_callable.as_ref().unwrap().call1(
                        py,
                        (
                            op,
                            extract_args(self.qubits.as_ref(py), qargs)?,
                            extract_args(self.clbits.as_ref(py), cargs)?,
                        ),
                    )
                } else {
                    Err(PyIndexError::new_err(format!(
                        "No element at index {index} in circuit data",
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
                    let cached_entry = self.data.remove(index);
                    self.drop_from_cache(py, cached_entry)
                } else {
                    Err(PyIndexError::new_err(format!(
                        "No element at index {index} in circuit data",
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
                let values = value.iter()?.collect::<PyResult<Vec<_>>>()?;
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
                        let v: ElementType = (*v).extract()?;
                        self.insert(py, indices.stop, v)?;
                    }
                }

                Ok(())
            }
            Int(index) => {
                let index = self.convert_py_index(index, IndexFor::Lookup)?;
                let value: ElementType = value.extract()?;
                let mut cached_entry = self.get_or_cache(py, value)?;
                swap(&mut cached_entry, &mut self.data[index]);
                self.drop_from_cache(py, cached_entry)
            }
        }
    }

    pub fn insert(&mut self, py: Python<'_>, index: isize, value: ElementType) -> PyResult<()> {
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

    pub fn append(&mut self, py: Python<'_>, value: ElementType) -> PyResult<()> {
        self.insert(py, self.data.len() as isize, value)
    }

    pub fn extend(&mut self, py: Python<'_>, itr: &PyAny) -> PyResult<()> {
        if let Ok(len) = itr.len() {
            self.data.reserve(len);
        }
        for v in itr.iter()? {
            self.append(py, v?.extract()?)?;
        }
        Ok(())
    }

    pub fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut to_drop = vec![];
        swap(&mut self.data, &mut to_drop);
        for entry in to_drop.into_iter() {
            self.drop_from_cache(py, entry)?;
        }
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
            visit.call(op)?;
        }
        visit.call(&self.new_callable)?;
        visit.call(&self.qubits)?;
        visit.call(&self.clbits)?;
        visit.call(&self.qubit_indices)?;
        visit.call(&self.clbit_indices)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        // TODO: do we need to explicitly clear qubits, clbit, qubit_indices, and clbit_indices?
        for InternedInstruction(op, _, _) in self.data.iter_mut() {
            *op = None;
        }
        self.new_callable = None;
    }
}

enum IndexFor {
    Lookup,
    Insertion,
}

impl CircuitData {
    fn context<'a>(&'a self, py: Python<'a>) -> PyResult<PyRef<InternContext>> {
        let cell: &'a PyCell<InternContext> = self.intern_context.as_ref(py);
        Ok(cell.try_borrow()?)
    }

    fn context_mut<'a>(&'a self, py: Python<'a>) -> PyResult<PyRefMut<InternContext>> {
        let cell: &PyCell<InternContext> = self.intern_context.as_ref(py);
        Ok(cell.try_borrow_mut()?)
    }

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
                        "Index {index} is out of bounds.",
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

    fn drop_from_cache(&self, py: Python<'_>, entry: InternedInstruction) -> PyResult<()> {
        let mut context = self.context_mut(py)?;
        let InternedInstruction(_, qargs_idx, cargs_idx) = entry;
        context.drop_use(qargs_idx);
        context.drop_use(cargs_idx);
        Ok(())
    }

    fn get_or_cache(&mut self, py: Python<'_>, elem: ElementType) -> PyResult<InternedInstruction> {
        let mut context = self.context_mut(py)?;
        let mut cache_args = |indices: &PyDict, bits: Py<PyTuple>| -> PyResult<IndexType> {
            let args = bits
                .as_ref(py)
                .into_iter()
                .map(|b| {
                    let py_idx = indices.as_ref().get_item(b)?;
                    let bit_locations = py_idx.extract::<(BitType, PyObject)>()?;
                    Ok(bit_locations.0)
                })
                .collect::<PyResult<Vec<BitType>>>()?;
            // TODO: handle context being full instead of just faithfully unwrapping
            Ok(context.intern(args).unwrap())
        };

        Ok(InternedInstruction(
            Some(elem.operation),
            cache_args(self.qubit_indices.as_ref(py), elem.qubits)?,
            cache_args(self.clbit_indices.as_ref(py), elem.clbits)?,
        ))
    }
}

impl Drop for CircuitData {
    fn drop(&mut self) {
        Python::with_gil(|py| self.clear(py)).unwrap();
    }
}
