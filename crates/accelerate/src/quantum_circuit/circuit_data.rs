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
use pyo3::types::{IntoPyDict, PySlice};
use pyo3::{intern, PyObject, PyResult};
use std::cmp::max;
use std::iter::zip;
use std::mem::swap;

#[pyclass(sequence, module = "qiskit._accelerate.quantum_circuit")]
#[derive(Clone, Debug)]
pub struct CircuitData {
    data: Vec<(PyObject, IndexType, IndexType)>,
    intern_context: Py<InternContext>,
    new_callable: PyObject,
    fn_idx_to_qubit: PyObject,
    fn_idx_to_clbit: PyObject,
    fn_qubit_to_idx: PyObject,
    fn_clbit_to_idx: PyObject,
}

#[derive(FromPyObject)]
pub struct ElementType {
    operation: PyObject,
    qubits: Vec<PyObject>,
    clbits: Vec<PyObject>,
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
        py: Python<'_>,
        intern_context: Py<InternContext>,
        new_callable: PyObject,
        qubits: PyObject,
        clbits: PyObject,
        qubit_indices: PyObject,
        clbit_indices: PyObject,
    ) -> PyResult<Self> {
        let fn_get_item = intern!(py, "__getitem__");
        Ok(CircuitData {
            new_callable,
            intern_context,
            data: Vec::new(),
            fn_idx_to_qubit: qubits.getattr(py, fn_get_item)?,
            fn_idx_to_clbit: clbits.getattr(py, fn_get_item)?,
            fn_qubit_to_idx: qubit_indices.getattr(py, fn_get_item)?,
            fn_clbit_to_idx: clbit_indices.getattr(py, fn_get_item)?,
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
                    |fn_idx_to_bit: &PyObject, args: &Vec<BitType>| -> PyResult<Vec<PyObject>> {
                        args.iter()
                            .map(|i| fn_idx_to_bit.call1(py, (*i,)))
                            .collect()
                    };

                if let Some((op, qargs_slot, cargs_slot)) = self.data.get(index) {
                    let cell: &PyCell<InternContext> = self.intern_context.as_ref(py);
                    let pyref: PyRef<'_, InternContext> = cell.try_borrow()?;
                    let context = &*pyref;
                    let qargs = context.lookup(*qargs_slot);
                    let cargs = context.lookup(*cargs_slot);
                    self.new_callable.call1(
                        py,
                        (
                            op,
                            extract_args(&self.fn_idx_to_qubit, qargs)?,
                            extract_args(&self.fn_idx_to_clbit, cargs)?,
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
                if let Some(_) = self.data.get(index) {
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
                let values = value.iter()?.take(slice.len()).collect::<Vec<_>>();
                if indices.step != 1 && slice.len() != values.len() {
                    return Err(PyValueError::new_err(format!(
                        "attempt to assign sequence of size {:?} to extended slice of size {:?}",
                        values.len(),
                        slice.len(),
                    )));
                }

                let mut slice_itr = slice.into_iter();
                let mut value_itr = values.into_iter();
                let enumerated = zip(&mut slice_itr, &mut value_itr);
                for (i, v) in enumerated {
                    self.__setitem__(py, Int(i), v?)?;
                }

                // Delete any extras.
                for _ in slice_itr {
                    let res = self.__delitem__(py, Int(indices.stop - 1));
                    if res.is_err() {
                        // We're empty!
                        break;
                    }
                }

                // Insert any extra values.
                for v in value_itr.rev() {
                    self.insert(py, indices.stop - 1, v?.extract()?)?;
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
        let index = index.unwrap_or(max(0, self.data.len() as isize - 1));
        let item = self.__getitem__(py, Int(index))?;
        self.__delitem__(py, Int(index))?;
        return Ok(item);
    }

    pub fn append(&mut self, py: Python<'_>, value: ElementType) -> PyResult<()> {
        self.insert(py, self.data.len() as isize, value)
    }

    pub fn extend(&mut self, py: Python<'_>, itr: Vec<ElementType>) -> PyResult<()> {
        for v in itr {
            self.append(py, v)?;
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

    pub fn __repr__(slf: &PyCell<Self>) -> PyResult<String> {
        let data = slf.iter()?.collect::<PyResult<Vec<_>>>()?;
        let reprs = data
            .into_iter()
            .map(|d| d.repr())
            .collect::<PyResult<Vec<_>>>()?;
        let strs = reprs
            .into_iter()
            .map(|x| x.to_str())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(format!("{:?}", &strs))
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

        let out_of_bounds = match kind {
            IndexFor::Lookup => index < 0 || index >= self.data.len() as isize,
            IndexFor::Insertion => index < 0 || index > self.data.len() as isize,
        };

        if out_of_bounds {
            return Err(PyIndexError::new_err(format!(
                "Index {index} is out of bounds.",
            )));
        }
        Ok(index as usize)
    }

    fn equals(slf: &PyAny, other: &PyAny) -> PyResult<bool> {
        println!("eq was called");
        let slf_len = slf.len();
        let other_len = other.len();
        if slf_len.is_ok() && other_len.is_ok() {
            if slf_len.unwrap() != other_len.unwrap() {
                println!("wrong len");
                return Ok(false);
            }
        }

        let ours_itr = match slf.iter() {
            Ok(i) => i,
            Err(_) => {
                println!("not iter!");
                return Ok(false);
            }
        };

        let theirs_itr = match other.iter() {
            Ok(i) => i,
            Err(_) => {
                println!("not iter!");
                return Ok(false);
            }
        };

        let zipped = ours_itr.zip(theirs_itr);
        for (ours, theirs) in zipped {
            if !ours?.eq(theirs?)? {
                println!("elem not equal!");
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn drop_from_cache(
        &self,
        py: Python<'_>,
        entry: (PyObject, IndexType, IndexType),
    ) -> PyResult<()> {
        let cell: &PyCell<InternContext> = self.intern_context.as_ref(py);
        let mut py_ref: PyRefMut<'_, InternContext> = cell.try_borrow_mut()?;
        let context = &mut *py_ref;
        let (_, qargs_idx, carg_idx) = entry;
        context.drop_use(qargs_idx);
        context.drop_use(carg_idx);
        Ok(())
    }

    fn get_or_cache(
        &mut self,
        py: Python<'_>,
        elem: ElementType,
    ) -> PyResult<(PyObject, IndexType, IndexType)> {
        let cell: &PyCell<InternContext> = self.intern_context.as_ref(py);
        let mut py_ref: PyRefMut<'_, InternContext> = cell.try_borrow_mut()?;
        let context = &mut *py_ref;
        let mut cache_args = |fn_bit_idx: &PyObject, bits: Vec<PyObject>| -> PyResult<IndexType> {
            let args = bits
                .into_iter()
                .map(|b| {
                    let py_idx = fn_bit_idx.call1(py, (b.clone(),))?;
                    let bit_locations = py_idx.extract::<(BitType, PyObject)>(py)?;
                    Ok(bit_locations.0)
                })
                .collect::<PyResult<Vec<BitType>>>()?;
            Ok(context.intern(args).unwrap())
        };

        Ok((
            elem.operation,
            cache_args(&self.fn_qubit_to_idx, elem.qubits)?,
            cache_args(&self.fn_clbit_to_idx, elem.clbits)?,
        ))
    }
}

impl Drop for CircuitData {
    fn drop(&mut self) {
        Python::with_gil(|py| self.clear(py)).unwrap();
    }
}
