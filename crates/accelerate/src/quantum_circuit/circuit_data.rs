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
use crate::quantum_circuit::py_ext;
use hashbrown::HashMap;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyIterator, PyList, PySlice, PyTuple, PyType};
use pyo3::{AsPyPointer, PyObject, PyResult, PyTraverseError, PyVisit};
use std::hash::{Hash, Hasher};
use std::iter::zip;

// Private type used to store instructions with interned arg lists.
#[derive(Clone, Debug)]
struct InternedInstruction {
    op: Option<PyObject>,
    qubits_id: IndexType,
    clbits_id: IndexType,
}

#[derive(Clone, Debug)]
struct _BitAsKey {
    hash: isize,
    id: u64,
    bit: PyObject,
}

impl _BitAsKey {
    fn new(bit: &PyAny) -> PyResult<Self> {
        Ok(_BitAsKey {
            hash: bit.hash()?,
            id: bit.as_ptr() as u64,
            bit: bit.into_py(bit.py()),
        })
    }
}

impl Hash for _BitAsKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_isize(self.hash);
    }
}

impl PartialEq for _BitAsKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            || Python::with_gil(|py| {
                self.bit
                    .as_ref(py)
                    .repr()
                    .unwrap()
                    .eq(other.bit.as_ref(py).repr().unwrap())
                    .unwrap()
            })
    }
}

impl Eq for _BitAsKey {}

#[pyclass(sequence, module = "qiskit._accelerate.quantum_circuit")]
#[derive(Clone, Debug)]
pub struct CircuitData {
    data: Vec<InternedInstruction>,
    intern_context: InternContext,
    #[pyo3(get)]
    qubits: Py<PyList>,
    #[pyo3(get)]
    clbits: Py<PyList>,
    qubits_native: Vec<PyObject>,
    clbits_native: Vec<PyObject>,
    qubit_indices_native: HashMap<_BitAsKey, BitType>,
    clbit_indices_native: HashMap<_BitAsKey, BitType>,
}

#[derive(FromPyObject)]
pub enum SliceOrInt<'a> {
    Slice(&'a PySlice),
    Int(isize),
}

#[pymethods]
impl CircuitData {
    #[new]
    #[pyo3(signature = (qubits=None, clbits=None, data=None, reserve=0))]
    pub fn new(
        py: Python<'_>,
        qubits: Option<&PyAny>,
        clbits: Option<&PyAny>,
        data: Option<&PyAny>,
        reserve: usize,
    ) -> PyResult<Self> {
        let mut self_ = CircuitData {
            data: if reserve > 0 {
                Vec::with_capacity(reserve)
            } else {
                Vec::new()
            },
            intern_context: InternContext::new(),
            qubits: PyList::empty(py).into_py(py),
            clbits: PyList::empty(py).into_py(py),
            qubits_native: Vec::new(),
            clbits_native: Vec::new(),
            qubit_indices_native: HashMap::new(),
            clbit_indices_native: HashMap::new(),
        };
        if let Some(qubits) = qubits {
            for bit in qubits.iter()? {
                self_.add_qubit(py, bit?)?;
            }
        }
        if let Some(clbits) = clbits {
            for bit in clbits.iter()? {
                self_.add_clbit(py, bit?)?;
            }
        }
        if let Some(data) = data {
            self_.extend(py, data)?;
        }
        Ok(self_)
    }

    pub fn __reduce__(self_: &PyCell<CircuitData>, py: Python<'_>) -> PyResult<PyObject> {
        let ty: &PyType = self_.get_type();
        let args = {
            let self_ = self_.borrow();
            (
                self_.qubits.clone_ref(py),
                self_.clbits.clone_ref(py),
                None::<()>,
                self_.data.len(),
            )
        };
        Ok((ty, args, None::<()>, self_.iter()?).into_py(py))
    }

    pub fn add_qubit(&mut self, py: Python<'_>, bit: &PyAny) -> PyResult<()> {
        let idx = self.qubits_native.len() as u32;
        self.qubit_indices_native.insert(_BitAsKey::new(bit)?, idx);
        self.qubits_native.push(bit.into_py(py));
        self.qubits.as_ref(py).append(bit)
    }

    pub fn add_clbit(&mut self, py: Python<'_>, bit: &PyAny) -> PyResult<()> {
        let idx = self.clbits_native.len() as u32;
        self.clbit_indices_native.insert(_BitAsKey::new(bit)?, idx);
        self.clbits_native.push(bit.into_py(py));
        self.clbits.as_ref(py).append(bit)
    }

    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(CircuitData {
            data: self.data.clone(),
            // TODO: reuse intern context once concurrency is properly
            //  handled.
            intern_context: self.intern_context.clone(),
            qubits: self.qubits.clone_ref(py),
            clbits: self.clbits.clone_ref(py),
            qubits_native: self.qubits_native.clone(),
            clbits_native: self.clbits_native.clone(),
            qubit_indices_native: self.qubit_indices_native.clone(),
            clbit_indices_native: self.clbit_indices_native.clone(),
        })
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    // Note: we also rely on this to make us iterable!
    pub fn __getitem__<'py>(&self, py: Python<'py>, index: &PyAny) -> PyResult<PyObject> {
        fn get_at(
            self_: &CircuitData,
            py: Python<'_>,
            index: isize,
        ) -> PyResult<Py<CircuitInstruction>> {
            let index = self_.convert_py_index(index, IndexFor::Lookup)?;
            if let Some(inst) = self_.data.get(index) {
                Py::new(
                    py,
                    CircuitInstruction {
                        operation: inst.op.as_ref().unwrap().clone_ref(py),
                        qubits: py_ext::tuple_new(
                            py,
                            self_
                                .intern_context
                                .lookup(inst.qubits_id)
                                .iter()
                                .map(|i| self_.qubits_native[*i as usize].clone_ref(py))
                                .collect(),
                        ),
                        clbits: py_ext::tuple_new(
                            py,
                            self_
                                .intern_context
                                .lookup(inst.clbits_id)
                                .iter()
                                .map(|i| self_.clbits_native[*i as usize].clone_ref(py))
                                .collect(),
                        ),
                    },
                )
            } else {
                Err(PyIndexError::new_err(format!(
                    "No element at index {:?} in circuit data",
                    index
                )))
            }
        }

        if index.is_exact_instance_of::<PySlice>() {
            let slice = self.convert_py_slice(py, index.downcast_exact::<PySlice>()?)?;
            let result = slice
                .into_iter()
                .map(|i| get_at(self, py, i))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(result.into_py(py))
        } else {
            Ok(get_at(self, py, index.extract()?)?.into_py(py))
        }
    }

    pub fn __delitem__(&mut self, py: Python<'_>, index: SliceOrInt) -> PyResult<()> {
        match index {
            Slice(slice) => {
                let slice = self.convert_py_slice(py, slice)?;
                for (i, x) in slice.into_iter().enumerate() {
                    self.__delitem__(py, Int(x - i as isize))?;
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
                std::mem::swap(&mut cached_entry, &mut self.data[index]);
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

    pub fn pop(&mut self, py: Python<'_>, index: Option<PyObject>) -> PyResult<PyObject> {
        let index =
            index.unwrap_or_else(|| std::cmp::max(0, self.data.len() as isize - 1).into_py(py));
        let item = self.__getitem__(py, index.as_ref(py))?;
        self.__delitem__(py, index.as_ref(py).extract()?)?;
        Ok(item)
    }

    pub fn append(&mut self, py: Python<'_>, value: PyRef<CircuitInstruction>) -> PyResult<()> {
        let cache_entry = self.get_or_cache(py, value)?;
        self.data.push(cache_entry);
        Ok(())
    }

    pub fn reserve(&mut self, _py: Python<'_>, additional: usize) {
        self.data.reserve(additional);
    }

    pub fn extend(&mut self, py: Python<'_>, itr: &PyAny) -> PyResult<()> {
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
        std::mem::swap(&mut self.data, &mut to_drop);
        Ok(())
    }

    // TODO: consider finishing impl to speed up circuit_to_instruction.py
    // pub fn iter_remap(&self, py: Python<'_>, qubits: Py<PyList>, clbits: Py<PyList>) -> PyResult<Py<PyIterator>> {
    //     let data: Py<CircuitData> = Py::new(py, CircuitData {
    //         data: self.data.clone(),
    //         // TODO: reuse intern context once concurrency is properly
    //         //  handled.
    //         intern_context: self.intern_context.clone(),
    //         qubits,
    //         clbits,
    //         qubit_indices: self.qubit_indices.clone_ref(py),
    //         clbit_indices: self.clbit_indices.clone_ref(py),
    //     })?;
    //
    //     let cell: &PyCell<CircuitData> = data.as_ref(py);
    //     Ok(cell.iter()?.into_py(py))
    // }

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
        for inst in self.data.iter() {
            if let Some(op) = inst.op.as_ref() {
                visit.call(op)?;
            }
        }
        for bit in self.qubits_native.iter().chain(self.clbits_native.iter()) {
            visit.call(bit)?;
        }

        visit.call(&self.qubits)?;
        visit.call(&self.clbits)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        // Clear anything that could have a reference cycle.
        for inst in self.data.iter_mut() {
            inst.op = None;
        }
        self.qubits_native.clear();
        self.clbits_native.clear();
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
            IndexFor::Insertion => std::cmp::min(std::cmp::max(0, index), self.data.len() as isize),
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
        // TODO: raise error if bit is not in self
        let mut cache_args =
            |indices: &HashMap<_BitAsKey, u32>, bits: &PyTuple| -> PyResult<IndexType> {
                let args = bits
                    .into_iter()
                    .map(|b| {
                        let native = _BitAsKey::new(b)?;
                        Ok(indices[&native])
                    })
                    .collect::<PyResult<Vec<BitType>>>()?;
                Ok(self.intern_context.intern(args))
            };
        Ok(InternedInstruction {
            op: Some(elem.operation.clone_ref(py)),
            qubits_id: cache_args(&self.qubit_indices_native, elem.qubits.as_ref(py))?,
            clbits_id: cache_args(&self.clbit_indices_native, elem.clbits.as_ref(py))?,
        })
    }
}
