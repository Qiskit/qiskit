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

#[cfg(feature = "cache_pygates")]
use std::cell::RefCell;

use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::imports::QUANTUM_CIRCUIT;
use crate::operations::Operation;

use numpy::IntoPyArray;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence, PyString, PyTuple};
use pyo3::{intern, IntoPy, PyObject, PyResult, ToPyObject};

/// Parent class for DAGOpNode, DAGInNode, and DAGOutNode.
#[pyclass(module = "qiskit._accelerate.circuit", subclass)]
#[derive(Clone, Debug)]
pub struct DAGNode {
    #[pyo3(get, set)]
    pub _node_id: isize,
}

#[pymethods]
impl DAGNode {
    #[new]
    #[pyo3(signature=(nid=-1))]
    fn new(nid: isize) -> Self {
        DAGNode { _node_id: nid }
    }

    fn __getstate__(&self) -> isize {
        self._node_id
    }

    fn __setstate__(&mut self, nid: isize) {
        self._node_id = nid;
    }

    fn __lt__(&self, other: &DAGNode) -> bool {
        self._node_id < other._node_id
    }

    fn __gt__(&self, other: &DAGNode) -> bool {
        self._node_id > other._node_id
    }

    fn __str__(_self: &Bound<DAGNode>) -> String {
        format!("{}", _self.as_ptr() as usize)
    }

    fn __hash__(&self, py: Python) -> PyResult<isize> {
        self._node_id.into_py(py).bind(py).hash()
    }
}

/// Object to represent an Instruction at a node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGOpNode {
    pub instruction: CircuitInstruction,
    #[pyo3(get)]
    pub sort_key: PyObject,
}

#[pymethods]
impl DAGOpNode {
    #[new]
    #[pyo3(signature = (op, qargs=None, cargs=None, *, dag=None))]
    fn new(
        py: Python,
        op: &Bound<PyAny>,
        qargs: Option<&Bound<PySequence>>,
        cargs: Option<&Bound<PySequence>>,
        dag: Option<&Bound<PyAny>>,
    ) -> PyResult<(Self, DAGNode)> {
        let qargs =
            qargs.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        let cargs =
            cargs.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;

        let sort_key = match dag {
            Some(dag) => {
                let cache = dag
                    .getattr(intern!(py, "_key_cache"))?
                    .downcast_into_exact::<PyDict>()?;
                let cache_key = PyTuple::new_bound(py, [&qargs, &cargs]);
                match cache.get_item(&cache_key)? {
                    Some(key) => key,
                    None => {
                        let indices: PyResult<Vec<_>> = qargs
                            .iter()
                            .chain(cargs.iter())
                            .map(|bit| {
                                dag.call_method1(intern!(py, "find_bit"), (bit,))?
                                    .getattr(intern!(py, "index"))
                            })
                            .collect();
                        let index_strs: Vec<_> =
                            indices?.into_iter().map(|i| format!("{:04}", i)).collect();
                        let key = PyString::new_bound(py, index_strs.join(",").as_str());
                        cache.set_item(&cache_key, &key)?;
                        key.into_any()
                    }
                }
            }
            None => qargs.str()?.into_any(),
        };
        Ok((
            DAGOpNode {
                instruction: CircuitInstruction::py_new(
                    op,
                    Some(qargs.into_any()),
                    Some(cargs.into_any()),
                )?,
                sort_key: sort_key.unbind(),
            },
            DAGNode { _node_id: -1 },
        ))
    }

    #[pyo3(signature = (instruction, /, *, dag=None, deepcopy=false))]
    #[staticmethod]
    fn from_instruction(
        py: Python,
        mut instruction: CircuitInstruction,
        dag: Option<&Bound<PyAny>>,
        deepcopy: bool,
    ) -> PyResult<PyObject> {
        let qargs = instruction.qubits.bind(py);
        let cargs = instruction.clbits.bind(py);

        let sort_key = match dag {
            Some(dag) => {
                let cache = dag
                    .getattr(intern!(py, "_key_cache"))?
                    .downcast_into_exact::<PyDict>()?;
                let cache_key = PyTuple::new_bound(py, [&qargs, &cargs]);
                match cache.get_item(&cache_key)? {
                    Some(key) => key,
                    None => {
                        let indices: PyResult<Vec<_>> = qargs
                            .iter()
                            .chain(cargs.iter())
                            .map(|bit| {
                                dag.call_method1(intern!(py, "find_bit"), (bit,))?
                                    .getattr(intern!(py, "index"))
                            })
                            .collect();
                        let index_strs: Vec<_> =
                            indices?.into_iter().map(|i| format!("{:04}", i)).collect();
                        let key = PyString::new_bound(py, index_strs.join(",").as_str());
                        cache.set_item(&cache_key, &key)?;
                        key.into_any()
                    }
                }
            }
            None => qargs.str()?.into_any(),
        };
        if deepcopy {
            instruction.operation = instruction.operation.py_deepcopy(py, None)?;
            #[cfg(feature = "cache_pygates")]
            {
                *instruction.py_op.borrow_mut() = None;
            }
        }
        let base = PyClassInitializer::from(DAGNode { _node_id: -1 });
        let sub = base.add_subclass(DAGOpNode {
            instruction,
            sort_key: sort_key.unbind(),
        });
        Ok(Py::new(py, sub)?.to_object(py))
    }

    fn __reduce__(slf: PyRef<Self>) -> PyResult<PyObject> {
        let py = slf.py();
        let state = (slf.as_ref()._node_id, &slf.sort_key);
        Ok((
            py.get_type_bound::<Self>(),
            (
                slf.instruction.get_operation(py)?,
                &slf.instruction.qubits,
                &slf.instruction.clbits,
            ),
            state,
        )
            .into_py(py))
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (nid, sort_key): (isize, PyObject) = state.extract()?;
        slf.as_mut()._node_id = nid;
        slf.sort_key = sort_key;
        Ok(())
    }

    /// Get a `CircuitInstruction` that represents the same information as this `DAGOpNode`.  If
    /// `deepcopy`, any internal Python objects are deep-copied.
    ///
    /// Note: this ought to be a temporary method, while the DAG/QuantumCircuit converters still go
    /// via Python space; this still involves copy-out and copy-in of the data, whereas doing it all
    /// within Rust space could directly re-pack the instruction from a `DAGOpNode` to a
    /// `PackedInstruction` with no intermediate copy.
    #[pyo3(signature = (/, *, deepcopy=false))]
    fn _to_circuit_instruction(&self, py: Python, deepcopy: bool) -> PyResult<CircuitInstruction> {
        Ok(CircuitInstruction {
            operation: if deepcopy {
                self.instruction.operation.py_deepcopy(py, None)?
            } else {
                self.instruction.operation.clone()
            },
            qubits: self.instruction.qubits.clone_ref(py),
            clbits: self.instruction.clbits.clone_ref(py),
            params: self.instruction.params.clone(),
            extra_attrs: self.instruction.extra_attrs.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: RefCell::new(None),
        })
    }

    #[getter]
    fn get_op(&self, py: Python) -> PyResult<PyObject> {
        self.instruction.get_operation(py)
    }

    #[setter]
    fn set_op(&mut self, op: &Bound<PyAny>) -> PyResult<()> {
        let res = op.extract::<OperationFromPython>()?;
        self.instruction.operation = res.operation;
        self.instruction.params = res.params;
        self.instruction.extra_attrs = res.extra_attrs;
        #[cfg(feature = "cache_pygates")]
        {
            *self.instruction.py_op.borrow_mut() = Some(op.into_py(op.py()));
        }
        Ok(())
    }

    #[getter]
    fn num_qubits(&self) -> u32 {
        self.instruction.op().num_qubits()
    }

    #[getter]
    fn num_clbits(&self) -> u32 {
        self.instruction.op().num_clbits()
    }

    #[getter]
    fn get_qargs(&self, py: Python) -> Py<PyTuple> {
        self.instruction.qubits.clone_ref(py)
    }

    #[setter]
    fn set_qargs(&mut self, qargs: Py<PyTuple>) {
        self.instruction.qubits = qargs;
    }

    #[getter]
    fn get_cargs(&self, py: Python) -> Py<PyTuple> {
        self.instruction.clbits.clone_ref(py)
    }

    #[setter]
    fn set_cargs(&mut self, cargs: Py<PyTuple>) {
        self.instruction.clbits = cargs;
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self, py: Python) -> Py<PyString> {
        self.instruction.op().name().into_py(py)
    }

    #[getter]
    fn get_params(&self, py: Python) -> PyObject {
        self.instruction.params.to_object(py)
    }

    #[setter]
    fn set_params(&mut self, val: smallvec::SmallVec<[crate::operations::Param; 3]>) {
        self.instruction.params = val;
    }

    pub fn is_parameterized(&self) -> bool {
        self.instruction.is_parameterized()
    }

    #[getter]
    fn matrix(&self, py: Python) -> Option<PyObject> {
        let matrix = self.instruction.op().matrix(&self.instruction.params);
        matrix.map(|mat| mat.into_pyarray_bound(py).into())
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.instruction
            .extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.label.as_deref())
    }

    #[getter]
    fn condition(&self, py: Python) -> Option<PyObject> {
        self.instruction
            .extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.condition.as_ref().map(|x| x.clone_ref(py)))
    }

    #[getter]
    fn duration(&self, py: Python) -> Option<PyObject> {
        self.instruction
            .extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.duration.as_ref().map(|x| x.clone_ref(py)))
    }

    #[getter]
    fn unit(&self) -> Option<&str> {
        self.instruction
            .extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.unit.as_deref())
    }

    #[getter]
    pub fn is_standard_gate(&self) -> bool {
        self.instruction.is_standard_gate()
    }

    #[setter]
    fn set_label(&mut self, val: Option<String>) {
        match self.instruction.extra_attrs.as_mut() {
            Some(attrs) => attrs.label = val,
            None => {
                if val.is_some() {
                    self.instruction.extra_attrs = Some(Box::new(
                        crate::circuit_instruction::ExtraInstructionAttributes {
                            label: val,
                            duration: None,
                            unit: None,
                            condition: None,
                        },
                    ))
                }
            }
        };
        if let Some(attrs) = &self.instruction.extra_attrs {
            if attrs.label.is_none()
                && attrs.duration.is_none()
                && attrs.unit.is_none()
                && attrs.condition.is_none()
            {
                self.instruction.extra_attrs = None;
            }
        }
    }

    #[getter]
    fn definition<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.instruction
            .op()
            .definition(&self.instruction.params)
            .map(|data| {
                QUANTUM_CIRCUIT
                    .get_bound(py)
                    .call_method1(intern!(py, "_from_circuit_data"), (data,))
            })
            .transpose()
    }

    /// Sets the Instruction name corresponding to the op for this node
    #[setter]
    fn set_name(&mut self, py: Python, new_name: PyObject) -> PyResult<()> {
        let op = self.instruction.get_operation_mut(py)?.into_bound(py);
        op.setattr(intern!(py, "name"), new_name)?;
        self.instruction.operation = op.extract::<OperationFromPython>()?.operation;
        Ok(())
    }

    /// Returns a representation of the DAGOpNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "DAGOpNode(op={}, qargs={}, cargs={})",
            self.instruction.get_operation(py)?.bind(py).repr()?,
            self.instruction.qubits.bind(py).repr()?,
            self.instruction.clbits.bind(py).repr()?
        ))
    }
}

/// Object to represent an incoming wire node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGInNode {
    #[pyo3(get)]
    wire: PyObject,
    #[pyo3(get)]
    sort_key: PyObject,
}

#[pymethods]
impl DAGInNode {
    #[new]
    fn new(py: Python, wire: PyObject) -> PyResult<(Self, DAGNode)> {
        Ok((
            DAGInNode {
                wire,
                sort_key: PyList::empty_bound(py).str()?.into_any().unbind(),
            },
            DAGNode { _node_id: -1 },
        ))
    }

    fn __reduce__(slf: PyRef<Self>, py: Python) -> PyObject {
        let state = (slf.as_ref()._node_id, &slf.sort_key);
        (py.get_type_bound::<Self>(), (&slf.wire,), state).into_py(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (nid, sort_key): (isize, PyObject) = state.extract()?;
        slf.as_mut()._node_id = nid;
        slf.sort_key = sort_key;
        Ok(())
    }

    /// Returns a representation of the DAGInNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGInNode(wire={})", self.wire.bind(py).repr()?))
    }
}

/// Object to represent an outgoing wire node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGOutNode {
    #[pyo3(get)]
    wire: PyObject,
    #[pyo3(get)]
    sort_key: PyObject,
}

#[pymethods]
impl DAGOutNode {
    #[new]
    fn new(py: Python, wire: PyObject) -> PyResult<(Self, DAGNode)> {
        Ok((
            DAGOutNode {
                wire,
                sort_key: PyList::empty_bound(py).str()?.into_any().unbind(),
            },
            DAGNode { _node_id: -1 },
        ))
    }

    fn __reduce__(slf: PyRef<Self>, py: Python) -> PyObject {
        let state = (slf.as_ref()._node_id, &slf.sort_key);
        (py.get_type_bound::<Self>(), (&slf.wire,), state).into_py(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (nid, sort_key): (isize, PyObject) = state.extract()?;
        slf.as_mut()._node_id = nid;
        slf.sort_key = sort_key;
        Ok(())
    }

    /// Returns a representation of the DAGOutNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGOutNode(wire={})", self.wire.bind(py).repr()?))
    }
}
