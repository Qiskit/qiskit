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

use crate::circuit_instruction::{
    convert_py_to_operation_type, operation_type_to_py, CircuitInstruction,
    ExtraInstructionAttributes,
};
use crate::operations::{Operation, OperationType, Param};
use crate::TupleLikeArg;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence, PyString, PyTuple};
use pyo3::{intern, PyObject, PyResult};
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::SmallVec;

/// Parent class for DAGOpNode, DAGInNode, and DAGOutNode.
#[pyclass(module = "qiskit._accelerate.circuit", subclass)]
#[derive(Clone, Debug)]
pub struct DAGNode {
    pub node: Option<NodeIndex>,
}

impl DAGNode {
    #[inline]
    pub fn py_nid(&self) -> isize {
        self.node
            .map(|node| node.index().try_into().unwrap())
            .unwrap_or(-1)
    }
}

#[pymethods]
impl DAGNode {
    #[new]
    #[pyo3(signature=(nid=-1))]
    fn py_new(nid: isize) -> Self {
        DAGNode {
            node: match nid {
                -1 => None,
                nid => Some(NodeIndex::new(nid.try_into().unwrap())),
            },
        }
    }

    #[getter]
    fn get__node_id(&self) -> isize {
        self.py_nid()
    }

    #[setter]
    fn set__node_id(&mut self, nid: isize) {
        self.node = match nid {
            -1 => None,
            nid => Some(NodeIndex::new(nid.try_into().unwrap())),
        }
    }

    fn __getstate__(&self) -> Option<usize> {
        self.node.map(|node| node.index())
    }

    fn __setstate__(&mut self, index: Option<usize>) {
        self.node = index.map(|index| NodeIndex::new(index));
    }

    fn __lt__(&self, other: &DAGNode) -> bool {
        self.py_nid() < other.py_nid()
    }

    fn __gt__(&self, other: &DAGNode) -> bool {
        self.py_nid() > other.py_nid()
    }

    fn __str__(_self: &Bound<DAGNode>) -> String {
        format!("{}", _self.as_ptr() as usize)
    }

    fn __hash__(&self, py: Python) -> PyResult<isize> {
        self.py_nid().into_py(py).bind(py).hash()
    }
}

/// Object to represent an Instruction at a node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGOpNode {
    pub instruction: CircuitInstruction,
    #[pyo3(get)]
    pub sort_key: PyObject,
}

impl DAGOpNode {
    pub fn new<T1, T2, U1, U2>(
        py: Python,
        node: NodeIndex,
        op: OperationType,
        qargs: impl IntoIterator<Item = T1, IntoIter = U1>,
        cargs: impl IntoIterator<Item = T2, IntoIter = U2>,
        params: SmallVec<[Param; 3]>,
        extra_attrs: Option<Box<ExtraInstructionAttributes>>,
        sort_key: Py<PyAny>,
    ) -> (Self, DAGNode)
    where
        T1: ToPyObject,
        T2: ToPyObject,
        U1: ExactSizeIterator<Item = T1>,
        U2: ExactSizeIterator<Item = T2>,
    {
        (
            DAGOpNode {
                instruction: CircuitInstruction::new(py, op, qargs, cargs, params, extra_attrs),
                sort_key,
            },
            DAGNode { node: Some(node) },
        )
    }
}

#[pymethods]
impl DAGOpNode {
    #[new]
    pub fn py_new(
        py: Python,
        op: PyObject,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        dag: Option<&Bound<PyAny>>,
    ) -> PyResult<Py<Self>> {
        let qargs = qargs.map_or_else(|| PyTuple::empty_bound(py), |q| q.value);
        let cargs = cargs.map_or_else(|| PyTuple::empty_bound(py), |c| c.value);

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
        let res = convert_py_to_operation_type(py, op.clone_ref(py))?;

        let extra_attrs = if res.label.is_some()
            || res.duration.is_some()
            || res.unit.is_some()
            || res.condition.is_some()
        {
            Some(Box::new(ExtraInstructionAttributes {
                label: res.label,
                duration: res.duration,
                unit: res.unit,
                condition: res.condition,
            }))
        } else {
            None
        };

        Py::new(
            py,
            (
                DAGOpNode {
                    instruction: CircuitInstruction {
                        operation: res.operation,
                        qubits: qargs.unbind(),
                        clbits: cargs.unbind(),
                        params: res.params,
                        extra_attrs,
                        #[cfg(feature = "cache_pygates")]
                        py_op: Some(op),
                    },
                    sort_key: sort_key.unbind(),
                },
                DAGNode { node: None },
            ),
        )
    }

    fn __reduce__(slf: PyRef<Self>, py: Python) -> PyResult<PyObject> {
        let state = (slf.as_ref().node.map(|node| node.index()), &slf.sort_key);
        Ok((
            py.get_type_bound::<Self>(),
            (
                operation_type_to_py(py, &slf.instruction)?,
                &slf.instruction.qubits,
                &slf.instruction.clbits,
            ),
            state,
        )
            .into_py(py))
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (index, sort_key): (Option<usize>, PyObject) = state.extract()?;
        slf.as_mut().node = index.map(|index| NodeIndex::new(index));
        slf.sort_key = sort_key;
        Ok(())
    }

    #[getter]
    fn get_op(&self, py: Python) -> PyResult<PyObject> {
        operation_type_to_py(py, &self.instruction)
    }

    #[setter]
    fn set_op(&mut self, py: Python, op: PyObject) -> PyResult<()> {
        let res = convert_py_to_operation_type(py, op)?;
        self.instruction.operation = res.operation;
        self.instruction.params = res.params;
        let extra_attrs = if res.label.is_some()
            || res.duration.is_some()
            || res.unit.is_some()
            || res.condition.is_some()
        {
            Some(Box::new(ExtraInstructionAttributes {
                label: res.label,
                duration: res.duration,
                unit: res.unit,
                condition: res.condition,
            }))
        } else {
            None
        };
        self.instruction.extra_attrs = extra_attrs;
        Ok(())
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
    fn get_name(&self, py: Python) -> PyObject {
        self.instruction.operation.name().to_object(py)
    }

    /// Sets the Instruction name corresponding to the op for this node
    #[setter]
    fn set_name(&mut self, py: Python, new_name: PyObject) -> PyResult<()> {
        let op = operation_type_to_py(py, &self.instruction)?;
        op.bind(py).setattr(intern!(py, "name"), new_name)?;
        let res = convert_py_to_operation_type(py, op)?;
        self.instruction.operation = res.operation;
        Ok(())
    }

    /// Returns a representation of the DAGOpNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "DAGOpNode(op={}, qargs={}, cargs={})",
            operation_type_to_py(py, &self.instruction)?
                .bind(py)
                .repr()?,
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

impl DAGInNode {
    pub fn new(py: Python, node: NodeIndex, wire: PyObject) -> (Self, DAGNode) {
        (
            DAGInNode {
                wire,
                sort_key: PyList::empty_bound(py).str().unwrap().into_any().unbind(),
            },
            DAGNode { node: Some(node) },
        )
    }
}

#[pymethods]
impl DAGInNode {
    #[new]
    fn py_new(py: Python, wire: PyObject) -> PyResult<(Self, DAGNode)> {
        Ok((
            DAGInNode {
                wire,
                sort_key: PyList::empty_bound(py).str()?.into_any().unbind(),
            },
            DAGNode { node: None },
        ))
    }

    fn __reduce__(slf: PyRef<Self>, py: Python) -> PyObject {
        let state = (slf.as_ref().node.map(|node| node.index()), &slf.sort_key);
        (py.get_type_bound::<Self>(), (&slf.wire,), state).into_py(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (index, sort_key): (Option<usize>, PyObject) = state.extract()?;
        slf.as_mut().node = index.map(|index| NodeIndex::new(index));
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

impl DAGOutNode {
    pub fn new(py: Python, node: NodeIndex, wire: PyObject) -> (Self, DAGNode) {
        (
            DAGOutNode {
                wire,
                sort_key: PyList::empty_bound(py).str().unwrap().into_any().unbind(),
            },
            DAGNode { node: Some(node) },
        )
    }
}

#[pymethods]
impl DAGOutNode {
    #[new]
    fn py_new(py: Python, wire: PyObject) -> PyResult<(Self, DAGNode)> {
        Ok((
            DAGOutNode {
                wire,
                sort_key: PyList::empty_bound(py).str()?.into_any().unbind(),
            },
            DAGNode { node: None },
        ))
    }

    fn __reduce__(slf: PyRef<Self>, py: Python) -> PyObject {
        let state = (slf.as_ref().node.map(|node| node.index()), &slf.sort_key);
        (py.get_type_bound::<Self>(), (&slf.wire,), state).into_py(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (index, sort_key): (Option<usize>, PyObject) = state.extract()?;
        slf.as_mut().node = index.map(|index| NodeIndex::new(index));
        slf.sort_key = sort_key;
        Ok(())
    }

    /// Returns a representation of the DAGOutNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGOutNode(wire={})", self.wire.bind(py).repr()?))
    }
}
