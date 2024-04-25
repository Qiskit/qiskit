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

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use pyo3::{intern, PyObject, PyResult};
use std::hash::Hash;

/// Parent class for DAGOpNode, DAGInNode, and DAGOutNode.
#[pyclass(module = "qiskit._accelerate.quantum_circuit", subclass)]
#[derive(Clone, Debug)]
pub struct DAGNode {
    #[pyo3(get, set)]
    _node_id: isize,
}

#[pymethods]
impl DAGNode {
    #[new]
    #[pyo3(signature=(nid=-1))]
    fn new(nid: isize) -> Self {
        DAGNode { _node_id: nid }
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

    /// Check if DAG nodes are considered equivalent, e.g., as a node_match for
    /// :func:`rustworkx.is_isomorphic_node_match`.
    ///
    /// Args:
    ///     node1 (DAGOpNode, DAGInNode, DAGOutNode): A node to compare.
    ///     node2 (DAGOpNode, DAGInNode, DAGOutNode): The other node to compare.
    ///     bit_indices1 (dict): Dictionary mapping Bit instances to their index
    ///         within the circuit containing node1
    ///     bit_indices2 (dict): Dictionary mapping Bit instances to their index
    ///         within the circuit containing node2
    ///
    /// Return:
    ///     Bool: If node1 == node2
    #[staticmethod]
    fn semantic_eq(
        py: Python,
        node1: &Bound<PyAny>,
        node2: &Bound<PyAny>,
        bit_indices1: &Bound<PyAny>,
        bit_indices2: &Bound<PyAny>,
    ) -> PyResult<bool> {
        if !node1.is_instance_of::<DAGOpNode>() || !node2.is_instance_of::<DAGOpNode>() {
            let same_type = node1.get_type().is(&node2.get_type());
            let indices1 = bit_indices1.get_item(node1.getattr(intern!(py, "wire"))?)?;
            let indices2 = bit_indices2.get_item(node2.getattr(intern!(py, "wire"))?)?;
            return Ok(same_type && indices1.eq(indices2)?)
        }

        let node1 = node1.downcast_exact::<DAGOpNode>()?;
        let node2 = node2.downcast_exact::<DAGOpNode>()?;

        // if node1.op.is_instance(PyType::)
    }
}

/// Object to represent an Instruction at a node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.quantum_circuit", extends=DAGNode)]
pub struct DAGOpNode {
    op: PyObject,
    qargs: Py<PyTuple>,
    cargs: Py<PyTuple>,
    sort_key: PyObject,
}

#[pymethods]
impl DAGOpNode {
    #[new]
    fn new(
        py: Python,
        op: PyObject,
        qargs: Option<&Bound<PyAny>>,
        cargs: Option<&Bound<PyAny>>,
        dag: Option<&Bound<PyAny>>,
    ) -> PyResult<(Self, DAGNode)> {
        fn as_tuple(py: Python<'_>, seq: Option<&Bound<PyAny>>) -> PyResult<Py<PyTuple>> {
            match seq {
                None => Ok(PyTuple::empty_bound(py).unbind()),
                Some(seq) => {
                    if seq.is_instance_of::<PyTuple>() {
                        Ok(seq.downcast_exact::<PyTuple>()?.into_py(py))
                    } else if seq.is_instance_of::<PyList>() {
                        let seq = seq.downcast_exact::<PyList>()?;
                        Ok(seq.to_tuple().unbind())
                    } else {
                        // New tuple from iterable.
                        Ok(PyTuple::new_bound(
                            py,
                            seq.iter()?
                                .map(|o| Ok(o?.unbind()))
                                .collect::<PyResult<Vec<PyObject>>>()?,
                        )
                        .unbind())
                    }
                }
            }
        }

        let qargs = as_tuple(py, qargs)?;
        let cargs = as_tuple(py, cargs)?;
        let sort_key: PyObject = match dag {
            Some(dag) => {
                todo!()
            }
            None => qargs.bind(py).str()?.as_any().unbind(),
        };

        Ok((
            DAGOpNode {
                op,
                qargs,
                cargs,
                sort_key,
            },
            DAGNode { _node_id: -1 },
        ))
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self, py: Python) -> PyResult<Bound<PyAny>> {
        self.op.bind(py).getattr(intern!(py, "name"))
    }

    /// Sets the Instruction name corresponding to the op for this node
    #[setter]
    fn set_name(&self, py: Python, new_name: PyObject) -> PyResult<()> {
        self.op.bind(py).setattr(intern!(py, "name"), new_name)
    }

    /// Returns a representation of the DAGOpNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "DAGOpNode(op={}, qargs={}, cargs={})",
            self.op.bind(py).str()?,
            self.qargs.bind(py).str()?,
            self.cargs.bind(py).str()?
        ))
    }
}

/// Object to represent an incoming wire node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.quantum_circuit", extends=DAGNode)]
pub struct DAGInNode {
    wire: PyObject,
    sort_key: PyObject,
}

#[pymethods]
impl DAGInNode {
    #[new]
    fn new(py: Python, wire: PyObject) -> PyResult<(Self, DAGNode)> {
        Ok((
            DAGInNode {
                wire,
                sort_key: PyList::empty_bound(py).str()?.as_any().unbind(),
            },
            DAGNode { _node_id: -1 },
        ))
    }

    /// Returns a representation of the DAGInNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGInNode(wire={})", self.wire.bind(py).str()?))
    }
}

/// Object to represent an outgoing wire node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.quantum_circuit", extends=DAGNode)]
pub struct DAGOutNode {
    wire: PyObject,
    sort_key: PyObject,
}

#[pymethods]
impl DAGOutNode {
    #[new]
    fn new(py: Python, wire: PyObject) -> PyResult<(Self, DAGNode)> {
        Ok((
            DAGOutNode {
                wire,
                sort_key: PyList::empty_bound(py).str()?.as_any().unbind(),
            },
            DAGNode { _node_id: -1 },
        ))
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        /// Returns a representation of the DAGOutNode
        Ok(format!("DAGOutNode(wire={})", self.wire.bind(py).str()?))
    }
}
