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
use crate::imports::QUANTUM_CIRCUIT;
use crate::operations::{Operation, OperationType, Param};
use crate::TupleLikeArg;

use ahash::AHasher;
use std::hash::Hasher;

use approx::relative_eq;

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{intern, IntoPy, PyObject, PyResult, ToPyObject};
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::smallvec;

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
        self.node = index.map(NodeIndex::new);
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
        params: smallvec::SmallVec<[Param; 3]>,
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
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (op, qargs=None, cargs=None, params=smallvec![], label=None, duration=None, unit=None, condition=None, dag=None))]
    pub fn py_new(
        py: Python,
        op: crate::circuit_instruction::OperationInput,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        params: smallvec::SmallVec<[crate::operations::Param; 3]>,
        label: Option<String>,
        duration: Option<PyObject>,
        unit: Option<String>,
        condition: Option<PyObject>,
        dag: Option<&Bound<PyAny>>,
    ) -> PyResult<Py<Self>> {
        let qargs = qargs.map_or_else(|| PyTuple::empty_bound(py), |q| q.value);
        let cargs = cargs.map_or_else(|| PyTuple::empty_bound(py), |c| c.value);

        let sort_key = py.None();

        let mut instruction = CircuitInstruction::py_new(
            py, op, None, None, params, label, duration, unit, condition,
        )?;
        instruction.qubits = qargs.into();
        instruction.clbits = cargs.into();

        Py::new(
            py,
            (
                DAGOpNode {
                    instruction,
                    sort_key: sort_key,
                },
                DAGNode { node: None },
            ),
        )
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python) -> PyResult<u64> {
        let super_ = slf.as_ref();
        let mut hasher = AHasher::default();
        hasher.write_isize(super_.py_nid());
        hasher.write(slf.instruction.operation.name().as_bytes());
        Ok(hasher.finish())
    }

    fn __eq__(slf: PyRef<Self>, py: Python, other: &Bound<PyAny>) -> PyResult<bool> {
        match other.downcast::<Self>() {
            Ok(other) => {
                let borrowed_other = other.borrow();
                let other_super = borrowed_other.as_ref();
                let super_ = slf.as_ref();

                if super_.py_nid() != other_super.py_nid() {
                    return Ok(false);
                }
                if !slf
                    .instruction
                    .operation
                    .eq(py, &borrowed_other.instruction.operation)?
                {
                    return Ok(false);
                }
                let params_eq = if let OperationType::Standard(_op) = slf.instruction.operation {
                    slf.instruction.params.iter().zip(borrowed_other.instruction.params.iter()).all(|(a, b)| {
                       match [a, b] {
                           [Param::Float(float_a), Param::Float(float_b)] => relative_eq!(float_a, float_b, max_relative = 1e-10),
                           [Param::ParameterExpression(param_a), Param::ParameterExpression(param_b)] => param_a.bind(py).eq(param_b).unwrap(),
                           _ => false,
                       }
                   })
                } else {
                    true
                };

                Ok(params_eq
                    && slf
                        .instruction
                        .qubits
                        .bind(py)
                        .eq(borrowed_other.instruction.qubits.clone_ref(py))?
                    && slf
                        .instruction
                        .clbits
                        .bind(py)
                        .eq(borrowed_other.instruction.clbits.clone_ref(py))?)
            }
            Err(_) => Ok(false),
        }
    }

    #[staticmethod]
    fn from_instruction(
        py: Python,
        instruction: CircuitInstruction,
        dag: Option<&Bound<PyAny>>,
    ) -> PyResult<PyObject> {
        let qargs = instruction.qubits.clone_ref(py).into_bound(py);
        let cargs = instruction.clbits.clone_ref(py).into_bound(py);

        let sort_key = py.None();
        let base = PyClassInitializer::from(DAGNode { node: None });
        let sub = base.add_subclass(DAGOpNode {
            instruction,
            sort_key,
        });
        Ok(Py::new(py, sub)?.to_object(py))
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
        slf.as_mut().node = index.map(NodeIndex::new);
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
    fn num_qubits(&self) -> u32 {
        self.instruction.operation.num_qubits()
    }

    #[getter]
    fn num_clbits(&self) -> u32 {
        self.instruction.operation.num_clbits()
    }

    #[getter]
    pub fn get_qargs(&self, py: Python) -> Py<PyTuple> {
        self.instruction.qubits.clone_ref(py)
    }

    #[setter]
    fn set_qargs(&mut self, qargs: Py<PyTuple>) {
        self.instruction.qubits = qargs;
    }

    #[getter]
    pub fn get_cargs(&self, py: Python) -> Py<PyTuple> {
        self.instruction.clbits.clone_ref(py)
    }

    #[setter]
    fn set_cargs(&mut self, cargs: Py<PyTuple>) {
        self.instruction.clbits = cargs;
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self) -> &str {
        self.instruction.operation.name()
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
        let matrix = self.instruction.operation.matrix(&self.instruction.params);
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
        let definition = self
            .instruction
            .operation
            .definition(&self.instruction.params);
        definition
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
        let op = operation_type_to_py(py, &self.instruction)?;
        op.bind(py).setattr(intern!(py, "name"), new_name)?;
        let res = convert_py_to_operation_type(py, op)?;
        self.instruction.operation = res.operation;
        Ok(())
    }

    #[getter]
    fn _raw_op(&self, py: Python) -> PyObject {
        self.instruction.operation.clone().into_py(py)
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
    pub wire: PyObject,
    #[pyo3(get)]
    sort_key: PyObject,
}

impl DAGInNode {
    pub fn new(py: Python, node: NodeIndex, wire: PyObject) -> (Self, DAGNode) {
        (
            DAGInNode {
                wire,
                sort_key: py.None(),
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
                sort_key: py.None(),
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
        slf.as_mut().node = index.map(NodeIndex::new);
        slf.sort_key = sort_key;
        Ok(())
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python) -> PyResult<u64> {
        let super_ = slf.as_ref();
        let mut hasher = AHasher::default();
        hasher.write_isize(super_.py_nid());
        hasher.write_isize(slf.wire.bind(py).hash()?);
        Ok(hasher.finish())
    }

    fn __eq__(slf: PyRef<Self>, py: Python, other: &Bound<PyAny>) -> PyResult<bool> {
        match other.downcast::<Self>() {
            Ok(other) => {
                let borrowed_other = other.borrow();
                let other_super = borrowed_other.as_ref();
                let super_ = slf.as_ref();
                Ok(super_.py_nid() == other_super.py_nid()
                    && slf.wire.bind(py).eq(borrowed_other.wire.clone_ref(py))?)
            }
            Err(_) => Ok(false),
        }
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
    pub wire: PyObject,
    #[pyo3(get)]
    sort_key: PyObject,
}

impl DAGOutNode {
    pub fn new(py: Python, node: NodeIndex, wire: PyObject) -> (Self, DAGNode) {
        (
            DAGOutNode {
                wire,
                sort_key: py.None(),
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
                sort_key: py.None(),
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
        slf.as_mut().node = index.map(NodeIndex::new);
        slf.sort_key = sort_key;
        Ok(())
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python) -> PyResult<u64> {
        let super_ = slf.as_ref();
        let mut hasher = AHasher::default();
        hasher.write_isize(super_.py_nid());
        hasher.write_isize(slf.wire.bind(py).hash()?);
        Ok(hasher.finish())
    }

    /// Returns a representation of the DAGOutNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGOutNode(wire={})", self.wire.bind(py).repr()?))
    }

    fn __eq__(slf: PyRef<Self>, py: Python, other: &Bound<PyAny>) -> PyResult<bool> {
        match other.downcast::<Self>() {
            Ok(other) => {
                let borrowed_other = other.borrow();
                let other_super = borrowed_other.as_ref();
                let super_ = slf.as_ref();
                Ok(super_.py_nid() == other_super.py_nid()
                    && slf.wire.bind(py).eq(borrowed_other.wire.clone_ref(py))?)
            }
            Err(_) => Ok(false),
        }
    }
}
