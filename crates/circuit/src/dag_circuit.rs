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

use std::hash::Hash;

use ahash::RandomState;
use approx::relative_eq;
use smallvec::SmallVec;

use crate::bit::{
    BitLocations, ClassicalRegister, PyClassicalRegister, PyClbit, PyQubit, QuantumRegister,
    Register, ShareableClbit, ShareableQubit,
};
use crate::bit_data::{BitData, VarAsKey};
use crate::bit_locator::BitLocator;
use crate::circuit_data::CircuitData;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::converters::QuantumCircuitData;
use crate::dag_node::{DAGInNode, DAGNode, DAGOpNode, DAGOutNode};
use crate::dot_utils::build_dot;
use crate::error::DAGCircuitError;
use crate::imports;
use crate::interner::{Interned, InternedMap, Interner};
use crate::operations::{ArrayType, Operation, OperationRef, Param, PyInstruction, StandardGate};
use crate::packed_instruction::{PackedInstruction, PackedOperation};
use crate::register_data::RegisterData;
use crate::rustworkx_core_vnext::isomorphism;
use crate::slice::PySequenceIndex;
use crate::{BitType, Clbit, Qubit, TupleLikeArg};

use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use itertools::Itertools;

use pyo3::exceptions::{
    PyDeprecationWarning, PyIndexError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use pyo3::types::{
    IntoPyDict, PyDict, PyInt, PyIterator, PyList, PySet, PyString, PyTuple, PyType,
};

use rustworkx_core::dag_algo::layers;
use rustworkx_core::err::ContractError;
use rustworkx_core::graph_ext::ContractNodesDirected;
use rustworkx_core::petgraph;
use rustworkx_core::petgraph::prelude::StableDiGraph;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::stable_graph::{EdgeReference, NodeIndex};
use rustworkx_core::petgraph::unionfind::UnionFind;
use rustworkx_core::petgraph::visit::{
    EdgeIndexable, IntoEdgeReferences, IntoNodeReferences, NodeFiltered, NodeIndexable,
};
use rustworkx_core::petgraph::Incoming;
use rustworkx_core::traversal::{
    ancestors as core_ancestors, bfs_predecessors as core_bfs_predecessors,
    bfs_successors as core_bfs_successors, descendants as core_descendants,
};

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::convert::Infallible;
use std::f64::consts::PI;

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

static CONTROL_FLOW_OP_NAMES: [&str; 4] = ["for_loop", "while_loop", "if_else", "switch_case"];
static SEMANTIC_EQ_SYMMETRIC: [&str; 4] = ["barrier", "swap", "break_loop", "continue_loop"];

/// An opaque key type that identifies a variable within a [DAGCircuit].
///
/// When a new variable is added to the DAG, it is associated internally
/// with one of these keys. When enumerating DAG nodes and edges, you can
/// retrieve the associated variable instance via [DAGCircuit::get_var].
///
/// These keys are [Eq], but this is semantically valid only for keys
/// from the same [DAGCircuit] instance.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Var(BitType);

impl Var {
    /// Construct a new [Var] object from a usize. if you have a u32 you can
    /// create a [Var] object directly with `Var(0u32)`. This will panic
    /// if the `usize` index exceeds `u32::MAX`.
    #[inline(always)]
    fn new(index: usize) -> Self {
        Var(index
            .try_into()
            .unwrap_or_else(|_| panic!("Index value '{}' exceeds the maximum bit width!", index)))
    }

    /// Get the index of the [Var]
    #[inline(always)]
    fn index(&self) -> usize {
        self.0 as usize
    }
}

impl From<BitType> for Var {
    fn from(value: BitType) -> Self {
        Var(value)
    }
}

impl From<Var> for BitType {
    fn from(value: Var) -> Self {
        value.0
    }
}

#[derive(Clone, Debug)]
pub enum NodeType {
    QubitIn(Qubit),
    QubitOut(Qubit),
    ClbitIn(Clbit),
    ClbitOut(Clbit),
    VarIn(Var),
    VarOut(Var),
    Operation(PackedInstruction),
}

impl NodeType {
    /// Unwraps this node as an operation and returns a reference to
    /// the contained [PackedInstruction].
    ///
    /// Panics if this is not an operation node.
    pub fn unwrap_operation(&self) -> &PackedInstruction {
        match self {
            NodeType::Operation(instr) => instr,
            _ => panic!("Node is not an operation!"),
        }
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub enum Wire {
    Qubit(Qubit),
    Clbit(Clbit),
    Var(Var),
}

impl Wire {
    fn to_pickle(&self, py: Python) -> PyResult<PyObject> {
        match self {
            Self::Qubit(bit) => (0, bit.0.into_py_any(py)?),
            Self::Clbit(bit) => (1, bit.0.into_py_any(py)?),
            Self::Var(var) => (2, var.0.into_py_any(py)?),
        }
        .into_py_any(py)
    }

    fn from_pickle(b: &Bound<PyAny>) -> PyResult<Self> {
        let tuple: Bound<PyTuple> = b.extract()?;
        let wire_type: usize = tuple.get_item(0)?.extract()?;
        if wire_type == 0 {
            Ok(Self::Qubit(Qubit(tuple.get_item(1)?.extract()?)))
        } else if wire_type == 1 {
            Ok(Self::Clbit(Clbit(tuple.get_item(1)?.extract()?)))
        } else if wire_type == 2 {
            Ok(Self::Var(Var(tuple.get_item(1)?.extract()?)))
        } else {
            Err(PyTypeError::new_err("Invalid wire type"))
        }
    }
}

/// Quantum circuit as a directed acyclic graph.
///
/// There are 3 types of nodes in the graph: inputs, outputs, and operations.
/// The nodes are connected by directed edges that correspond to qubits and
/// bits.
#[pyclass(module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct DAGCircuit {
    /// Circuit name.  Generally, this corresponds to the name
    /// of the QuantumCircuit from which the DAG was generated.
    #[pyo3(get, set)]
    name: Option<PyObject>,
    /// Circuit metadata
    #[pyo3(get, set)]
    metadata: Option<PyObject>,

    dag: StableDiGraph<NodeType, Wire>,

    qregs: RegisterData<QuantumRegister>,
    cregs: RegisterData<ClassicalRegister>,

    /// The cache used to intern instruction qargs.
    pub qargs_interner: Interner<[Qubit]>,
    /// The cache used to intern instruction cargs.
    pub cargs_interner: Interner<[Clbit]>,
    /// Qubits registered in the circuit.
    qubits: BitData<Qubit, ShareableQubit>,
    /// Clbits registered in the circuit.
    clbits: BitData<Clbit, ShareableClbit>,
    /// Variables registered in the circuit.
    vars: BitData<Var, VarAsKey>,
    /// Global phase.
    global_phase: Param,
    /// Duration.
    duration: Option<PyObject>,
    /// Unit of duration.
    unit: String,

    // Note: these are tracked separately from `qubits` and `clbits`
    // because it's not yet clear if the Rust concept of a native Qubit
    // and Clbit should correspond directly to the numerical Python
    // index that users see in the Python API.
    /// The index locations of bits, and their positions within
    /// registers.
    qubit_locations: BitLocator<ShareableQubit, QuantumRegister>,
    clbit_locations: BitLocator<ShareableClbit, ClassicalRegister>,

    /// Map from qubit to input and output nodes of the graph.
    qubit_io_map: Vec<[NodeIndex; 2]>,

    /// Map from clbit to input and output nodes of the graph.
    clbit_io_map: Vec<[NodeIndex; 2]>,

    /// Map from var to input and output nodes of the graph.
    var_io_map: Vec<[NodeIndex; 2]>,

    /// Operation kind to count
    op_names: IndexMap<String, usize, RandomState>,

    // Python modules we need to frequently access (for now).
    control_flow_module: PyControlFlowModule,
    vars_info: HashMap<String, DAGVarInfo>,
    vars_by_type: [Py<PySet>; 3],

    captured_stretches: IndexMap<String, Py<PyAny>, RandomState>,
    declared_stretches: IndexMap<String, Py<PyAny>, RandomState>,
}

#[derive(Clone, Debug)]
struct PyControlFlowModule {
    condition_resources: Py<PyAny>,
    node_resources: Py<PyAny>,
}

#[derive(Clone, Debug)]
struct PyLegacyResources {
    clbits: Py<PyTuple>,
    cregs: Py<PyTuple>,
}

impl PyControlFlowModule {
    fn new(py: Python) -> PyResult<Self> {
        let module = PyModule::import(py, "qiskit.circuit.controlflow")?;
        Ok(PyControlFlowModule {
            condition_resources: module.getattr("condition_resources")?.unbind(),
            node_resources: module.getattr("node_resources")?.unbind(),
        })
    }

    fn condition_resources(&self, condition: &Bound<PyAny>) -> PyResult<PyLegacyResources> {
        let res = self
            .condition_resources
            .bind(condition.py())
            .call1((condition,))?;
        Ok(PyLegacyResources {
            clbits: res.getattr("clbits")?.downcast_into_exact()?.unbind(),
            cregs: res.getattr("cregs")?.downcast_into_exact()?.unbind(),
        })
    }

    fn node_resources(&self, node: &Bound<PyAny>) -> PyResult<PyLegacyResources> {
        let res = self.node_resources.bind(node.py()).call1((node,))?;
        Ok(PyLegacyResources {
            clbits: res.getattr("clbits")?.downcast_into_exact()?.unbind(),
            cregs: res.getattr("cregs")?.downcast_into_exact()?.unbind(),
        })
    }
}

#[derive(IntoPyObject)]
struct PyVariableMapper {
    mapper: Py<PyAny>,
}

impl PyVariableMapper {
    fn new(
        py: Python,
        target_cregs: Bound<PyAny>,
        bit_map: Option<Bound<PyDict>>,
        var_map: Option<Bound<PyDict>>,
        add_register: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let kwargs: HashMap<&str, Option<Py<PyAny>>> =
            HashMap::from_iter([("add_register", add_register)]);
        Ok(PyVariableMapper {
            mapper: imports::VARIABLE_MAPPER
                .get_bound(py)
                .call(
                    (target_cregs, bit_map, var_map),
                    Some(&kwargs.into_py_dict(py)?),
                )?
                .unbind(),
        })
    }

    fn map_condition<'py>(
        &self,
        condition: &Bound<'py, PyAny>,
        allow_reorder: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = condition.py();
        let kwargs: HashMap<&str, bool> = HashMap::from_iter([("allow_reorder", allow_reorder)]);
        self.mapper.bind(py).call_method(
            intern!(py, "map_condition"),
            (condition,),
            Some(&kwargs.into_py_dict(py)?),
        )
    }

    fn map_target<'py>(&self, target: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = target.py();
        self.mapper
            .bind(py)
            .call_method1(intern!(py, "map_target"), (target,))
    }
}

#[pyfunction]
fn reject_new_register(reg: &Bound<PyAny>) -> PyResult<()> {
    Err(DAGCircuitError::new_err(format!(
        "No register with '{:?}' to map this expression onto.",
        reg.getattr("bits")?
    )))
}

#[pyclass(name = "BitLocations", module = "qiskit._accelerate.circuit", sequence)]
#[derive(Clone, Debug)]
pub struct PyBitLocations {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub registers: Py<PyList>,
}

#[pymethods]
impl PyBitLocations {
    #[new]
    /// Creates a new instance of [PyBitLocations]
    pub fn new(index: usize, registers: Py<PyList>) -> Self {
        Self { index, registers }
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        let borrowed = slf.borrow();
        if let Ok(other) = other.downcast::<Self>() {
            let other_borrowed = other.borrow();
            Ok(borrowed.index == other_borrowed.index
                && slf.getattr("registers")?.eq(other.getattr("registers")?)?)
        } else if let Ok(other) = other.downcast::<PyTuple>() {
            Ok(slf.getattr("index")?.eq(other.get_item(0)?)?
                && slf.getattr("registers")?.eq(other.get_item(1)?)?)
        } else {
            Ok(false)
        }
    }

    fn __iter__(slf: Bound<Self>) -> PyResult<Bound<PyIterator>> {
        (slf.getattr("index")?, slf.getattr("registers")?)
            .into_bound_py_any(slf.py())?
            .try_iter()
    }

    fn __repr__(slf: Bound<Self>) -> PyResult<String> {
        Ok(format!(
            "{}(index={} registers={})",
            slf.get_type().name()?,
            slf.getattr("index")?.repr()?,
            slf.getattr("registers")?.repr()?
        ))
    }

    fn __getnewargs__(slf: Bound<Self>) -> PyResult<(Bound<PyAny>, Bound<PyAny>)> {
        Ok((slf.getattr("index")?, slf.getattr("registers")?))
    }

    fn __getitem__(&self, py: Python, index: PySequenceIndex<'_>) -> PyResult<PyObject> {
        let getter = |index: usize| -> PyResult<PyObject> {
            match index {
                0 => self.index.into_py_any(py),
                1 => Ok(self.registers.clone_ref(py).into_any()),
                _ => Err(PyIndexError::new_err("index out of range")),
            }
        };
        if let Ok(index) = index.with_len(2) {
            match index {
                crate::slice::SequenceIndex::Int(index) => getter(index),
                _ => PyTuple::new(py, index.iter().map(|idx| getter(idx).unwrap()))
                    .map(|obj| obj.into_any().unbind()),
            }
        } else {
            Err(PyIndexError::new_err("index out of range"))
        }
    }

    #[staticmethod]
    fn __len__() -> usize {
        2
    }
}

#[derive(Copy, Clone, Debug)]
enum DAGVarType {
    Input = 0,
    Capture = 1,
    Declare = 2,
}

#[derive(Clone, Debug)]
struct DAGVarInfo {
    var: PyObject,
    type_: DAGVarType,
    in_node: NodeIndex,
    out_node: NodeIndex,
}

#[pymethods]
impl DAGCircuit {
    #[new]
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(DAGCircuit {
            name: None,
            metadata: Some(PyDict::new(py).unbind().into()),
            dag: StableDiGraph::default(),
            qregs: RegisterData::new(),
            cregs: RegisterData::new(),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: BitData::new(),
            clbits: BitData::new(),
            vars: BitData::new(),
            global_phase: Param::Float(0.),
            duration: None,
            unit: "dt".to_string(),
            qubit_locations: BitLocator::new(),
            clbit_locations: BitLocator::new(),
            qubit_io_map: Vec::new(),
            clbit_io_map: Vec::new(),
            var_io_map: Vec::new(),
            op_names: IndexMap::default(),
            control_flow_module: PyControlFlowModule::new(py)?,
            vars_info: HashMap::new(),
            vars_by_type: [
                PySet::empty(py)?.unbind(),
                PySet::empty(py)?.unbind(),
                PySet::empty(py)?.unbind(),
            ],
            captured_stretches: IndexMap::default(),
            declared_stretches: IndexMap::default(),
        })
    }

    /// Returns the dict containing the QuantumRegisters in the circuit
    #[getter]
    fn get_qregs(&self, py: Python) -> &Py<PyDict> {
        self.qregs.cached(py)
    }

    /// Returns a dict mapping Clbit instances to tuple comprised of 0) the
    /// corresponding index in circuit.clbits and 1) a list of
    /// Register-int pairs for each Register containing the Bit and its index
    /// within that register.
    #[getter("_qubit_indices")]
    pub fn get_qubit_locations(&self, py: Python) -> &Py<PyDict> {
        self.qubit_locations.cached(py)
    }

    /// Returns the dict containing the QuantumRegisters in the circuit
    #[getter]
    fn get_cregs(&self, py: Python) -> &Py<PyDict> {
        self.cregs.cached(py)
    }

    /// Returns a dict mapping Clbit instances to tuple comprised of 0) the
    /// corresponding index in circuit.clbits and 1) a list of
    /// Register-int pairs for each Register containing the Bit and its index
    /// within that register.
    #[getter("_clbit_indices")]
    pub fn get_clbit_locations(&self, py: Python) -> &Py<PyDict> {
        self.clbit_locations.cached(py)
    }

    /// Returns the total duration of the circuit, set by a scheduling transpiler pass. Its unit is
    /// specified by :attr:`.unit`
    ///
    /// DEPRECATED since Qiskit 1.3.0 and will be removed in Qiskit 3.0.0
    #[getter("duration")]
    fn get_duration(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        imports::WARNINGS_WARN.get_bound(py).call1((
            intern!(
                py,
                concat!(
                    "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.duration`` is ",
                    "deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 3.0.0.",
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ))?;
        self.get_internal_duration(py)
    }

    /// Returns the total duration of the circuit for internal use (no deprecation warning).
    ///
    /// To be removed with get_duration.
    #[getter("_duration")]
    fn get_internal_duration(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        Ok(self.duration.as_ref().map(|x| x.clone_ref(py)))
    }

    /// Sets the total duration of the circuit, set by a scheduling transpiler pass. Its unit is
    /// specified by :attr:`.unit`
    ///
    /// DEPRECATED since Qiskit 1.3.0 and will be removed in Qiskit 3.0.0
    #[setter("duration")]
    fn set_duration(&mut self, py: Python, duration: Option<PyObject>) -> PyResult<()> {
        imports::WARNINGS_WARN.get_bound(py).call1((
            intern!(
                py,
                concat!(
                    "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.duration`` is ",
                    "deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 3.0.0.",
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ))?;
        self.set_internal_duration(duration);
        Ok(())
    }

    /// Sets the total duration of the circuit for internal use (no deprecation warning).
    ///
    /// To be removed with set_duration.
    #[setter("_duration")]
    fn set_internal_duration(&mut self, duration: Option<PyObject>) {
        self.duration = duration
    }

    /// Returns the unit that duration is specified in.
    ///
    /// DEPRECATED since Qiskit 1.3.0 and will be removed in Qiskit 3.0.0
    #[getter]
    fn get_unit(&self, py: Python) -> PyResult<String> {
        imports::WARNINGS_WARN.get_bound(py).call1((
            intern!(
                py,
                concat!(
                    "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.unit`` is ",
                    "deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 3.0.0.",
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ))?;
        self.get_internal_unit()
    }

    /// Returns the unit that duration is specified in for internal use (no deprecation warning).
    ///
    /// To be removed with get_unit.
    #[getter("_unit")]
    fn get_internal_unit(&self) -> PyResult<String> {
        Ok(self.unit.clone())
    }

    /// Sets the unit that duration is specified in.
    ///
    /// DEPRECATED since Qiskit 1.3.0 and will be removed in Qiskit 3.0.0
    #[setter("unit")]
    fn set_unit(&mut self, py: Python, unit: String) -> PyResult<()> {
        imports::WARNINGS_WARN.get_bound(py).call1((
            intern!(
                py,
                concat!(
                    "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.unit`` is ",
                    "deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 3.0.0.",
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ))?;
        self.set_internal_unit(unit);
        Ok(())
    }

    /// Sets the unit that duration is specified in for internal use (no deprecation warning).
    ///
    /// To be removed with set_unit.
    #[setter("_unit")]
    fn set_internal_unit(&mut self, unit: String) {
        self.unit = unit
    }

    #[getter]
    fn input_map(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new(py);
        for (qubit, indices) in self
            .qubit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Qubit::new(idx), indices))
        {
            out_dict.set_item(
                self.qubits.get(qubit).unwrap(),
                self.get_node(py, indices[0])?,
            )?;
        }
        for (clbit, indices) in self
            .clbit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Clbit::new(idx), indices))
        {
            out_dict.set_item(
                self.clbits.get(clbit).unwrap(),
                self.get_node(py, indices[0])?,
            )?;
        }
        for (var, indices) in self
            .var_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Var::new(idx), indices))
        {
            out_dict.set_item(
                self.vars.get(var).unwrap().into_pyobject(py)?,
                self.get_node(py, indices[0])?,
            )?;
        }
        Ok(out_dict.unbind())
    }

    #[getter]
    fn output_map(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new(py);
        for (qubit, indices) in self
            .qubit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Qubit::new(idx), indices))
        {
            out_dict.set_item(
                self.qubits.get(qubit).unwrap(),
                self.get_node(py, indices[1])?,
            )?;
        }
        for (clbit, indices) in self
            .clbit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Clbit::new(idx), indices))
        {
            out_dict.set_item(
                self.clbits.get(clbit).unwrap(),
                self.get_node(py, indices[1])?,
            )?;
        }
        for (var, indices) in self
            .var_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Var::new(idx), indices))
        {
            out_dict.set_item(
                self.vars
                    .get(var)
                    .unwrap()
                    .clone_ref(py)
                    .into_pyobject(py)?,
                self.get_node(py, indices[1])?,
            )?;
        }
        Ok(out_dict.unbind())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new(py);
        out_dict.set_item("name", self.name.as_ref().map(|x| x.clone_ref(py)))?;
        out_dict.set_item("metadata", self.metadata.as_ref().map(|x| x.clone_ref(py)))?;
        out_dict.set_item("qregs", self.qregs.cached(py))?;
        out_dict.set_item("cregs", self.cregs.cached(py))?;
        out_dict.set_item("global_phase", self.global_phase.clone())?;
        out_dict.set_item(
            "qubit_io_map",
            self.qubit_io_map
                .iter()
                .enumerate()
                .map(|(k, v)| (k, [v[0].index(), v[1].index()]))
                .into_py_dict(py)?,
        )?;
        out_dict.set_item(
            "clbit_io_map",
            self.clbit_io_map
                .iter()
                .enumerate()
                .map(|(k, v)| (k, [v[0].index(), v[1].index()]))
                .into_py_dict(py)?,
        )?;
        out_dict.set_item(
            "var_io_map",
            self.var_io_map
                .iter()
                .enumerate()
                .map(|(k, v)| (k, [v[0].index(), v[1].index()]))
                .into_py_dict(py)?,
        )?;
        out_dict.set_item("op_name", self.op_names.clone())?;
        out_dict.set_item(
            "vars_info",
            self.vars_info
                .iter()
                .map(|(k, v)| {
                    (
                        k,
                        (
                            v.var.clone_ref(py),
                            v.type_ as u8,
                            v.in_node.index(),
                            v.out_node.index(),
                        ),
                    )
                })
                .into_py_dict(py)?,
        )?;
        out_dict.set_item(
            "captured_stretches",
            self.captured_stretches.iter().into_py_dict(py)?,
        )?;
        out_dict.set_item(
            "declared_stretches",
            self.declared_stretches.iter().into_py_dict(py)?,
        )?;
        out_dict.set_item("vars_by_type", self.vars_by_type.clone())?;
        out_dict.set_item("qubits", self.qubits.bits())?;
        out_dict.set_item("clbits", self.clbits.bits())?;
        out_dict.set_item("vars", self.vars.bits())?;
        let mut nodes: Vec<PyObject> = Vec::with_capacity(self.dag.node_count());
        for node_idx in self.dag.node_indices() {
            let node_data = self.get_node(py, node_idx)?;
            nodes.push((node_idx.index(), node_data).into_py_any(py)?);
        }
        out_dict.set_item("nodes", nodes)?;
        out_dict.set_item(
            "nodes_removed",
            self.dag.node_count() != self.dag.node_bound(),
        )?;
        let mut edges: Vec<PyObject> = Vec::with_capacity(self.dag.edge_bound());
        // edges are saved with none (deleted edges) instead of their index to save space
        for i in 0..self.dag.edge_bound() {
            let idx = EdgeIndex::new(i);
            let edge = match self.dag.edge_weight(idx) {
                Some(edge_w) => {
                    let endpoints = self.dag.edge_endpoints(idx).unwrap();
                    (
                        endpoints.0.index(),
                        endpoints.1.index(),
                        edge_w.clone().to_pickle(py)?,
                    )
                        .into_py_any(py)?
                }
                None => py.None(),
            };
            edges.push(edge);
        }
        out_dict.set_item("edges", edges)?;
        Ok(out_dict.unbind())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let dict_state = state.downcast_bound::<PyDict>(py)?;
        self.name = dict_state.get_item("name")?.unwrap().extract()?;
        self.metadata = dict_state.get_item("metadata")?.unwrap().extract()?;
        self.qregs = RegisterData::from_mapping(
            dict_state
                .get_item("qregs")?
                .unwrap()
                .extract::<IndexMap<String, QuantumRegister>>()?,
        );
        self.cregs = RegisterData::from_mapping(
            dict_state
                .get_item("cregs")?
                .unwrap()
                .extract::<IndexMap<String, ClassicalRegister>>()?,
        );
        self.global_phase = dict_state.get_item("global_phase")?.unwrap().extract()?;
        self.op_names = dict_state.get_item("op_name")?.unwrap().extract()?;
        self.vars_by_type = dict_state.get_item("vars_by_type")?.unwrap().extract()?;
        let binding = dict_state.get_item("vars_info")?.unwrap();
        let vars_info_raw = binding.downcast::<PyDict>().unwrap();
        self.vars_info = HashMap::with_capacity(vars_info_raw.len());
        for (key, value) in vars_info_raw.iter() {
            let val_tuple = value.downcast::<PyTuple>()?;
            let info = DAGVarInfo {
                var: val_tuple.get_item(0)?.unbind(),
                type_: match val_tuple.get_item(1)?.extract::<u8>()? {
                    0 => DAGVarType::Input,
                    1 => DAGVarType::Capture,
                    2 => DAGVarType::Declare,
                    _ => return Err(PyValueError::new_err("Invalid var type")),
                },
                in_node: NodeIndex::new(val_tuple.get_item(2)?.extract()?),
                out_node: NodeIndex::new(val_tuple.get_item(3)?.extract()?),
            };
            self.vars_info.insert(key.extract()?, info);
        }
        self.captured_stretches = dict_state
            .get_item("captured_stretches")?
            .unwrap()
            .extract()?;
        self.declared_stretches = dict_state
            .get_item("declared_stretches")?
            .unwrap()
            .extract()?;
        let binding = dict_state.get_item("qubits")?.unwrap();
        let qubits_raw = binding.extract::<Vec<ShareableQubit>>()?;
        for bit in qubits_raw.into_iter() {
            self.qubits.add(bit, false)?;
        }
        let binding = dict_state.get_item("clbits")?.unwrap();
        let clbits_raw = binding.extract::<Vec<ShareableClbit>>()?;
        for bit in clbits_raw.into_iter() {
            self.clbits.add(bit, false)?;
        }
        let binding = dict_state.get_item("vars")?.unwrap();
        let vars_raw = binding.downcast::<PyList>()?;
        for bit in vars_raw.iter() {
            self.vars.add(VarAsKey::new(&bit), false)?;
        }
        let binding = dict_state.get_item("qubit_io_map")?.unwrap();
        let qubit_index_map_raw = binding.downcast::<PyDict>().unwrap();
        self.qubit_io_map = Vec::with_capacity(qubit_index_map_raw.len());
        for (_k, v) in qubit_index_map_raw.iter() {
            let indices: [usize; 2] = v.extract()?;
            self.qubit_io_map
                .push([NodeIndex::new(indices[0]), NodeIndex::new(indices[1])]);
        }
        let binding = dict_state.get_item("clbit_io_map")?.unwrap();
        let clbit_index_map_raw = binding.downcast::<PyDict>().unwrap();
        self.clbit_io_map = Vec::with_capacity(clbit_index_map_raw.len());
        for (_k, v) in clbit_index_map_raw.iter() {
            let indices: [usize; 2] = v.extract()?;
            self.clbit_io_map
                .push([NodeIndex::new(indices[0]), NodeIndex::new(indices[1])]);
        }
        let binding = dict_state.get_item("var_io_map")?.unwrap();
        let var_index_map_raw = binding.downcast::<PyDict>().unwrap();
        self.var_io_map = Vec::with_capacity(var_index_map_raw.len());
        for (_k, v) in var_index_map_raw.iter() {
            let indices: [usize; 2] = v.extract()?;
            self.var_io_map
                .push([NodeIndex::new(indices[0]), NodeIndex::new(indices[1])]);
        }
        // Rebuild Graph preserving index holes:
        let binding = dict_state.get_item("nodes")?.unwrap();
        let nodes_lst = binding.downcast::<PyList>()?;
        let binding = dict_state.get_item("edges")?.unwrap();
        let edges_lst = binding.downcast::<PyList>()?;
        let node_removed: bool = dict_state.get_item("nodes_removed")?.unwrap().extract()?;
        self.dag = StableDiGraph::default();
        if !node_removed {
            for item in nodes_lst.iter() {
                let node_w = item.downcast::<PyTuple>().unwrap().get_item(1).unwrap();
                let weight = self.pack_into(py, &node_w)?;
                self.dag.add_node(weight);
            }
        } else if nodes_lst.len() == 1 {
            // graph has only one node, handle logic here to save one if in the loop later
            let binding = nodes_lst.get_item(0).unwrap();
            let item = binding.downcast::<PyTuple>().unwrap();
            let node_idx: usize = item.get_item(0).unwrap().extract().unwrap();
            let node_w = item.get_item(1).unwrap();

            for _i in 0..node_idx {
                self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));
            }
            let weight = self.pack_into(py, &node_w)?;
            self.dag.add_node(weight);
            for i in 0..node_idx {
                self.dag.remove_node(NodeIndex::new(i));
            }
        } else {
            let binding = nodes_lst.get_item(nodes_lst.len() - 1).unwrap();
            let last_item = binding.downcast::<PyTuple>().unwrap();

            // list of temporary nodes that will be removed later to re-create holes
            let node_bound_1: usize = last_item.get_item(0).unwrap().extract().unwrap();
            let mut tmp_nodes: Vec<NodeIndex> =
                Vec::with_capacity(node_bound_1 + 1 - nodes_lst.len());

            for item in nodes_lst {
                let item = item.downcast::<PyTuple>().unwrap();
                let next_index: usize = item.get_item(0).unwrap().extract().unwrap();
                let weight: PyObject = item.get_item(1).unwrap().extract().unwrap();
                while next_index > self.dag.node_bound() {
                    // node does not exist
                    let tmp_node = self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));
                    tmp_nodes.push(tmp_node);
                }
                // add node to the graph, and update the next available node index
                let weight = self.pack_into(py, weight.bind(py))?;
                self.dag.add_node(weight);
            }
            // Remove any temporary nodes we added
            for tmp_node in tmp_nodes {
                self.dag.remove_node(tmp_node);
            }
        }

        // to ensure O(1) on edge deletion, use a temporary node to store missing edges
        let tmp_node = self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));

        for item in edges_lst {
            if item.is_none() {
                // add a temporary edge that will be deleted later to re-create the hole
                self.dag
                    .add_edge(tmp_node, tmp_node, Wire::Qubit(Qubit(u32::MAX)));
            } else {
                let triple = item.downcast::<PyTuple>().unwrap();
                let edge_p: usize = triple.get_item(0).unwrap().extract().unwrap();
                let edge_c: usize = triple.get_item(1).unwrap().extract().unwrap();
                let edge_w = Wire::from_pickle(&triple.get_item(2).unwrap())?;
                self.dag
                    .add_edge(NodeIndex::new(edge_p), NodeIndex::new(edge_c), edge_w);
            }
        }
        self.dag.remove_node(tmp_node);
        self.qubit_locations = BitLocator::with_capacity(self.qubits.len());
        for (index, qubit) in self.qubits.bits().iter().enumerate() {
            let registers = self
                .qregs
                .registers()
                .iter()
                .filter_map(|x| x.index_of(qubit).map(|y| (x.clone(), y)));
            self.qubit_locations
                .insert(qubit.clone(), BitLocations::new(index as u32, registers));
        }
        self.clbit_locations = BitLocator::with_capacity(self.clbits.len());
        for (index, clbit) in self.clbits.bits().iter().enumerate() {
            let registers = self
                .cregs
                .registers()
                .iter()
                .filter_map(|x| x.index_of(clbit).map(|y| (x.clone(), y)));
            self.clbit_locations
                .insert(clbit.clone(), BitLocations::new(index as u32, registers));
        }
        Ok(())
    }

    /// Returns the current sequence of registered :class:`.Qubit` instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`DAGCircuit` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Qubit`): The current sequence of registered qubits.
    #[getter(qubits)]
    pub fn py_qubits(&self, py: Python<'_>) -> Py<PyList> {
        self.qubits.cached(py).clone_ref(py)
    }

    /// Returns the current sequence of registered :class:`.Clbit`
    /// instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`DAGCircuit` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Clbit`): The current sequence of registered clbits.
    #[getter(clbits)]
    pub fn py_clbits(&self, py: Python<'_>) -> Py<PyList> {
        self.clbits.cached(py).clone_ref(py)
    }

    /// Return a list of the wires in order.
    #[getter]
    fn get_wires(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let wires: Bound<PyList> = PyList::new(py, self.qubits.bits().iter())?;

        for clbit in self.clbits.bits().iter() {
            wires.append(clbit)?
        }

        let out_list = PyList::new(py, wires)?;
        for var_type_set in &self.vars_by_type {
            for var in var_type_set.bind(py).iter() {
                out_list.append(var)?;
            }
        }
        Ok(out_list.unbind())
    }

    /// Returns the number of nodes in the dag.
    #[getter]
    fn get_node_counter(&self) -> usize {
        self.dag.node_count()
    }

    /// Return the global phase of the circuit.
    #[getter]
    pub fn get_global_phase(&self) -> Param {
        self.global_phase.clone()
    }

    /// Set the global phase of the circuit.
    ///
    /// Args:
    ///     angle (float, :class:`.ParameterExpression`): The phase angle.
    #[setter]
    pub fn set_global_phase(&mut self, angle: Param) -> PyResult<()> {
        match angle {
            Param::Float(angle) => {
                self.global_phase = Param::Float(angle.rem_euclid(2. * PI));
            }
            Param::ParameterExpression(angle) => {
                self.global_phase = Param::ParameterExpression(angle);
            }
            Param::Obj(_) => return Err(PyTypeError::new_err("Invalid type for global phase")),
        }
        Ok(())
    }

    /// Remove all operation nodes with the given name.
    fn remove_all_ops_named(&mut self, opname: &str) {
        let mut to_remove = Vec::new();
        for (id, weight) in self.dag.node_references() {
            if let NodeType::Operation(packed) = &weight {
                if opname == packed.op.name() {
                    to_remove.push(id);
                }
            }
        }
        for node in to_remove {
            self.remove_op_node(node);
        }
    }

    /// Add individual qubit wires.
    fn add_qubits(&mut self, qubits: Vec<Bound<PyAny>>) -> PyResult<()> {
        for bit in qubits.into_iter() {
            let Ok(bit) = bit.extract::<ShareableQubit>() else {
                return Err(DAGCircuitError::new_err("not a Qubit instance."));
            };
            if self.qubits.find(&bit).is_some() {
                return Err(DAGCircuitError::new_err(format!(
                    "duplicate qubits {:?}",
                    bit
                )));
            }
            self.add_qubit_unchecked(bit)?;
        }
        Ok(())
    }

    /// Add individual qubit wires.
    fn add_clbits(&mut self, clbits: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        for bit in clbits.into_iter() {
            let Ok(bit) = bit.extract::<ShareableClbit>() else {
                return Err(DAGCircuitError::new_err("not a Clbit instance."));
            };
            if self.clbits.find(&bit).is_some() {
                return Err(DAGCircuitError::new_err(format!(
                    "duplicate clbits {:?}",
                    bit
                )));
            }
            self.add_clbit_unchecked(bit)?;
        }
        Ok(())
    }

    /// Add all wires in a quantum register.
    pub fn add_qreg(&mut self, qreg: QuantumRegister) -> PyResult<()> {
        self.qregs
            .add_register(qreg.clone(), true)
            .map_err(|_| DAGCircuitError::new_err(format!("duplicate register {}", qreg.name())))?;

        for (index, bit) in qreg.bits().enumerate() {
            if self.qubits.find(&bit).is_none() {
                self.add_qubit_unchecked(bit.clone())?;
            }
            let locations: &mut BitLocations<QuantumRegister> =
                self.qubit_locations.get_mut(&bit).unwrap();
            locations.add_register(qreg.clone(), index);
        }
        Ok(())
    }

    /// Add all wires in a classical register.
    fn add_creg(&mut self, creg: ClassicalRegister) -> PyResult<()> {
        self.cregs
            .add_register(creg.clone(), true)
            .map_err(|_| DAGCircuitError::new_err(format!("duplicate register {}", creg.name())))?;

        for (index, bit) in creg.bits().enumerate() {
            if self.clbits.find(&bit).is_none() {
                self.add_clbit_unchecked(bit.clone())?;
            }
            let locations: &mut BitLocations<ClassicalRegister> =
                self.clbit_locations.get_mut(&bit).unwrap();
            locations.add_register(creg.clone(), index);
        }
        Ok(())
    }

    /// Finds locations in the circuit, by mapping the Qubit and Clbit to positional index
    /// BitLocations is defined as: BitLocations = namedtuple("BitLocations", ("index", "registers"))
    ///
    /// Args:
    ///     bit (Bit): The bit to locate.
    ///
    /// Returns:
    ///     namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
    ///         contains the index at which the ``Bit`` can be found (in either
    ///         :obj:`~DAGCircuit.qubits`, :obj:`~DAGCircuit.clbits`, depending on its
    ///         type). The second element (``registers``) is a list of ``(register, index)``
    ///         pairs with an entry for each :obj:`~Register` in the circuit which contains the
    ///         :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).
    ///
    ///   Raises:
    ///     DAGCircuitError: If the supplied :obj:`~Bit` was of an unknown type.
    ///     DAGCircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.
    fn find_bit<'py>(
        &self,
        py: Python<'py>,
        bit: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBitLocations>> {
        if let Ok(qubit) = bit.extract::<ShareableQubit>() {
            self.qubit_locations
                .get(&qubit)
                .map(|location| location.clone().into_pyobject(py))
                .transpose()?
                .ok_or_else(|| {
                    DAGCircuitError::new_err(format!(
                        "Could not locate provided bit: {}. Has it been added to the DAGCircuit?",
                        bit
                    ))
                })
        } else if let Ok(clbit) = bit.extract::<ShareableClbit>() {
            self.clbit_locations
                .get(&clbit)
                .map(|location| location.clone().into_pyobject(py))
                .transpose()?
                .ok_or_else(|| {
                    DAGCircuitError::new_err(format!(
                        "Could not locate provided bit: {}. Has it been added to the DAGCircuit?",
                        bit
                    ))
                })
        } else {
            Err(DAGCircuitError::new_err(format!(
                "Could not locate bit of unknown type: {}",
                bit.get_type()
            )))
        }
    }

    /// Remove classical bits from the circuit. All bits MUST be idle.
    /// Any registers with references to at least one of the specified bits will
    /// also be removed.
    ///
    /// .. warning::
    ///     This method is rather slow, since it must iterate over the entire
    ///     DAG to fix-up bit indices.
    ///
    /// Args:
    ///     clbits (List[Clbit]): The bits to remove.
    ///
    /// Raises:
    ///     DAGCircuitError: a clbit is not a :obj:`.Clbit`, is not in the circuit,
    ///         or is not idle.
    #[pyo3(signature = (*clbits))]
    fn remove_clbits(&mut self, py: Python, clbits: Vec<ShareableClbit>) -> PyResult<()> {
        let bit_iter = match self.clbits.map_bits(clbits.iter().cloned()) {
            Ok(bit_iter) => bit_iter,
            Err(_) => {
                return Err(DAGCircuitError::new_err(format!(
                    "clbits not in circuit: {:?}",
                    clbits
                )))
            }
        };
        let clbits: HashSet<Clbit> = bit_iter.collect();
        let mut busy_bits = Vec::new();
        for bit in clbits.iter() {
            if !self.is_wire_idle(&Wire::Clbit(*bit))? {
                busy_bits.push(self.clbits.get(*bit).unwrap());
            }
        }

        if !busy_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "clbits not idle: {:?}",
                busy_bits
            )));
        }

        // Remove any references to bits.
        let mut cregs_to_remove = Vec::new();
        for creg in self.cregs.registers() {
            for bit in creg.bits() {
                if clbits.contains(&self.clbits.find(&bit).unwrap()) {
                    cregs_to_remove.push(creg.clone());
                    break;
                }
            }
        }
        self.remove_cregs(&PyTuple::new(py, cregs_to_remove)?)?;

        // Remove DAG in/out nodes etc.
        for bit in clbits.iter() {
            self.remove_idle_wire(Wire::Clbit(*bit))?;
        }

        // Copy the current clbit mapping so we can use it while remapping
        // wires used on edges and in operation cargs.
        let old_clbits = self.clbits.clone();

        // Remove the clbit indices, which will invalidate our mapping of Clbit to
        // Python bits throughout the entire DAG.
        self.clbits.remove_indices(clbits.clone())?;

        // Update input/output maps to use new Clbits.
        let io_mapping: HashMap<Clbit, [NodeIndex; 2]> = self
            .clbit_io_map
            .drain(..)
            .enumerate()
            .filter_map(|(k, v)| {
                let clbit = Clbit::new(k);
                if clbits.contains(&clbit) {
                    None
                } else {
                    Some((
                        self.clbits
                            .find(old_clbits.get(Clbit::new(k)).unwrap())
                            .unwrap(),
                        v,
                    ))
                }
            })
            .collect();

        self.clbit_io_map = (0..io_mapping.len())
            .map(|idx| {
                let clbit = Clbit::new(idx);
                io_mapping[&clbit]
            })
            .collect();

        // Update edges to use the new Clbits.
        for edge_weight in self.dag.edge_weights_mut() {
            if let Wire::Clbit(c) = edge_weight {
                *c = self.clbits.find(old_clbits.get(*c).unwrap()).unwrap();
            }
        }

        // Update operation cargs to use the new Clbits.
        for node_weight in self.dag.node_weights_mut() {
            match node_weight {
                NodeType::Operation(op) => {
                    let cargs = self.cargs_interner.get(op.clbits);
                    let carg_bits = old_clbits.map_indices(cargs).cloned();
                    op.clbits = self
                        .cargs_interner
                        .insert_owned(self.clbits.map_bits(carg_bits)?.collect());
                }
                NodeType::ClbitIn(c) | NodeType::ClbitOut(c) => {
                    *c = self.clbits.find(old_clbits.get(*c).unwrap()).unwrap();
                }
                _ => (),
            }
        }

        // Update bit locations.
        for (i, bit) in self.clbits.bits().iter().enumerate() {
            let raw_loc = self.clbit_locations.get_mut(bit).unwrap();
            raw_loc.index = i as u32;
        }
        Ok(())
    }

    /// Remove classical registers from the circuit, leaving underlying bits
    /// in place.
    ///
    /// Raises:
    ///     DAGCircuitError: a creg is not a ClassicalRegister, or is not in
    ///     the circuit.
    #[pyo3(signature = (*cregs))]
    fn remove_cregs(&mut self, cregs: &Bound<PyTuple>) -> PyResult<()> {
        let mut valid_regs: Vec<ClassicalRegister> = Vec::new();
        for reg in cregs.iter() {
            if let Ok(creg) = reg.extract::<ClassicalRegister>() {
                if let Some(reg) = self.cregs.get(creg.name()) {
                    if reg != &creg {
                        return Err(DAGCircuitError::new_err(format!(
                            "creg not in circuit: {:?}",
                            reg
                        )));
                    }
                    valid_regs.push(creg);
                } else {
                    return Err(DAGCircuitError::new_err(format!(
                        "creg not in circuit: {:?}",
                        reg
                    )));
                }
            } else {
                return Err(DAGCircuitError::new_err(format!(
                    "creg not of type ClassicalRegister: {:?}",
                    reg
                )));
            }
        }

        // Use an iterator that will remove the registers from the circuit as it iterates.
        let valid_names = valid_regs.iter().map(|reg| {
            for (index, bit) in reg.bits().enumerate() {
                let bit_position = self.clbit_locations.get_mut(&bit).unwrap();
                bit_position.remove_register(reg, index);
            }
            reg.name().to_string()
        });
        self.cregs.remove_registers(valid_names);
        Ok(())
    }

    /// Remove quantum bits from the circuit. All bits MUST be idle.
    /// Any registers with references to at least one of the specified bits will
    /// also be removed.
    ///
    /// .. warning::
    ///     This method is rather slow, since it must iterate over the entire
    ///     DAG to fix-up bit indices.
    ///
    /// Args:
    ///     qubits (List[~qiskit.circuit.Qubit]): The bits to remove.
    ///
    /// Raises:
    ///     DAGCircuitError: a qubit is not a :obj:`~.circuit.Qubit`, is not in the circuit,
    ///         or is not idle.
    #[pyo3(signature = (*qubits))]
    fn remove_qubits(&mut self, py: Python, qubits: Vec<ShareableQubit>) -> PyResult<()> {
        let bit_iter = match self.qubits.map_bits(qubits.iter().cloned()) {
            Ok(bit_iter) => bit_iter,
            Err(_) => {
                return Err(DAGCircuitError::new_err(format!(
                    "qubits not in circuit: {:?}",
                    qubits
                )))
            }
        };
        let qubits: HashSet<Qubit> = bit_iter.collect();

        let mut busy_bits = Vec::new();
        for bit in qubits.iter() {
            if !self.is_wire_idle(&Wire::Qubit(*bit))? {
                busy_bits.push(self.qubits.get(*bit).unwrap());
            }
        }

        if !busy_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qubits not idle: {:?}",
                busy_bits
            )));
        }

        // Remove any references to bits.
        let mut qregs_to_remove = Vec::new();
        for qreg in self.qregs.registers() {
            for bit in qreg.bits() {
                if qubits.contains(&self.qubits.find(&bit).unwrap()) {
                    qregs_to_remove.push(qreg.clone());
                    break;
                }
            }
        }
        self.remove_qregs(&PyTuple::new(py, qregs_to_remove)?)?;

        // Remove DAG in/out nodes etc.
        for bit in qubits.iter() {
            self.remove_idle_wire(Wire::Qubit(*bit))?;
        }

        // Copy the current qubit mapping so we can use it while remapping
        // wires used on edges and in operation qargs.
        let old_qubits = self.qubits.clone();

        // Remove the qubit indices, which will invalidate our mapping of Qubit to
        // Python bits throughout the entire DAG.
        self.qubits.remove_indices(qubits.clone())?;

        // Update input/output maps to use new Qubits.
        let io_mapping: HashMap<Qubit, [NodeIndex; 2]> = self
            .qubit_io_map
            .drain(..)
            .enumerate()
            .filter_map(|(k, v)| {
                let qubit = Qubit::new(k);
                if qubits.contains(&qubit) {
                    None
                } else {
                    Some((self.qubits.find(old_qubits.get(qubit).unwrap()).unwrap(), v))
                }
            })
            .collect();

        self.qubit_io_map = (0..io_mapping.len())
            .map(|idx| {
                let qubit = Qubit::new(idx);
                io_mapping[&qubit]
            })
            .collect();

        // Update edges to use the new Qubits.
        for edge_weight in self.dag.edge_weights_mut() {
            if let Wire::Qubit(b) = edge_weight {
                *b = self.qubits.find(old_qubits.get(*b).unwrap()).unwrap();
            }
        }

        // Update operation qargs to use the new Qubits.
        for node_weight in self.dag.node_weights_mut() {
            match node_weight {
                NodeType::Operation(op) => {
                    let qargs = self.qargs_interner.get(op.qubits);
                    let qarg_bits = old_qubits.map_indices(qargs).cloned();
                    op.qubits = self
                        .qargs_interner
                        .insert_owned(self.qubits.map_bits(qarg_bits)?.collect());
                }
                NodeType::QubitIn(q) | NodeType::QubitOut(q) => {
                    *q = self.qubits.find(old_qubits.get(*q).unwrap()).unwrap();
                }
                _ => (),
            }
        }

        // Update bit locations.
        for (i, bit) in self.qubits.bits().iter().enumerate() {
            let raw_loc = self.qubit_locations.get_mut(bit).unwrap();
            raw_loc.index = i as u32;
        }
        Ok(())
    }

    /// Remove quantum registers from the circuit, leaving underlying bits
    /// in place.
    ///
    /// Raises:
    ///     DAGCircuitError: a qreg is not a QuantumRegister, or is not in
    ///     the circuit.
    #[pyo3(signature = (*qregs))]
    fn remove_qregs(&mut self, qregs: &Bound<PyTuple>) -> PyResult<()> {
        // let self_bound_cregs = self.cregs.bind(py);
        let mut valid_regs: Vec<QuantumRegister> = Vec::new();
        for reg in qregs.iter() {
            if let Ok(qregs) = reg.extract::<QuantumRegister>() {
                if let Some(reg) = self.qregs.get(qregs.name()) {
                    if reg != &qregs {
                        return Err(DAGCircuitError::new_err(format!(
                            "creg not in circuit: {:?}",
                            reg
                        )));
                    }
                    valid_regs.push(qregs);
                } else {
                    return Err(DAGCircuitError::new_err(format!(
                        "creg not in circuit: {:?}",
                        reg
                    )));
                }
            } else {
                return Err(DAGCircuitError::new_err(format!(
                    "creg not of type ClassicalRegister: {:?}",
                    reg
                )));
            }
        }

        // Use an iterator that will remove the registers from the circuit as it iterates.
        let valid_names = valid_regs.iter().map(|reg| {
            for (index, bit) in reg.bits().enumerate() {
                let bit_position = self.qubit_locations.get_mut(&bit).unwrap();
                bit_position.remove_register(reg, index);
            }
            reg.name().to_string()
        });
        self.qregs.remove_registers(valid_names);
        Ok(())
    }

    /// Verify that the condition is valid.
    ///
    /// Args:
    ///     name (string): used for error reporting
    ///     condition (tuple or None): a condition tuple (ClassicalRegister, int) or (Clbit, bool)
    ///
    /// Raises:
    ///     DAGCircuitError: if conditioning on an invalid register
    fn _check_condition(&self, py: Python, name: &str, condition: &Bound<PyAny>) -> PyResult<()> {
        if condition.is_none() {
            return Ok(());
        }

        let resources = self.control_flow_module.condition_resources(condition)?;
        for reg in resources.cregs.bind(py) {
            if !self
                .cregs
                .contains_key(reg.getattr(intern!(py, "name"))?.to_string().as_str())
            {
                return Err(DAGCircuitError::new_err(format!(
                    "invalid creg in condition for {}",
                    name
                )));
            }
        }

        for bit in resources.clbits.bind(py) {
            let bit: ShareableClbit = bit.extract()?;
            if self.clbits.find(&bit).is_none() {
                return Err(DAGCircuitError::new_err(format!(
                    "invalid clbits in condition for {}",
                    name
                )));
            }
        }

        Ok(())
    }

    /// Return a copy of self with the same structure but empty.
    ///
    /// That structure includes:
    ///     * name and other metadata
    ///     * global phase
    ///     * duration
    ///     * all the qubits and clbits, including the registers.
    ///
    /// Returns:
    ///     DAGCircuit: An empty copy of self.
    #[pyo3(signature = (*, vars_mode="alike"))]
    pub fn copy_empty_like(&self, py: Python, vars_mode: &str) -> PyResult<Self> {
        let mut target_dag = DAGCircuit::with_capacity(
            py,
            self.num_qubits(),
            self.num_clbits(),
            Some(self.num_vars()),
            None,
            None,
        )?;
        target_dag.name = self.name.as_ref().map(|n| n.clone_ref(py));
        target_dag.global_phase = self.global_phase.clone();
        target_dag.duration = self.duration.as_ref().map(|d| d.clone_ref(py));
        target_dag.unit.clone_from(&self.unit);
        target_dag.metadata = self.metadata.as_ref().map(|m| m.clone_ref(py));
        target_dag.qargs_interner = self.qargs_interner.clone();
        target_dag.cargs_interner = self.cargs_interner.clone();

        for bit in self.qubits.bits() {
            target_dag.add_qubit_unchecked(bit.clone())?;
        }
        for bit in self.clbits.bits() {
            target_dag.add_clbit_unchecked(bit.clone())?;
        }
        for reg in self.qregs.registers() {
            target_dag.add_qreg(reg.clone())?;
        }
        for reg in self.cregs.registers() {
            target_dag.add_creg(reg.clone())?;
        }
        if vars_mode == "alike" {
            for var in self.vars_by_type[DAGVarType::Input as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Input)?;
            }
            for var in self.vars_by_type[DAGVarType::Capture as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Capture)?;
            }
            for var in self.vars_by_type[DAGVarType::Declare as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Declare)?;
            }
            for stretch in self.captured_stretches.values() {
                target_dag.add_captured_stretch(py, stretch.bind(py))?;
            }
            for stretch in self.declared_stretches.values() {
                target_dag.add_declared_stretch(stretch.bind(py))?;
            }
        } else if vars_mode == "captures" {
            for var in self.vars_by_type[DAGVarType::Input as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Capture)?;
            }
            for var in self.vars_by_type[DAGVarType::Capture as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Capture)?;
            }
            for var in self.vars_by_type[DAGVarType::Declare as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Capture)?;
            }
            for stretch in self.captured_stretches.values() {
                target_dag.add_captured_stretch(py, stretch.bind(py))?;
            }
            for stretch in self.declared_stretches.values() {
                target_dag.add_captured_stretch(py, stretch.bind(py))?;
            }
        } else if vars_mode != "drop" {
            return Err(PyValueError::new_err(format!(
                "unknown vars_mode: '{}'",
                vars_mode
            )));
        }

        Ok(target_dag)
    }

    #[pyo3(signature=(node, check=false))]
    fn _apply_op_node_back(
        &mut self,
        py: Python,
        node: &Bound<PyAny>,
        check: bool,
    ) -> PyResult<()> {
        if let NodeType::Operation(inst) = self.pack_into(py, node)? {
            if check {
                self.check_op_addition(py, &inst)?;
            }

            self.push_back(py, inst)?;
            Ok(())
        } else {
            Err(PyTypeError::new_err("Invalid node type input"))
        }
    }

    /// Apply an operation to the output of the circuit.
    ///
    /// Args:
    ///     op (qiskit.circuit.Operation): the operation associated with the DAG node
    ///     qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
    ///     cargs (tuple[Clbit]): cbits that op will be applied to
    ///     check (bool): If ``True`` (default), this function will enforce that the
    ///         :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
    ///         :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
    ///         uphold these invariants itself, but the cost of several checks will be skipped.
    ///         This is most useful when building a new DAG from a source of known-good nodes.
    /// Returns:
    ///     DAGOpNode: the node for the op that was added to the dag
    ///
    /// Raises:
    ///     DAGCircuitError: if a leaf node is connected to multiple outputs
    #[pyo3(name = "apply_operation_back", signature = (op, qargs=None, cargs=None, *, check=true))]
    pub fn py_apply_operation_back(
        &mut self,
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        check: bool,
    ) -> PyResult<Py<PyAny>> {
        let py_op = op.extract::<OperationFromPython>()?;
        let qargs = qargs
            .map(|q| q.value.extract::<Vec<ShareableQubit>>())
            .transpose()?;
        let cargs = cargs
            .map(|c| c.value.extract::<Vec<ShareableClbit>>())
            .transpose()?;
        let node = {
            let qubits_id = self
                .qargs_interner
                .insert_owned(self.qubits.map_bits(qargs.into_iter().flatten())?.collect());
            let clbits_id = self
                .cargs_interner
                .insert_owned(self.clbits.map_bits(cargs.into_iter().flatten())?.collect());
            let instr = PackedInstruction {
                op: py_op.operation,
                qubits: qubits_id,
                clbits: clbits_id,
                params: (!py_op.params.is_empty()).then(|| Box::new(py_op.params)),
                label: py_op.label,
                #[cfg(feature = "cache_pygates")]
                py_op: op.unbind().into(),
            };

            if check {
                self.check_op_addition(py, &instr)?;
            }
            self.push_back(py, instr)?
        };

        self.get_node(py, node)
    }

    /// Apply an operation to the input of the circuit.
    ///
    /// Args:
    ///     op (qiskit.circuit.Operation): the operation associated with the DAG node
    ///     qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
    ///     cargs (tuple[Clbit]): cbits that op will be applied to
    ///     check (bool): If ``True`` (default), this function will enforce that the
    ///         :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
    ///         :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
    ///         uphold these invariants itself, but the cost of several checks will be skipped.
    ///         This is most useful when building a new DAG from a source of known-good nodes.
    /// Returns:
    ///     DAGOpNode: the node for the op that was added to the dag
    ///
    /// Raises:
    ///     DAGCircuitError: if initial nodes connected to multiple out edges
    #[pyo3(name = "apply_operation_front", signature = (op, qargs=None, cargs=None, *, check=true))]
    fn py_apply_operation_front(
        &mut self,
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        check: bool,
    ) -> PyResult<Py<PyAny>> {
        let py_op = op.extract::<OperationFromPython>()?;
        let qargs = qargs
            .map(|q| q.value.extract::<Vec<ShareableQubit>>())
            .transpose()?;
        let cargs = cargs
            .map(|c| c.value.extract::<Vec<ShareableClbit>>())
            .transpose()?;
        let node = {
            let qubits_id = self
                .qargs_interner
                .insert_owned(self.qubits.map_bits(qargs.into_iter().flatten())?.collect());
            let clbits_id = self
                .cargs_interner
                .insert_owned(self.clbits.map_bits(cargs.into_iter().flatten())?.collect());
            let instr = PackedInstruction {
                op: py_op.operation,
                qubits: qubits_id,
                clbits: clbits_id,
                params: (!py_op.params.is_empty()).then(|| Box::new(py_op.params)),
                label: py_op.label,
                #[cfg(feature = "cache_pygates")]
                py_op: op.unbind().into(),
            };

            if check {
                self.check_op_addition(py, &instr)?;
            }
            self.push_front(py, instr)?
        };

        self.get_node(py, node)
    }

    /// Compose the ``other`` circuit onto the output of this circuit.
    ///
    /// A subset of input wires of ``other`` are mapped
    /// to a subset of output wires of this circuit.
    ///
    /// ``other`` can be narrower or of equal width to ``self``.
    ///
    /// Args:
    ///     other (DAGCircuit): circuit to compose with self
    ///     qubits (list[~qiskit.circuit.Qubit|int]): qubits of self to compose onto.
    ///     clbits (list[Clbit|int]): clbits of self to compose onto.
    ///     front (bool): If True, front composition will be performed (not implemented yet)
    ///     inplace (bool): If True, modify the object. Otherwise return composed circuit.
    ///     inline_captures (bool): If ``True``, variables marked as "captures" in the ``other`` DAG
    ///         will be inlined onto existing uses of those same variables in ``self``.  If ``False``,
    ///         all variables in ``other`` are required to be distinct from ``self``, and they will
    ///         be added to ``self``.
    ///
    /// ..
    ///     Note: unlike `QuantumCircuit.compose`, there's no `var_remap` argument here.  That's
    ///     because the `DAGCircuit` inner-block structure isn't set up well to allow the recursion,
    ///     and `DAGCircuit.compose` is generally only used to rebuild a DAG from layers within
    ///     itself than to join unrelated circuits.  While there's no strong motivating use-case
    ///     (unlike the `QuantumCircuit` equivalent), it's safer and more performant to not provide
    ///     the option.
    ///
    /// Returns:
    ///    DAGCircuit: the composed dag (returns None if inplace==True).
    ///
    /// Raises:
    ///     DAGCircuitError: if ``other`` is wider or there are duplicate edge mappings.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (other, qubits=None, clbits=None, front=false, inplace=true, *, inline_captures=false))]
    fn compose(
        slf: PyRefMut<Self>,
        py: Python,
        other: &DAGCircuit,
        qubits: Option<Bound<PyList>>,
        clbits: Option<Bound<PyList>>,
        front: bool,
        inplace: bool,
        inline_captures: bool,
    ) -> PyResult<Option<PyObject>> {
        if front {
            return Err(DAGCircuitError::new_err(
                "Front composition not supported yet.",
            ));
        }

        if other.qubits.len() > slf.qubits.len() || other.clbits.len() > slf.clbits.len() {
            return Err(DAGCircuitError::new_err(
                "Trying to compose with another DAGCircuit which has more 'in' edges.",
            ));
        }

        // Number of qubits and clbits must match number in circuit or None
        let identity_qubit_map = other
            .qubits
            .bits()
            .iter()
            .zip(slf.qubits.bits())
            .into_py_dict(py)?;
        let identity_clbit_map = other
            .clbits
            .bits()
            .iter()
            .zip(slf.clbits.bits())
            .into_py_dict(py)?;

        let qubit_map: Bound<PyDict> = match qubits {
            None => identity_qubit_map.clone(),
            Some(qubits) => {
                if qubits.len() != other.qubits.len() {
                    return Err(DAGCircuitError::new_err(concat!(
                        "Number of items in qubits parameter does not",
                        " match number of qubits in the circuit."
                    )));
                }

                let self_qubits = slf.qubits.cached(py).bind(py);
                let other_qubits = other.qubits.cached(py).bind(py);
                let dict = PyDict::new(py);
                for (i, q) in qubits.iter().enumerate() {
                    let q = if q.is_instance_of::<PyInt>() {
                        self_qubits.get_item(q.extract()?)?
                    } else {
                        q
                    };

                    dict.set_item(other_qubits.get_item(i)?, q)?;
                }
                dict
            }
        };

        let clbit_map: Bound<PyDict> = match clbits {
            None => identity_clbit_map.clone(),
            Some(clbits) => {
                if clbits.len() != other.clbits.len() {
                    return Err(DAGCircuitError::new_err(concat!(
                        "Number of items in clbits parameter does not",
                        " match number of clbits in the circuit."
                    )));
                }

                let self_clbits = slf.clbits.cached(py).bind(py);
                let other_clbits = other.clbits.cached(py).bind(py);
                let dict = PyDict::new(py);
                for (i, q) in clbits.iter().enumerate() {
                    let q = if q.is_instance_of::<PyInt>() {
                        self_clbits.get_item(q.extract()?)?
                    } else {
                        q
                    };

                    dict.set_item(other_clbits.get_item(i)?, q)?;
                }
                dict
            }
        };

        let edge_map = if qubit_map.is_empty() && clbit_map.is_empty() {
            // try to do a 1-1 mapping in order
            identity_qubit_map
                .iter()
                .chain(identity_clbit_map.iter())
                .into_py_dict(py)?
        } else {
            qubit_map.iter().chain(clbit_map.iter()).into_py_dict(py)?
        };

        // Chck duplicates in wire map.
        {
            let edge_map_values: Vec<_> = edge_map.values().iter().collect();
            if PySet::new(py, edge_map_values.as_slice())?.len() != edge_map.len() {
                return Err(DAGCircuitError::new_err("duplicates in wire_map"));
            }
        }

        // Compose
        let mut dag: PyRefMut<DAGCircuit> = if inplace {
            slf
        } else {
            Py::new(py, slf.clone())?.into_bound(py).borrow_mut()
        };

        dag.global_phase = add_global_phase(py, &dag.global_phase, &other.global_phase)?;

        // This is all the handling we need for realtime variables, if there's no remapping. They:
        //
        // * get added to the DAG and then operations involving them get appended on normally.
        // * get inlined onto an existing variable, then operations get appended normally.
        // * there's a clash or a failed inlining, and we just raise an error.
        //
        // Notably if there's no remapping, there's no need to recurse into control-flow or to do any
        // Var rewriting during the Expr visits.
        for var in other.iter_input_vars(py)?.bind(py) {
            dag.add_input_var(py, &var?)?;
        }
        if inline_captures {
            for var in other.iter_captures(py)?.bind(py) {
                let var = var?;
                if !dag.has_identifier(&var)? {
                    return Err(DAGCircuitError::new_err(format!("Variable '{}' to be inlined is not in the base DAG. If you wanted it to be automatically added, use `inline_captures=False`.", var)));
                }
            }
        } else {
            for var in other.iter_captured_vars(py)?.bind(py) {
                dag.add_captured_var(py, &var?)?;
            }
            for stretch in other.iter_captured_stretches(py)?.bind(py) {
                dag.add_captured_stretch(py, &stretch?)?;
            }
        }
        for var in other.iter_declared_vars(py)?.bind(py) {
            dag.add_declared_var(py, &var?)?;
        }
        for var in other.iter_declared_stretches(py)?.bind(py) {
            dag.add_declared_stretch(&var?)?;
        }

        let variable_mapper = PyVariableMapper::new(
            py,
            dag.cregs.registers().to_vec().into_bound_py_any(py)?,
            Some(edge_map.clone()),
            None,
            Some(wrap_pyfunction!(reject_new_register, py)?.into_py_any(py)?),
        )?;

        for node in other.topological_nodes()? {
            match &other.dag[node] {
                NodeType::QubitIn(q) => {
                    let bit = other.qubits.get(*q).unwrap();
                    let m_wire = edge_map
                        .get_item(bit)?
                        .map(|py_bit| py_bit.extract::<ShareableQubit>())
                        .transpose()?
                        .unwrap_or_else(|| bit.clone());
                    let wire_in_dag = dag.qubits.find(&m_wire);

                    if wire_in_dag.is_none()
                        || (dag.qubit_io_map.len() - 1 < wire_in_dag.unwrap().index())
                    {
                        return Err(DAGCircuitError::new_err(format!(
                            "wire {:?} not in self",
                            m_wire,
                        )));
                    }
                    // TODO: Python code has check here if node.wire is in other._wires. Why?
                }
                NodeType::ClbitIn(c) => {
                    let bit = other.clbits.get(*c).unwrap();
                    let m_wire = edge_map
                        .get_item(bit)?
                        .map(|py_bit| py_bit.extract::<ShareableClbit>())
                        .transpose()?
                        .unwrap_or_else(|| bit.clone());
                    let wire_in_dag = dag.clbits.find(&m_wire);
                    if wire_in_dag.is_none()
                        || dag.clbit_io_map.len() - 1 < wire_in_dag.unwrap().index()
                    {
                        return Err(DAGCircuitError::new_err(format!(
                            "wire {:?} not in self",
                            m_wire,
                        )));
                    }
                    // TODO: Python code has check here if node.wire is in other._wires. Why?
                }
                NodeType::Operation(op) => {
                    let m_qargs = {
                        let qubits = other
                            .qubits
                            .map_indices(other.qargs_interner.get(op.qubits));
                        let mut mapped = Vec::with_capacity(qubits.len());
                        for bit in qubits {
                            mapped.push(
                                edge_map
                                    .get_item(bit)?
                                    .map(|bit| bit.extract())
                                    .transpose()?
                                    .unwrap_or_else(|| bit.clone()),
                            );
                        }
                        PyTuple::new(py, mapped)
                    };
                    let m_cargs = {
                        let clbits = other
                            .clbits
                            .map_indices(other.cargs_interner.get(op.clbits));
                        let mut mapped = Vec::with_capacity(clbits.len());
                        for bit in clbits {
                            mapped.push(
                                edge_map
                                    .get_item(bit)?
                                    .map(|bit| bit.extract())
                                    .transpose()?
                                    .unwrap_or_else(|| bit.clone()),
                            );
                        }
                        PyTuple::new(py, mapped)
                    };

                    // We explicitly create a mutable py_op here since we might
                    // update the condition.
                    let mut py_op = op.unpack_py_op(py)?.into_bound(py);
                    if py_op.getattr(intern!(py, "mutable"))?.extract::<bool>()? {
                        py_op = py_op.call_method0(intern!(py, "to_mutable"))?;
                    }

                    if op.op.control_flow() {
                        if py_op.is_instance(imports::IF_ELSE_OP.get_bound(py))?
                            || py_op.is_instance(imports::WHILE_LOOP_OP.get_bound(py))?
                        {
                            if let Ok(condition) = py_op.getattr(intern!(py, "condition")) {
                                let condition = variable_mapper.map_condition(&condition, true)?;
                                py_op.setattr(intern!(py, "condition"), condition)?;
                            }
                        } else if py_op.is_instance(imports::SWITCH_CASE_OP.get_bound(py))? {
                            py_op.setattr(
                                intern!(py, "target"),
                                variable_mapper
                                    .map_target(&py_op.getattr(intern!(py, "target"))?)?,
                            )?;
                        };
                    }

                    dag.py_apply_operation_back(
                        py,
                        py_op,
                        Some(TupleLikeArg { value: m_qargs? }),
                        Some(TupleLikeArg { value: m_cargs? }),
                        false,
                    )?;
                }
                // If its a Var wire, we already checked that it exists in the destination.
                NodeType::VarIn(_)
                | NodeType::VarOut(_)
                | NodeType::QubitOut(_)
                | NodeType::ClbitOut(_) => (),
            }
        }

        if !inplace {
            let out_obj = dag.into_py_any(py)?;
            Ok(Some(out_obj))
        } else {
            Ok(None)
        }
    }

    /// Reverse the operations in the ``self`` circuit.
    ///
    /// Returns:
    ///     DAGCircuit: the reversed dag.
    fn reverse_ops<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let qc = imports::DAG_TO_CIRCUIT.get_bound(py).call1((slf,))?;
        let reversed = qc.call_method0("reverse_ops")?;
        imports::CIRCUIT_TO_DAG.get_bound(py).call1((reversed,))
    }

    /// Return idle wires.
    ///
    /// Args:
    ///     ignore (list(str)): List of node names to ignore. Default: []
    ///
    /// Yields:
    ///     Bit: Bit in idle wire.
    ///
    /// Raises:
    ///     DAGCircuitError: If the DAG is invalid
    #[pyo3(signature=(ignore=None))]
    fn idle_wires(&self, py: Python, ignore: Option<&Bound<PyList>>) -> PyResult<Py<PyIterator>> {
        let mut result: Vec<PyObject> = Vec::new();
        let wires = (0..self.qubit_io_map.len())
            .map(|idx| Wire::Qubit(Qubit::new(idx)))
            .chain((0..self.clbit_io_map.len()).map(|idx| Wire::Clbit(Clbit::new(idx))))
            .chain((0..self.var_io_map.len()).map(|idx| Wire::Var(Var::new(idx))));
        match ignore {
            Some(ignore) => {
                // Convert the list to a Rust set.
                let ignore_set = ignore
                    .into_iter()
                    .map(|s| s.extract())
                    .collect::<PyResult<HashSet<String>>>()?;
                for wire in wires {
                    let nodes_found = self.nodes_on_wire(&wire, true).into_iter().any(|node| {
                        let weight = self.dag.node_weight(node).unwrap();
                        if let NodeType::Operation(packed) = weight {
                            !ignore_set.contains(packed.op.name())
                        } else {
                            false
                        }
                    });

                    if !nodes_found {
                        result.push(match wire {
                            Wire::Qubit(qubit) => {
                                self.qubits.get(qubit).unwrap().into_py_any(py)?
                            }
                            Wire::Clbit(clbit) => {
                                self.clbits.get(clbit).unwrap().into_py_any(py)?
                            }
                            Wire::Var(var) => self.vars.get(var).unwrap().into_py_any(py)?,
                        });
                    }
                }
            }
            None => {
                for wire in wires {
                    if self.is_wire_idle(&wire)? {
                        result.push(match wire {
                            Wire::Qubit(qubit) => {
                                self.qubits.get(qubit).unwrap().into_py_any(py)?
                            }
                            Wire::Clbit(clbit) => {
                                self.clbits.get(clbit).unwrap().into_py_any(py)?
                            }
                            Wire::Var(var) => self.vars.get(var).unwrap().into_py_any(py)?,
                        });
                    }
                }
            }
        }
        Ok(PyTuple::new(py, result)?.into_any().try_iter()?.unbind())
    }

    /// Return the number of operations.  If there is control flow present, this count may only
    /// be an estimate, as the complete control-flow path cannot be statically known.
    ///
    /// Args:
    ///     recurse: if ``True``, then recurse into control-flow operations.  For loops with
    ///         known-length iterators are counted unrolled.  If-else blocks sum both of the two
    ///         branches.  While loops are counted as if the loop body runs once only.  Defaults to
    ///         ``False`` and raises :class:`.DAGCircuitError` if any control flow is present, to
    ///         avoid silently returning a mostly meaningless number.
    ///
    /// Returns:
    ///     int: the circuit size
    ///
    /// Raises:
    ///     DAGCircuitError: if an unknown :class:`.ControlFlowOp` is present in a call with
    ///         ``recurse=True``, or any control flow is present in a non-recursive call.
    #[pyo3(signature= (*, recurse=false))]
    fn size(&self, py: Python, recurse: bool) -> PyResult<usize> {
        let mut length = self.dag.node_count() - (self.width() * 2);
        if !self.has_control_flow() {
            return Ok(length);
        }
        if !recurse {
            return Err(DAGCircuitError::new_err(concat!(
                "Size with control flow is ambiguous.",
                " You may use `recurse=True` to get a result",
                " but see this method's documentation for the meaning of this."
            )));
        }

        // Handle recursively.
        let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py);
        for node in self.dag.node_weights() {
            let NodeType::Operation(node) = node else {
                continue;
            };
            if !node.op.control_flow() {
                continue;
            }
            let OperationRef::Instruction(inst) = node.op.view() else {
                panic!("control flow op must be an instruction");
            };
            let inst_bound = inst.instruction.bind(py);
            if inst_bound.is_instance(imports::FOR_LOOP_OP.get_bound(py))? {
                let blocks = inst_bound.getattr("blocks")?;
                let block_zero = blocks.get_item(0)?;
                let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block_zero,))?.extract()?;
                length += node.params_view().len() * inner_dag.size(py, true)?
            } else if inst_bound.is_instance(imports::WHILE_LOOP_OP.get_bound(py))? {
                let blocks = inst_bound.getattr("blocks")?;
                let block_zero = blocks.get_item(0)?;
                let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block_zero,))?.extract()?;
                length += inner_dag.size(py, true)?
            } else if inst_bound.is_instance(imports::IF_ELSE_OP.get_bound(py))?
                || inst_bound.is_instance(imports::SWITCH_CASE_OP.get_bound(py))?
            {
                let blocks = inst_bound.getattr("blocks")?;
                for block in blocks.try_iter()? {
                    let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block?,))?.extract()?;
                    length += inner_dag.size(py, true)?;
                }
            } else {
                continue;
            }
            // We don't count a control-flow node itself!
            length -= 1;
        }
        Ok(length)
    }

    /// Return the circuit depth.  If there is control flow present, this count may only be an
    /// estimate, as the complete control-flow path cannot be statically known.
    ///
    /// Args:
    ///     recurse: if ``True``, then recurse into control-flow operations.  For loops
    ///         with known-length iterators are counted as if the loop had been manually unrolled
    ///         (*i.e.* with each iteration of the loop body written out explicitly).
    ///         If-else blocks take the longer case of the two branches.  While loops are counted as
    ///         if the loop body runs once only.  Defaults to ``False`` and raises
    ///         :class:`.DAGCircuitError` if any control flow is present, to avoid silently
    ///         returning a nonsensical number.
    ///
    /// Returns:
    ///     int: the circuit depth
    ///
    /// Raises:
    ///     DAGCircuitError: if not a directed acyclic graph
    ///     DAGCircuitError: if unknown control flow is present in a recursive call, or any control
    ///         flow is present in a non-recursive call.
    #[pyo3(signature= (*, recurse=false))]
    fn depth(&self, py: Python, recurse: bool) -> PyResult<usize> {
        if self.qubits.is_empty() && self.clbits.is_empty() && self.vars_info.is_empty() {
            return Ok(0);
        }
        if !self.has_control_flow() {
            let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
            return match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
                Some(res) => Ok(res.1 - 1),
                None => Err(DAGCircuitError::new_err("not a DAG")),
            };
        }
        if !recurse {
            return Err(DAGCircuitError::new_err(concat!(
                "Depth with control flow is ambiguous.",
                " You may use `recurse=True` to get a result",
                " but see this method's documentation for the meaning of this."
            )));
        }

        // Handle recursively.
        let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py);
        let mut node_lookup: HashMap<NodeIndex, usize> = HashMap::new();
        for (node_index, node) in self.dag.node_references() {
            let NodeType::Operation(node) = node else {
                continue;
            };
            if !node.op.control_flow() {
                continue;
            }
            let OperationRef::Instruction(inst) = node.op.view() else {
                panic!("control flow op must be an instruction")
            };
            let inst_bound = inst.instruction.bind(py);
            let weight = if inst_bound.is_instance(imports::FOR_LOOP_OP.get_bound(py))? {
                node.params_view().len()
            } else {
                1
            };
            if weight == 0 {
                node_lookup.insert(node_index, 0);
            } else {
                let blocks = inst_bound.getattr("blocks")?;
                let mut block_weights: Vec<usize> = Vec::with_capacity(blocks.len()?);
                for block in blocks.try_iter()? {
                    let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block?,))?.extract()?;
                    block_weights.push(inner_dag.depth(py, true)?);
                }
                node_lookup.insert(node_index, weight * block_weights.iter().max().unwrap());
            }
        }

        let weight_fn = |edge: EdgeReference<'_, Wire>| -> Result<usize, Infallible> {
            Ok(*node_lookup.get(&edge.target()).unwrap_or(&1))
        };
        match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
            Some(res) => Ok(res.1 - 1),
            None => Err(DAGCircuitError::new_err("not a DAG")),
        }
    }

    /// Return the total number of qubits + clbits used by the circuit.
    /// This function formerly returned the number of qubits by the calculation
    /// return len(self._wires) - self.num_clbits()
    /// but was changed by issue #2564 to return number of qubits + clbits
    /// with the new function DAGCircuit.num_qubits replacing the former
    /// semantic of DAGCircuit.width().
    fn width(&self) -> usize {
        self.qubits.len() + self.clbits.len() + self.vars_info.len()
    }

    /// Return the total number of qubits used by the circuit.
    /// num_qubits() replaces former use of width().
    /// DAGCircuit.width() now returns qubits + clbits for
    /// consistency with Circuit.width() [qiskit-terra #2564].
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Return the total number of classical bits used by the circuit.
    pub fn num_clbits(&self) -> usize {
        self.clbits.len()
    }

    /// Compute how many components the circuit can decompose into.
    fn num_tensor_factors(&self) -> usize {
        // This function was forked from rustworkx's
        // number_weekly_connected_components() function as of 0.15.0:
        // https://github.com/Qiskit/rustworkx/blob/0.15.0/src/connectivity/mod.rs#L215-L235

        let mut weak_components = self.dag.node_count();
        let mut vertex_sets = UnionFind::new(self.dag.node_bound());
        for edge in self.dag.edge_references() {
            let (a, b) = (edge.source(), edge.target());
            // union the two vertices of the edge
            if vertex_sets.union(a.index(), b.index()) {
                weak_components -= 1
            };
        }
        weak_components
    }

    fn __eq__(&self, py: Python, other: &DAGCircuit) -> PyResult<bool> {
        // Try to convert to float, but in case of unbound ParameterExpressions
        // a TypeError will be raise, fallback to normal equality in those
        // cases.
        let phase_is_close = |self_phase: f64, other_phase: f64| -> bool {
            ((self_phase - other_phase + PI).rem_euclid(2. * PI) - PI).abs() <= 1.0e-10
        };
        let normalize_param = |param: &Param| {
            if let Param::ParameterExpression(ob) = param {
                ob.bind(py)
                    .call_method0(intern!(py, "numeric"))
                    .ok()
                    .map(|ob| ob.extract::<Param>())
                    .unwrap_or_else(|| Ok(param.clone()))
            } else {
                Ok(param.clone())
            }
        };

        let phase_eq = match [
            normalize_param(&self.global_phase)?,
            normalize_param(&other.global_phase)?,
        ] {
            [Param::Float(self_phase), Param::Float(other_phase)] => {
                Ok(phase_is_close(self_phase, other_phase))
            }
            _ => self.global_phase.eq(py, &other.global_phase),
        }?;
        if !phase_eq {
            return Ok(false);
        }

        // We don't do any semantic equivalence between Var nodes, as things stand; DAGs can only be
        // equal in our mind if they use the exact same UUID vars.
        for (our_vars, their_vars) in self.vars_by_type.iter().zip(&other.vars_by_type) {
            if !our_vars.bind(py).eq(their_vars)? {
                return Ok(false);
            }
        }

        if self.captured_stretches.len() != other.captured_stretches.len()
            || self.declared_stretches.len() != other.declared_stretches.len()
        {
            return Ok(false);
        }

        // Note that `captured_stretches` is a set and thus order of captured stretches
        // does not influence equality.
        let our_captured_stretches = PySet::new(py, self.captured_stretches.values())?;
        for their_stretch in other.captured_stretches.values() {
            if !our_captured_stretches.contains(their_stretch)? {
                return Ok(false);
            }
        }

        for (our_stretch, their_stretch) in self
            .declared_stretches
            .values()
            .zip(other.declared_stretches.values())
        {
            if !our_stretch.bind(py).eq(their_stretch)? {
                return Ok(false);
            }
        }

        let self_bit_indices = {
            let indices = self
                .qubits
                .bits()
                .into_pyobject(py)?
                .try_iter()?
                .chain(self.clbits.bits().into_pyobject(py)?.try_iter()?)
                .enumerate()
                .map(|(idx, bit)| -> PyResult<_> { Ok((bit?, idx)) });
            indices.collect::<PyResult<Vec<_>>>()?.into_py_dict(py)?
        };

        let other_bit_indices = {
            let indices = other
                .qubits
                .bits()
                .into_pyobject(py)?
                .try_iter()?
                .chain(other.clbits.bits().clone().into_pyobject(py)?.try_iter()?)
                .enumerate()
                .map(|(idx, bit)| -> PyResult<_> { Ok((bit?, idx)) });
            indices.collect::<PyResult<Vec<_>>>()?.into_py_dict(py)?
        };

        // Check if qregs are the same.
        let self_qregs = self.qregs.registers();
        let other_qregs = &other.qregs;
        if self_qregs.len() != other_qregs.len() {
            return Ok(false);
        }
        for (regname, self_bits) in self_qregs.iter().map(|reg| (reg.name(), reg)) {
            let self_bits: Vec<ShareableQubit> = self_bits.bits().collect();
            let other_bits: Vec<ShareableQubit> = match other_qregs.get(regname) {
                Some(bits) => bits.bits().collect(),
                None => return Ok(false),
            };
            if !self
                .qubits
                .map_bits(self_bits)?
                .eq(other.qubits.map_bits(other_bits)?)
            {
                return Ok(false);
            }
        }

        // Check if cregs are the same.
        let self_cregs = self.cregs.registers();
        let other_cregs = &other.cregs;
        if self_cregs.len() != other_cregs.len() {
            return Ok(false);
        }

        for (regname, self_bits) in self_cregs.iter().map(|reg| (reg.name(), reg)) {
            let self_bits: Vec<ShareableClbit> = self_bits.bits().collect();
            let other_bits: Vec<ShareableClbit> = match other_cregs.get(regname) {
                Some(bits) => bits.bits().collect(),
                None => return Ok(false),
            };
            if !self
                .clbits
                .map_bits(self_bits)?
                .eq(other.clbits.map_bits(other_bits)?)
            {
                return Ok(false);
            }
        }

        // Check for VF2 isomorphic match.
        let condition_op_check = imports::CONDITION_OP_CHECK.get_bound(py);
        let switch_case_op_check = imports::SWITCH_CASE_OP_CHECK.get_bound(py);
        let for_loop_op_check = imports::FOR_LOOP_OP_CHECK.get_bound(py);
        let box_op_check = imports::BOX_OP_CHECK.get_bound(py);
        let node_match = |n1: &NodeType, n2: &NodeType| -> PyResult<bool> {
            match [n1, n2] {
                [NodeType::Operation(inst1), NodeType::Operation(inst2)] => {
                    if inst1.op.name() != inst2.op.name() {
                        return Ok(false);
                    }
                    let check_args = || -> bool {
                        let node1_qargs = self.qargs_interner.get(inst1.qubits);
                        let node2_qargs = other.qargs_interner.get(inst2.qubits);
                        let node1_cargs = self.cargs_interner.get(inst1.clbits);
                        let node2_cargs = other.cargs_interner.get(inst2.clbits);
                        if SEMANTIC_EQ_SYMMETRIC.contains(&inst1.op.name()) {
                            let node1_qargs =
                                node1_qargs.iter().copied().collect::<HashSet<Qubit>>();
                            let node2_qargs =
                                node2_qargs.iter().copied().collect::<HashSet<Qubit>>();
                            let node1_cargs =
                                node1_cargs.iter().copied().collect::<HashSet<Clbit>>();
                            let node2_cargs =
                                node2_cargs.iter().copied().collect::<HashSet<Clbit>>();
                            if node1_qargs != node2_qargs || node1_cargs != node2_cargs {
                                return false;
                            }
                        } else if node1_qargs != node2_qargs || node1_cargs != node2_cargs {
                            return false;
                        }
                        true
                    };
                    match [inst1.op.view(), inst2.op.view()] {
                        [OperationRef::StandardGate(_), OperationRef::StandardGate(_)]
                        | [OperationRef::StandardInstruction(_), OperationRef::StandardInstruction(_)] => {
                            Ok(inst1.py_op_eq(py, inst2)?
                                && check_args()
                                && inst1
                                    .params_view()
                                    .iter()
                                    .zip(inst2.params_view().iter())
                                    .all(|(a, b)| a.is_close(py, b, 1e-10).unwrap()))
                        }
                        [OperationRef::Instruction(op1), OperationRef::Instruction(op2)] => {
                            if op1.control_flow() && op2.control_flow() {
                                let n1 = self.unpack_into(py, NodeIndex::new(0), n1)?;
                                let n2 = other.unpack_into(py, NodeIndex::new(0), n2)?;
                                let name = op1.name();
                                if name == "if_else" || name == "while_loop" {
                                    condition_op_check
                                        .call1((n1, n2, &self_bit_indices, &other_bit_indices))?
                                        .extract()
                                } else if name == "switch_case" {
                                    switch_case_op_check
                                        .call1((n1, n2, &self_bit_indices, &other_bit_indices))?
                                        .extract()
                                } else if name == "for_loop" {
                                    for_loop_op_check
                                        .call1((n1, n2, &self_bit_indices, &other_bit_indices))?
                                        .extract()
                                } else if name == "box" {
                                    box_op_check
                                        .call1((n1, n2, &self_bit_indices, &other_bit_indices))?
                                        .extract()
                                } else {
                                    Err(PyRuntimeError::new_err(format!(
                                        "unhandled control-flow operation: {}",
                                        name
                                    )))
                                }
                            } else {
                                Ok(inst1.py_op_eq(py, inst2)? && check_args())
                            }
                        }
                        [OperationRef::Gate(_op1), OperationRef::Gate(_op2)] => {
                            Ok(inst1.py_op_eq(py, inst2)? && check_args())
                        }
                        [OperationRef::Operation(_op1), OperationRef::Operation(_op2)] => {
                            Ok(inst1.py_op_eq(py, inst2)? && check_args())
                        }
                        // Handle the edge case where we end up with a Python object and a standard
                        // gate/instruction.
                        // This typically only happens if we have a ControlledGate in Python
                        // and we have mutable state set.
                        [OperationRef::StandardGate(_), OperationRef::Gate(_)]
                        | [OperationRef::Gate(_), OperationRef::StandardGate(_)]
                        | [OperationRef::StandardInstruction(_), OperationRef::Instruction(_)]
                        | [OperationRef::Instruction(_), OperationRef::StandardInstruction(_)] => {
                            Ok(inst1.py_op_eq(py, inst2)? && check_args())
                        }
                        [OperationRef::Unitary(op_a), OperationRef::Unitary(op_b)] => {
                            match [&op_a.array, &op_b.array] {
                                [ArrayType::NDArray(a), ArrayType::NDArray(b)] => {
                                    Ok(relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8))
                                }
                                [ArrayType::OneQ(a), ArrayType::NDArray(b)]
                                | [ArrayType::NDArray(b), ArrayType::OneQ(a)] => {
                                    if b.shape()[0] == 2 {
                                        for i in 0..2 {
                                            for j in 0..2 {
                                                if !relative_eq!(
                                                    b[[i, j]],
                                                    a[(i, j)],
                                                    max_relative = 1e-5,
                                                    epsilon = 1e-8
                                                ) {
                                                    return Ok(false);
                                                }
                                            }
                                        }
                                        Ok(true)
                                    } else {
                                        Ok(false)
                                    }
                                }
                                [ArrayType::TwoQ(a), ArrayType::NDArray(b)]
                                | [ArrayType::NDArray(b), ArrayType::TwoQ(a)] => {
                                    if b.shape()[0] == 4 {
                                        for i in 0..4 {
                                            for j in 0..4 {
                                                if !relative_eq!(
                                                    b[[i, j]],
                                                    a[(i, j)],
                                                    max_relative = 1e-5,
                                                    epsilon = 1e-8
                                                ) {
                                                    return Ok(false);
                                                }
                                            }
                                        }
                                        Ok(true)
                                    } else {
                                        Ok(false)
                                    }
                                }
                                [ArrayType::OneQ(a), ArrayType::OneQ(b)] => {
                                    Ok(relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8))
                                }
                                [ArrayType::TwoQ(a), ArrayType::TwoQ(b)] => {
                                    Ok(relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8))
                                }
                                _ => Ok(false),
                            }
                        }
                        _ => Ok(false),
                    }
                }
                [NodeType::QubitIn(bit1), NodeType::QubitIn(bit2)] => Ok(bit1 == bit2),
                [NodeType::ClbitIn(bit1), NodeType::ClbitIn(bit2)] => Ok(bit1 == bit2),
                [NodeType::QubitOut(bit1), NodeType::QubitOut(bit2)] => Ok(bit1 == bit2),
                [NodeType::ClbitOut(bit1), NodeType::ClbitOut(bit2)] => Ok(bit1 == bit2),
                [NodeType::VarIn(var1), NodeType::VarIn(var2)] => self
                    .vars
                    .get(*var1)
                    .into_pyobject(py)?
                    .eq(other.vars.get(*var2).unwrap().into_pyobject(py)?),
                [NodeType::VarOut(var1), NodeType::VarOut(var2)] => {
                    self.vars.get(*var1).into_pyobject(py)?.eq(other
                        .vars
                        .get(*var2)
                        .unwrap()
                        .clone_ref(py)
                        .into_pyobject(py)?)
                }
                _ => Ok(false),
            }
        };

        isomorphism::vf2::is_isomorphic(
            &self.dag,
            &other.dag,
            node_match,
            isomorphism::vf2::NoSemanticMatch,
            true,
            Ordering::Equal,
            true,
            None,
        )
        .map_err(|e| match e {
            isomorphism::vf2::IsIsomorphicError::NodeMatcherErr(e) => e,
            _ => {
                unreachable!()
            }
        })
    }

    /// Yield nodes in topological order.
    ///
    /// Args:
    ///     key (Callable): A callable which will take a DAGNode object and
    ///         return a string sort key. If not specified the bit qargs and
    ///         cargs of a node will be used for sorting.
    ///
    /// Returns:
    ///     generator(DAGOpNode, DAGInNode, or DAGOutNode): node in topological order
    #[pyo3(name = "topological_nodes", signature=(key=None))]
    fn py_topological_nodes(
        &self,
        py: Python,
        key: Option<Bound<PyAny>>,
    ) -> PyResult<Py<PyIterator>> {
        let nodes: PyResult<Vec<_>> = if let Some(key) = key {
            self.topological_key_sort(py, &key)?
                .map(|node| self.get_node(py, node))
                .collect()
        } else {
            // Good path, using interner IDs.
            self.topological_nodes()?
                .map(|n| self.get_node(py, n))
                .collect()
        };

        Ok(PyTuple::new(py, nodes?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Yield op nodes in topological order.
    ///
    /// Allowed to pass in specific key to break ties in top order
    ///
    /// Args:
    ///     key (Callable): A callable which will take a DAGNode object and
    ///         return a string sort key. If not specified the qargs and
    ///         cargs of a node will be used for sorting.
    ///
    /// Returns:
    ///     generator(DAGOpNode): op node in topological order
    #[pyo3(name = "topological_op_nodes", signature=(key=None))]
    fn py_topological_op_nodes(
        &self,
        py: Python,
        key: Option<Bound<PyAny>>,
    ) -> PyResult<Py<PyIterator>> {
        let nodes: PyResult<Vec<_>> = if let Some(key) = key {
            self.topological_key_sort(py, &key)?
                .filter_map(|node| match self.dag.node_weight(node) {
                    Some(NodeType::Operation(_)) => Some(self.get_node(py, node)),
                    _ => None,
                })
                .collect()
        } else {
            // Good path, using interner IDs.
            self.topological_op_nodes()?
                .map(|n| self.get_node(py, n))
                .collect()
        };

        Ok(PyTuple::new(py, nodes?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Replace a block of nodes with a single node.
    ///
    /// This is used to consolidate a block of DAGOpNodes into a single
    /// operation. A typical example is a block of gates being consolidated
    /// into a single ``UnitaryGate`` representing the unitary matrix of the
    /// block.
    ///
    /// Args:
    ///     node_block (List[DAGNode]): A list of dag nodes that represents the
    ///         node block to be replaced
    ///     op (qiskit.circuit.Operation): The operation to replace the
    ///         block with
    ///     wire_pos_map (Dict[Bit, int]): The dictionary mapping the bits to their positions in the
    ///         output ``qargs`` or ``cargs``. This is necessary to reconstruct the arg order over
    ///         multiple gates in the combined single op node.  If a :class:`.Bit` is not in the
    ///         dictionary, it will not be added to the args; this can be useful when dealing with
    ///         control-flow operations that have inherent bits in their ``condition`` or ``target``
    ///         fields.
    ///     cycle_check (bool): When set to True this method will check that
    ///         replacing the provided ``node_block`` with a single node
    ///         would introduce a cycle (which would invalidate the
    ///         ``DAGCircuit``) and will raise a ``DAGCircuitError`` if a cycle
    ///         would be introduced. This checking comes with a run time
    ///         penalty. If you can guarantee that your input ``node_block`` is
    ///         a contiguous block and won't introduce a cycle when it's
    ///         contracted to a single node, this can be set to ``False`` to
    ///         improve the runtime performance of this method.
    ///
    /// Raises:
    ///     DAGCircuitError: if ``cycle_check`` is set to ``True`` and replacing
    ///         the specified block introduces a cycle or if ``node_block`` is
    ///         empty.
    ///
    /// Returns:
    ///     DAGOpNode: The op node that replaces the block.
    #[pyo3(signature = (node_block, op, wire_pos_map, cycle_check=true))]
    fn replace_block_with_op(
        &mut self,
        py: Python,
        node_block: Vec<PyRef<DAGNode>>,
        op: Bound<PyAny>,
        wire_pos_map: &Bound<PyDict>,
        cycle_check: bool,
    ) -> PyResult<Py<PyAny>> {
        // If node block is empty return early
        if node_block.is_empty() {
            return Err(DAGCircuitError::new_err(
                "Can't replace an empty 'node_block'",
            ));
        }

        let mut qubit_pos_map: HashMap<Qubit, usize> = HashMap::new();
        let mut clbit_pos_map: HashMap<Clbit, usize> = HashMap::new();
        for (bit, index) in wire_pos_map.iter() {
            if bit.downcast::<PyQubit>().is_ok() {
                qubit_pos_map.insert(
                    self.qubits.find(&bit.extract::<ShareableQubit>()?).unwrap(),
                    index.extract()?,
                );
            } else if bit.downcast::<PyClbit>().is_ok() {
                clbit_pos_map.insert(
                    self.clbits.find(&bit.extract::<ShareableClbit>()?).unwrap(),
                    index.extract()?,
                );
            } else {
                return Err(DAGCircuitError::new_err(
                    "Wire map keys must be Qubit or Clbit instances.",
                ));
            }
        }

        let block_ids: Vec<_> = node_block.iter().map(|n| n.node.unwrap()).collect();
        let py_op = op.extract::<OperationFromPython>()?;

        let new_node = self.replace_block(
            &block_ids,
            py_op.operation,
            py_op.params,
            py_op.label.as_ref().map(|v| v.as_str()),
            cycle_check,
            &qubit_pos_map,
            &clbit_pos_map,
        )?;
        self.get_node(py, new_node)
    }

    /// Replace one node with dag.
    ///
    /// Args:
    ///     node (DAGOpNode): node to substitute
    ///     input_dag (DAGCircuit): circuit that will substitute the node
    ///     wires (list[Bit] | Dict[Bit, Bit]): gives an order for (qu)bits
    ///         in the input circuit. If a list, then the bits refer to those in the ``input_dag``,
    ///         and the order gets matched to the node wires by qargs first, then cargs, then
    ///         conditions.  If a dictionary, then a mapping of bits in the ``input_dag`` to those
    ///         that the ``node`` acts on.
    ///     propagate_condition (bool): DEPRECATED a legacy option that used
    ///         to control the behavior of handling control flow. It has no
    ///         effect anymore, left it for backwards compatibility. Will be
    ///         removed in Qiskit 3.0.
    ///
    /// Returns:
    ///     dict: maps node IDs from `input_dag` to their new node incarnations in `self`.
    ///
    /// Raises:
    ///     DAGCircuitError: if met with unexpected predecessor/successors
    #[pyo3(name = "substitute_node_with_dag", signature = (node, input_dag, wires=None, propagate_condition=None))]
    pub fn py_substitute_node_with_dag(
        &mut self,
        py: Python,
        node: &Bound<PyAny>,
        input_dag: &DAGCircuit,
        wires: Option<Bound<PyAny>>,
        propagate_condition: Option<bool>,
    ) -> PyResult<Py<PyDict>> {
        if propagate_condition.is_some() {
            imports::WARNINGS_WARN.get_bound(py).call1((
                intern!(
                    py,
                    concat!(
                        "The propagate_condition argument is deprecated as of Qiskit 2.0.0.",
                        "It has no effect anymore and will be removed in Qiskit 3.0.0.",
                    )
                ),
                py.get_type::<PyDeprecationWarning>(),
                2,
            ))?;
        }
        let (node_index, bound_node) = match node.downcast::<DAGOpNode>() {
            Ok(bound_node) => (bound_node.borrow().as_ref().node.unwrap(), bound_node),
            Err(_) => return Err(DAGCircuitError::new_err("expected node DAGOpNode")),
        };

        let node = match &self.dag[node_index] {
            NodeType::Operation(op) => op.clone(),
            _ => return Err(DAGCircuitError::new_err("expected node")),
        };

        type WireMapsTuple = (HashMap<Qubit, Qubit>, HashMap<Clbit, Clbit>, Py<PyDict>);

        let build_wire_map = |wires: &Bound<PyList>| -> PyResult<WireMapsTuple> {
            let qargs_list = imports::BUILTIN_LIST
                .get_bound(py)
                .call1((bound_node.borrow().get_qargs(py),))?;
            let qargs_list = qargs_list.downcast::<PyList>().unwrap();
            let cargs_list = imports::BUILTIN_LIST
                .get_bound(py)
                .call1((bound_node.borrow().get_cargs(py),))?;
            let cargs_list = cargs_list.downcast::<PyList>().unwrap();
            let cargs_set = imports::BUILTIN_SET.get_bound(py).call1((cargs_list,))?;
            let cargs_set = cargs_set.downcast::<PySet>().unwrap();
            if self.may_have_additional_wires(py, &node) {
                let (add_cargs, _add_vars) = self.additional_wires(py, node.op.view())?;
                for wire in add_cargs.iter() {
                    let clbit = self.clbits.get(*wire).unwrap();
                    if !cargs_set.contains(clbit)? {
                        cargs_list.append(clbit)?;
                    }
                }
            }
            let qargs_len = qargs_list.len();
            let cargs_len = cargs_list.len();

            if qargs_len + cargs_len != wires.len() {
                return Err(DAGCircuitError::new_err(format!(
                    "bit mapping invalid: expected {}, got {}",
                    qargs_len + cargs_len,
                    wires.len()
                )));
            }
            let mut qubit_wire_map = HashMap::new();
            let mut clbit_wire_map = HashMap::new();
            let var_map = PyDict::new(py);
            for (index, wire) in wires.iter().enumerate() {
                if wire.downcast::<PyQubit>().is_ok() {
                    if index >= qargs_len {
                        unreachable!()
                    }
                    let input_qubit: Qubit = input_dag
                        .qubits
                        .find(&wire.extract::<ShareableQubit>()?)
                        .unwrap();
                    let self_qubit: Qubit = self
                        .qubits
                        .find(&qargs_list.get_item(index)?.extract::<ShareableQubit>()?)
                        .unwrap();
                    qubit_wire_map.insert(input_qubit, self_qubit);
                } else if wire.downcast::<PyClbit>().is_ok() {
                    if index < qargs_len {
                        unreachable!()
                    }
                    clbit_wire_map.insert(
                        input_dag
                            .clbits
                            .find(&wire.extract::<ShareableClbit>()?)
                            .unwrap(),
                        self.clbits
                            .find(
                                &cargs_list
                                    .get_item(index - qargs_len)?
                                    .extract::<ShareableClbit>()?,
                            )
                            .unwrap(),
                    );
                } else {
                    return Err(DAGCircuitError::new_err(
                        "`Var` nodes cannot be remapped during substitution",
                    ));
                }
            }
            Ok((qubit_wire_map, clbit_wire_map, var_map.unbind()))
        };

        let (qubit_wire_map, clbit_wire_map, var_map): (
            HashMap<Qubit, Qubit>,
            HashMap<Clbit, Clbit>,
            Py<PyDict>,
        ) = match wires {
            Some(wires) => match wires.downcast::<PyDict>() {
                Ok(bound_wires) => {
                    let mut qubit_wire_map = HashMap::new();
                    let mut clbit_wire_map = HashMap::new();
                    let var_map = PyDict::new(py);
                    for (source_wire, target_wire) in bound_wires.iter() {
                        if source_wire.downcast::<PyQubit>().is_ok() {
                            qubit_wire_map.insert(
                                input_dag
                                    .qubits
                                    .find(&source_wire.extract::<ShareableQubit>()?)
                                    .unwrap(),
                                self.qubits
                                    .find(&target_wire.extract::<ShareableQubit>()?)
                                    .unwrap(),
                            );
                        } else if source_wire.downcast::<PyClbit>().is_ok() {
                            clbit_wire_map.insert(
                                input_dag
                                    .clbits
                                    .find(&source_wire.extract::<ShareableClbit>()?)
                                    .unwrap(),
                                self.clbits
                                    .find(&target_wire.extract::<ShareableClbit>()?)
                                    .unwrap(),
                            );
                        } else {
                            var_map.set_item(source_wire, target_wire)?;
                        }
                    }
                    (qubit_wire_map, clbit_wire_map, var_map.unbind())
                }
                Err(_) => {
                    let wires: Bound<PyList> = match wires.downcast::<PyList>() {
                        Ok(bound_list) => bound_list.clone(),
                        // If someone passes a sequence instead of an exact list (tuple is
                        // occasionally used) cast that to a list and then use it.
                        Err(_) => {
                            let raw_wires = imports::BUILTIN_LIST.get_bound(py).call1((wires,))?;
                            raw_wires.extract()?
                        }
                    };
                    build_wire_map(&wires)?
                }
            },
            None => {
                let raw_wires = input_dag.get_wires(py);
                let binding = raw_wires?;
                let wires = binding.bind(py);
                build_wire_map(wires)?
            }
        };

        let var_iter = input_dag.iter_vars(py)?;
        let raw_set = imports::BUILTIN_SET.get_bound(py).call1((var_iter,))?;
        let input_dag_var_set: &Bound<PySet> = raw_set.downcast()?;

        let node_vars = if self.may_have_additional_wires(py, &node) {
            let (_additional_clbits, additional_vars) =
                self.additional_wires(py, node.op.view())?;
            let var_set = PySet::new(py, &additional_vars)?;
            if input_dag_var_set
                .call_method1(intern!(py, "difference"), (var_set.clone(),))?
                .is_truthy()?
            {
                return Err(DAGCircuitError::new_err(format!(
                    "Cannot replace a node with a DAG with more variables. Variables in node: {:?}. Variables in dag: {:?}",
                    var_set.str(), input_dag_var_set.str(),
                )));
            }
            var_set
        } else {
            PySet::empty(py)?
        };
        let bound_var_map = var_map.bind(py);
        for var in input_dag_var_set.iter() {
            bound_var_map.set_item(var.clone(), var)?;
        }

        for contracted_var in node_vars
            .call_method1(intern!(py, "difference"), (input_dag_var_set,))?
            .downcast::<PySet>()?
            .iter()
        {
            let pred = self
                .dag
                .edges_directed(node_index, Incoming)
                .find(|edge| {
                    if let Wire::Var(var) = edge.weight() {
                        contracted_var.eq(self.vars.get(*var)).unwrap()
                    } else {
                        false
                    }
                })
                .unwrap();
            let succ = self
                .dag
                .edges_directed(node_index, Outgoing)
                .find(|edge| {
                    if let Wire::Var(var) = edge.weight() {
                        contracted_var.eq(self.vars.get(*var)).unwrap()
                    } else {
                        false
                    }
                })
                .unwrap();
            self.dag.add_edge(
                pred.source(),
                succ.target(),
                Wire::Var(self.vars.find(&VarAsKey::new(&contracted_var)).unwrap()),
            );
        }

        let new_input_dag: Option<DAGCircuit> = None;
        // It doesn't make sense to try and propagate a condition from a control-flow op; a
        // replacement for the control-flow op should implement the operation completely.
        let node_map = self.substitute_node_with_subgraph(
            py,
            node_index,
            input_dag,
            &qubit_wire_map,
            &clbit_wire_map,
            &var_map,
        )?;
        self.global_phase = add_global_phase(py, &self.global_phase, &input_dag.global_phase)?;

        let wire_map_dict = PyDict::new(py);
        for (source, target) in clbit_wire_map.iter() {
            let source_bit = match new_input_dag {
                Some(ref in_dag) => in_dag.clbits.get(*source),
                None => input_dag.clbits.get(*source),
            };
            let target_bit = self.clbits.get(*target);
            wire_map_dict.set_item(source_bit, target_bit)?;
        }
        let bound_var_map = var_map.bind(py);

        // Note: creating this list to hold new registers created by the mapper is a temporary
        // measure until qiskit.expr is ported to Rust. It is necessary because we cannot easily
        // have Python call back to DAGCircuit::add_creg while we're currently borrowing
        // the DAGCircuit.
        let new_registers = PyList::empty(py);
        let add_new_register = new_registers.getattr("append")?.unbind();
        let flush_new_registers = |dag: &mut DAGCircuit| -> PyResult<()> {
            for reg in &new_registers {
                dag.add_creg(reg.extract()?)?;
            }
            new_registers.del_slice(0, new_registers.len())?;
            Ok(())
        };

        let variable_mapper = PyVariableMapper::new(
            py,
            self.cregs.registers().to_vec().into_bound_py_any(py)?,
            Some(wire_map_dict),
            Some(bound_var_map.clone()),
            Some(add_new_register),
        )?;

        for (old_node_index, new_node_index) in node_map.iter() {
            let old_node = match new_input_dag {
                Some(ref in_dag) => &in_dag.dag[*old_node_index],
                None => &input_dag.dag[*old_node_index],
            };
            if let NodeType::Operation(old_inst) = old_node {
                if let OperationRef::Instruction(old_op) = old_inst.op.view() {
                    if old_op.name() == "switch_case" {
                        let raw_target = old_op.instruction.getattr(py, "target")?;
                        let target = raw_target.bind(py);
                        let kwargs = PyDict::new(py);
                        kwargs.set_item(
                            "label",
                            old_inst
                                .label
                                .as_ref()
                                .map(|x| PyString::new(py, x.as_str())),
                        )?;

                        let new_op = imports::SWITCH_CASE_OP.get_bound(py).call(
                            (
                                variable_mapper.map_target(target)?,
                                old_op.instruction.call_method0(py, "cases_specifier")?,
                            ),
                            Some(&kwargs),
                        )?;
                        flush_new_registers(self)?;

                        if let NodeType::Operation(ref mut new_inst) =
                            &mut self.dag[*new_node_index]
                        {
                            new_inst.op = PyInstruction {
                                qubits: old_op.num_qubits(),
                                clbits: old_op.num_clbits(),
                                params: old_op.num_params(),
                                control_flow: old_op.control_flow(),
                                op_name: old_op.name().to_string(),
                                instruction: new_op.clone().unbind(),
                            }
                            .into();
                            #[cfg(feature = "cache_pygates")]
                            {
                                new_inst.py_op = new_op.unbind().into();
                            }
                        }
                    } else if old_inst.op.control_flow() {
                        if let Ok(condition) =
                            old_op.instruction.getattr(py, intern!(py, "condition"))
                        {
                            if old_inst.op.name() != "switch_case" {
                                let new_condition: Option<PyObject> = variable_mapper
                                    .map_condition(condition.bind(py), false)?
                                    .extract()?;
                                flush_new_registers(self)?;

                                if let NodeType::Operation(ref mut new_inst) =
                                    &mut self.dag[*new_node_index]
                                {
                                    #[cfg(feature = "cache_pygates")]
                                    {
                                        new_inst.py_op.take();
                                    }
                                    match new_inst.op.view() {
                                        OperationRef::Instruction(py_inst) => {
                                            py_inst.instruction.setattr(
                                                py,
                                                "condition",
                                                new_condition,
                                            )?;
                                        }
                                        _ => panic!("Instruction mismatch"),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let out_dict = PyDict::new(py);
        for (old_index, new_index) in node_map {
            out_dict.set_item(old_index.index(), self.get_node(py, new_index)?)?;
        }
        Ok(out_dict.unbind())
    }

    /// Replace a DAGOpNode with a single operation. qargs, cargs and
    /// conditions for the new operation will be inferred from the node to be
    /// replaced. The new operation will be checked to match the shape of the
    /// replaced operation.
    ///
    /// Args:
    ///     node (DAGOpNode): Node to be replaced
    ///     op (qiskit.circuit.Operation): The :class:`qiskit.circuit.Operation`
    ///         instance to be added to the DAG
    ///     inplace (bool): Optional, default False. If True, existing DAG node
    ///         will be modified to include op. Otherwise, a new DAG node will
    ///         be used.
    ///     propagate_condition (bool): DEPRECATED a legacy option that used
    ///         to control the behavior of handling control flow. It has no
    ///         effect anymore, left it for backwards compatibility. Will be
    ///         removed in Qiskit 3.0.
    ///
    ///
    /// Returns:
    ///     DAGOpNode: the new node containing the added operation.
    ///
    /// Raises:
    ///     DAGCircuitError: If replacement operation was incompatible with
    ///     location of target node.
    #[pyo3(name = "substitute_node", signature = (node, op, inplace=false, propagate_condition=None))]
    pub fn py_substitute_node(
        &mut self,
        py: Python,
        node: &Bound<PyAny>,
        op: &Bound<PyAny>,
        inplace: bool,
        propagate_condition: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if propagate_condition.is_some() {
            imports::WARNINGS_WARN.get_bound(py).call1((
                intern!(
                    py,
                    concat!(
                        "The propagate_condition argument is deprecated as of Qiskit 2.0.0.",
                        "It has no effect anymore and will be removed in Qiskit 3.0.0.",
                    )
                ),
                py.get_type::<PyDeprecationWarning>(),
                2,
            ))?;
        }
        let mut node: PyRefMut<DAGOpNode> = match node.downcast() {
            Ok(node) => node.borrow_mut(),
            Err(_) => return Err(DAGCircuitError::new_err("Only DAGOpNodes can be replaced.")),
        };
        let py = op.py();
        let node_index = node.as_ref().node.unwrap();
        self.substitute_node_with_py_op(py, node_index, op)?;
        if inplace {
            let new_weight = self.dag[node_index].unwrap_operation();
            let temp: OperationFromPython = op.extract()?;
            node.instruction.operation = temp.operation;
            node.instruction.params = new_weight.params_view().iter().cloned().collect();
            node.instruction.label.clone_from(&new_weight.label);
            #[cfg(feature = "cache_pygates")]
            {
                node.instruction.py_op = new_weight.py_op.clone();
            }
            node.into_py_any(py)
        } else {
            self.get_node(py, node_index)
        }
    }

    /// Decompose the circuit into sets of qubits with no gates connecting them.
    ///
    /// Args:
    ///     remove_idle_qubits (bool): Flag denoting whether to remove idle qubits from
    ///         the separated circuits. If ``False``, each output circuit will contain the
    ///         same number of qubits as ``self``.
    ///
    /// Returns:
    ///     List[DAGCircuit]: The circuits resulting from separating ``self`` into sets
    ///         of disconnected qubits
    ///
    /// Each :class:`~.DAGCircuit` instance returned by this method will contain the same number of
    /// clbits as ``self``. The global phase information in ``self`` will not be maintained
    /// in the subcircuits returned by this method.
    #[pyo3(signature = (remove_idle_qubits=false, *, vars_mode="alike"))]
    fn separable_circuits(
        &self,
        py: Python,
        remove_idle_qubits: bool,
        vars_mode: &str,
    ) -> PyResult<Py<PyList>> {
        let connected_components = rustworkx_core::connectivity::connected_components(&self.dag);
        let dags = PyList::empty(py);

        for comp_nodes in connected_components.iter() {
            let mut new_dag = self.copy_empty_like(py, vars_mode)?;
            new_dag.global_phase = Param::Float(0.);

            // A map from nodes in the this DAGCircuit to nodes in the new dag. Used for adding edges
            let mut node_map: HashMap<NodeIndex, NodeIndex> =
                HashMap::with_capacity(comp_nodes.len());

            // Adding the nodes to the new dag
            let mut non_classical = false;
            for node in comp_nodes {
                match self.dag.node_weight(*node) {
                    Some(w) => match w {
                        NodeType::ClbitIn(b) => {
                            let clbit_in = new_dag.clbit_io_map[b.index()][0];
                            node_map.insert(*node, clbit_in);
                        }
                        NodeType::ClbitOut(b) => {
                            let clbit_out = new_dag.clbit_io_map[b.index()][1];
                            node_map.insert(*node, clbit_out);
                        }
                        NodeType::QubitIn(q) => {
                            let qbit_in = new_dag.qubit_io_map[q.index()][0];
                            node_map.insert(*node, qbit_in);
                            non_classical = true;
                        }
                        NodeType::QubitOut(q) => {
                            let qbit_out = new_dag.qubit_io_map[q.index()][1];
                            node_map.insert(*node, qbit_out);
                            non_classical = true;
                        }
                        NodeType::VarIn(v) => {
                            let var_in = new_dag.var_io_map[v.index()][0];
                            node_map.insert(*node, var_in);
                        }
                        NodeType::VarOut(v) => {
                            let var_out = new_dag.var_io_map[v.index()][1];
                            node_map.insert(*node, var_out);
                        }
                        NodeType::Operation(pi) => {
                            let new_node = new_dag.dag.add_node(NodeType::Operation(pi.clone()));
                            new_dag.increment_op(pi.op.name());
                            node_map.insert(*node, new_node);
                            non_classical = true;
                        }
                    },
                    None => panic!("DAG node without payload!"),
                }
            }
            if !non_classical {
                continue;
            }
            let node_filter = |node: NodeIndex| -> bool { node_map.contains_key(&node) };

            let filtered = NodeFiltered(&self.dag, node_filter);

            // Remove the edges added by copy_empty_like (as idle wires) to avoid duplication
            new_dag.dag.clear_edges();
            for edge in filtered.edge_references() {
                let new_source = node_map[&edge.source()];
                let new_target = node_map[&edge.target()];
                new_dag
                    .dag
                    .add_edge(new_source, new_target, edge.weight().clone());
            }
            // Add back any edges for idle wires
            for (qubit, [in_node, out_node]) in new_dag
                .qubit_io_map
                .iter()
                .enumerate()
                .map(|(idx, indices)| (Qubit::new(idx), indices))
            {
                if new_dag.dag.edges(*in_node).next().is_none() {
                    new_dag
                        .dag
                        .add_edge(*in_node, *out_node, Wire::Qubit(qubit));
                }
            }
            for (clbit, [in_node, out_node]) in new_dag
                .clbit_io_map
                .iter()
                .enumerate()
                .map(|(idx, indices)| (Clbit::new(idx), indices))
            {
                if new_dag.dag.edges(*in_node).next().is_none() {
                    new_dag
                        .dag
                        .add_edge(*in_node, *out_node, Wire::Clbit(clbit));
                }
            }
            for (var_index, &[in_node, out_node]) in new_dag.var_io_map.iter().enumerate() {
                if new_dag.dag.edges(in_node).next().is_none() {
                    new_dag
                        .dag
                        .add_edge(in_node, out_node, Wire::Var(Var::new(var_index)));
                }
            }
            if remove_idle_qubits {
                let idle_wires: Vec<Bound<PyAny>> = new_dag
                    .idle_wires(py, None)?
                    .into_bound(py)
                    .map(|q| q.unwrap())
                    .filter(|e| e.downcast::<PyQubit>().is_ok())
                    .collect();

                let qubits = PyTuple::new(py, idle_wires)?;
                new_dag.remove_qubits(py, qubits.extract()?)?; // TODO: this does not really work, some issue with remove_qubits itself
            }

            dags.append(pyo3::Py::new(py, new_dag)?)?;
        }

        Ok(dags.unbind())
    }

    /// Swap connected nodes e.g. due to commutation.
    ///
    /// Args:
    ///     node1 (OpNode): predecessor node
    ///     node2 (OpNode): successor node
    ///
    /// Raises:
    ///     DAGCircuitError: if either node is not an OpNode or nodes are not connected
    fn swap_nodes(&mut self, node1: &DAGNode, node2: &DAGNode) -> PyResult<()> {
        let node1 = node1.node.unwrap();
        let node2 = node2.node.unwrap();

        // Check that both nodes correspond to operations
        if !matches!(self.dag.node_weight(node1).unwrap(), NodeType::Operation(_))
            || !matches!(self.dag.node_weight(node2).unwrap(), NodeType::Operation(_))
        {
            return Err(DAGCircuitError::new_err(
                "Nodes to swap are not both DAGOpNodes",
            ));
        }

        // Gather all wires connecting node1 and node2.
        // This functionality was extracted from rustworkx's 'get_edge_data'
        let wires: Vec<Wire> = self
            .dag
            .edges(node1)
            .filter(|edge| edge.target() == node2)
            .map(|edge| edge.weight().clone())
            .collect();

        if wires.is_empty() {
            return Err(DAGCircuitError::new_err(
                "Attempt to swap unconnected nodes",
            ));
        };

        // Closure that finds the first parent/child node connected to a reference node by given wire
        // and returns relevant edge information depending on the specified direction:
        //  - Incoming -> parent -> outputs (parent_edge_id, parent_source_node_id)
        //  - Outgoing -> child -> outputs (child_edge_id, child_target_node_id)
        // This functionality was inspired in rustworkx's 'find_predecessors_by_edge' and 'find_successors_by_edge'.
        let directed_edge_for_wire = |node: NodeIndex, direction: Direction, wire: &Wire| {
            for edge in self.dag.edges_directed(node, direction) {
                if wire == edge.weight() {
                    match direction {
                        Incoming => return Some((edge.id(), edge.source())),
                        Outgoing => return Some((edge.id(), edge.target())),
                    }
                }
            }
            None
        };

        // Vector that contains a tuple of (wire, edge_info, parent_info, child_info) per wire in wires
        let relevant_edges = wires
            .iter()
            .rev()
            .map(|wire| {
                (
                    wire,
                    directed_edge_for_wire(node1, Outgoing, wire).unwrap(),
                    directed_edge_for_wire(node1, Incoming, wire).unwrap(),
                    directed_edge_for_wire(node2, Outgoing, wire).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        // Iterate over relevant edges and modify self.dag
        for (wire, (node1_to_node2, _), (parent_to_node1, parent), (node2_to_child, child)) in
            relevant_edges
        {
            self.dag.remove_edge(parent_to_node1);
            self.dag.add_edge(parent, node2, wire.clone());
            self.dag.remove_edge(node1_to_node2);
            self.dag.add_edge(node2, node1, wire.clone());
            self.dag.remove_edge(node2_to_child);
            self.dag.add_edge(node1, child, wire.clone());
        }
        Ok(())
    }

    /// Get the node in the dag.
    ///
    /// Args:
    ///     node_id(int): Node identifier.
    ///
    /// Returns:
    ///     node: the node.
    fn node(&self, py: Python, node_id: isize) -> PyResult<Py<PyAny>> {
        self.get_node(py, NodeIndex::new(node_id as usize))
    }

    /// Iterator for node values.
    ///
    /// Yield:
    ///     node: the node.
    fn nodes(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result: PyResult<Vec<_>> = self
            .dag
            .node_references()
            .map(|(node, weight)| self.unpack_into(py, node, weight))
            .collect();
        let tup = PyTuple::new(py, result?)?;
        Ok(tup.into_any().try_iter().unwrap().unbind())
    }

    /// Iterator for edge values with source and destination node.
    ///
    /// This works by returning the outgoing edges from the specified nodes. If
    /// no nodes are specified all edges from the graph are returned.
    ///
    /// Args:
    ///     nodes(DAGOpNode, DAGInNode, or DAGOutNode|list(DAGOpNode, DAGInNode, or DAGOutNode):
    ///         Either a list of nodes or a single input node. If none is specified,
    ///         all edges are returned from the graph.
    ///
    /// Yield:
    ///     edge: the edge as a tuple with the format
    ///         (source node, destination node, edge wire)
    #[pyo3(signature=(nodes=None))]
    fn edges(&self, py: Python, nodes: Option<Bound<PyAny>>) -> PyResult<Py<PyIterator>> {
        let get_node_index = |obj: &Bound<PyAny>| -> PyResult<NodeIndex> {
            Ok(obj.downcast::<DAGNode>()?.borrow().node.unwrap())
        };

        let actual_nodes: Vec<_> = match nodes {
            None => self.dag.node_indices().collect(),
            Some(nodes) => {
                let mut out = Vec::new();
                if let Ok(node) = get_node_index(&nodes) {
                    out.push(node);
                } else {
                    for node in nodes.try_iter()? {
                        out.push(get_node_index(&node?)?);
                    }
                }
                out
            }
        };

        let mut edges = Vec::new();
        for node in actual_nodes {
            for edge in self.dag.edges_directed(node, Outgoing) {
                edges.push((
                    self.get_node(py, edge.source())?,
                    self.get_node(py, edge.target())?,
                    match edge.weight() {
                        Wire::Qubit(qubit) => {
                            self.qubits.get(*qubit).unwrap().into_bound_py_any(py)?
                        }
                        Wire::Clbit(clbit) => {
                            self.clbits.get(*clbit).unwrap().into_bound_py_any(py)?
                        }
                        Wire::Var(var) => self.vars.get(*var).unwrap().into_bound_py_any(py)?,
                    },
                ))
            }
        }

        Ok(PyTuple::new(py, edges)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Get the list of "op" nodes in the dag.
    ///
    /// Args:
    ///     op (Type): :class:`qiskit.circuit.Operation` subclass op nodes to
    ///         return. If None, return all op nodes.
    ///     include_directives (bool): include `barrier`, `snapshot` etc.
    ///
    /// Returns:
    ///     list[DAGOpNode]: the list of dag nodes containing the given op.
    #[pyo3(name= "op_nodes", signature=(op=None, include_directives=true))]
    fn py_op_nodes(
        &self,
        py: Python,
        op: Option<&Bound<PyType>>,
        include_directives: bool,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        let filter_is_nonstandard = if let Some(op) = op {
            op.getattr(intern!(py, "_standard_gate")).ok().is_none()
        } else {
            true
        };
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(packed) = &weight {
                if !include_directives && packed.op.directive() {
                    continue;
                }
                if let Some(op_type) = op {
                    // This middle catch is to avoid Python-space operation creation for most uses of
                    // `op`; we're usually just looking for control-flow ops, and standard gates
                    // aren't control-flow ops.
                    if !(filter_is_nonstandard && packed.op.try_standard_gate().is_some())
                        && packed.op.py_op_is_instance(op_type)?
                    {
                        nodes.push(self.unpack_into(py, node, weight)?);
                    }
                } else {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Get a list of "op" nodes in the dag that contain control flow instructions.
    ///
    /// Returns:
    ///     list[DAGOpNode]: The list of dag nodes containing control flow ops.
    fn control_flow_op_nodes(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        if !self.has_control_flow() {
            return Ok(vec![]);
        }
        self.dag
            .node_references()
            .filter_map(|(node_index, node_type)| match node_type {
                NodeType::Operation(ref node) => {
                    if node.op.control_flow() {
                        Some(self.unpack_into(py, node_index, node_type))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect()
    }

    /// Get the list of gate nodes in the dag.
    ///
    /// Returns:
    ///     list[DAGOpNode]: the list of DAGOpNodes that represent gates.
    fn gate_nodes(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.dag
            .node_references()
            .filter_map(|(node, weight)| match weight {
                NodeType::Operation(ref packed) => match packed.op.view() {
                    OperationRef::Gate(_) | OperationRef::StandardGate(_) => {
                        Some(self.unpack_into(py, node, weight))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// Get the set of "op" nodes with the given name.
    #[pyo3(signature = (*names))]
    fn named_nodes(&self, py: Python<'_>, names: &Bound<PyTuple>) -> PyResult<Vec<Py<PyAny>>> {
        let mut names_set: HashSet<String> = HashSet::with_capacity(names.len());
        for name_obj in names.iter() {
            names_set.insert(name_obj.extract::<String>()?);
        }
        let mut result: Vec<Py<PyAny>> = Vec::new();
        for (id, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if names_set.contains(packed.op.name()) {
                    result.push(self.unpack_into(py, id, weight)?);
                }
            }
        }
        Ok(result)
    }

    /// Get list of 2 qubit operations. Ignore directives like snapshot and barrier.
    #[pyo3(name = "two_qubit_ops")]
    pub fn py_two_qubit_ops(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.two_qubit_ops()
            .map(|(index, _)| self.unpack_into(py, index, &self.dag[index]))
            .collect()
    }

    /// Get list of 3+ qubit operations. Ignore directives like snapshot and barrier.
    fn multi_qubit_ops(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if packed.op.directive() {
                    continue;
                }

                let qargs = self.qargs_interner.get(packed.qubits);
                if qargs.len() >= 3 {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Returns the longest path in the dag as a list of DAGOpNodes, DAGInNodes, and DAGOutNodes.
    fn longest_path(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
        match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
            Some(res) => res.0,
            None => return Err(DAGCircuitError::new_err("not a DAG")),
        }
        .into_iter()
        .map(|node_index| self.get_node(py, node_index))
        .collect()
    }

    /// Returns iterator of the successors of a node as DAGOpNodes and DAGOutNodes."""
    fn successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Outgoing)
            .unique()
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new(py, successors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the predecessors of a node as DAGOpNodes and DAGInNodes.
    fn predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Incoming)
            .unique()
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of "op" successors of a node in the dag.
    fn op_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Outgoing)
            .unique()
            .filter_map(|i| match self.dag[i] {
                NodeType::Operation(_) => Some(self.get_node(py, i)),
                _ => None,
            })
            .collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns the iterator of "op" predecessors of a node in the dag.
    fn op_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Incoming)
            .unique()
            .filter_map(|i| match self.dag[i] {
                NodeType::Operation(_) => Some(self.get_node(py, i)),
                _ => None,
            })
            .collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Checks if a second node is in the successors of node.
    fn is_successor(&self, node: &DAGNode, node_succ: &DAGNode) -> bool {
        self.dag
            .find_edge(node.node.unwrap(), node_succ.node.unwrap())
            .is_some()
    }

    /// Checks if a second node is in the predecessors of node.
    fn is_predecessor(&self, node: &DAGNode, node_pred: &DAGNode) -> bool {
        self.dag
            .find_edge(node_pred.node.unwrap(), node.node.unwrap())
            .is_some()
    }

    /// Returns iterator of the predecessors of a node that are
    /// connected by a quantum edge as DAGOpNodes and DAGInNodes.
    #[pyo3(name = "quantum_predecessors")]
    fn py_quantum_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .quantum_predecessors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the successors of a node that are
    /// connected by a quantum edge as DAGOpNodes and DAGOutNodes.
    #[pyo3(name = "quantum_successors")]
    fn py_quantum_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successors: PyResult<Vec<_>> = self
            .quantum_successors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new(py, successors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the predecessors of a node that are
    /// connected by a classical edge as DAGOpNodes and DAGInNodes.
    fn classical_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let edges = self.dag.edges_directed(node.node.unwrap(), Incoming);
        let filtered = edges.filter_map(|e| match e.weight() {
            Wire::Qubit(_) => None,
            _ => Some(e.source()),
        });
        let predecessors: PyResult<Vec<_>> =
            filtered.unique().map(|i| self.get_node(py, i)).collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns set of the ancestors of a node as DAGOpNodes and DAGInNodes.
    #[pyo3(name = "ancestors")]
    fn py_ancestors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PySet>> {
        let ancestors: PyResult<Vec<PyObject>> = self
            .ancestors(node.node.unwrap())
            .map(|node| self.get_node(py, node))
            .collect();
        Ok(PySet::new(py, &ancestors?)?.unbind())
    }

    /// Returns set of the descendants of a node as DAGOpNodes and DAGOutNodes.
    #[pyo3(name = "descendants")]
    fn py_descendants(&self, py: Python, node: &DAGNode) -> PyResult<Py<PySet>> {
        let descendants: PyResult<Vec<PyObject>> = self
            .descendants(node.node.unwrap())
            .map(|node| self.get_node(py, node))
            .collect();
        Ok(PySet::new(py, &descendants?)?.unbind())
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its successors in  BFS order.
    #[pyo3(name = "bfs_successors")]
    fn py_bfs_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successor_index: PyResult<Vec<(PyObject, Vec<PyObject>)>> = self
            .bfs_successors(node.node.unwrap())
            .map(|(node, nodes)| -> PyResult<(PyObject, Vec<PyObject>)> {
                Ok((
                    self.get_node(py, node)?,
                    nodes
                        .iter()
                        .map(|sub_node| self.get_node(py, *sub_node))
                        .collect::<PyResult<Vec<_>>>()?,
                ))
            })
            .collect();
        Ok(PyList::new(py, successor_index?)?
            .into_any()
            .try_iter()?
            .unbind())
    }

    /// Returns iterator of the successors of a node that are
    /// connected by a classical edge as DAGOpNodes and DAGOutNodes.
    fn classical_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let edges = self.dag.edges_directed(node.node.unwrap(), Outgoing);
        let filtered = edges.filter_map(|e| match e.weight() {
            Wire::Qubit(_) => None,
            _ => Some(e.target()),
        });
        let predecessors: PyResult<Vec<_>> =
            filtered.unique().map(|i| self.get_node(py, i)).collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Remove an operation node n.
    ///
    /// Add edges from predecessors to successors.
    #[pyo3(name = "remove_op_node")]
    fn py_remove_op_node(&mut self, node: &Bound<PyAny>) -> PyResult<()> {
        let node: PyRef<DAGOpNode> = match node.downcast::<DAGOpNode>() {
            Ok(node) => node.borrow(),
            Err(_) => return Err(DAGCircuitError::new_err("Node not an DAGOpNode")),
        };
        let index = node.as_ref().node.unwrap();
        if self.dag.node_weight(index).is_none() {
            return Err(DAGCircuitError::new_err("Node not in DAG"));
        }
        self.remove_op_node(index);
        Ok(())
    }

    /// Remove all of the ancestor operation nodes of node.
    fn remove_ancestors_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let ancestors: Vec<_> = core_ancestors(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        for a in ancestors {
            self.dag.remove_node(a);
        }
        Ok(())
    }

    /// Remove all of the descendant operation nodes of node.
    fn remove_descendants_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let descendants: Vec<_> = core_descendants(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        for d in descendants {
            self.dag.remove_node(d);
        }
        Ok(())
    }

    /// Remove all of the non-ancestors operation nodes of node.
    fn remove_nonancestors_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let ancestors: HashSet<_> = core_ancestors(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        let non_ancestors: Vec<_> = self
            .dag
            .node_indices()
            .filter(|node_id| !ancestors.contains(node_id))
            .collect();
        for na in non_ancestors {
            self.dag.remove_node(na);
        }
        Ok(())
    }

    /// Remove all of the non-descendants operation nodes of node.
    fn remove_nondescendants_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let descendants: HashSet<_> = core_descendants(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        let non_descendants: Vec<_> = self
            .dag
            .node_indices()
            .filter(|node_id| !descendants.contains(node_id))
            .collect();
        for nd in non_descendants {
            self.dag.remove_node(nd);
        }
        Ok(())
    }

    /// Return a list of op nodes in the first layer of this dag.
    #[pyo3(name = "front_layer")]
    fn py_front_layer(&self, py: Python) -> PyResult<Py<PyList>> {
        let native_front_layer = self.front_layer();
        let front_layer_list = PyList::empty(py);
        for node in native_front_layer {
            front_layer_list.append(self.get_node(py, node)?)?;
        }
        Ok(front_layer_list.into())
    }

    /// Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.
    ///
    /// A layer is a circuit whose gates act on disjoint qubits, i.e.,
    /// a layer has depth 1. The total number of layers equals the
    /// circuit depth d. The layers are indexed from 0 to d-1 with the
    /// earliest layer at index 0. The layers are constructed using a
    /// greedy algorithm. Each returned layer is a dict containing
    /// {"graph": circuit graph, "partition": list of qubit lists}.
    ///
    /// The returned layer contains new (but semantically equivalent) DAGOpNodes, DAGInNodes,
    /// and DAGOutNodes. These are not the same as nodes of the original dag, but are equivalent
    /// via DAGNode.semantic_eq(node1, node2).
    ///
    /// TODO: Gates that use the same cbits will end up in different
    /// layers as this is currently implemented. This may not be
    /// the desired behavior.
    #[pyo3(signature = (*, vars_mode="captures"))]
    fn layers(&self, py: Python, vars_mode: &str) -> PyResult<Py<PyIterator>> {
        let layer_list = PyList::empty(py);
        let mut graph_layers = self.multigraph_layers();
        if graph_layers.next().is_none() {
            return Ok(PyIterator::from_object(&layer_list)?.into());
        }

        for graph_layer in graph_layers {
            let layer_dict = PyDict::new(py);
            // Sort to make sure they are in the order they were added to the original DAG
            // It has to be done by node_id as graph_layer is just a list of nodes
            // with no implied topology
            // Drawing tools rely on _node_id to infer order of node creation
            // so we need this to be preserved by layers()
            // Get the op nodes from the layer, removing any input and output nodes.
            let mut op_nodes: Vec<(&PackedInstruction, &NodeIndex)> = graph_layer
                .iter()
                .filter_map(|node| self.dag.node_weight(*node).map(|dag_node| (dag_node, node)))
                .filter_map(|(node, index)| match node {
                    NodeType::Operation(oper) => Some((oper, index)),
                    _ => None,
                })
                .collect();
            op_nodes.sort_by_key(|(_, node_index)| **node_index);

            if op_nodes.is_empty() {
                return Ok(PyIterator::from_object(&layer_list)?.into());
            }

            let mut new_layer = self.copy_empty_like(py, vars_mode)?;

            new_layer.extend(py, op_nodes.iter().map(|(inst, _)| (*inst).clone()))?;

            let support_iter = new_layer.op_nodes(false).map(|(_, instruction)| {
                PyTuple::new(
                    py,
                    new_layer
                        .qubits
                        .map_indices(new_layer.qargs_interner.get(instruction.qubits)),
                )
                .unwrap()
            });
            let support_list = PyList::empty(py);
            for support_qarg in support_iter {
                support_list.append(support_qarg)?;
            }
            layer_dict.set_item("graph", new_layer)?;
            layer_dict.set_item("partition", support_list)?;
            layer_list.append(layer_dict)?;
        }
        Ok(layer_list.into_any().try_iter()?.into())
    }

    /// Yield a layer for all gates of this circuit.
    ///
    /// A serial layer is a circuit with one gate. The layers have the
    /// same structure as in layers().
    #[pyo3(signature = (*, vars_mode="captures"))]
    fn serial_layers(&self, py: Python, vars_mode: &str) -> PyResult<Py<PyIterator>> {
        let layer_list = PyList::empty(py);
        for next_node in self.topological_op_nodes()? {
            let retrieved_node: &PackedInstruction = match self.dag.node_weight(next_node) {
                Some(NodeType::Operation(node)) => node,
                _ => unreachable!("A non-operation node was obtained from topological_op_nodes."),
            };
            let mut new_layer = self.copy_empty_like(py, vars_mode)?;

            // Save the support of the operation we add to the layer
            let support_list = PyList::empty(py);
            let qubits = PyTuple::new(
                py,
                self.qargs_interner
                    .get(retrieved_node.qubits)
                    .iter()
                    .map(|qubit| self.qubits.get(*qubit)),
            )?
            .unbind();
            new_layer.push_back(py, retrieved_node.clone())?;

            if !retrieved_node.op.directive() {
                support_list.append(qubits)?;
            }

            let layer_dict = [
                ("graph", new_layer.into_py_any(py)?),
                ("partition", support_list.into_any().unbind()),
            ]
            .into_py_dict(py)?;
            layer_list.append(layer_dict)?;
        }

        Ok(layer_list.into_any().try_iter()?.into())
    }

    /// Yield layers of the multigraph.
    #[pyo3(name = "multigraph_layers")]
    fn py_multigraph_layers(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let graph_layers = self.multigraph_layers().map(|layer| -> Vec<PyObject> {
            layer
                .into_iter()
                .filter_map(|index| self.get_node(py, index).ok())
                .collect()
        });
        let list: Bound<PyList> = PyList::new(py, graph_layers.collect::<Vec<Vec<PyObject>>>())?;
        Ok(PyIterator::from_object(&list)?.unbind())
    }

    /// Return a set of non-conditional runs of "op" nodes with the given names.
    ///
    /// For example, "... h q[0]; cx q[0],q[1]; cx q[0],q[1]; h q[1]; .."
    /// would produce the tuple of cx nodes as an element of the set returned
    /// from a call to collect_runs(["cx"]). If instead the cx nodes were
    /// "cx q[0],q[1]; cx q[1],q[0];", the method would still return the
    /// pair in a tuple. The namelist can contain names that are not
    /// in the circuit's basis.
    ///
    /// Nodes must have only one successor to continue the run.
    #[pyo3(name = "collect_runs")]
    fn py_collect_runs(&self, py: Python, namelist: &Bound<PyList>) -> PyResult<Py<PySet>> {
        let mut name_list_set = HashSet::with_capacity(namelist.len());
        for name in namelist.iter() {
            name_list_set.insert(name.extract::<String>()?);
        }

        let out_set = PySet::empty(py)?;

        for run in self.collect_runs(name_list_set) {
            let run_tuple = PyTuple::new(
                py,
                run.into_iter()
                    .map(|node_index| self.get_node(py, node_index).unwrap()),
            )?;
            out_set.add(run_tuple)?;
        }
        Ok(out_set.unbind())
    }

    /// Return a set of non-conditional runs of 1q "op" nodes.
    #[pyo3(name = "collect_1q_runs")]
    fn py_collect_1q_runs(&self, py: Python) -> PyResult<Py<PyList>> {
        match self.collect_1q_runs() {
            Some(runs) => {
                let runs_iter = runs.map(|node_indices| {
                    PyList::new(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unwrap()
                    .unbind()
                });
                let out_list = PyList::empty(py);
                for run_list in runs_iter {
                    out_list.append(run_list)?;
                }
                Ok(out_list.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Return a set of non-conditional runs of 2q "op" nodes.
    #[pyo3(name = "collect_2q_runs")]
    fn py_collect_2q_runs(&self, py: Python) -> PyResult<Py<PyList>> {
        match self.collect_2q_runs() {
            Some(runs) => {
                let runs_iter = runs.into_iter().map(|node_indices| {
                    PyList::new(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unwrap()
                    .unbind()
                });
                let out_list = PyList::empty(py);
                for run_list in runs_iter {
                    out_list.append(run_list)?;
                }
                Ok(out_list.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Iterator for nodes that affect a given wire.
    ///
    /// Args:
    ///     wire (Bit): the wire to be looked at.
    ///     only_ops (bool): True if only the ops nodes are wanted;
    ///                 otherwise, all nodes are returned.
    /// Yield:
    ///      Iterator: the successive nodes on the given wire
    ///
    /// Raises:
    ///     DAGCircuitError: if the given wire doesn't exist in the DAG
    #[pyo3(name = "nodes_on_wire", signature = (wire, only_ops=false))]
    fn py_nodes_on_wire(
        &self,
        py: Python,
        wire: &Bound<PyAny>,
        only_ops: bool,
    ) -> PyResult<Py<PyIterator>> {
        let wire = if wire.downcast::<PyQubit>().is_ok() {
            let wire = wire.extract::<ShareableQubit>()?;
            self.qubits.find(&wire).map(Wire::Qubit)
        } else if wire.downcast::<PyClbit>().is_ok() {
            let wire = wire.extract::<ShareableClbit>()?;
            self.clbits.find(&wire).map(Wire::Clbit)
        } else {
            let wire = VarAsKey::new(wire);
            self.vars.find(&wire).map(Wire::Var)
        }
        .ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given wire {:?} is not present in the circuit",
                wire
            ))
        })?;

        let nodes = self
            .nodes_on_wire(&wire, only_ops)
            .into_iter()
            .map(|n| self.get_node(py, n))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, nodes)?.into_any().try_iter()?.unbind())
    }

    /// Count the occurrences of operation names.
    ///
    /// Args:
    ///     recurse: if ``True`` (default), then recurse into control-flow operations.  In all
    ///         cases, this counts only the number of times the operation appears in any possible
    ///         block; both branches of if-elses are counted, and for- and while-loop blocks are
    ///         only counted once.
    ///
    /// Returns:
    ///     Mapping[str, int]: a mapping of operation names to the number of times it appears.
    #[pyo3(name = "count_ops", signature = (*, recurse=true))]
    fn py_count_ops(&self, py: Python, recurse: bool) -> PyResult<PyObject> {
        self.count_ops(py, recurse)?.into_py_any(py)
    }

    /// Count the occurrences of operation names on the longest path.
    ///
    /// Returns a dictionary of counts keyed on the operation name.
    fn count_ops_longest_path(&self) -> PyResult<HashMap<&str, usize>> {
        if self.dag.node_count() == 0 {
            return Ok(HashMap::new());
        }
        let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
        let longest_path =
            match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
                Some(res) => res.0,
                None => return Err(DAGCircuitError::new_err("not a DAG")),
            };
        // Allocate for worst case where all operations are unique
        let mut op_counts: HashMap<&str, usize> = HashMap::with_capacity(longest_path.len() - 2);
        for node_index in &longest_path[1..longest_path.len() - 1] {
            if let NodeType::Operation(ref packed) = self.dag[*node_index] {
                let name = packed.op.name();
                op_counts
                    .entry(name)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }
        Ok(op_counts)
    }

    /// Returns causal cone of a qubit.
    ///
    /// A qubit's causal cone is the set of qubits that can influence the output of that
    /// qubit through interactions, whether through multi-qubit gates or operations. Knowing
    /// the causal cone of a qubit can be useful when debugging faulty circuits, as it can
    /// help identify which wire(s) may be causing the problem.
    ///
    /// This method does not consider any classical data dependency in the ``DAGCircuit``,
    /// classical bit wires are ignored for the purposes of building the causal cone.
    ///
    /// Args:
    ///     qubit (~qiskit.circuit.Qubit): The output qubit for which we want to find the causal cone.
    ///
    /// Returns:
    ///     Set[~qiskit.circuit.Qubit]: The set of qubits whose interactions affect ``qubit``.
    fn quantum_causal_cone(&self, py: Python, qubit: &Bound<PyAny>) -> PyResult<Py<PySet>> {
        // Retrieve the output node from the qubit
        let qubit_nat: ShareableQubit = qubit.extract()?;
        let output_qubit = self.qubits.find(&qubit_nat).ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given qubit {:?} is not present in the circuit",
                qubit
            ))
        })?;
        let output_node_index = self
            .qubit_io_map
            .get(output_qubit.index())
            .map(|x| x[1])
            .ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "The given qubit {:?} is not present in qubit_output_map",
                    qubit
                ))
            })?;

        let mut qubits_in_cone: HashSet<&Qubit> = HashSet::from([&output_qubit]);
        let mut queue: VecDeque<NodeIndex> = self.quantum_predecessors(output_node_index).collect();

        // The processed_non_directive_nodes stores the set of processed non-directive nodes.
        // This is an optimization to avoid considering the same non-directive node multiple
        // times when reached from different paths.
        // The directive nodes (such as barriers or measures) are trickier since when processing
        // them we only add their predecessors that intersect qubits_in_cone. Hence, directive
        // nodes have to be considered multiple times.
        let mut processed_non_directive_nodes: HashSet<NodeIndex> = HashSet::new();

        while !queue.is_empty() {
            let cur_index = queue.pop_front().unwrap();

            if let NodeType::Operation(packed) = self.dag.node_weight(cur_index).unwrap() {
                if !packed.op.directive() {
                    // If the operation is not a directive (in particular not a barrier nor a measure),
                    // we do not do anything if it was already processed. Otherwise, we add its qubits
                    // to qubits_in_cone, and append its predecessors to queue.
                    if processed_non_directive_nodes.contains(&cur_index) {
                        continue;
                    }
                    qubits_in_cone.extend(self.qargs_interner.get(packed.qubits));
                    processed_non_directive_nodes.insert(cur_index);

                    for pred_index in self.quantum_predecessors(cur_index) {
                        if let NodeType::Operation(_pred_packed) =
                            self.dag.node_weight(pred_index).unwrap()
                        {
                            queue.push_back(pred_index);
                        }
                    }
                } else {
                    // Directives (such as barriers and measures) may be defined over all the qubits,
                    // yet not all of these qubits should be considered in the causal cone. So we
                    // only add those predecessors that have qubits in common with qubits_in_cone.
                    for pred_index in self.quantum_predecessors(cur_index) {
                        if let NodeType::Operation(pred_packed) =
                            self.dag.node_weight(pred_index).unwrap()
                        {
                            if self
                                .qargs_interner
                                .get(pred_packed.qubits)
                                .iter()
                                .any(|x| qubits_in_cone.contains(x))
                            {
                                queue.push_back(pred_index);
                            }
                        }
                    }
                }
            }
        }

        let qubits_in_cone_vec: Vec<_> = qubits_in_cone.iter().map(|&&qubit| qubit).collect();
        let elements = self.qubits.map_indices(&qubits_in_cone_vec);
        Ok(PySet::new(py, elements)?.unbind())
    }

    /// Return a dictionary of circuit properties.
    fn properties(&self, py: Python) -> PyResult<HashMap<&str, PyObject>> {
        Ok(HashMap::from_iter([
            ("size", self.size(py, false)?.into_py_any(py)?),
            ("depth", self.depth(py, false)?.into_py_any(py)?),
            ("width", self.width().into_py_any(py)?),
            ("qubits", self.num_qubits().into_py_any(py)?),
            ("bits", self.num_clbits().into_py_any(py)?),
            ("factors", self.num_tensor_factors().into_py_any(py)?),
            ("operations", self.py_count_ops(py, true)?),
        ]))
    }

    /// Draws the dag circuit.
    ///
    /// This function needs `Graphviz <https://www.graphviz.org/>`_ to be
    /// installed. Graphviz is not a python package and can't be pip installed
    /// (the ``graphviz`` package on PyPI is a Python interface library for
    /// Graphviz and does not actually install Graphviz). You can refer to
    /// `the Graphviz documentation <https://www.graphviz.org/download/>`__ on
    /// how to install it.
    ///
    /// Args:
    ///     scale (float): scaling factor
    ///     filename (str): file path to save image to (format inferred from name)
    ///     style (str):
    ///         'plain': B&W graph;
    ///         'color' (default): color input/output/op nodes
    ///
    /// Returns:
    ///     Ipython.display.Image: if in Jupyter notebook and not saving to file,
    ///     otherwise None.
    #[pyo3(signature=(scale=0.7, filename=None, style="color"))]
    fn draw<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        scale: f64,
        filename: Option<&str>,
        style: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = PyModule::import(py, "qiskit.visualization.dag_visualization")?;
        module.call_method1("dag_drawer", (slf, scale, filename, style))
    }

    #[pyo3(signature=(graph_attrs=None, node_attrs=None, edge_attrs=None))]
    fn _to_dot<'py>(
        &self,
        py: Python<'py>,
        graph_attrs: Option<BTreeMap<String, String>>,
        node_attrs: Option<PyObject>,
        edge_attrs: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyString>> {
        let mut buffer = Vec::<u8>::new();
        build_dot(py, self, &mut buffer, graph_attrs, node_attrs, edge_attrs)?;
        Ok(PyString::new(py, std::str::from_utf8(&buffer)?))
    }

    /// Add an input variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    fn add_input_var(&mut self, py: Python, var: &Bound<PyAny>) -> PyResult<()> {
        if !self.vars_by_type[DAGVarType::Capture as usize]
            .bind(py)
            .is_empty()
            || !self.captured_stretches.is_empty()
        {
            return Err(DAGCircuitError::new_err(
                "cannot add inputs to a circuit with captures",
            ));
        }
        self.add_var(py, var, DAGVarType::Input)?;
        Ok(())
    }

    /// Add a captured variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    fn add_captured_var(&mut self, py: Python, var: &Bound<PyAny>) -> PyResult<()> {
        if !self.vars_by_type[DAGVarType::Input as usize]
            .bind(py)
            .is_empty()
        {
            return Err(DAGCircuitError::new_err(
                "cannot add captures to a circuit with inputs",
            ));
        }
        self.add_var(py, var, DAGVarType::Capture)?;
        Ok(())
    }

    /// Add a captured stretch to the circuit.
    ///
    /// Args:
    ///     var: the stretch to add.
    fn add_captured_stretch(&mut self, py: Python, var: &Bound<PyAny>) -> PyResult<()> {
        if !self.vars_by_type[DAGVarType::Input as usize]
            .bind(py)
            .is_empty()
        {
            return Err(DAGCircuitError::new_err(
                "cannot add captures to a circuit with inputs",
            ));
        }
        let var_name: String = var.getattr("name")?.extract::<String>()?;
        if self.vars_info.contains_key(&var_name) {
            return Err(DAGCircuitError::new_err(
                "cannot add stretch as its name shadows an existing identifier",
            ));
        }
        if let Some(previous) = self.declared_stretches.get(&var_name) {
            if var.eq(previous)? {
                return Err(DAGCircuitError::new_err("already present in the circuit"));
            }
            return Err(DAGCircuitError::new_err(
                "cannot add stretch as its name shadows an existing identifier",
            ));
        }
        if let Some(previous) = self.captured_stretches.get(&var_name) {
            if var.eq(previous)? {
                return Err(DAGCircuitError::new_err("already present in the circuit"));
            }
            return Err(DAGCircuitError::new_err(
                "cannot add stretch as its name shadows an existing identifier",
            ));
        }

        self.captured_stretches
            .insert(var_name, var.clone().unbind());
        Ok(())
    }

    /// Add a declared local variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    fn add_declared_var(&mut self, py: Python, var: &Bound<PyAny>) -> PyResult<()> {
        self.add_var(py, var, DAGVarType::Declare)?;
        Ok(())
    }

    /// Add a declared stretch to the circuit.
    ///
    /// Args:
    ///     var: the stretch to add.
    fn add_declared_stretch(&mut self, var: &Bound<PyAny>) -> PyResult<()> {
        let var_name: String = var.getattr("name")?.extract::<String>()?;
        if self.vars_info.contains_key(&var_name) {
            return Err(DAGCircuitError::new_err(
                "cannot add stretch as its name shadows an existing identifier",
            ));
        }
        if let Some(previous) = self.declared_stretches.get(&var_name) {
            if var.eq(previous)? {
                return Err(DAGCircuitError::new_err("already present in the circuit"));
            }
            return Err(DAGCircuitError::new_err(
                "cannot add stretch as its name shadows an existing identifier",
            ));
        }
        if let Some(previous) = self.captured_stretches.get(&var_name) {
            if var.eq(previous)? {
                return Err(DAGCircuitError::new_err("already present in the circuit"));
            }
            return Err(DAGCircuitError::new_err(
                "cannot add stretch as its name shadows an existing identifier",
            ));
        }

        self.declared_stretches
            .insert(var_name, var.clone().unbind());
        Ok(())
    }

    /// Total number of classical variables tracked by the circuit.
    #[getter]
    fn num_vars(&self) -> usize {
        self.vars_info.len()
    }

    /// Number of input classical variables tracked by the circuit.
    #[getter]
    fn num_input_vars(&self, py: Python) -> usize {
        self.vars_by_type[DAGVarType::Input as usize].bind(py).len()
    }

    /// Number of captured classical variables tracked by the circuit.
    #[getter]
    fn num_captured_vars(&self, py: Python) -> usize {
        self.vars_by_type[DAGVarType::Capture as usize]
            .bind(py)
            .len()
    }

    /// Number of declared local classical variables tracked by the circuit.
    #[getter]
    fn num_declared_vars(&self, py: Python) -> usize {
        self.vars_by_type[DAGVarType::Declare as usize]
            .bind(py)
            .len()
    }

    // These three stretch getter methods are for testing purposes, they'll be public
    // APIs in Qiskit 2.1
    #[getter]
    fn _num_stretches(&self) -> usize {
        self._num_captured_stretches() + self._num_declared_stretches()
    }

    #[getter]
    fn _num_captured_stretches(&self) -> usize {
        self.captured_stretches.len()
    }

    #[getter]
    fn _num_declared_stretches(&self) -> usize {
        self.declared_stretches.len()
    }

    /// Is this realtime variable in the DAG?
    ///
    /// Args:
    ///     var: the variable or name to check.
    fn has_var(&self, var: &Bound<PyAny>) -> PyResult<bool> {
        match var.extract::<String>() {
            Ok(name) => Ok(self.vars_info.contains_key(&name)),
            Err(_) => {
                let raw_name = var.getattr("name")?;
                let var_name: String = raw_name.extract()?;
                match self.vars_info.get(&var_name) {
                    Some(var_in_dag) => Ok(var_in_dag.var.is(var)),
                    None => Ok(false),
                }
            }
        }
    }

    /// Is this stretch in the DAG?
    ///
    /// Args:
    ///     var: the stretch or name to check.
    fn has_stretch(&self, var: &Bound<PyAny>) -> PyResult<bool> {
        match var.extract::<String>() {
            Ok(name) => Ok(self.declared_stretches.contains_key(&name)
                || self.captured_stretches.contains_key(&name)),
            Err(_) => {
                let raw_name = var.getattr("name")?;
                let var_name: String = raw_name.extract()?;
                if let Some(stretch) = self.declared_stretches.get(&var_name) {
                    return Ok(stretch.is(var));
                }
                if let Some(stretch) = self.captured_stretches.get(&var_name) {
                    return Ok(stretch.is(var));
                }
                Ok(false)
            }
        }
    }

    /// Is this identifier in the DAG?
    ///
    /// Args:
    ///     var: the identifier or name to check.
    fn has_identifier(&self, var: &Bound<PyAny>) -> PyResult<bool> {
        Ok(self.has_var(var)? || self.has_stretch(var)?)
    }

    /// Iterable over the input classical variables tracked by the circuit.
    fn iter_input_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        Ok(self.vars_by_type[DAGVarType::Input as usize]
            .bind(py)
            .clone()
            .into_any()
            .try_iter()?
            .unbind())
    }

    /// Iterable over the captured classical variables tracked by the circuit.
    fn iter_captured_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        Ok(self.vars_by_type[DAGVarType::Capture as usize]
            .bind(py)
            .clone()
            .into_any()
            .try_iter()?
            .unbind())
    }

    /// Iterable over the captured stretches tracked by the circuit.
    fn iter_captured_stretches(&self, py: Python) -> PyResult<Py<PyIterator>> {
        Ok(PyList::new(py, self.captured_stretches.values())?
            .into_any()
            .try_iter()?
            .unbind())
    }

    /// Iterable over all captured identifiers tracked by the circuit.
    fn iter_captures(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let out_set = PySet::empty(py)?;
        for var in self.vars_by_type[DAGVarType::Capture as usize]
            .bind(py)
            .iter()
        {
            out_set.add(var)?;
        }
        for stretch in self.captured_stretches.values() {
            out_set.add(stretch)?;
        }
        Ok(out_set.into_any().try_iter()?.unbind())
    }

    /// Iterable over the declared classical variables tracked by the circuit.
    fn iter_declared_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        Ok(self.vars_by_type[DAGVarType::Declare as usize]
            .bind(py)
            .clone()
            .into_any()
            .try_iter()?
            .unbind())
    }

    /// Iterable over the declared stretches tracked by the circuit.
    fn iter_declared_stretches(&self, py: Python) -> PyResult<Py<PyIterator>> {
        Ok(PyList::new(py, self.declared_stretches.values())?
            .into_any()
            .try_iter()?
            .unbind())
    }

    /// Iterable over all the classical variables tracked by the circuit.
    fn iter_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let out_set = PySet::empty(py)?;
        for var_type_set in &self.vars_by_type {
            for var in var_type_set.bind(py).iter() {
                out_set.add(var)?;
            }
        }
        Ok(out_set.into_any().try_iter()?.unbind())
    }

    /// Iterable over all the stretches tracked by the circuit.
    fn iter_stretches(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let out_set = PySet::empty(py)?;
        for s in self.captured_stretches.values() {
            out_set.add(s)?;
        }
        for s in self.declared_stretches.values() {
            out_set.add(s)?;
        }
        Ok(out_set.into_any().try_iter()?.unbind())
    }

    fn _has_edge(&self, source: usize, target: usize) -> bool {
        self.dag
            .contains_edge(NodeIndex::new(source), NodeIndex::new(target))
    }

    fn _is_dag(&self) -> bool {
        rustworkx_core::petgraph::algo::toposort(&self.dag, None).is_ok()
    }

    fn _in_edges(&self, py: Python, node_index: usize) -> Vec<Py<PyTuple>> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Incoming)
            .map(|wire| {
                (
                    wire.source().index(),
                    wire.target().index(),
                    match wire.weight() {
                        Wire::Qubit(qubit) => {
                            self.qubits.get(*qubit).into_bound_py_any(py).unwrap()
                        }
                        Wire::Clbit(clbit) => {
                            self.clbits.get(*clbit).into_bound_py_any(py).unwrap()
                        }
                        Wire::Var(var) => self.vars.get(*var).into_bound_py_any(py).unwrap(),
                    },
                )
                    .into_pyobject(py)
                    .unwrap()
                    .unbind()
            })
            .collect()
    }

    fn _out_edges(&self, py: Python, node_index: usize) -> Vec<Py<PyTuple>> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .map(|wire| {
                (
                    wire.source().index(),
                    wire.target().index(),
                    match wire.weight() {
                        Wire::Qubit(qubit) => {
                            self.qubits.get(*qubit).into_bound_py_any(py).unwrap()
                        }
                        Wire::Clbit(clbit) => {
                            self.clbits.get(*clbit).into_bound_py_any(py).unwrap()
                        }
                        Wire::Var(var) => self.vars.get(*var).into_bound_py_any(py).unwrap(),
                    },
                )
                    .into_pyobject(py)
                    .unwrap()
                    .unbind()
            })
            .collect()
    }

    fn _in_wires(&self, py: Python, node_index: usize) -> Vec<PyObject> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Incoming)
            .map(|wire| match wire.weight() {
                Wire::Qubit(qubit) => self.qubits.get(*qubit).into_py_any(py).unwrap(),
                Wire::Clbit(clbit) => self.clbits.get(*clbit).into_py_any(py).unwrap(),
                Wire::Var(var) => self.vars.get(*var).into_py_any(py).unwrap(),
            })
            .collect()
    }

    fn _out_wires(&self, py: Python, node_index: usize) -> Vec<PyObject> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .map(|wire| match wire.weight() {
                Wire::Qubit(qubit) => self.qubits.get(*qubit).into_py_any(py).unwrap(),
                Wire::Clbit(clbit) => self.clbits.get(*clbit).into_py_any(py).unwrap(),
                Wire::Var(var) => self.vars.get(*var).into_py_any(py).unwrap(),
            })
            .collect()
    }

    fn _find_successors_by_edge(
        &self,
        py: Python,
        node_index: usize,
        edge_checker: &Bound<PyAny>,
    ) -> PyResult<Vec<PyObject>> {
        let mut result = Vec::new();
        for e in self
            .dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .unique_by(|e| e.id())
        {
            let weight = match e.weight() {
                Wire::Qubit(qubit) => self.qubits.get(*qubit).into_py_any(py)?,
                Wire::Clbit(clbit) => self.clbits.get(*clbit).into_py_any(py)?,
                Wire::Var(var) => self.vars.get(*var).into_py_any(py)?,
            };
            if edge_checker.call1((weight,))?.extract::<bool>()? {
                result.push(self.get_node(py, e.target())?);
            }
        }
        Ok(result)
    }

    fn _edges(&self, py: Python) -> PyResult<Vec<PyObject>> {
        self.dag
            .edge_indices()
            .map(|index| {
                let wire = self.dag.edge_weight(index).unwrap();
                match wire {
                    Wire::Qubit(qubit) => self.qubits.get(*qubit).into_py_any(py),
                    Wire::Clbit(clbit) => self.clbits.get(*clbit).into_py_any(py),
                    Wire::Var(var) => self.vars.get(*var).into_py_any(py),
                }
            })
            .collect()
    }
}

impl DAGCircuit {
    /// Returns an immutable view of the [QuantumRegister] instances in the circuit.
    #[inline(always)]
    pub fn qregs(&self) -> &[QuantumRegister] {
        self.qregs.registers()
    }

    /// Returns an immutable view of the [ClassicalRegister] instances in the circuit.
    #[inline(always)]
    pub fn cregs(&self) -> &[ClassicalRegister] {
        self.cregs.registers()
    }

    /// Returns an immutable view of the [QuantumRegister] data struct in the circuit.
    #[inline(always)]
    pub fn qregs_data(&self) -> &RegisterData<QuantumRegister> {
        &self.qregs
    }

    /// Returns an immutable view of the [ClassicalRegister] data struct in the circuit.
    #[inline(always)]
    pub fn cregs_data(&self) -> &RegisterData<ClassicalRegister> {
        &self.cregs
    }

    /// Returns an immutable view of the qubit locations of the [DAGCircuit]
    #[inline(always)]
    pub fn qubit_locations(&self) -> &BitLocator<ShareableQubit, QuantumRegister> {
        &self.qubit_locations
    }

    /// Returns an immutable view of the clbit locations of the [DAGCircuit]
    #[inline(always)]
    pub fn clbit_locations(&self) -> &BitLocator<ShareableClbit, ClassicalRegister> {
        &self.clbit_locations
    }

    /// Returns an immutable view of the qubit io map
    #[inline(always)]
    pub fn qubit_io_map(&self) -> &[[NodeIndex; 2]] {
        &self.qubit_io_map
    }

    /// Returns an immutable view of the inner StableGraph managed by the circuit.
    #[inline(always)]
    pub fn dag(&self) -> &StableDiGraph<NodeType, Wire> {
        &self.dag
    }

    /// Returns an immutable view of the Interner used for Qargs
    #[inline(always)]
    pub fn qargs_interner(&self) -> &Interner<[Qubit]> {
        &self.qargs_interner
    }

    /// Returns an immutable view of the Interner used for Cargs
    #[inline(always)]
    pub fn cargs_interner(&self) -> &Interner<[Clbit]> {
        &self.cargs_interner
    }

    /// Returns an immutable view of the Global Phase `Param` of the circuit
    #[inline(always)]
    pub fn global_phase(&self) -> &Param {
        &self.global_phase
    }

    /// Returns an immutable view of the Qubits registered in the circuit
    #[inline(always)]
    pub fn qubits(&self) -> &BitData<Qubit, ShareableQubit> {
        &self.qubits
    }

    /// Returns an immutable view of the Classical bits registered in the circuit
    #[inline(always)]
    pub fn clbits(&self) -> &BitData<Clbit, ShareableClbit> {
        &self.clbits
    }

    /// Returns an immutable view of the Variable wires registered in the circuit
    #[inline(always)]
    pub fn vars(&self) -> &BitData<Var, VarAsKey> {
        &self.vars
    }

    /// Merge the `qargs` in a different [Interner] into this DAG, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    ///
    /// The input [InternedMap] is cleared of its previous entries by this method, and then we
    /// re-use the allocation.
    pub fn merge_qargs_using(
        &mut self,
        other: &Interner<[Qubit]>,
        map_fn: impl FnMut(&Qubit) -> Option<Qubit>,
        map: &mut InternedMap<[Qubit]>,
    ) {
        // 4 is an arbitrary guess for the amount of stack space to allocate for mapping the
        // `qargs`, but it doesn't matter if it's too short because it'll safely spill to the heap.
        self.qargs_interner
            .merge_map_slice_using::<4>(other, map_fn, map);
    }

    /// Merge the `qargs` in a different [Interner] into this DAG, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    pub fn merge_qargs(
        &mut self,
        other: &Interner<[Qubit]>,
        map_fn: impl FnMut(&Qubit) -> Option<Qubit>,
    ) -> InternedMap<[Qubit]> {
        let mut out = InternedMap::new();
        self.merge_qargs_using(other, map_fn, &mut out);
        out
    }

    /// Merge the `cargs` in a different [Interner] into this DAG, remapping the clbits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    ///
    /// The input [InternedMap] is cleared of its previous entries by this method, and then we
    /// re-use the allocation.
    pub fn merge_cargs_using(
        &mut self,
        other: &Interner<[Clbit]>,
        map_fn: impl FnMut(&Clbit) -> Option<Clbit>,
        map: &mut InternedMap<[Clbit]>,
    ) {
        // 4 is an arbitrary guess for the amount of stack space to allocate for mapping the
        // `cargs`, but it doesn't matter if it's too short because it'll safely spill to the heap.
        self.cargs_interner
            .merge_map_slice_using::<4>(other, map_fn, map);
    }

    /// Merge the `cargs` in a different [Interner] into this DAG, remapping the clbits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    pub fn merge_cargs(
        &mut self,
        other: &Interner<[Clbit]>,
        map_fn: impl FnMut(&Clbit) -> Option<Clbit>,
    ) -> InternedMap<[Clbit]> {
        let mut out = InternedMap::new();
        self.merge_cargs_using(other, map_fn, &mut out);
        out
    }

    /// Return an iterator of gate runs with non-conditional op nodes of given names
    pub fn collect_runs(
        &self,
        namelist: HashSet<String>,
    ) -> impl Iterator<Item = Vec<NodeIndex>> + '_ {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(namelist.contains(inst.op.name())),
                _ => Ok(false),
            }
        };

        match rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn) {
            Some(iter) => iter.map(|result| result.unwrap()),
            None => panic!("invalid DAG: cycle(s) detected!"),
        }
    }

    /// Return a set of non-conditional runs of 1q "op" nodes.
    pub fn collect_1q_runs(&self) -> Option<impl Iterator<Item = Vec<NodeIndex>> + '_> {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(inst.op.num_qubits() == 1
                    && inst.op.num_clbits() == 0
                    && !inst.is_parameterized()
                    && (inst.op.try_standard_gate().is_some()
                        || inst.op.matrix(inst.params_view()).is_some())),
                _ => Ok(false),
            }
        };
        rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn)
            .map(|node_iter| node_iter.map(|x| x.unwrap()))
    }

    /// Return a set of non-conditional runs of 2q "op" nodes.
    pub fn collect_2q_runs(&self) -> Option<Vec<Vec<NodeIndex>>> {
        let filter_fn = move |node_index: NodeIndex| -> Result<Option<bool>, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => match inst.op.view() {
                    OperationRef::StandardGate(gate) => {
                        Ok(Some(gate.num_qubits() <= 2 && !inst.is_parameterized()))
                    }
                    OperationRef::Gate(gate) => {
                        Ok(Some(gate.num_qubits() <= 2 && !inst.is_parameterized()))
                    }
                    OperationRef::Unitary(gate) => Ok(Some(gate.num_qubits() <= 2)),
                    _ => Ok(Some(false)),
                },
                _ => Ok(None),
            }
        };

        let color_fn = move |edge_index: EdgeIndex| -> Result<Option<usize>, Infallible> {
            let wire = self.dag.edge_weight(edge_index).unwrap();
            match wire {
                Wire::Qubit(index) => Ok(Some(index.index())),
                _ => Ok(None),
            }
        };
        rustworkx_core::dag_algo::collect_bicolor_runs(&self.dag, filter_fn, color_fn).unwrap()
    }

    fn increment_op(&mut self, op: &str) {
        match self.op_names.get_mut(op) {
            Some(count) => {
                *count += 1;
            }
            None => {
                self.op_names.insert(op.to_string(), 1);
            }
        }
    }

    fn decrement_op(&mut self, op: &str) {
        match self.op_names.get_mut(op) {
            Some(count) => {
                if *count > 1 {
                    *count -= 1;
                } else {
                    self.op_names.swap_remove(op);
                }
            }
            None => panic!("Cannot decrement something not added!"),
        }
    }

    pub fn quantum_predecessors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.dag
            .edges_directed(node, Incoming)
            .filter_map(|e| match e.weight() {
                Wire::Qubit(_) => Some(e.source()),
                _ => None,
            })
            .unique()
    }

    pub fn quantum_successors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.dag
            .edges_directed(node, Outgoing)
            .filter_map(|e| match e.weight() {
                Wire::Qubit(_) => Some(e.target()),
                _ => None,
            })
            .unique()
    }

    /// Apply a [PackedInstruction] to the back of the circuit.
    ///
    /// The provided `instr` MUST be valid for this DAG, e.g. its
    /// bits, registers, vars, and interner IDs must be valid in
    /// this DAG.
    ///
    /// This is mostly used to apply operations from one DAG to
    /// another that was created from the first via
    /// [DAGCircuit::copy_empty_like].
    pub fn push_back(&mut self, py: Python, instr: PackedInstruction) -> PyResult<NodeIndex> {
        let op_name = instr.op.name();
        let (all_cbits, vars): (Vec<Clbit>, Option<Vec<PyObject>>) = {
            if self.may_have_additional_wires(py, &instr) {
                let mut clbits: HashSet<Clbit> =
                    HashSet::from_iter(self.cargs_interner.get(instr.clbits).iter().copied());
                let (additional_clbits, additional_vars) =
                    self.additional_wires(py, instr.op.view())?;
                for clbit in additional_clbits {
                    clbits.insert(clbit);
                }
                (clbits.into_iter().collect(), Some(additional_vars))
            } else {
                (self.cargs_interner.get(instr.clbits).to_vec(), None)
            }
        };

        self.increment_op(op_name);

        let qubits_id = instr.qubits;
        let new_node = self.dag.add_node(NodeType::Operation(instr));

        // Put the new node in-between the previously "last" nodes on each wire
        // and the output map.
        let output_nodes: HashSet<NodeIndex> = self
            .qargs_interner
            .get(qubits_id)
            .iter()
            .map(|q| self.qubit_io_map.get(q.index()).map(|x| x[1]).unwrap())
            .chain(
                all_cbits
                    .iter()
                    .map(|c| self.clbit_io_map.get(c.index()).map(|x| x[1]).unwrap()),
            )
            .chain(vars.iter().flatten().map(|v| {
                self.var_io_map
                    .get(self.vars.find(&v.bind(py).into()).unwrap().index())
                    .map(|x| x[1])
                    .unwrap()
            }))
            .collect();

        for output_node in output_nodes {
            let last_edges: Vec<_> = self
                .dag
                .edges_directed(output_node, Incoming)
                .map(|e| (e.source(), e.id(), e.weight().clone()))
                .collect();
            for (source, old_edge, weight) in last_edges.into_iter() {
                self.dag.add_edge(source, new_node, weight.clone());
                self.dag.add_edge(new_node, output_node, weight);
                self.dag.remove_edge(old_edge);
            }
        }

        Ok(new_node)
    }

    /// Apply a [PackedInstruction] to the front of the circuit.
    ///
    /// The provided `instr` MUST be valid for this DAG, e.g. its
    /// bits, registers, vars, and interner IDs must be valid in
    /// this DAG.
    ///
    /// This is mostly used to apply operations from one DAG to
    /// another that was created from the first via
    /// [DAGCircuit::copy_empty_like].
    fn push_front(&mut self, py: Python, inst: PackedInstruction) -> PyResult<NodeIndex> {
        let op_name = inst.op.name();
        let (all_cbits, vars): (Vec<Clbit>, Option<Vec<PyObject>>) = {
            if self.may_have_additional_wires(py, &inst) {
                let mut clbits: HashSet<Clbit> =
                    HashSet::from_iter(self.cargs_interner.get(inst.clbits).iter().copied());
                let (additional_clbits, additional_vars) =
                    self.additional_wires(py, inst.op.view())?;
                for clbit in additional_clbits {
                    clbits.insert(clbit);
                }
                (clbits.into_iter().collect(), Some(additional_vars))
            } else {
                (self.cargs_interner.get(inst.clbits).to_vec(), None)
            }
        };

        self.increment_op(op_name);

        let qubits_id = inst.qubits;
        let new_node = self.dag.add_node(NodeType::Operation(inst));

        // Put the new node in-between the input map and the previously
        // "first" nodes on each wire.
        let mut input_nodes: Vec<NodeIndex> = self
            .qargs_interner
            .get(qubits_id)
            .iter()
            .map(|q| self.qubit_io_map[q.index()][0])
            .chain(all_cbits.iter().map(|c| self.clbit_io_map[c.index()][0]))
            .collect();
        if let Some(vars) = vars {
            for var in vars {
                input_nodes.push(
                    self.var_io_map[self
                        .vars
                        .find(&VarAsKey::new(var.bind(py)))
                        .unwrap()
                        .index()][0],
                );
            }
        }

        for input_node in input_nodes {
            let first_edges: Vec<_> = self
                .dag
                .edges_directed(input_node, Outgoing)
                .map(|e| (e.target(), e.id(), e.weight().clone()))
                .collect();
            for (target, old_edge, weight) in first_edges.into_iter() {
                self.dag.add_edge(input_node, new_node, weight.clone());
                self.dag.add_edge(new_node, target, weight);
                self.dag.remove_edge(old_edge);
            }
        }

        Ok(new_node)
    }

    /// Apply a [PackedOperation] to the back of the circuit.
    pub fn apply_operation_back(
        &mut self,
        py: Python,
        op: PackedOperation,
        qargs: &[Qubit],
        cargs: &[Clbit],
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<String>,
        #[cfg(feature = "cache_pygates")] py_op: Option<PyObject>,
    ) -> PyResult<NodeIndex> {
        self.inner_apply_op(
            py,
            op,
            qargs,
            cargs,
            params,
            label,
            #[cfg(feature = "cache_pygates")]
            py_op,
            false,
        )
    }

    /// Apply a [PackedOperation] to the front of the circuit.
    pub fn apply_operation_front(
        &mut self,
        py: Python,
        op: PackedOperation,
        qargs: &[Qubit],
        cargs: &[Clbit],
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<String>,
        #[cfg(feature = "cache_pygates")] py_op: Option<PyObject>,
    ) -> PyResult<NodeIndex> {
        self.inner_apply_op(
            py,
            op,
            qargs,
            cargs,
            params,
            label,
            #[cfg(feature = "cache_pygates")]
            py_op,
            true,
        )
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn inner_apply_op(
        &mut self,
        py: Python,
        op: PackedOperation,
        qargs: &[Qubit],
        cargs: &[Clbit],
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<String>,
        #[cfg(feature = "cache_pygates")] py_op: Option<PyObject>,
        front: bool,
    ) -> PyResult<NodeIndex> {
        // Check that all qargs are within an acceptable range
        qargs.iter().try_for_each(|qarg| {
            if qarg.index() >= self.num_qubits() {
                return Err(PyValueError::new_err(format!(
                    "Qubit index {} is out of range. This DAGCircuit currently has only {} qubits.",
                    qarg.0,
                    self.num_qubits()
                )));
            }
            Ok(())
        })?;

        // Check that all cargs are within an acceptable range
        cargs.iter().try_for_each(|carg| {
            if carg.index() >= self.num_clbits() {
                return Err(PyValueError::new_err(format!(
                    "Clbit index {} is out of range. This DAGCircuit currently has only {} clbits.",
                    carg.0,
                    self.num_clbits()
                )));
            }
            Ok(())
        })?;

        #[cfg(feature = "cache_pygates")]
        let py_op = if let Some(py_op) = py_op {
            py_op.into()
        } else {
            OnceLock::new()
        };
        let packed_instruction = PackedInstruction {
            op,
            qubits: self.qargs_interner.insert(qargs),
            clbits: self.cargs_interner.insert(cargs),
            params: params.map(Box::new),
            label: label.map(Box::new),
            #[cfg(feature = "cache_pygates")]
            py_op,
        };

        if front {
            self.push_front(py, packed_instruction)
        } else {
            self.push_back(py, packed_instruction)
        }
    }

    fn sort_key(&self, node: NodeIndex) -> SortKeyType {
        match &self.dag[node] {
            NodeType::Operation(packed) => (
                self.qargs_interner.get(packed.qubits),
                self.cargs_interner.get(packed.clbits),
            ),
            NodeType::QubitIn(q) => (std::slice::from_ref(q), &[Clbit(u32::MAX)]),
            NodeType::QubitOut(_q) => (&[Qubit(u32::MAX)], &[Clbit(u32::MAX)]),
            NodeType::ClbitIn(c) => (&[Qubit(u32::MAX)], std::slice::from_ref(c)),
            NodeType::ClbitOut(_c) => (&[Qubit(u32::MAX)], &[Clbit(u32::MAX)]),
            _ => (&[], &[]),
        }
    }

    fn topological_nodes(&self) -> PyResult<impl Iterator<Item = NodeIndex>> {
        let key = |node: NodeIndex| -> Result<SortKeyType, Infallible> { Ok(self.sort_key(node)) };
        let nodes =
            rustworkx_core::dag_algo::lexicographical_topological_sort(&self.dag, key, false, None)
                .map_err(|e| match e {
                    rustworkx_core::dag_algo::TopologicalSortError::CycleOrBadInitialState => {
                        PyValueError::new_err(format!("{}", e))
                    }
                    rustworkx_core::dag_algo::TopologicalSortError::KeyError(_) => {
                        unreachable!()
                    }
                })?;
        Ok(nodes.into_iter())
    }

    pub fn topological_op_nodes(&self) -> PyResult<impl Iterator<Item = NodeIndex> + '_> {
        Ok(self.topological_nodes()?.filter(|node: &NodeIndex| {
            matches!(self.dag.node_weight(*node), Some(NodeType::Operation(_)))
        }))
    }

    fn topological_key_sort(
        &self,
        py: Python,
        key: &Bound<PyAny>,
    ) -> PyResult<impl Iterator<Item = NodeIndex>> {
        // This path (user provided key func) is not ideal, since we no longer
        // use a string key after moving to Rust, in favor of using a tuple
        // of the qargs and cargs interner IDs of the node.
        let key = |node: NodeIndex| -> PyResult<String> {
            let node = self.get_node(py, node)?;
            key.call1((node,))?.extract()
        };
        Ok(
            rustworkx_core::dag_algo::lexicographical_topological_sort(&self.dag, key, false, None)
                .map_err(|e| match e {
                    rustworkx_core::dag_algo::TopologicalSortError::CycleOrBadInitialState => {
                        PyValueError::new_err(format!("{}", e))
                    }
                    rustworkx_core::dag_algo::TopologicalSortError::KeyError(ref e) => {
                        e.clone_ref(py)
                    }
                })?
                .into_iter(),
        )
    }

    #[inline]
    fn has_control_flow(&self) -> bool {
        CONTROL_FLOW_OP_NAMES
            .iter()
            .any(|x| self.op_names.contains_key(&x.to_string()))
    }

    fn is_wire_idle(&self, wire: &Wire) -> PyResult<bool> {
        let (input_node, output_node) = match wire {
            Wire::Qubit(qubit) => (
                self.qubit_io_map[qubit.index()][0],
                self.qubit_io_map[qubit.index()][1],
            ),
            Wire::Clbit(clbit) => (
                self.clbit_io_map[clbit.index()][0],
                self.clbit_io_map[clbit.index()][1],
            ),
            Wire::Var(var) => (
                self.var_io_map[var.index()][0],
                self.var_io_map[var.index()][1],
            ),
        };

        let child = self
            .dag
            .neighbors_directed(input_node, Outgoing)
            .next()
            .ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "Invalid dagcircuit input node {:?} has no output",
                    input_node
                ))
            })?;

        Ok(child == output_node)
    }

    fn may_have_additional_wires(&self, py: Python, instr: &PackedInstruction) -> bool {
        let OperationRef::Instruction(inst) = instr.op.view() else {
            return false;
        };
        inst.control_flow()
            || inst
                .instruction
                .bind(py)
                .is_instance(imports::STORE_OP.get_bound(py))
                .unwrap()
    }

    fn additional_wires(
        &self,
        py: Python,
        op: OperationRef,
    ) -> PyResult<(Vec<Clbit>, Vec<PyObject>)> {
        let wires_from_expr = |node: &Bound<PyAny>| -> PyResult<(Vec<Clbit>, Vec<PyObject>)> {
            let mut clbits = Vec::new();
            let mut vars = Vec::new();
            for var in imports::ITER_VARS
                .get_bound(py)
                .call1((node,))?
                .try_iter()?
            {
                let var = var?;
                let var_var = var.getattr("var")?;
                if var_var.downcast::<PyClbit>().is_ok() {
                    let var_clbit: ShareableClbit = var_var.extract()?;
                    clbits.push(self.clbits.find(&var_clbit).unwrap());
                } else if var_var.is_instance_of::<PyClassicalRegister>() {
                    for bit in var_var.try_iter().unwrap() {
                        let clbit: ShareableClbit = bit?.extract()?;
                        clbits.push(self.clbits.find(&clbit).unwrap());
                    }
                } else {
                    vars.push(var.unbind());
                }
            }
            Ok((clbits, vars))
        };

        let mut clbits = Vec::new();
        let mut vars = Vec::new();

        if let OperationRef::Instruction(inst) = op {
            let op = inst.instruction.bind(py);
            if inst.control_flow() {
                // The `condition` field might not exist, for example if this a `for` loop, and
                // that's not an exceptional state for us.
                if let Ok(condition) = op.getattr(intern!(py, "condition")) {
                    if !condition.is_none() {
                        if condition.is_instance(imports::EXPR.get_bound(py)).unwrap() {
                            let (expr_clbits, expr_vars) = wires_from_expr(&condition)?;
                            for bit in expr_clbits {
                                clbits.push(bit);
                            }
                            for var in expr_vars {
                                vars.push(var);
                            }
                        } else {
                            for bit in self
                                .control_flow_module
                                .condition_resources(&condition)?
                                .clbits
                                .bind(py)
                            {
                                clbits.push(self.clbits.find(&bit.extract()?).unwrap());
                            }
                        }
                    }
                }

                for var in op.call_method0("iter_captured_vars")?.try_iter()? {
                    vars.push(var?.unbind())
                }
                if op.is_instance(imports::SWITCH_CASE_OP.get_bound(py))? {
                    let target = op.getattr(intern!(py, "target"))?;
                    if target.downcast::<PyClbit>().is_ok() {
                        let target_clbit: ShareableClbit = target.extract()?;
                        clbits.push(self.clbits.find(&target_clbit).unwrap());
                    } else if target.is_instance_of::<PyClassicalRegister>() {
                        for bit in target.try_iter()? {
                            let clbit: ShareableClbit = bit?.extract()?;
                            clbits.push(self.clbits.find(&clbit).unwrap());
                        }
                    } else {
                        let (expr_clbits, expr_vars) = wires_from_expr(&target)?;
                        for bit in expr_clbits {
                            clbits.push(bit);
                        }
                        for var in expr_vars {
                            vars.push(var);
                        }
                    }
                }
            } else if op.is_instance(imports::STORE_OP.get_bound(py))? {
                let (expr_clbits, expr_vars) = wires_from_expr(&op.getattr("lvalue")?)?;
                for bit in expr_clbits {
                    clbits.push(bit);
                }
                for var in expr_vars {
                    vars.push(var);
                }
                let (expr_clbits, expr_vars) = wires_from_expr(&op.getattr("rvalue")?)?;
                for bit in expr_clbits {
                    clbits.push(bit);
                }
                for var in expr_vars {
                    vars.push(var);
                }
            }
        }
        Ok((clbits, vars))
    }

    /// Add a qubit or bit to the circuit.
    ///
    /// Args:
    ///     wire: the wire to be added
    ///
    ///     This adds a pair of in and out nodes connected by an edge.
    ///
    /// Returns:
    ///     The input and output node indices of the added wire, respectively.
    ///
    /// Raises:
    ///     DAGCircuitError: if trying to add duplicate wire
    fn add_wire(&mut self, wire: Wire) -> PyResult<(NodeIndex, NodeIndex)> {
        let (in_node, out_node) = match wire {
            Wire::Qubit(qubit) => {
                if (qubit.index()) >= self.qubit_io_map.len() {
                    let input_node = self.dag.add_node(NodeType::QubitIn(qubit));
                    let output_node = self.dag.add_node(NodeType::QubitOut(qubit));
                    self.qubit_io_map.push([input_node, output_node]);
                    Ok((input_node, output_node))
                } else {
                    Err(DAGCircuitError::new_err("qubit wire already exists!"))
                }
            }
            Wire::Clbit(clbit) => {
                if (clbit.index()) >= self.clbit_io_map.len() {
                    let input_node = self.dag.add_node(NodeType::ClbitIn(clbit));
                    let output_node = self.dag.add_node(NodeType::ClbitOut(clbit));
                    self.clbit_io_map.push([input_node, output_node]);
                    Ok((input_node, output_node))
                } else {
                    Err(DAGCircuitError::new_err("classical wire already exists!"))
                }
            }
            Wire::Var(var) => {
                if var.index() >= self.var_io_map.len() {
                    let in_node = self.dag.add_node(NodeType::VarIn(var));
                    let out_node = self.dag.add_node(NodeType::VarOut(var));
                    self.var_io_map.push([in_node, out_node]);
                    Ok((in_node, out_node))
                } else {
                    return Err(DAGCircuitError::new_err("var wire already exists!"));
                }
            }
        }?;

        self.dag.add_edge(in_node, out_node, wire);
        Ok((in_node, out_node))
    }

    /// Get the nodes on the given wire.
    ///
    /// Note: result is empty if the wire is not in the DAG.
    pub fn nodes_on_wire(&self, wire: &Wire, only_ops: bool) -> Vec<NodeIndex> {
        let mut nodes = Vec::new();
        let mut current_node = match wire {
            Wire::Qubit(qubit) => self.qubit_io_map.get(qubit.index()).map(|x| x[0]),
            Wire::Clbit(clbit) => self.clbit_io_map.get(clbit.index()).map(|x| x[0]),
            Wire::Var(var) => self.var_io_map.get(var.index()).map(|x| x[0]),
        };

        while let Some(node) = current_node {
            if only_ops {
                let node_weight = self.dag.node_weight(node).unwrap();
                if let NodeType::Operation(_) = node_weight {
                    nodes.push(node);
                }
            } else {
                nodes.push(node);
            }

            let edges = self.dag.edges_directed(node, Outgoing);
            current_node = edges.into_iter().find_map(|edge| {
                if edge.weight() == wire {
                    Some(edge.target())
                } else {
                    None
                }
            });
        }
        nodes
    }

    fn remove_idle_wire(&mut self, wire: Wire) -> PyResult<()> {
        let [in_node, out_node] = match wire {
            Wire::Qubit(qubit) => self.qubit_io_map[qubit.index()],
            Wire::Clbit(clbit) => self.clbit_io_map[clbit.index()],
            Wire::Var(var) => self.var_io_map[var.index()],
        };
        self.dag.remove_node(in_node);
        self.dag.remove_node(out_node);
        Ok(())
    }

    fn add_qubit_unchecked(&mut self, bit: ShareableQubit) -> PyResult<Qubit> {
        let qubit = self.qubits.add(bit.clone(), false)?;
        self.qubit_locations
            .insert(bit, BitLocations::new((self.qubits.len() - 1) as u32, []));
        self.add_wire(Wire::Qubit(qubit))?;
        Ok(qubit)
    }

    fn add_clbit_unchecked(&mut self, bit: ShareableClbit) -> PyResult<Clbit> {
        let clbit = self.clbits.add(bit.clone(), false)?;
        self.clbit_locations
            .insert(bit, BitLocations::new((self.clbits.len() - 1) as u32, []));
        self.add_wire(Wire::Clbit(clbit))?;
        Ok(clbit)
    }

    pub fn get_node(&self, py: Python, node: NodeIndex) -> PyResult<Py<PyAny>> {
        self.unpack_into(py, node, self.dag.node_weight(node).unwrap())
    }

    /// Remove an operation node n.
    ///
    /// Add edges from predecessors to successors.
    ///
    /// # Returns
    ///
    /// The removed [PackedInstruction] is returned
    pub fn remove_op_node(&mut self, index: NodeIndex) -> PackedInstruction {
        let mut edge_list: Vec<(NodeIndex, NodeIndex, Wire)> = Vec::new();
        for (source, in_weight) in self
            .dag
            .edges_directed(index, Incoming)
            .map(|x| (x.source(), x.weight()))
        {
            for (target, out_weight) in self
                .dag
                .edges_directed(index, Outgoing)
                .map(|x| (x.target(), x.weight()))
            {
                if in_weight == out_weight {
                    edge_list.push((source, target, in_weight.clone()));
                }
            }
        }
        for (source, target, weight) in edge_list {
            self.dag.add_edge(source, target, weight);
        }

        match self.dag.remove_node(index) {
            Some(NodeType::Operation(packed)) => {
                let op_name = packed.op.name();
                self.decrement_op(op_name);
                packed
            }
            _ => panic!("Must be called with valid operation node!"),
        }
    }

    /// Returns an iterator of the ancestors indices of a node.
    pub fn ancestors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        core_ancestors(&self.dag, node).filter(move |next| next != &node)
    }

    /// Returns an iterator of the descendants of a node as DAGOpNodes and DAGOutNodes.
    pub fn descendants(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        core_descendants(&self.dag, node).filter(move |next| next != &node)
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its successors in  BFS order.
    pub fn bfs_successors(
        &self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, Vec<NodeIndex>)> + '_ {
        core_bfs_successors(&self.dag, node).filter(move |(_, others)| !others.is_empty())
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its predecessors in BFS order.
    pub fn bfs_predecessors(
        &self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, Vec<NodeIndex>)> + '_ {
        core_bfs_predecessors(&self.dag, node).filter(move |(_, others)| !others.is_empty())
    }

    fn pack_into(&mut self, py: Python, b: &Bound<PyAny>) -> Result<NodeType, PyErr> {
        Ok(if let Ok(in_node) = b.downcast::<DAGInNode>() {
            let in_node = in_node.borrow();
            let wire = in_node.wire.bind(py);
            if let Ok(qubit) = wire.extract::<ShareableQubit>() {
                NodeType::QubitIn(self.qubits.find(&qubit).unwrap())
            } else if let Ok(clbit) = wire.extract::<ShareableClbit>() {
                NodeType::ClbitIn(self.clbits.find(&clbit).unwrap())
            } else {
                let var = VarAsKey::new(wire);
                NodeType::VarIn(self.vars.find(&var).unwrap())
            }
        } else if let Ok(out_node) = b.downcast::<DAGOutNode>() {
            let out_node = out_node.borrow();
            let wire = out_node.wire.bind(py);
            if let Ok(qubit) = wire.extract::<ShareableQubit>() {
                NodeType::QubitOut(self.qubits.find(&qubit).unwrap())
            } else if let Ok(clbit) = wire.extract::<ShareableClbit>() {
                NodeType::ClbitOut(self.clbits.find(&clbit).unwrap())
            } else {
                let var = VarAsKey::new(wire);
                NodeType::VarOut(self.vars.find(&var).unwrap())
            }
        } else if let Ok(op_node) = b.downcast::<DAGOpNode>() {
            let op_node = op_node.borrow();
            let qubits = self.qargs_interner.insert_owned(
                self.qubits
                    .map_bits(
                        op_node
                            .instruction
                            .qubits
                            .extract::<Vec<ShareableQubit>>(py)?
                            .into_iter(),
                    )?
                    .collect(),
            );
            let clbits = self.cargs_interner.insert_owned(
                self.clbits
                    .map_bits(
                        op_node
                            .instruction
                            .clbits
                            .extract::<Vec<ShareableClbit>>(py)?
                            .into_iter(),
                    )?
                    .collect(),
            );
            let params = (!op_node.instruction.params.is_empty())
                .then(|| Box::new(op_node.instruction.params.clone()));
            let inst = PackedInstruction {
                op: op_node.instruction.operation.clone(),
                qubits,
                clbits,
                params,
                label: op_node.instruction.label.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: op_node.instruction.py_op.clone(),
            };
            NodeType::Operation(inst)
        } else {
            return Err(PyTypeError::new_err("Invalid type for DAGNode"));
        })
    }

    fn unpack_into(&self, py: Python, id: NodeIndex, weight: &NodeType) -> PyResult<Py<PyAny>> {
        let dag_node = match weight {
            NodeType::QubitIn(qubit) => Py::new(
                py,
                DAGInNode::new(id, self.qubits.get(*qubit).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::QubitOut(qubit) => Py::new(
                py,
                DAGOutNode::new(id, self.qubits.get(*qubit).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::ClbitIn(clbit) => Py::new(
                py,
                DAGInNode::new(id, self.clbits.get(*clbit).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::ClbitOut(clbit) => Py::new(
                py,
                DAGOutNode::new(id, self.clbits.get(*clbit).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::Operation(packed) => {
                let qubits = self.qargs_interner.get(packed.qubits);
                let clbits = self.cargs_interner.get(packed.clbits);
                Py::new(
                    py,
                    (
                        DAGOpNode {
                            instruction: CircuitInstruction {
                                operation: packed.op.clone(),
                                qubits: PyTuple::new(py, self.qubits.map_indices(qubits))?.unbind(),
                                clbits: PyTuple::new(py, self.clbits.map_indices(clbits))?.unbind(),
                                params: packed.params_view().iter().cloned().collect(),
                                label: packed.label.clone(),
                                #[cfg(feature = "cache_pygates")]
                                py_op: packed.py_op.clone(),
                            },
                        },
                        DAGNode { node: Some(id) },
                    ),
                )?
                .into_any()
            }
            NodeType::VarIn(var) => Py::new(
                py,
                DAGInNode::new(id, self.vars.get(*var).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::VarOut(var) => Py::new(
                py,
                DAGOutNode::new(id, self.vars.get(*var).unwrap().into_py_any(py)?),
            )?
            .into_any(),
        };
        Ok(dag_node)
    }

    /// An iterator of the DAG indices and corresponding `PackedInstruction` references for
    /// the `NodeType::Operation` variants stored in the DAG.
    ///
    /// See also [op_node_indices], which provides only the indices.
    pub fn op_nodes(
        &self,
        include_directives: bool,
    ) -> impl Iterator<Item = (NodeIndex, &PackedInstruction)> + '_ {
        self.dag
            .node_references()
            .filter_map(move |(node_index, node_type)| match node_type {
                NodeType::Operation(ref node) => {
                    (include_directives || !node.op.directive()).then_some((node_index, node))
                }
                _ => None,
            })
    }

    /// An iterator of the DAG indices corresponding to `NodeType::Operation` variants.
    ///
    /// See also [op_nodes], which also provides a reference to the contained `PackedInstruction`.
    pub fn op_node_indices(
        &self,
        include_directives: bool,
    ) -> impl Iterator<Item = NodeIndex> + '_ {
        self.op_nodes(include_directives).map(|(index, _)| index)
    }

    /// Return an iterator of 2 qubit operations. Ignore directives like snapshot and barrier.
    pub fn two_qubit_ops(&self) -> impl Iterator<Item = (NodeIndex, &PackedInstruction)> + '_ {
        self.op_nodes(false)
            .filter(|(_, instruction)| self.qargs_interner.get(instruction.qubits).len() == 2)
    }

    // Filter any nodes that don't match a given predicate function
    pub fn filter_op_nodes<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&PackedInstruction) -> bool,
    {
        let remove_indices = self
            .op_nodes(true)
            .filter_map(|(index, instruction)| (!predicate(instruction)).then_some(index))
            .collect::<Vec<_>>();
        for node in remove_indices {
            self.remove_op_node(node);
        }
    }

    /// Returns an iterator over a list layers of the `DAGCircuit``.
    pub fn multigraph_layers(&self) -> impl Iterator<Item = Vec<NodeIndex>> + '_ {
        let mut first_layer: Vec<_> = self.qubit_io_map.iter().map(|x| x[0]).collect();
        first_layer.extend(self.clbit_io_map.iter().map(|x| x[0]));
        first_layer.extend(self.var_io_map.iter().map(|x| x[0]));
        // A DAG is by definition acyclical, therefore unwrapping the layer should never fail.
        layers(&self.dag, first_layer).map(|layer| match layer {
            Ok(layer) => layer,
            Err(_) => unreachable!("Not a DAG."),
        })
    }

    /// Returns an iterator over the first layer of the `DAGCircuit``.
    pub fn front_layer(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        let mut graph_layers = self.multigraph_layers();
        graph_layers.next();
        graph_layers
            .next()
            .into_iter()
            .flatten()
            .filter(|node| matches!(self.dag.node_weight(*node).unwrap(), NodeType::Operation(_)))
    }

    fn substitute_node_with_subgraph(
        &mut self,
        py: Python,
        node: NodeIndex,
        other: &DAGCircuit,
        qubit_map: &HashMap<Qubit, Qubit>,
        clbit_map: &HashMap<Clbit, Clbit>,
        var_map: &Py<PyDict>,
    ) -> PyResult<IndexMap<NodeIndex, NodeIndex, RandomState>> {
        if self.dag.node_weight(node).is_none() {
            return Err(PyIndexError::new_err(format!(
                "Specified node {} is not in this graph",
                node.index()
            )));
        }

        // Add wire from pred to succ if no ops on mapped wire on ``other``
        for (in_dag_wire, self_wire) in qubit_map.iter() {
            let [input_node, out_node] = other.qubit_io_map[in_dag_wire.index()];
            if other.dag.find_edge(input_node, out_node).is_some() {
                let pred = self
                    .dag
                    .edges_directed(node, Incoming)
                    .find(|edge| {
                        if let Wire::Qubit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                let succ = self
                    .dag
                    .edges_directed(node, Outgoing)
                    .find(|edge| {
                        if let Wire::Qubit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                self.dag
                    .add_edge(pred.source(), succ.target(), Wire::Qubit(*self_wire));
            }
        }
        for (in_dag_wire, self_wire) in clbit_map.iter() {
            let [input_node, out_node] = other.clbit_io_map[in_dag_wire.index()];
            if other.dag.find_edge(input_node, out_node).is_some() {
                let pred = self
                    .dag
                    .edges_directed(node, Incoming)
                    .find(|edge| {
                        if let Wire::Clbit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                let succ = self
                    .dag
                    .edges_directed(node, Outgoing)
                    .find(|edge| {
                        if let Wire::Clbit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                self.dag
                    .add_edge(pred.source(), succ.target(), Wire::Clbit(*self_wire));
            }
        }

        let bound_var_map = var_map.bind(py);
        let node_filter = |node: NodeIndex| -> bool {
            match other.dag[node] {
                NodeType::Operation(_) => !other
                    .dag
                    .edges_directed(node, petgraph::Direction::Outgoing)
                    .any(|edge| match edge.weight() {
                        Wire::Qubit(qubit) => !qubit_map.contains_key(qubit),
                        Wire::Clbit(clbit) => !clbit_map.contains_key(clbit),
                        Wire::Var(var) => !bound_var_map
                            .contains(other.vars.get(*var).cloned())
                            .unwrap(),
                    }),
                _ => false,
            }
        };
        let reverse_qubit_map: HashMap<Qubit, Qubit> =
            qubit_map.iter().map(|(x, y)| (*y, *x)).collect();
        let reverse_clbit_map: HashMap<Clbit, Clbit> =
            clbit_map.iter().map(|(x, y)| (*y, *x)).collect();
        let reverse_var_map = PyDict::new(py);
        for (k, v) in bound_var_map.iter() {
            reverse_var_map.set_item(v, k)?;
        }
        // Copy nodes from other to self
        let mut out_map: IndexMap<NodeIndex, NodeIndex, RandomState> =
            IndexMap::with_capacity_and_hasher(other.dag.node_count(), RandomState::default());
        for old_index in other.dag.node_indices() {
            if !node_filter(old_index) {
                continue;
            }
            let mut new_node = other.dag[old_index].clone();
            if let NodeType::Operation(ref mut new_inst) = new_node {
                let new_qubit_indices: Vec<Qubit> = other
                    .qargs_interner
                    .get(new_inst.qubits)
                    .iter()
                    .map(|old_qubit| qubit_map[old_qubit])
                    .collect();
                let new_clbit_indices: Vec<Clbit> = other
                    .cargs_interner
                    .get(new_inst.clbits)
                    .iter()
                    .map(|old_clbit| clbit_map[old_clbit])
                    .collect();
                new_inst.qubits = self.qargs_interner.insert_owned(new_qubit_indices);
                new_inst.clbits = self.cargs_interner.insert_owned(new_clbit_indices);
                self.increment_op(new_inst.op.name());
            }
            let new_index = self.dag.add_node(new_node);
            out_map.insert(old_index, new_index);
        }
        // If no nodes are copied bail here since there is nothing left
        // to do.
        if out_map.is_empty() {
            match self.dag.remove_node(node) {
                Some(NodeType::Operation(packed)) => {
                    let op_name = packed.op.name();
                    self.decrement_op(op_name);
                }
                _ => unreachable!("Must be called with valid operation node!"),
            }
            // Return a new empty map to clear allocation from out_map
            return Ok(IndexMap::default());
        }
        // Copy edges from other to self
        for edge in other.dag.edge_references().filter(|edge| {
            out_map.contains_key(&edge.target()) && out_map.contains_key(&edge.source())
        }) {
            self.dag.add_edge(
                out_map[&edge.source()],
                out_map[&edge.target()],
                match edge.weight() {
                    Wire::Qubit(qubit) => Wire::Qubit(qubit_map[qubit]),
                    Wire::Clbit(clbit) => Wire::Clbit(clbit_map[clbit]),
                    Wire::Var(var) => Wire::Var(
                        self.vars
                            .find(
                                &bound_var_map
                                    .get_item(other.vars.get(*var))?
                                    .unwrap()
                                    .into(),
                            )
                            .unwrap(),
                    ),
                },
            );
        }
        // Add edges to/from node to nodes in other
        let edges: Vec<(NodeIndex, NodeIndex, Wire)> = self
            .dag
            .edges_directed(node, Incoming)
            .map(|x| (x.source(), x.target(), x.weight().clone()))
            .collect();
        for (source, _target, weight) in edges {
            let wire_input_id = match weight {
                Wire::Qubit(qubit) => other
                    .qubit_io_map
                    .get(reverse_qubit_map[&qubit].index())
                    .map(|x| x[0]),
                Wire::Clbit(clbit) => other
                    .clbit_io_map
                    .get(reverse_clbit_map[&clbit].index())
                    .map(|x| x[0]),
                Wire::Var(var) => {
                    let index = other
                        .vars
                        .find(
                            &reverse_var_map
                                .get_item(self.vars.get(var))?
                                .unwrap()
                                .into(),
                        )
                        .unwrap()
                        .index();
                    other.var_io_map.get(index).map(|x| x[0])
                }
            };
            let old_index =
                wire_input_id.and_then(|x| other.dag.neighbors_directed(x, Outgoing).next());
            let target_out = match old_index {
                Some(old_index) => match out_map.get(&old_index) {
                    Some(new_index) => *new_index,
                    None => {
                        // If the index isn't in the node map we've already added the edges as
                        // part of the idle wire handling at the top of this method so just
                        // move on.
                        continue;
                    }
                },
                None => continue,
            };
            self.dag.add_edge(source, target_out, weight);
        }
        let edges: Vec<(NodeIndex, NodeIndex, Wire)> = self
            .dag
            .edges_directed(node, Outgoing)
            .map(|x| (x.source(), x.target(), x.weight().clone()))
            .collect();
        for (_source, target, weight) in edges {
            let wire_output_id = match weight {
                Wire::Qubit(qubit) => other
                    .qubit_io_map
                    .get(reverse_qubit_map[&qubit].index())
                    .map(|x| x[1]),
                Wire::Clbit(clbit) => other
                    .clbit_io_map
                    .get(reverse_clbit_map[&clbit].index())
                    .map(|x| x[1]),
                Wire::Var(var) => {
                    let index = other
                        .vars
                        .find(
                            &reverse_var_map
                                .get_item(self.vars.get(var))?
                                .unwrap()
                                .into(),
                        )
                        .unwrap()
                        .index();
                    other.var_io_map.get(index).map(|x| x[1])
                }
            };
            let old_index =
                wire_output_id.and_then(|x| other.dag.neighbors_directed(x, Incoming).next());
            let source_out = match old_index {
                Some(old_index) => match out_map.get(&old_index) {
                    Some(new_index) => *new_index,
                    None => {
                        // If the index isn't in the node map we've already added the edges as
                        // part of the idle wire handling at the top of this method so just
                        // move on.
                        continue;
                    }
                },
                None => continue,
            };
            self.dag.add_edge(source_out, target, weight);
        }
        // Remove node
        if let NodeType::Operation(inst) = &self.dag[node] {
            self.decrement_op(inst.op.name().to_string().as_str());
        }
        self.dag.remove_node(node);
        Ok(out_map)
    }

    /// Retrieve a variable given its unique [Var] key within the DAG.
    ///
    /// The provided [Var] must be from this [DAGCircuit].
    pub fn get_var<'py>(&self, py: Python<'py>, var: Var) -> Option<Bound<'py, PyAny>> {
        self.vars.get(var).map(|v| v.into_bound_py_any(py).ok())?
    }

    fn add_var(&mut self, py: Python, var: &Bound<PyAny>, type_: DAGVarType) -> PyResult<Var> {
        // The setup of the initial graph structure between an "in" and an "out" node is the same as
        // the bit-related `_add_wire`, but this logically needs to do different bookkeeping around
        // tracking the properties
        if !var.getattr("standalone")?.extract::<bool>()? {
            return Err(DAGCircuitError::new_err(
                "cannot add variables that wrap `Clbit` or `ClassicalRegister` instances",
            ));
        }
        let var_name: String = var.getattr("name")?.extract::<String>()?;
        if let Some(previous) = self.vars_info.get(&var_name) {
            if var.eq(previous.var.clone_ref(py))? {
                return Err(DAGCircuitError::new_err("already present in the circuit"));
            }
            return Err(DAGCircuitError::new_err(
                "cannot add var as its name shadows an existing identifier",
            ));
        }
        if self.declared_stretches.contains_key(&var_name)
            || self.captured_stretches.contains_key(&var_name)
        {
            return Err(DAGCircuitError::new_err(
                "cannot add var as its name shadows an existing identifier",
            ));
        }

        let var_idx = self.vars.add(var.into(), true)?;
        let (in_index, out_index) = self.add_wire(Wire::Var(var_idx))?;
        self.vars_by_type[type_ as usize]
            .bind(py)
            .add(var.clone().unbind())?;
        self.vars_info.insert(
            var_name,
            DAGVarInfo {
                var: var.clone().unbind(),
                type_,
                in_node: in_index,
                out_node: out_index,
            },
        );
        Ok(var_idx)
    }

    fn check_op_addition(&self, py: Python, inst: &PackedInstruction) -> PyResult<()> {
        for b in self.qargs_interner.get(inst.qubits) {
            if self.qubit_io_map.len() - 1 < b.index() {
                return Err(DAGCircuitError::new_err(format!(
                    "qubit {:?} not found in output map",
                    self.qubits.get(*b).unwrap()
                )));
            }
        }

        for b in self.cargs_interner.get(inst.clbits) {
            if !self.clbit_io_map.len() - 1 < b.index() {
                return Err(DAGCircuitError::new_err(format!(
                    "clbit {:?} not found in output map",
                    self.clbits.get(*b).unwrap()
                )));
            }
        }

        if self.may_have_additional_wires(py, inst) {
            let (clbits, vars) = self.additional_wires(py, inst.op.view())?;
            for b in clbits {
                if !self.clbit_io_map.len() - 1 < b.index() {
                    return Err(DAGCircuitError::new_err(format!(
                        "clbit {:?} not found in output map",
                        self.clbits.get(b).unwrap()
                    )));
                }
            }
            for v in vars {
                let var_idx = self.vars.find(&v.bind(py).into()).unwrap();
                if !self.var_io_map.len() - 1 < var_idx.index() {
                    return Err(DAGCircuitError::new_err(format!(
                        "var {} not found in output map",
                        v
                    )));
                }
            }
        }
        Ok(())
    }

    /// Alternative constructor, builds a DAGCircuit with a fixed capacity.
    ///
    /// # Arguments:
    /// - `py`: Python GIL token
    /// - `num_qubits`: Number of qubits in the circuit
    /// - `num_clbits`: Number of classical bits in the circuit.
    /// - `num_vars`: (Optional) number of variables in the circuit.
    /// - `num_ops`: (Optional) number of operations in the circuit.
    /// - `num_edges`: (Optional) If known, number of edges in the circuit.
    pub fn with_capacity(
        py: Python,
        num_qubits: usize,
        num_clbits: usize,
        num_vars: Option<usize>,
        num_ops: Option<usize>,
        num_edges: Option<usize>,
    ) -> PyResult<Self> {
        let num_ops: usize = num_ops.unwrap_or_default();
        let num_vars = num_vars.unwrap_or_default();
        let num_edges = num_edges.unwrap_or(
            num_qubits +    // 1 edge between the input node and the output node or 1st op node.
            num_clbits +    // 1 edge between the input node and the output node or 1st op node.
            num_vars +      // 1 edge between the input node and the output node or 1st op node.
            num_ops, // In Average there will be 3 edges (2 qubits and 1 clbit, or 3 qubits) per op_node.
        );

        let num_nodes = num_qubits * 2 + // One input + One output node per qubit
            num_clbits * 2 +    // One input + One output node per clbit
            num_vars * 2 +  // One input + output node per variable
            num_ops;

        Ok(Self {
            name: None,
            metadata: Some(PyDict::new(py).unbind().into()),
            dag: StableDiGraph::with_capacity(num_nodes, num_edges),
            qregs: RegisterData::new(),
            cregs: RegisterData::new(),
            qargs_interner: Interner::with_capacity(num_qubits),
            cargs_interner: Interner::with_capacity(num_clbits),
            qubits: BitData::with_capacity(num_qubits),
            clbits: BitData::with_capacity(num_clbits),
            vars: BitData::with_capacity(num_vars),
            global_phase: Param::Float(0.),
            duration: None,
            unit: "dt".to_string(),
            qubit_locations: BitLocator::with_capacity(num_qubits),
            clbit_locations: BitLocator::with_capacity(num_clbits),
            qubit_io_map: Vec::with_capacity(num_qubits),
            clbit_io_map: Vec::with_capacity(num_clbits),
            var_io_map: Vec::with_capacity(num_vars),
            op_names: IndexMap::default(),
            control_flow_module: PyControlFlowModule::new(py)?,
            vars_info: HashMap::with_capacity(num_vars),
            vars_by_type: [
                PySet::empty(py)?.unbind(),
                PySet::empty(py)?.unbind(),
                PySet::empty(py)?.unbind(),
            ],
            captured_stretches: IndexMap::default(),
            declared_stretches: IndexMap::default(),
        })
    }

    /// Get qargs from an intern index
    pub fn get_qargs(&self, index: Interned<[Qubit]>) -> &[Qubit] {
        self.qargs_interner.get(index)
    }

    /// Get cargs from an intern index
    pub fn get_cargs(&self, index: Interned<[Clbit]>) -> &[Clbit] {
        self.cargs_interner.get(index)
    }

    /// Insert a new 1q standard gate on incoming qubit
    pub fn insert_1q_on_incoming_qubit(
        &mut self,
        new_gate: (StandardGate, &[f64]),
        old_index: NodeIndex,
    ) {
        self.increment_op(new_gate.0.name());
        let old_node = &self.dag[old_index];
        let inst = if let NodeType::Operation(old_node) = old_node {
            PackedInstruction {
                op: new_gate.0.into(),
                qubits: old_node.qubits,
                clbits: old_node.clbits,
                params: (!new_gate.1.is_empty())
                    .then(|| Box::new(new_gate.1.iter().map(|x| Param::Float(*x)).collect())),
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            }
        } else {
            panic!("This method only works if provided index is an op node");
        };
        let new_index = self.dag.add_node(NodeType::Operation(inst));
        let (parent_index, edge_index, weight) = self
            .dag
            .edges_directed(old_index, Incoming)
            .map(|edge| (edge.source(), edge.id(), edge.weight().clone()))
            .next()
            .unwrap();
        self.dag.add_edge(parent_index, new_index, weight.clone());
        self.dag.add_edge(new_index, old_index, weight);
        self.dag.remove_edge(edge_index);
    }

    /// Remove a sequence of 1 qubit nodes from the dag
    /// This must only be called if all the nodes operate
    /// on a single qubit with no other wires in or out of any nodes
    pub fn remove_1q_sequence(&mut self, sequence: &[NodeIndex]) {
        let (parent_index, weight) = self
            .dag
            .edges_directed(*sequence.first().unwrap(), Incoming)
            .map(|edge| (edge.source(), edge.weight().clone()))
            .next()
            .unwrap();
        let child_index = self
            .dag
            .edges_directed(*sequence.last().unwrap(), Outgoing)
            .map(|edge| edge.target())
            .next()
            .unwrap();
        self.dag.add_edge(parent_index, child_index, weight);
        for node in sequence {
            match self.dag.remove_node(*node) {
                Some(NodeType::Operation(packed)) => {
                    let op_name = packed.op.name();
                    self.decrement_op(op_name);
                }
                _ => panic!("Must be called with valid operation node!"),
            }
        }
    }

    /// Replace a node with individual operations from a provided callback
    /// function on each qubit of that node.
    #[allow(unused_variables)]
    pub fn replace_node_with_1q_ops<F>(
        &mut self,
        py: Python, // Unused if cache_pygates isn't enabled
        node: NodeIndex,
        insert: F,
    ) -> PyResult<()>
    where
        F: Fn(&Wire) -> (PackedOperation, SmallVec<[Param; 3]>),
    {
        let mut edge_list: Vec<(NodeIndex, NodeIndex, Wire)> = Vec::with_capacity(2);
        for (source, in_weight) in self
            .dag
            .edges_directed(node, Incoming)
            .map(|x| (x.source(), x.weight()))
        {
            for (target, out_weight) in self
                .dag
                .edges_directed(node, Outgoing)
                .map(|x| (x.target(), x.weight()))
            {
                if in_weight == out_weight {
                    edge_list.push((source, target, in_weight.clone()));
                }
            }
        }
        for (source, target, weight) in edge_list {
            let (new_op, params) = insert(&weight);
            self.increment_op(new_op.name());
            let qubits = if let Wire::Qubit(qubit) = weight {
                vec![qubit]
            } else {
                panic!("This method only works if the gate being replaced has no classical incident wires")
            };
            #[cfg(feature = "cache_pygates")]
            let py_op = match new_op.view() {
                OperationRef::StandardGate(_)
                | OperationRef::StandardInstruction(_)
                | OperationRef::Unitary(_) => OnceLock::new(),
                OperationRef::Gate(gate) => OnceLock::from(gate.gate.clone_ref(py)),
                OperationRef::Instruction(instruction) => {
                    OnceLock::from(instruction.instruction.clone_ref(py))
                }
                OperationRef::Operation(op) => OnceLock::from(op.operation.clone_ref(py)),
            };
            let inst = PackedInstruction {
                op: new_op,
                qubits: self.qargs_interner.insert_owned(qubits),
                clbits: self.cargs_interner.get_default(),
                params: (!params.is_empty()).then(|| Box::new(params)),
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op,
            };
            let new_index = self.dag.add_node(NodeType::Operation(inst));
            self.dag.add_edge(source, new_index, weight.clone());
            self.dag.add_edge(new_index, target, weight);
        }

        match self.dag.remove_node(node) {
            Some(NodeType::Operation(packed)) => {
                let op_name = packed.op.name();
                self.decrement_op(op_name);
            }
            _ => panic!("Must be called with valid operation node"),
        }
        Ok(())
    }

    pub fn add_global_phase(&mut self, py: Python, value: &Param) -> PyResult<()> {
        match value {
            Param::Obj(_) => {
                return Err(PyTypeError::new_err(
                    "Invalid parameter type, only float and parameter expression are supported",
                ))
            }
            _ => self.set_global_phase(add_global_phase(py, &self.global_phase, value)?)?,
        }
        Ok(())
    }

    /// Return the op name counts in the circuit
    ///
    /// Args:
    ///     py: The python token necessary for control flow recursion
    ///     recurse: Whether to recurse into control flow ops or not
    pub fn count_ops(
        &self,
        py: Python,
        recurse: bool,
    ) -> PyResult<IndexMap<String, usize, RandomState>> {
        if !recurse || !self.has_control_flow() {
            Ok(self.op_names.clone())
        } else {
            fn inner(
                py: Python,
                dag: &DAGCircuit,
                counts: &mut IndexMap<String, usize, RandomState>,
            ) -> PyResult<()> {
                for (key, value) in dag.op_names.iter() {
                    counts
                        .entry(key.clone())
                        .and_modify(|count| *count += value)
                        .or_insert(*value);
                }
                let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py);
                for node in dag.dag.node_weights() {
                    let NodeType::Operation(node) = node else {
                        continue;
                    };
                    if !node.op.control_flow() {
                        continue;
                    }
                    let OperationRef::Instruction(inst) = node.op.view() else {
                        panic!("control flow op must be an instruction")
                    };
                    let blocks = inst.instruction.bind(py).getattr("blocks")?;
                    for block in blocks.try_iter()? {
                        let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block?,))?.extract()?;
                        inner(py, inner_dag, counts)?;
                    }
                }
                Ok(())
            }
            let mut counts =
                IndexMap::with_capacity_and_hasher(self.op_names.len(), RandomState::default());
            inner(py, self, &mut counts)?;
            Ok(counts)
        }
    }

    /// Get an immutable reference to the op counts for this DAGCircuit
    ///
    /// This differs from count_ops() in that it doesn't handle control flow recursion at all
    /// and it returns a reference instead of an owned copy. If you don't need to work with
    /// control flow or ownership of the counts this is a more efficient alternative to
    /// `DAGCircuit::count_ops(py, false)`
    pub fn get_op_counts(&self) -> &IndexMap<String, usize, RandomState> {
        &self.op_names
    }

    /// Extends the DAG with valid instances of [PackedInstruction].
    pub fn extend<I>(&mut self, py: Python, iter: I) -> PyResult<Vec<NodeIndex>>
    where
        I: IntoIterator<Item = PackedInstruction>,
    {
        self.try_extend(
            py,
            iter.into_iter()
                .map(|inst| -> Result<PackedInstruction, Infallible> { Ok(inst) }),
        )
    }

    /// Extends the DAG with valid instances of [PackedInstruction], where the iterator produces the
    /// results in a fallible manner.
    pub fn try_extend<I, E>(&mut self, py: Python, iter: I) -> PyResult<Vec<NodeIndex>>
    where
        I: IntoIterator<Item = Result<PackedInstruction, E>>,
        PyErr: From<E>,
    {
        // Create HashSets to keep track of each bit/var's last node
        let mut qubit_last_nodes: HashMap<Qubit, NodeIndex> = HashMap::default();
        let mut clbit_last_nodes: HashMap<Clbit, NodeIndex> = HashMap::default();
        // TODO: Refactor once Vars are in rust
        // Dict [ Var: (int, VarWeight)]
        let vars_last_nodes: Bound<PyDict> = PyDict::new(py);

        // Consume into iterator to obtain size hint
        let iter = iter.into_iter();
        // Store new nodes to return
        let mut new_nodes = Vec::with_capacity(iter.size_hint().1.unwrap_or_default());
        for instr in iter {
            let instr = instr?;
            let op_name = instr.op.name();
            let (all_cbits, vars): (Vec<Clbit>, Option<Vec<PyObject>>) = {
                if self.may_have_additional_wires(py, &instr) {
                    let mut clbits: HashSet<Clbit> =
                        HashSet::from_iter(self.cargs_interner.get(instr.clbits).iter().copied());
                    let (additional_clbits, additional_vars) =
                        self.additional_wires(py, instr.op.view())?;
                    for clbit in additional_clbits {
                        clbits.insert(clbit);
                    }
                    (clbits.into_iter().collect(), Some(additional_vars))
                } else {
                    (self.cargs_interner.get(instr.clbits).to_vec(), None)
                }
            };

            // Increment the operation count
            self.increment_op(op_name);

            // Get the correct qubit indices
            let qubits_id = instr.qubits;

            // Insert op-node to graph.
            let new_node = self.dag.add_node(NodeType::Operation(instr));
            new_nodes.push(new_node);

            // Check all the qubits in this instruction.
            for qubit in self.qargs_interner.get(qubits_id) {
                // Retrieve each qubit's last node
                let qubit_last_node = *qubit_last_nodes.entry(*qubit).or_insert_with(|| {
                    // If the qubit is not in the last nodes collection, the edge between the output node and its predecessor.
                    // Then, store the predecessor's NodeIndex in the last nodes collection.
                    let output_node = self.qubit_io_map[qubit.index()][1];
                    let (edge_id, predecessor_node) = self
                        .dag
                        .edges_directed(output_node, Incoming)
                        .next()
                        .map(|edge| (edge.id(), edge.source()))
                        .unwrap();
                    self.dag.remove_edge(edge_id);
                    predecessor_node
                });
                qubit_last_nodes
                    .entry(*qubit)
                    .and_modify(|val| *val = new_node);
                self.dag
                    .add_edge(qubit_last_node, new_node, Wire::Qubit(*qubit));
            }

            // Check all the clbits in this instruction.
            for clbit in all_cbits {
                let clbit_last_node = *clbit_last_nodes.entry(clbit).or_insert_with(|| {
                    // If the qubit is not in the last nodes collection, the edge between the output node and its predecessor.
                    // Then, store the predecessor's NodeIndex in the last nodes collection.
                    let output_node = self.clbit_io_map[clbit.index()][1];
                    let (edge_id, predecessor_node) = self
                        .dag
                        .edges_directed(output_node, Incoming)
                        .next()
                        .map(|edge| (edge.id(), edge.source()))
                        .unwrap();
                    self.dag.remove_edge(edge_id);
                    predecessor_node
                });
                clbit_last_nodes
                    .entry(clbit)
                    .and_modify(|val| *val = new_node);
                self.dag
                    .add_edge(clbit_last_node, new_node, Wire::Clbit(clbit));
            }

            // If available, check all the vars in this instruction
            for var in vars.iter().flatten() {
                let var_last_node = if let Some(result) = vars_last_nodes.get_item(var)? {
                    let node: usize = result.extract()?;
                    vars_last_nodes.del_item(var)?;
                    NodeIndex::new(node)
                } else {
                    // If the var is not in the last nodes collection, the edge between the output node and its predecessor.
                    // Then, store the predecessor's NodeIndex in the last nodes collection.
                    let var_idx = self.vars.find(&var.bind(py).into()).unwrap();
                    let output_node = self.var_io_map.get(var_idx.index()).unwrap()[1];
                    let (edge_id, predecessor_node) = self
                        .dag
                        .edges_directed(output_node, Incoming)
                        .next()
                        .map(|edge| (edge.id(), edge.source()))
                        .unwrap();
                    self.dag.remove_edge(edge_id);
                    predecessor_node
                };

                // Because `DAGCircuit::additional_wires` can return repeated instances of vars,
                // we need to make sure to skip those to avoid cycles.
                vars_last_nodes.set_item(var, new_node.index())?;
                if var_last_node == new_node {
                    continue;
                }
                self.dag.add_edge(
                    var_last_node,
                    new_node,
                    Wire::Var(self.vars.find(&var.bind(py).into()).unwrap()),
                );
            }
        }

        // Add the output_nodes back to qargs
        for (qubit, node) in qubit_last_nodes {
            let output_node = self.qubit_io_map[qubit.index()][1];
            self.dag.add_edge(node, output_node, Wire::Qubit(qubit));
        }

        // Add the output_nodes back to cargs
        for (clbit, node) in clbit_last_nodes {
            let output_node = self.clbit_io_map[clbit.index()][1];
            self.dag.add_edge(node, output_node, Wire::Clbit(clbit));
        }

        // Add the output_nodes back to vars
        for item in vars_last_nodes.items() {
            let (var, node): (PyObject, usize) = item.extract()?;
            let var = self.vars.find(&var.bind(py).into()).unwrap();
            let output_node = self.var_io_map.get(var.index()).unwrap()[1];
            self.dag
                .add_edge(NodeIndex::new(node), output_node, Wire::Var(var));
        }

        Ok(new_nodes)
    }

    /// Alternative constructor to build an instance of [DAGCircuit] from a `QuantumCircuit`.
    pub(crate) fn from_circuit(
        py: Python,
        qc: QuantumCircuitData,
        copy_op: bool,
        qubit_order: Option<Vec<Bound<PyAny>>>,
        clbit_order: Option<Vec<Bound<PyAny>>>,
    ) -> PyResult<DAGCircuit> {
        // Extract necessary attributes
        let qc_data = qc.data;
        let num_qubits = qc_data.num_qubits();
        let num_clbits = qc_data.num_clbits();
        let num_ops = qc_data.__len__();
        let num_vars = qc.declared_vars.len() + qc.input_vars.len() + qc.captured_vars.len();

        // Build DAGCircuit with capacity
        let mut new_dag = DAGCircuit::with_capacity(
            py,
            num_qubits,
            num_clbits,
            Some(num_vars),
            Some(num_ops),
            None,
        )?;

        // Assign other necessary data
        new_dag.name = qc.name.map(|ob| ob.unbind());

        // Avoid manually acquiring the GIL.
        new_dag.global_phase = match qc_data.global_phase() {
            Param::ParameterExpression(exp) => Param::ParameterExpression(exp.clone_ref(py)),
            Param::Float(float) => Param::Float(*float),
            _ => unreachable!("Incorrect parameter assigned for global phase"),
        };

        new_dag.metadata = qc.metadata.map(|meta| meta.unbind());

        // Add the qubits depending on order, and produce the qargs map.
        let qarg_map = if let Some(qubit_ordering) = qubit_order {
            let mut ordered_vec = Vec::from_iter((0..num_qubits as u32).map(Qubit));
            qubit_ordering
                .into_iter()
                .try_for_each(|qubit| -> PyResult<()> {
                    let qubit_nat: ShareableQubit = qubit.extract()?;
                    if new_dag.qubits.find(&qubit_nat).is_some() {
                        return Err(DAGCircuitError::new_err(format!(
                            "duplicate qubits {}",
                            &qubit
                        )));
                    }
                    let qubit_index = qc_data.qubits().find(&qubit_nat).unwrap();
                    ordered_vec[qubit_index.index()] = new_dag.add_qubit_unchecked(qubit_nat)?;
                    Ok(())
                })?;
            // The `Vec::get` use is because an arbitrary interner might contain old references to
            // bit instances beyond `num_qubits`, such as if it's from a DAG that had wires removed.
            new_dag.merge_qargs(qc_data.qargs_interner(), |bit| {
                ordered_vec.get(bit.index()).copied()
            })
        } else {
            qc_data
                .qubits()
                .bits()
                .iter()
                .try_for_each(|qubit| -> PyResult<_> {
                    new_dag.add_qubit_unchecked(qubit.clone())?;
                    Ok(())
                })?;
            new_dag.merge_qargs(qc_data.qargs_interner(), |bit| Some(*bit))
        };

        // Add the clbits depending on order, and produce the cargs map.
        let carg_map = if let Some(clbit_ordering) = clbit_order {
            let mut ordered_vec = Vec::from_iter((0..num_clbits as u32).map(Clbit));
            clbit_ordering
                .into_iter()
                .try_for_each(|clbit| -> PyResult<()> {
                    let clbit_nat: ShareableClbit = clbit.extract()?;
                    if new_dag.clbits.find(&clbit_nat).is_some() {
                        return Err(DAGCircuitError::new_err(format!(
                            "duplicate clbits {}",
                            &clbit
                        )));
                    };
                    let clbit_index = qc_data.clbits().find(&clbit_nat).unwrap();
                    ordered_vec[clbit_index.index()] = new_dag.add_clbit_unchecked(clbit_nat)?;
                    Ok(())
                })?;
            // The `Vec::get` use is because an arbitrary interner might contain old references to
            // bit instances beyond `num_clbits`, such as if it's from a DAG that had wires removed.
            new_dag.merge_cargs(qc_data.cargs_interner(), |bit| {
                ordered_vec.get(bit.index()).copied()
            })
        } else {
            qc_data
                .clbits()
                .bits()
                .iter()
                .try_for_each(|clbit| -> PyResult<()> {
                    new_dag.add_clbit_unchecked(clbit.clone())?;
                    Ok(())
                })?;
            new_dag.merge_cargs(qc_data.cargs_interner(), |bit| Some(*bit))
        };

        // Add all of the new vars.
        for var in &qc.declared_vars {
            new_dag.add_var(py, var, DAGVarType::Declare)?;
        }

        for var in &qc.input_vars {
            new_dag.add_var(py, var, DAGVarType::Input)?;
        }

        for var in &qc.captured_vars {
            new_dag.add_var(py, var, DAGVarType::Capture)?;
        }

        new_dag
            .captured_stretches
            .reserve(qc.captured_stretches.len());

        for var in qc.captured_stretches {
            let name: String = var.getattr("name")?.extract::<String>()?;
            new_dag.captured_stretches.insert(name, var.unbind());
        }

        new_dag
            .declared_stretches
            .reserve(qc.declared_stretches.len());

        for var in qc.declared_stretches {
            let name: String = var.getattr("name")?.extract::<String>()?;
            new_dag.declared_stretches.insert(name, var.unbind());
        }

        // Add all the registers
        for qreg in qc_data.qregs() {
            new_dag.add_qreg(qreg.clone())?;
        }

        for creg in qc_data.cregs() {
            new_dag.add_creg(creg.clone())?;
        }

        // After bits and registers are added, copy bitlocations
        new_dag.qubit_locations = qc_data.qubit_indices().clone();
        new_dag.clbit_locations = qc_data.clbit_indices().clone();

        new_dag.try_extend(
            py,
            qc_data.iter().map(|instr| -> PyResult<PackedInstruction> {
                Ok(PackedInstruction {
                    op: if copy_op {
                        instr.op.py_deepcopy(py, None)?
                    } else {
                        instr.op.clone()
                    },
                    qubits: qarg_map[instr.qubits],
                    clbits: carg_map[instr.clbits],
                    params: instr.params.clone(),
                    label: instr.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                })
            }),
        )?;
        Ok(new_dag)
    }

    /// Builds a [DAGCircuit] based on an instance of [CircuitData].
    pub fn from_circuit_data(
        py: Python,
        circuit_data: CircuitData,
        copy_op: bool,
    ) -> PyResult<Self> {
        let circ = QuantumCircuitData {
            data: circuit_data,
            name: None,
            metadata: None,
            input_vars: Vec::new(),
            captured_vars: Vec::new(),
            declared_vars: Vec::new(),
            captured_stretches: Vec::new(),
            declared_stretches: Vec::new(),
        };
        Self::from_circuit(py, circ, copy_op, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    /// Replace a block of node indices with a new packed operation
    pub fn replace_block(
        &mut self,
        block_ids: &[NodeIndex],
        op: PackedOperation,
        params: SmallVec<[Param; 3]>,
        label: Option<&str>,
        cycle_check: bool,
        qubit_pos_map: &HashMap<Qubit, usize>,
        clbit_pos_map: &HashMap<Clbit, usize>,
    ) -> PyResult<NodeIndex> {
        let mut block_op_names = Vec::with_capacity(block_ids.len());
        let mut block_qargs: HashSet<Qubit> = HashSet::new();
        let mut block_cargs: HashSet<Clbit> = HashSet::new();
        for nd in block_ids {
            let weight = self.dag.node_weight(*nd);
            match weight {
                Some(NodeType::Operation(packed)) => {
                    block_op_names.push(packed.op.name().to_string());
                    block_qargs.extend(self.qargs_interner.get(packed.qubits));
                    block_cargs.extend(self.cargs_interner.get(packed.clbits));
                    // Add classical bits from SwitchCaseOp, if applicable.
                    if let OperationRef::Instruction(op) = packed.op.view() {
                        if op.name() == "switch_case" {
                            Python::with_gil(|py| -> PyResult<()> {
                                let op_bound = op.instruction.bind(py);
                                let target = op_bound.getattr(intern!(py, "target"))?;
                                if target.downcast::<PyClbit>().is_ok() {
                                    let target_clbit: ShareableClbit = target.extract()?;
                                    block_cargs.insert(self.clbits.find(&target_clbit).unwrap());
                                } else if target.is_instance_of::<PyClassicalRegister>() {
                                    block_cargs.extend(self.clbits.map_bits(
                                        target.extract::<Vec<ShareableClbit>>()?.into_iter(),
                                    )?);
                                } else {
                                    block_cargs.extend(
                                        self.clbits.map_bits(
                                            self.control_flow_module
                                                .node_resources(&target)?
                                                .clbits
                                                .extract::<Vec<ShareableClbit>>(py)?
                                                .into_iter(),
                                        )?,
                                    );
                                }
                                Ok(())
                            })?;
                        }
                    }
                }
                Some(_) => {
                    return Err(DAGCircuitError::new_err(
                        "Nodes in 'node_block' must be of type 'DAGOpNode'.",
                    ))
                }
                None => {
                    return Err(DAGCircuitError::new_err(
                        "Node in 'node_block' not found in DAG.",
                    ))
                }
            }
        }

        let mut block_qargs: Vec<Qubit> = block_qargs
            .into_iter()
            .filter(|q| qubit_pos_map.contains_key(q))
            .collect();
        block_qargs.sort_by_key(|q| qubit_pos_map[q]);

        let mut block_cargs: Vec<Clbit> = block_cargs
            .into_iter()
            .filter(|c| clbit_pos_map.contains_key(c))
            .collect();
        block_cargs.sort_by_key(|c| clbit_pos_map[c]);

        if op.num_qubits() as usize != block_qargs.len() {
            return Err(DAGCircuitError::new_err(format!(
                "Number of qubits in the replacement operation ({}) is not equal to the number of qubits in the block ({})!", op.num_qubits(), block_qargs.len()
            )));
        }

        let op_name = op.name().to_string();
        let qubits = self.qargs_interner.insert_owned(block_qargs);
        let clbits = self.cargs_interner.insert_owned(block_cargs);
        let weight = NodeType::Operation(PackedInstruction {
            op,
            qubits,
            clbits,
            params: (!params.is_empty()).then(|| Box::new(params)),
            label: label.map(|label| Box::new(label.to_string())),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        });

        let new_node = self
            .dag
            .contract_nodes(block_ids.iter().copied(), weight, cycle_check)
            .map_err(|e| match e {
                ContractError::DAGWouldCycle => DAGCircuitError::new_err(
                    "Replacing the specified node block would introduce a cycle",
                ),
            })?;

        self.increment_op(op_name.as_str());
        for name in block_op_names {
            self.decrement_op(name.as_str());
        }
        Ok(new_node)
    }

    /// Substitute an operation in a node with a new one. The wire counts must match and the same
    /// argument order will be used.
    pub fn substitute_op(
        &mut self,
        node_index: NodeIndex,
        new_op: PackedOperation,
        params: SmallVec<[Param; 3]>,
        label: Option<&str>,
    ) -> PyResult<()> {
        let old_packed = self.dag[node_index].unwrap_operation();
        let op_name = old_packed.op.name().to_string();

        if old_packed.op.num_qubits() != new_op.num_qubits()
            || old_packed.op.num_clbits() != new_op.num_clbits()
        {
            return Err(DAGCircuitError::new_err(
                format!(
                    "Cannot replace node of width ({} qubits, {} clbits) with operation of mismatched width ({} qubits, {} clbits)",
                    old_packed.op.num_qubits(), old_packed.op.num_clbits(), new_op.num_qubits(), new_op.num_clbits()
                )));
        }
        let new_op_name = new_op.name().to_string();
        let new_weight = NodeType::Operation(PackedInstruction {
            op: new_op,
            qubits: old_packed.qubits,
            clbits: old_packed.clbits,
            params: (!params.is_empty()).then(|| params.into()),
            label: label.map(|label| Box::new(label.to_string())),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        });
        if let Some(weight) = self.dag.node_weight_mut(node_index) {
            *weight = new_weight;
        }

        // Update self.op_names
        self.decrement_op(op_name.as_str());
        self.increment_op(new_op_name.as_str());
        Ok(())
    }

    /// Substitute a give node in the dag with a new operation from python
    pub fn substitute_node_with_py_op(
        &mut self,
        py: Python,
        node_index: NodeIndex,
        op: &Bound<PyAny>,
    ) -> PyResult<()> {
        // Extract information from node that is going to be replaced
        let old_packed = self.dag[node_index].unwrap_operation();
        let op_name = old_packed.op.name().to_string();
        // Extract information from new op
        let new_op = op.extract::<OperationFromPython>()?;
        let current_wires: HashSet<Wire> = self
            .dag
            .edges(node_index)
            .map(|e| e.weight().clone())
            .collect();
        let mut new_wires: HashSet<Wire> = self
            .qargs_interner
            .get(old_packed.qubits)
            .iter()
            .map(|x| Wire::Qubit(*x))
            .chain(
                self.cargs_interner
                    .get(old_packed.clbits)
                    .iter()
                    .map(|x| Wire::Clbit(*x)),
            )
            .collect();
        let (additional_clbits, additional_vars) =
            self.additional_wires(py, new_op.operation.view())?;
        new_wires.extend(additional_clbits.iter().map(|x| Wire::Clbit(*x)));
        new_wires.extend(
            additional_vars
                .iter()
                .map(|x| Wire::Var(self.vars.find(&x.bind(py).into()).unwrap())),
        );

        if old_packed.op.num_qubits() != new_op.operation.num_qubits()
            || old_packed.op.num_clbits() != new_op.operation.num_clbits()
        {
            return Err(DAGCircuitError::new_err(
                format!(
                    "Cannot replace node of width ({} qubits, {} clbits) with operation of mismatched width ({} qubits, {} clbits)",
                    old_packed.op.num_qubits(), old_packed.op.num_clbits(), new_op.operation.num_qubits(), new_op.operation.num_clbits()
                )));
        }

        #[cfg(feature = "cache_pygates")]
        let py_op_cache = Some(op.clone().unbind());

        let label = new_op.label.clone();
        if new_wires != current_wires {
            // The new wires must be a non-strict subset of the current wires; if they add new
            // wires, we'd not know where to cut the existing wire to insert the new dependency.
            return Err(DAGCircuitError::new_err(format!(
                "New operation '{:?}' does not span the same wires as the old node '{:?}'. New wires: {:?}, old_wires: {:?}.", op.str(), old_packed.op.view(), new_wires, current_wires
            )));
        }
        let new_op_name = new_op.operation.name().to_string();
        let new_weight = NodeType::Operation(PackedInstruction {
            op: new_op.operation,
            qubits: old_packed.qubits,
            clbits: old_packed.clbits,
            params: (!new_op.params.is_empty()).then(|| new_op.params.into()),
            label,
            #[cfg(feature = "cache_pygates")]
            py_op: py_op_cache.map(OnceLock::from).unwrap_or_default(),
        });
        if let Some(weight) = self.dag.node_weight_mut(node_index) {
            *weight = new_weight;
        }

        // Update self.op_names
        self.decrement_op(op_name.as_str());
        self.increment_op(new_op_name.as_str());
        Ok(())
    }
}

impl ::std::ops::Index<NodeIndex> for DAGCircuit {
    type Output = NodeType;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.dag.index(index)
    }
}

/// Add to global phase. Global phase can only be Float or ParameterExpression so this
/// does not handle the full possibility of parameter values.
pub(crate) fn add_global_phase(py: Python, phase: &Param, other: &Param) -> PyResult<Param> {
    Ok(match [phase, other] {
        [Param::Float(a), Param::Float(b)] => Param::Float(a + b),
        [Param::Float(a), Param::ParameterExpression(b)] => Param::ParameterExpression(
            b.clone_ref(py)
                .call_method1(py, intern!(py, "__radd__"), (*a,))?,
        ),
        [Param::ParameterExpression(a), Param::Float(b)] => Param::ParameterExpression(
            a.clone_ref(py)
                .call_method1(py, intern!(py, "__add__"), (*b,))?,
        ),
        [Param::ParameterExpression(a), Param::ParameterExpression(b)] => {
            Param::ParameterExpression(a.clone_ref(py).call_method1(
                py,
                intern!(py, "__add__"),
                (b,),
            )?)
        }
        _ => panic!("Invalid global phase"),
    })
}

type SortKeyType<'a> = (&'a [Qubit], &'a [Clbit]);

#[cfg(all(test, not(miri)))]
mod test {
    use crate::bit::{ClassicalRegister, QuantumRegister};
    use crate::dag_circuit::{DAGCircuit, Wire};
    use crate::operations::{StandardGate, StandardInstruction};
    use crate::packed_instruction::{PackedInstruction, PackedOperation};
    use crate::{Clbit, Qubit};
    use hashbrown::HashSet;
    use pyo3::prelude::*;
    use rustworkx_core::petgraph::prelude::*;
    use rustworkx_core::petgraph::visit::IntoEdgeReferences;

    fn new_dag(py: Python, qubits: u32, clbits: u32) -> DAGCircuit {
        let qreg = QuantumRegister::new_owning("q".to_owned(), qubits);
        let creg = ClassicalRegister::new_owning("c".to_owned(), clbits);
        let mut dag = DAGCircuit::new(py).unwrap();
        dag.add_qreg(qreg).unwrap();
        dag.add_creg(creg).unwrap();
        dag
    }

    macro_rules! cx_gate {
        ($dag:expr, $q0:expr, $q1:expr) => {
            PackedInstruction {
                op: PackedOperation::from_standard_gate(StandardGate::CXGate),
                qubits: $dag
                    .qargs_interner
                    .insert_owned(vec![Qubit($q0), Qubit($q1)]),
                clbits: $dag.cargs_interner.get_default(),
                params: None,
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op: Default::default(),
            }
        };
    }

    macro_rules! measure {
        ($dag:expr, $qarg:expr, $carg:expr) => {{
            let qubits = $dag.qargs_interner.insert_owned(vec![Qubit($qarg)]);
            let clbits = $dag.cargs_interner.insert_owned(vec![Clbit($qarg)]);
            PackedInstruction {
                op: PackedOperation::from_standard_instruction(StandardInstruction::Measure),
                qubits,
                clbits,
                params: None,
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op: Default::default(),
            }
        }};
    }

    #[test]
    fn test_push_back() -> PyResult<()> {
        Python::with_gil(|py| {
            let mut dag = new_dag(py, 2, 2);

            // IO nodes.
            let [q0_in_node, q0_out_node] = dag.qubit_io_map[0];
            let [q1_in_node, q1_out_node] = dag.qubit_io_map[1];
            let [c0_in_node, c0_out_node] = dag.clbit_io_map[0];
            let [c1_in_node, c1_out_node] = dag.clbit_io_map[1];

            // Add a CX to the otherwise empty circuit.
            let cx = cx_gate!(dag, 0, 1);
            let cx_node = dag.push_back(py, cx)?;
            assert!(matches!(dag.op_names.get("cx"), Some(1)));

            let expected_wires = HashSet::from_iter([
                // q0In => CX => q0Out
                (q0_in_node, cx_node, Wire::Qubit(Qubit(0))),
                (cx_node, q0_out_node, Wire::Qubit(Qubit(0))),
                // q1In => CX => q1Out
                (q1_in_node, cx_node, Wire::Qubit(Qubit(1))),
                (cx_node, q1_out_node, Wire::Qubit(Qubit(1))),
                // No clbits used, so in goes straight to out.
                (c0_in_node, c0_out_node, Wire::Clbit(Clbit(0))),
                (c1_in_node, c1_out_node, Wire::Clbit(Clbit(1))),
            ]);

            let actual_wires: HashSet<_> = dag
                .dag
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight().clone()))
                .collect();

            assert_eq!(actual_wires, expected_wires, "unexpected DAG structure");

            // Add measures after CX.
            let measure_q0 = measure!(dag, 0, 0);
            let measure_q0_node = dag.push_back(py, measure_q0)?;

            let measure_q1 = measure!(dag, 1, 1);
            let measure_q1_node = dag.push_back(py, measure_q1)?;

            let expected_wires = HashSet::from_iter([
                // q0In -> CX -> M -> q0Out
                (q0_in_node, cx_node, Wire::Qubit(Qubit(0))),
                (cx_node, measure_q0_node, Wire::Qubit(Qubit(0))),
                (measure_q0_node, q0_out_node, Wire::Qubit(Qubit(0))),
                // q1In -> CX -> M -> q1Out
                (q1_in_node, cx_node, Wire::Qubit(Qubit(1))),
                (cx_node, measure_q1_node, Wire::Qubit(Qubit(1))),
                (measure_q1_node, q1_out_node, Wire::Qubit(Qubit(1))),
                // c0In -> M -> c0Out
                (c0_in_node, measure_q0_node, Wire::Clbit(Clbit(0))),
                (measure_q0_node, c0_out_node, Wire::Clbit(Clbit(0))),
                // c1In -> M -> c1Out
                (c1_in_node, measure_q1_node, Wire::Clbit(Clbit(1))),
                (measure_q1_node, c1_out_node, Wire::Clbit(Clbit(1))),
            ]);

            let actual_wires: HashSet<_> = dag
                .dag
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight().clone()))
                .collect();

            assert_eq!(actual_wires, expected_wires, "unexpected DAG structure");
            Ok(())
        })
    }

    #[test]
    fn test_push_front() -> PyResult<()> {
        Python::with_gil(|py| {
            let mut dag = new_dag(py, 2, 2);

            // IO nodes.
            let [q0_in_node, q0_out_node] = dag.qubit_io_map[0];
            let [q1_in_node, q1_out_node] = dag.qubit_io_map[1];
            let [c0_in_node, c0_out_node] = dag.clbit_io_map[0];
            let [c1_in_node, c1_out_node] = dag.clbit_io_map[1];

            // Add measures first (we'll add something before them afterwards).
            let measure_q0 = measure!(dag, 0, 0);
            let measure_q0_node = dag.push_back(py, measure_q0)?;

            let measure_q1 = measure!(dag, 1, 1);
            let measure_q1_node = dag.push_back(py, measure_q1)?;

            let expected_wires = HashSet::from_iter([
                // q0In => M => q0Out
                (q0_in_node, measure_q0_node, Wire::Qubit(Qubit(0))),
                (measure_q0_node, q0_out_node, Wire::Qubit(Qubit(0))),
                // q1In => M => q1Out
                (q1_in_node, measure_q1_node, Wire::Qubit(Qubit(1))),
                (measure_q1_node, q1_out_node, Wire::Qubit(Qubit(1))),
                // c0In -> M -> c0Out
                (c0_in_node, measure_q0_node, Wire::Clbit(Clbit(0))),
                (measure_q0_node, c0_out_node, Wire::Clbit(Clbit(0))),
                // c1In -> M -> c1Out
                (c1_in_node, measure_q1_node, Wire::Clbit(Clbit(1))),
                (measure_q1_node, c1_out_node, Wire::Clbit(Clbit(1))),
            ]);

            let actual_wires: HashSet<_> = dag
                .dag
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight().clone()))
                .collect();

            assert_eq!(actual_wires, expected_wires);

            // Add a CX before the measures.
            let cx = cx_gate!(dag, 0, 1);
            let cx_node = dag.push_front(py, cx)?;
            assert!(matches!(dag.op_names.get("cx"), Some(1)));

            let expected_wires = HashSet::from_iter([
                // q0In -> CX -> M -> q0Out
                (q0_in_node, cx_node, Wire::Qubit(Qubit(0))),
                (cx_node, measure_q0_node, Wire::Qubit(Qubit(0))),
                (measure_q0_node, q0_out_node, Wire::Qubit(Qubit(0))),
                // q1In -> CX -> M -> q1Out
                (q1_in_node, cx_node, Wire::Qubit(Qubit(1))),
                (cx_node, measure_q1_node, Wire::Qubit(Qubit(1))),
                (measure_q1_node, q1_out_node, Wire::Qubit(Qubit(1))),
                // c0In -> M -> c0Out
                (c0_in_node, measure_q0_node, Wire::Clbit(Clbit(0))),
                (measure_q0_node, c0_out_node, Wire::Clbit(Clbit(0))),
                // c1In -> M -> c1Out
                (c1_in_node, measure_q1_node, Wire::Clbit(Clbit(1))),
                (measure_q1_node, c1_out_node, Wire::Clbit(Clbit(1))),
            ]);

            let actual_wires: HashSet<_> = dag
                .dag
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight().clone()))
                .collect();

            assert_eq!(actual_wires, expected_wires, "unexpected DAG structure");
            Ok(())
        })
    }
}
