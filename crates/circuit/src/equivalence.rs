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

use itertools::Itertools;

use pyo3::exceptions::PyTypeError;
use rustworkx_core::petgraph::csr::IndexType;
use rustworkx_core::petgraph::stable_graph::StableDiGraph;
use rustworkx_core::petgraph::visit::IntoEdgeReferences;

use smallvec::SmallVec;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{error::Error, fmt::Display};

use exceptions::CircuitError;
use hashbrown::{HashMap, HashSet};
use pyo3::types::{PyDict, PyList, PySet, PyString};
use pyo3::{prelude::*, types::IntoPyDict};

use rustworkx_core::petgraph::{
    graph::{EdgeIndex, NodeIndex},
    visit::EdgeRef,
};

use crate::circuit_instruction::{convert_py_to_operation_type, CircuitInstruction};
use crate::imports::ImportOnceCell;
use crate::operations::Param;
use crate::operations::{Operation, OperationType};

mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.circuit.exceptions, CircuitError}
}
pub static PYDIGRAPH: ImportOnceCell = ImportOnceCell::new("rustworkx", "PyDiGraph");
pub static QUANTUMCIRCUIT: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.quantumcircuit", "QuantumCircuit");

// Custom Structs

#[pyclass(sequence, module = "qiskit._accelerate.circuit.equivalence")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub num_qubits: u32,
}

#[pymethods]
impl Key {
    #[new]
    #[pyo3(signature = (name, num_qubits))]
    fn new(name: String, num_qubits: u32) -> Self {
        Self { name, num_qubits }
    }

    fn __eq__(&self, other: Self) -> bool {
        self.eq(&other)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        (self.name.to_string(), self.num_qubits).hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(slf: PyRef<Self>) -> String {
        slf.to_string()
    }

    fn __getnewargs__(slf: PyRef<Self>) -> (Bound<PyString>, u32) {
        (
            PyString::new_bound(slf.py(), slf.name.as_str()),
            slf.num_qubits,
        )
    }
}

impl Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Key(name=\'{}\', num_qubits={})",
            self.name, self.num_qubits
        )
    }
}

#[pyclass(sequence, module = "qiskit._accelerate.circuit.equivalence")]
#[derive(Debug, Clone, PartialEq)]
pub struct Equivalence {
    #[pyo3(get)]
    pub params: SmallVec<[Param; 3]>,
    #[pyo3(get)]
    pub circuit: CircuitRep,
}

#[pymethods]
impl Equivalence {
    #[new]
    #[pyo3(signature = (params, circuit))]
    fn new(params: SmallVec<[Param; 3]>, circuit: CircuitRep) -> Self {
        Self { circuit, params }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    fn __eq__(&self, other: Self) -> bool {
        self.eq(&other)
    }

    fn __getnewargs__(&self, py: Python) -> (Py<PyList>, PyObject) {
        let params = PyList::new_bound(py, self.params.iter().map(|param| param.to_object(py)));
        (params.unbind(), self.circuit.object.clone_ref(py))
    }
}

impl Display for Equivalence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Equivalence(params=[{}], circuit={})",
            self.params.iter().format(", "),
            self.circuit
        )
    }
}

#[pyclass(sequence, module = "qiskit._accelerate.circuit.equivalence")]
#[derive(Debug, Clone, PartialEq)]
pub struct NodeData {
    #[pyo3(get)]
    key: Key,
    #[pyo3(get)]
    equivs: Vec<Equivalence>,
}

#[pymethods]
impl NodeData {
    #[new]
    #[pyo3(signature = (key, equivs))]
    fn new(key: Key, equivs: Vec<Equivalence>) -> Self {
        Self { key, equivs }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    fn __eq__(&self, other: Self) -> bool {
        self.eq(&other)
    }

    fn __getnewargs__(&self, py: Python) -> (Key, Py<PyList>) {
        (
            self.key.clone(),
            PyList::new_bound(
                py,
                self.equivs.iter().map(|equiv| equiv.clone().into_py(py)),
            )
            .unbind(),
        )
    }
}

impl Display for NodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NodeData(key={}, equivs=[{}])",
            self.key,
            self.equivs.iter().format(", ")
        )
    }
}

#[pyclass(sequence, module = "qiskit._accelerate.circuit.equivalence")]
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeData {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub num_gates: usize,
    #[pyo3(get)]
    pub rule: Equivalence,
    #[pyo3(get)]
    pub source: Key,
}

#[pymethods]
impl EdgeData {
    #[new]
    #[pyo3(signature = (index, num_gates, rule, source))]
    fn new(index: usize, num_gates: usize, rule: Equivalence, source: Key) -> Self {
        Self {
            index,
            num_gates,
            rule,
            source,
        }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    fn __eq__(&self, other: Self) -> bool {
        self.eq(&other)
    }

    fn __getnewargs__(slf: PyRef<Self>) -> (usize, usize, Equivalence, Key) {
        (
            slf.index,
            slf.num_gates,
            slf.rule.clone(),
            slf.source.clone(),
        )
    }
}

impl Display for EdgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EdgeData(index={}, num_gates={}, rule={}, source={})",
            self.index, self.num_gates, self.rule, self.source
        )
    }
}

// Enum to extract circuit instructions more broadly
#[derive(Debug, Clone)]
pub struct GateOper {
    operation: OperationType,
    params: SmallVec<[Param; 3]>,
}

impl<'py> FromPyObject<'py> for GateOper {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        let op_struct = convert_py_to_operation_type(ob.py(), ob.into())?;
        Ok(Self {
            operation: op_struct.operation,
            params: op_struct.params,
        })
    }
}

/// Temporary interpretation of QuantumCircuit
#[derive(Debug, Clone)]
pub struct CircuitRep {
    object: PyObject,
    pub num_qubits: u32,
    pub num_clbits: u32,
    params: Option<SmallVec<[Param; 3]>>,
    data: Option<Vec<CircuitInstruction>>,
    // TODO: Have a valid implementation of CircuiData that's usable in Rust.
}

impl CircuitRep {
    pub fn parameters(&mut self) -> &[Param] {
        if self.params.is_none() {
            let params = Python::with_gil(|py| -> PyResult<SmallVec<[Param; 3]>> {
                self.object
                    .bind(py)
                    .getattr("parameters")?
                    .getattr("data")?
                    .extract()
            })
            .unwrap_or_default();
            self.params = Some(params);
            return self.params.as_ref().unwrap();
        }
        return self.params.as_ref().unwrap();
    }

    pub fn data(&mut self) -> &[CircuitInstruction] {
        if self.data.is_none() {
            let data = Python::with_gil(|py| -> PyResult<Vec<CircuitInstruction>> {
                self.object.bind(py).getattr("data")?.extract()
            })
            .unwrap_or_default();
            self.data = Some(data);
            return self.data.as_ref().unwrap();
        }
        return self.data.as_ref().unwrap();
    }
}

impl FromPyObject<'_> for CircuitRep {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if ob.is_instance(QUANTUMCIRCUIT.get_bound(ob.py()))? {
            let num_qubits = ob.getattr("num_qubits")?.extract()?;
            let num_clbits = ob.getattr("num_clbits")?.extract()?;
            Ok(Self {
                object: ob.into_py(ob.py()),
                num_qubits,
                num_clbits,
                params: None,
                data: None,
            })
        } else {
            Err(PyTypeError::new_err(
                "Provided object was not an instance of QuantumCircuit",
            ))
        }
    }
}

impl Display for CircuitRep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let py_rep_str = Python::with_gil(|py| -> PyResult<String> {
            let bound = self.object.bind(py);
            bound.repr().map(|pystring| pystring.to_string())
        })
        .unwrap();
        write!(f, "{}", py_rep_str)
    }
}

impl PartialEq for CircuitRep {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| -> PyResult<bool> {
            let bound = other.object.bind(py);
            bound.eq(&self.object)
        })
        .unwrap_or_default()
    }
}

impl IntoPy<PyObject> for CircuitRep {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.object
    }
}

// Custom Types
type GraphType = StableDiGraph<NodeData, EdgeData>;
type KTIType = HashMap<Key, NodeIndex>;

#[pyclass(
    subclass,
    name = "BaseEquivalenceLibrary",
    module = "qiskit._accelerate.circuit.equivalence"
)]
#[derive(Debug, Clone)]
pub struct EquivalenceLibrary {
    _graph: GraphType,
    key_to_node_index: KTIType,
    rule_id: usize,
    graph: Option<PyObject>,
}

#[pymethods]
impl EquivalenceLibrary {
    /// Create a new equivalence library.
    ///
    /// Args:
    ///     base (Optional[EquivalenceLibrary]):  Base equivalence library to
    ///         be referenced if an entry is not found in this library.
    #[new]
    fn new(base: Option<&EquivalenceLibrary>) -> Self {
        if let Some(base) = base {
            Self {
                _graph: base._graph.clone(),
                key_to_node_index: base.key_to_node_index.clone(),
                rule_id: base.rule_id,
                graph: None,
            }
        } else {
            Self {
                _graph: GraphType::new(),
                key_to_node_index: KTIType::new(),
                rule_id: 0_usize,
                graph: None,
            }
        }
    }

    // TODO: Add a way of returning a graph

    /// Add a new equivalence to the library. Future queries for the Gate
    /// will include the given circuit, in addition to all existing equivalences
    /// (including those from base).
    ///
    /// Parameterized Gates (those including `qiskit.circuit.Parameters` in their
    /// `Gate.params`) can be marked equivalent to parameterized circuits,
    /// provided the parameters match.
    ///
    /// Args:
    ///     gate (Gate): A Gate instance.
    ///     equivalent_circuit (QuantumCircuit): A circuit equivalently
    ///         implementing the given Gate.
    #[pyo3(text_signature = "(gate, equivalent_circuit, /,")]
    fn add_equivalence(&mut self, gate: GateOper, equivalent_circuit: CircuitRep) -> PyResult<()> {
        match self.add_equiv(gate, equivalent_circuit) {
            Ok(_) => Ok(()),
            Err(e) => Err(CircuitError::new_err(e.message)),
        }
    }

    /// Check if a library contains any decompositions for gate.

    ///     Args:
    ///         gate (Gate): A Gate instance.

    ///     Returns:
    ///         Bool: True if gate has a known decomposition in the library.
    ///             False otherwise.
    #[pyo3(text_signature = "(gate, /,)")]
    pub fn has_entry(&self, gate: GateOper) -> bool {
        let key = Key {
            name: gate.operation.name().to_string(),
            num_qubits: gate.operation.num_qubits(),
        };
        self.key_to_node_index.contains_key(&key)
    }

    /// Set the equivalence record for a Gate. Future queries for the Gate
    /// will return only the circuits provided.
    ///
    /// Parameterized Gates (those including `qiskit.circuit.Parameters` in their
    /// `Gate.params`) can be marked equivalent to parameterized circuits,
    /// provided the parameters match.
    ///
    /// Args:
    ///     gate (Gate): A Gate instance.
    ///     entry (List['QuantumCircuit']) : A list of QuantumCircuits, each
    ///         equivalently implementing the given Gate.
    #[pyo3(text_signature = "(gate, entry, /,)")]
    fn set_entry(&mut self, gate: GateOper, entry: Vec<CircuitRep>) -> PyResult<()> {
        match self.set_entry_native(gate, entry) {
            Ok(_) => Ok(()),
            Err(e) => Err(CircuitError::new_err(e.message)),
        }
    }

    /// Gets the set of QuantumCircuits circuits from the library which
    /// equivalently implement the given Gate.
    ///
    /// Parameterized circuits will have their parameters replaced with the
    /// corresponding entries from Gate.params.
    ///
    /// Args:
    ///     gate (Gate) - Gate: A Gate instance.
    ///
    /// Returns:
    ///     List[QuantumCircuit]: A list of equivalent QuantumCircuits. If empty,
    ///         library contains no known decompositions of Gate.
    ///
    ///         Returned circuits will be ordered according to their insertion in
    ///         the library, from earliest to latest, from top to base. The
    ///         ordering of the StandardEquivalenceLibrary will not generally be
    ///         consistent across Qiskit versions.
    #[pyo3(text_signature = "(gate, /,)")]
    pub fn get_entry(&self, py: Python, gate: GateOper) -> PyResult<Py<PyList>> {
        let key = Key {
            name: gate.operation.name().to_string(),
            num_qubits: gate.operation.num_qubits(),
        };
        let query_params = gate.params;

        let bound_equivalencies = self
            ._get_equivalences(&key)
            .into_iter()
            .filter_map(|equivalence| rebind_equiv(py, equivalence, &query_params).ok());
        let return_list = PyList::empty_bound(py);
        for equiv in bound_equivalencies {
            return_list.append(equiv)?;
        }
        Ok(return_list.unbind())
    }

    #[getter]
    fn get_graph(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(graph) = &self.graph {
            Ok(graph.clone_ref(py))
        } else {
            self.graph = Some(to_pygraph(py, &self._graph)?);
            Ok(self
                .graph
                .as_ref()
                .map(|graph| graph.clone_ref(py))
                .unwrap())
        }
    }

    /// Get all the equivalences for the given key
    pub fn _get_equivalences(&self, key: &Key) -> Vec<Equivalence> {
        if let Some(key_in) = self.key_to_node_index.get(key) {
            self._graph[*key_in].equivs.clone()
        } else {
            vec![]
        }
    }

    #[pyo3(name = "keys")]
    pub fn py_keys(slf: PyRef<Self>) -> PyResult<Bound<PySet>> {
        let py_set = PySet::empty_bound(slf.py())?;
        for key in slf.keys() {
            py_set.add(key.clone().into_py(slf.py()))?;
        }
        Ok(py_set)
    }

    fn node_index(&self, key: Key) -> usize {
        self.key_to_node_index[&key].index()
    }

    fn __getstate__(slf: PyRef<Self>) -> PyResult<Bound<'_, PyDict>> {
        let ret = PyDict::new_bound(slf.py());
        ret.set_item("rule_id", slf.rule_id)?;
        let key_to_usize_node: Bound<PyDict> = PyDict::new_bound(slf.py());
        for (key, val) in slf.key_to_node_index.iter() {
            key_to_usize_node.set_item((&key.name, key.num_qubits), val.index())?;
        }
        ret.set_item("key_to_node_index", key_to_usize_node)?;
        let graph_nodes: Bound<PyList> = PyList::empty_bound(slf.py());
        for weight in slf._graph.node_weights() {
            graph_nodes.append(weight.clone().into_py(slf.py()))?;
        }
        ret.set_item("graph_nodes", graph_nodes.unbind())?;
        let edges = slf._graph.edge_references().map(|edge| {
            (
                edge.source().index(),
                edge.target().index(),
                edge.weight().clone().into_py(slf.py()),
            )
        });
        let graph_edges = PyList::empty_bound(slf.py());
        for edge in edges {
            graph_edges.add(edge)?;
        }
        ret.set_item("graph_edges", graph_edges.unbind())?;
        Ok(ret)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<'_, PyDict>) -> PyResult<()> {
        slf.rule_id = state.get_item("rule_id")?.unwrap().extract()?;
        let graph_nodes_ref: Bound<PyAny> = state.get_item("graph_nodes")?.unwrap();
        let graph_nodes: &Bound<PyList> = graph_nodes_ref.downcast()?;
        let graph_edge_ref: Bound<PyAny> = state.get_item("graph_edges")?.unwrap();
        let graph_edges: &Bound<PyList> = graph_edge_ref.downcast()?;
        slf._graph = GraphType::new();
        for node_weight in graph_nodes {
            slf._graph.add_node(node_weight.extract()?);
        }
        for edge in graph_edges {
            let (source_node, target_node, edge_weight) = edge.extract()?;
            slf._graph.add_edge(
                NodeIndex::new(source_node),
                NodeIndex::new(target_node),
                edge_weight,
            );
        }
        slf.key_to_node_index = state
            .get_item("key_to_node_index")?
            .unwrap()
            .extract::<HashMap<(String, u32), usize>>()?
            .into_iter()
            .map(|((name, num_qubits), val)| (Key::new(name, num_qubits), NodeIndex::new(val)))
            .collect();
        slf.graph = None;
        Ok(())
    }
}

// Rust native methods
impl EquivalenceLibrary {
    pub fn keys(&self) -> impl Iterator<Item = &Key> {
        self.key_to_node_index.keys()
    }

    /// Create a new node if key not found
    fn set_default_node(&mut self, key: Key) -> NodeIndex {
        if let Some(value) = self.key_to_node_index.get(&key) {
            *value
        } else {
            let node = self._graph.add_node(NodeData {
                key: key.clone(),
                equivs: vec![],
            });
            self.key_to_node_index.insert(key, node);
            node
        }
    }

    /// Rust native equivalent to `EquivalenceLibrary.add_equivalence()`
    ///
    /// Add a new equivalence to the library. Future queries for the Gate
    /// will include the given circuit, in addition to all existing equivalences
    /// (including those from base).
    ///
    /// Parameterized Gates (those including `qiskit.circuit.Parameters` in their
    /// `Gate.params`) can be marked equivalent to parameterized circuits,
    /// provided the parameters match.
    ///
    /// Args:
    ///     gate (Gate): A Gate instance.
    ///     equivalent_circuit (QuantumCircuit): A circuit equivalently
    ///         implementing the given Gate.
    pub fn add_equiv(
        &mut self,
        gate: GateOper,
        mut equivalent_circuit: CircuitRep,
    ) -> Result<(), EquivalenceError> {
        raise_if_shape_mismatch(&gate, &equivalent_circuit)?;
        raise_if_param_mismatch(&gate.params, equivalent_circuit.parameters())?;

        let key: Key = Key {
            name: gate.operation.name().to_string(),
            num_qubits: gate.operation.num_qubits(),
        };
        let equiv = Equivalence {
            params: gate.params,
            circuit: equivalent_circuit.clone(),
        };

        let target = self.set_default_node(key);
        if let Some(node) = self._graph.node_weight_mut(target) {
            node.equivs.push(equiv.clone());
        }
        let sources: HashSet<Key> =
            HashSet::from_iter(equivalent_circuit.data().iter().map(|inst| Key {
                name: inst.operation.name().to_string(),
                num_qubits: inst.operation.num_qubits(),
            }));
        let edges = Vec::from_iter(sources.iter().map(|source| {
            (
                self.set_default_node(source.clone()),
                target,
                EdgeData {
                    index: self.rule_id,
                    num_gates: sources.len(),
                    rule: equiv.clone(),
                    source: source.clone(),
                },
            )
        }));
        for edge in edges {
            self._graph.add_edge(edge.0, edge.1, edge.2);
        }
        self.rule_id += 1;
        self.graph = None;
        Ok(())
    }

    /// Rust native equivalent to `EquivalenceLibrary.set_entry()`
    /// Set the equivalence record for a Gate. Future queries for the Gate
    /// will return only the circuits provided.
    ///
    /// Parameterized Gates (those including `qiskit.circuit.Parameters` in their
    /// `Gate.params`) can be marked equivalent to parameterized circuits,
    /// provided the parameters match.
    ///
    /// Args:
    ///     gate (Gate): A Gate instance.
    ///     entry (List['QuantumCircuit']) : A list of QuantumCircuits, each
    ///         equivalently implementing the given Gate.
    pub fn set_entry_native(
        &mut self,
        gate: GateOper,
        mut entry: Vec<CircuitRep>,
    ) -> Result<(), EquivalenceError> {
        for equiv in entry.iter_mut() {
            raise_if_shape_mismatch(&gate, equiv)?;
            raise_if_param_mismatch(&gate.params, equiv.parameters())?;
        }

        let key = Key {
            name: gate.operation.name().to_string(),
            num_qubits: gate.operation.num_qubits(),
        };
        let node_index = self.set_default_node(key);

        if let Some(graph_ind) = self._graph.node_weight_mut(node_index) {
            graph_ind.equivs.clear();
        }

        let edges: Vec<EdgeIndex> = self
            ._graph
            .edges_directed(node_index, rustworkx_core::petgraph::Direction::Incoming)
            .map(|x| x.id())
            .collect();
        for edge in edges {
            self._graph.remove_edge(edge);
        }
        for equiv in entry {
            self.add_equiv(gate.clone(), equiv.clone())?
        }
        self.graph = None;
        Ok(())
    }
}

fn raise_if_param_mismatch(
    gate_params: &[Param],
    circuit_parameters: &[Param],
) -> Result<(), EquivalenceError> {
    let gate_params = gate_params
        .iter()
        .filter(|param| matches!(param, Param::ParameterExpression(_)))
        .collect_vec();
    if gate_params.len() == circuit_parameters.len()
        && gate_params.iter().any(|x| !circuit_parameters.contains(x))
    {
        return Err(EquivalenceError::new_err(format!(
            "Cannot add equivalence between circuit and gate \
            of different parameters. Gate params: {:#?}. \
            Circuit params: {:#?}.",
            gate_params, circuit_parameters
        )));
    }
    Ok(())
}

fn raise_if_shape_mismatch(gate: &GateOper, circuit: &CircuitRep) -> Result<(), EquivalenceError> {
    if gate.operation.num_qubits() != circuit.num_qubits
        || gate.operation.num_clbits() != circuit.num_clbits
    {
        return Err(EquivalenceError::new_err(format!(
            "Cannot add equivalence between circuit and gate \
            of different shapes. Gate: {} qubits and {} clbits. \
            Circuit: {} qubits and {} clbits.",
            gate.operation.num_qubits(),
            gate.operation.num_clbits(),
            circuit.num_qubits,
            circuit.num_clbits
        )));
    }
    Ok(())
}

fn rebind_equiv(py: Python, equiv: Equivalence, query_params: &[Param]) -> PyResult<PyObject> {
    let (equiv_params, equiv_circuit) = (equiv.params, equiv.circuit);
    let param_iter = equiv_params
        .into_iter()
        .zip(query_params.iter().cloned())
        .filter_map(|(param_x, param_y)| match param_x {
            Param::ParameterExpression(_) => Some((param_x, param_y)),
            _ => None,
        });
    let param_map = PyDict::new_bound(py);
    for (param_key, param_val) in param_iter {
        param_map.set_item(param_key, param_val)?;
    }
    let kwargs = [("inplace", false), ("flat_input", true)].into_py_dict_bound(py);
    let new_equiv = equiv_circuit.object.call_method_bound(
        py,
        "assign_parameters",
        (param_map,),
        Some(&kwargs),
    )?;
    Ok(new_equiv)
}

// Errors

#[derive(Debug, Clone)]
pub struct EquivalenceError {
    pub message: String,
}

impl EquivalenceError {
    pub fn new_err(message: String) -> Self {
        Self { message }
    }
}

impl Error for EquivalenceError {}

impl Display for EquivalenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

fn to_pygraph<N, E>(py: Python<'_>, pet_graph: &StableDiGraph<N, E>) -> PyResult<PyObject>
where
    N: IntoPy<PyObject> + Clone,
    E: IntoPy<PyObject> + Clone,
{
    let graph = PYDIGRAPH.get_bound(py).call0()?;
    let node_weights: Vec<N> = pet_graph.node_weights().cloned().collect();
    graph.call_method1("add_nodes_from", (node_weights,))?;
    let edge_weights: Vec<(usize, usize, E)> = pet_graph
        .edge_references()
        .map(|edge| {
            (
                edge.source().index(),
                edge.target().index(),
                edge.weight().clone(),
            )
        })
        .collect();
    graph.call_method1("add_edges_from", (edge_weights,))?;
    Ok(graph.unbind())
}

#[pymodule]
pub fn equivalence(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EquivalenceLibrary>()?;
    m.add_class::<NodeData>()?;
    m.add_class::<EdgeData>()?;
    m.add_class::<Equivalence>()?;
    m.add_class::<Key>()?;
    Ok(())
}
