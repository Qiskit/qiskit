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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{error::Error, fmt::Display};

use exceptions::CircuitError;
use hashbrown::{HashMap, HashSet};
use pyo3::sync::GILOnceCell;
use pyo3::{prelude::*, types::IntoPyDict};
use rustworkx_core::petgraph::{
    graph::{DiGraph, EdgeIndex, NodeIndex},
    visit::EdgeRef,
};

mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.circuit.exceptions, CircuitError}
}

/// Helper wrapper around `GILOnceCell` instances that are just intended to store a Python object
/// that is lazily imported.
pub struct ImportOnceCell {
    module: &'static str,
    object: &'static str,
    cell: GILOnceCell<Py<PyAny>>,
}

impl ImportOnceCell {
    const fn new(module: &'static str, object: &'static str) -> Self {
        Self {
            module,
            object,
            cell: GILOnceCell::new(),
        }
    }

    /// Get the underlying GIL-independent reference to the contained object, importing if
    /// required.
    #[inline]
    pub fn get(&self, py: Python) -> &Py<PyAny> {
        self.cell.get_or_init(py, || {
            py.import_bound(self.module)
                .unwrap()
                .getattr(self.object)
                .unwrap()
                .unbind()
        })
    }

    /// Get a GIL-bound reference to the contained object, importing if required.
    #[inline]
    pub fn get_bound<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.get(py).bind(py)
    }
}

pub static PARAMETER_EXPRESSION: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.parameterexpression", "ParameterExpression");
pub static QUANTUM_CIRCUIT: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.quantumcircuit", "QuantumCircuit");
pub static PYDIGRAPH: ImportOnceCell = ImportOnceCell::new("rustworkx", "PyDiGraph");

// Custom Structs

#[pyclass(sequence)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub num_qubits: usize,
}

#[pymethods]
impl Key {
    #[new]
    fn new(name: String, num_qubits: usize) -> Self {
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

    fn __repr__(slf: PyRef<'_, Self>) -> String {
        slf.to_string()
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

#[pyclass(sequence)]
#[derive(Debug, Clone)]
pub struct Equivalence {
    #[pyo3(get)]
    pub params: Vec<Param>,
    #[pyo3(get)]
    pub circuit: CircuitRep,
}

#[pymethods]
impl Equivalence {
    #[new]
    fn new(params: Vec<Param>, circuit: CircuitRep) -> Self {
        Self { circuit, params }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl Display for Equivalence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Equivalence(params={:?}, circuits={:?})",
            self.params, self.circuit
        )
    }
}

#[pyclass(sequence)]
#[derive(Debug, Clone)]
pub struct NodeData {
    #[pyo3(get)]
    key: Key,
    #[pyo3(get)]
    equivs: Vec<Equivalence>,
}

#[pymethods]
impl NodeData {
    #[new]
    fn new(key: Key, equivs: Vec<Equivalence>) -> Self {
        Self { key, equivs }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl Display for NodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeData(key={}, equivs={:#?})", self.key, self.equivs)
    }
}

#[pyclass(sequence)]
#[derive(Debug, Clone)]
pub struct EdgeData {
    #[pyo3(get)]
    pub index: u32,
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
    fn new(index: u32, num_gates: usize, rule: Equivalence, source: Key) -> Self {
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

// REPRESENTATIONS of non rust objects
// Temporary definition of Parameter
#[derive(Clone, Debug)]
pub enum Param {
    ParameterExpression(ParamExpRep),
    Float(f64),
    Obj(PyObject),
}

impl<'py> FromPyObject<'py> for Param {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        Ok(
            if b.is_instance(PARAMETER_EXPRESSION.get_bound(b.py()))?
                || b.is_instance(QUANTUM_CIRCUIT.get_bound(b.py()))?
            {
                let id = b.getattr("_uuid")?.getattr("hex")?.extract::<String>()?;
                Param::ParameterExpression(ParamExpRep {
                    id,
                    object: b.clone().unbind(),
                })
            } else if let Ok(val) = b.extract::<f64>() {
                Param::Float(val)
            } else {
                Param::Obj(b.clone().unbind())
            },
        )
    }
}

impl IntoPy<PyObject> for Param {
    fn into_py(self, py: Python) -> PyObject {
        match &self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.object.clone_ref(py),
            Self::Obj(val) => val.clone_ref(py),
        }
    }
}

impl ToPyObject for Param {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.object.clone_ref(py),
            Self::Obj(val) => val.clone_ref(py),
        }
    }
}

impl Param {
    fn compare(one: &PyObject, other: &PyObject) -> bool {
        Python::with_gil(|py| -> PyResult<bool> {
            let other_bound = other.bind(py);
            other_bound.eq(one)
        })
        .unwrap_or_default()
    }
}

impl PartialEq for Param {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Param::Float(s), Param::Float(other)) => s == other,
            (Param::Float(_), Param::ParameterExpression(_)) => false,
            (Param::ParameterExpression(_), Param::Float(_)) => false,
            (Param::ParameterExpression(s), Param::ParameterExpression(other)) => s.id == other.id,
            (Param::ParameterExpression(_), Param::Obj(_)) => false,
            (Param::Float(_), Param::Obj(_)) => false,
            (Param::Obj(_), Param::ParameterExpression(_)) => false,
            (Param::Obj(_), Param::Float(_)) => false,
            (Param::Obj(one), Param::Obj(other)) => Self::compare(one, other),
        }
    }
}

/// ParamExpIdentifier
#[derive(Debug, Clone)]
pub struct ParamExpRep {
    id: String,
    object: PyObject,
}

/// Temporary interpretation of Gate
#[derive(Debug, Clone)]
pub struct GateRep {
    object: PyObject,
    pub num_qubits: Option<usize>,
    pub num_clbits: Option<usize>,
    pub name: Option<String>,
    pub label: Option<String>,
    pub params: Vec<Param>,
}

impl FromPyObject<'_> for GateRep {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let num_qubits = match ob.getattr("num_qubits") {
            Ok(num_qubits) => num_qubits.extract::<usize>().ok(),
            Err(_) => None,
        };
        let num_clbits = match ob.getattr("num_clbits") {
            Ok(num_clbits) => num_clbits.extract::<usize>().ok(),
            Err(_) => None,
        };
        let name = match ob.getattr("name") {
            Ok(name) => name.extract::<String>().ok(),
            Err(_) => None,
        };
        let label = match ob.getattr("label") {
            Ok(label) => label.extract::<String>().ok(),
            Err(_) => None,
        };
        let params = match ob.getattr("params") {
            Ok(params) => params.extract::<Vec<Param>>().ok(),
            Err(_) => Some(vec![]),
        }
        .unwrap_or_default();
        Ok(Self {
            object: ob.into(),
            num_qubits,
            num_clbits,
            name,
            label,
            params,
        })
    }
}
impl IntoPy<PyObject> for GateRep {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.object
    }
}

/// Temporary interpretation of QuantumCircuit
#[derive(Debug, Clone)]
pub struct CircuitRep {
    object: PyObject,
    pub num_qubits: Option<usize>,
    pub num_clbits: Option<usize>,
    pub label: Option<String>,
    pub params: Vec<Param>,
    pub data: Vec<CircuitInstructionRep>,
}

impl FromPyObject<'_> for CircuitRep {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let num_qubits = match ob.getattr("num_qubits") {
            Ok(num_qubits) => num_qubits.extract::<usize>().ok(),
            Err(_) => None,
        };
        let num_clbits = match ob.getattr("num_clbits") {
            Ok(num_clbits) => num_clbits.extract::<usize>().ok(),
            Err(_) => None,
        };
        let label = match ob.getattr("label") {
            Ok(label) => label.extract::<String>().ok(),
            Err(_) => None,
        };
        let params = ob
            .getattr("parameters")?
            .getattr("data")?
            .extract::<Vec<Param>>()
            .unwrap_or_default();
        let data = match ob.getattr("data") {
            Ok(data) => data.extract::<Vec<CircuitInstructionRep>>().ok(),
            Err(_) => Some(vec![]),
        }
        .unwrap_or_default();
        Ok(Self {
            object: ob.into(),
            num_qubits,
            num_clbits,
            label,
            params,
            data,
        })
    }
}

impl IntoPy<PyObject> for CircuitRep {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.object
    }
}

// Temporary Representation of CircuitInstruction
#[derive(Debug, Clone)]
pub struct CircuitInstructionRep {
    operation: GateRep,
    qubits: Vec<PyObject>,
}

impl FromPyObject<'_> for CircuitInstructionRep {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let qubits = ob.getattr("qubits")?.extract::<Vec<PyObject>>()?;
        let operation = ob.getattr("operation")?.extract::<GateRep>()?;
        Ok(Self { operation, qubits })
    }
}

// Custom Types
type GraphType = DiGraph<NodeData, EdgeData>;
type KTIType = HashMap<Key, NodeIndex>;

#[pyclass(subclass, name = "BaseEquivalenceLibrary")]
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
    fn add_equivalence(&mut self, gate: GateRep, equivalent_circuit: CircuitRep) -> PyResult<()> {
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
    pub fn has_entry(&self, gate: GateRep) -> bool {
        let key = Key {
            name: gate.name.unwrap_or_default(),
            num_qubits: gate.num_qubits.unwrap_or_default(),
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
    fn set_entry(&mut self, gate: GateRep, entry: Vec<CircuitRep>) -> PyResult<()> {
        match self.set_entry_native(&gate, &entry) {
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
    pub fn get_entry(&self, py: Python<'_>, gate: GateRep) -> Vec<CircuitRep> {
        let key = Key {
            name: gate.name.unwrap_or_default(),
            num_qubits: gate.num_qubits.unwrap_or_default(),
        };
        let query_params = gate.params;

        self.get_equivalences(&key)
            .into_iter()
            .filter_map(|equivalence| rebind_equiv(py, equivalence, &query_params).ok())
            .collect()
    }

    #[getter]
    fn get_graph(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(graph) = &self.graph {
            Ok(graph.to_owned())
        } else {
            self.graph = Some(to_pygraph(py, &self._graph)?);
            Ok(self.graph.to_object(py))
        }
    }

    fn keys(&self) -> HashSet<Key> {
        self.key_to_node_index.keys().cloned().collect()
    }

    fn node_index(&self, key: Key) -> usize {
        self.key_to_node_index[&key].index()
    }
}

// Rust native methods
impl EquivalenceLibrary {
    /// Create a new node if key not found
    fn set_default_node(&mut self, key: Key) -> NodeIndex {
        if let Some(value) = self.key_to_node_index.get(&key) {
            *value
        } else {
            let node = self._graph.add_node(NodeData {
                key: key.to_owned(),
                equivs: vec![],
            });
            self.key_to_node_index.insert(key, node);
            node
        }
        // *self
        //     .key_to_node_index
        //     .entry(key.to_owned())
        //     .or_insert(self._graph.add_node(NodeData {
        //         key,
        //         equivs: vec![],
        //     }))
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
        gate: GateRep,
        equivalent_circuit: CircuitRep,
    ) -> Result<(), EquivalenceError> {
        raise_if_shape_mismatch(&gate, &equivalent_circuit)?;
        raise_if_param_mismatch(&gate.params, &equivalent_circuit.params)?;

        let key: Key = Key {
            name: gate.name.unwrap_or_default(),
            num_qubits: gate.num_qubits.unwrap_or_default(),
        };
        let equiv = Equivalence {
            params: gate.params,
            circuit: equivalent_circuit.to_owned(),
        };

        let target = self.set_default_node(key);
        if let Some(node) = self._graph.node_weight_mut(target) {
            node.equivs.push(equiv.to_owned());
        }
        let sources: HashSet<Key> =
            HashSet::from_iter(equivalent_circuit.data.iter().map(|inst| Key {
                name: inst.operation.name.to_owned().unwrap_or_default(),
                num_qubits: inst.qubits.len(),
            }));
        let edges = Vec::from_iter(sources.iter().map(|source| {
            (
                self.set_default_node(source.to_owned()),
                target,
                EdgeData {
                    index: self.rule_id as u32,
                    num_gates: sources.len(),
                    rule: equiv.to_owned(),
                    source: source.to_owned(),
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
        gate: &GateRep,
        entry: &Vec<CircuitRep>,
    ) -> Result<(), EquivalenceError> {
        for equiv in entry {
            raise_if_shape_mismatch(gate, equiv)?;
            raise_if_param_mismatch(&gate.params, &equiv.params)?;
        }

        let key = Key {
            name: gate.name.to_owned().unwrap_or_default(),
            num_qubits: gate.num_qubits.unwrap_or_default(),
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
            self.add_equiv(gate.to_owned(), equiv.to_owned())?
        }
        self.graph = None;
        Ok(())
    }

    /// Get all the equivalences for the given key
    fn get_equivalences(&self, key: &Key) -> Vec<Equivalence> {
        if let Some(key_in) = self.key_to_node_index.get(key) {
            self._graph[*key_in].equivs.to_owned()
        } else {
            vec![]
        }
    }
}

fn raise_if_param_mismatch(
    gate_params: &[Param],
    circuit_parameters: &[Param],
) -> Result<(), EquivalenceError> {
    let uid_gate_parameters: HashSet<String> = gate_params
        .iter()
        .filter_map(|x| match x {
            Param::ParameterExpression(param) => Some(param.id.to_string()),
            Param::Float(_) => None,
            Param::Obj(_) => None,
        })
        .collect();
    let uid_circuit_parameters: HashSet<String> = circuit_parameters
        .iter()
        .filter_map(|x| match x {
            Param::ParameterExpression(param) => Some(param.id.to_string()),
            Param::Float(_) => None,
            Param::Obj(_) => None,
        })
        .collect();
    if uid_gate_parameters != uid_circuit_parameters {
        return Err(EquivalenceError::new_err(format!(
            "Cannot add equivalence between circuit and gate \
            of different parameters. Gate params: {:#?}. \
            Circuit params: {:#?}.",
            gate_params, circuit_parameters
        )));
    }
    Ok(())
}

fn raise_if_shape_mismatch(gate: &GateRep, circuit: &CircuitRep) -> Result<(), EquivalenceError> {
    if gate.num_qubits != circuit.num_qubits || gate.num_clbits != circuit.num_clbits {
        return Err(EquivalenceError::new_err(format!(
            "Cannot add equivalence between circuit and gate \
            of different shapes. Gate: {} qubits and {} clbits. \
            Circuit: {} qubits and {} clbits.",
            gate.num_qubits.unwrap_or_default(),
            gate.num_clbits.unwrap_or_default(),
            circuit.num_qubits.unwrap_or_default(),
            circuit.num_clbits.unwrap_or_default()
        )));
    }
    Ok(())
}

fn rebind_equiv(
    py: Python<'_>,
    equiv: Equivalence,
    query_params: &[Param],
) -> PyResult<CircuitRep> {
    let (equiv_params, equiv_circuit) = (equiv.params, equiv.circuit);
    let param_map: Vec<(Param, Param)> = equiv_params
        .into_iter()
        .filter_map(|param| {
            println!(
                "{:#?}: is expr: {}",
                param,
                matches!(param, Param::ParameterExpression(_))
            );
            if matches!(param, Param::ParameterExpression(_)) {
                Some(param)
            } else {
                None
            }
        })
        .zip(query_params.iter().cloned())
        .collect();
    let dict = param_map.as_slice().into_py_dict_bound(py);
    let kwargs = [("inplace", false), ("flat_input", true)].into_py_dict_bound(py);
    let new_equiv =
        equiv_circuit
            .object
            .call_method_bound(py, "assign_parameters", (dict,), Some(&kwargs))?;
    new_equiv.extract::<CircuitRep>(py)
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

fn to_pygraph<N, E>(py: Python<'_>, pet_graph: &DiGraph<N, E>) -> PyResult<PyObject>
where
    N: IntoPy<PyObject> + Clone,
    E: IntoPy<PyObject> + Clone,
{
    let graph = PYDIGRAPH.get_bound(py).call0()?;
    let node_weights = pet_graph.node_weights();
    for node in node_weights {
        graph.call_method1("add_node", (node.to_owned(),))?;
    }
    let edge_weights = pet_graph.edge_indices().map(|edge| {
        (
            pet_graph.edge_endpoints(edge).unwrap(),
            pet_graph.edge_weight(edge).unwrap(),
        )
    });
    for ((source, target), weight) in edge_weights {
        graph.call_method1(
            "add_edge",
            (source.index(), target.index(), weight.to_owned()),
        )?;
    }
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
