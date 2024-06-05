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

use std::{error::Error, fmt::Display};

use exceptions::CircuitError;
use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use rustworkx_core::petgraph::{
    graph::{DiGraph, EdgeIndex, NodeIndex},
    visit::EdgeRef,
};



mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.exceptions, CircuitError}
}
// Custom Structs

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub name: String,
    pub num_qubits: usize,
}

#[derive(Debug, Clone)]
pub struct Equivalence {
    pub params: Vec<Param>,
    pub circuit: CircuitRep,
}

// Temporary interpretation of Param
#[derive(Debug, Clone, FromPyObject)]
pub enum Param {
    Float(f64),
    ParameterExpression(PyObject),
}

impl Param {
    fn compare(one: &PyObject, other: &PyObject) -> bool {
        Python::with_gil(|py| -> PyResult<bool> {
            let other_bound = other.bind(py);
            other_bound.eq(one)
        })
        .unwrap()
    }
}

impl PartialEq for Param {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Param::Float(s), Param::Float(other)) => s == other,
            (Param::Float(_), Param::ParameterExpression(_)) => false,
            (Param::ParameterExpression(_), Param::Float(_)) => false,
            (Param::ParameterExpression(s), Param::ParameterExpression(other)) => {
                Self::compare(s, other)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeData {
    key: Key,
    equivs: Vec<Equivalence>,
}

#[derive(Debug, Clone)]
pub struct EdgeData {
    pub index: u32,
    pub num_gates: usize,
    pub rule: Equivalence,
    pub source: Key,
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

/// Temporary interpretation of Gate
#[derive(Debug, Clone)]
pub struct CircuitRep {
    object: PyObject,
    pub num_qubits: Option<usize>,
    pub num_clbits: Option<usize>,
    pub label: Option<String>,
    pub params: Vec<Param>,
    pub data: Vec<GateRep>,
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
        let params = match ob.getattr("params") {
            Ok(params) => params.extract::<Vec<Param>>().ok(),
            Err(_) => Some(vec![]),
        }
        .unwrap_or_default();
        let data = match ob.getattr("data") {
            Ok(data) => data.extract::<Vec<GateRep>>().ok(),
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

// Custom Types
type GraphType = DiGraph<NodeData, EdgeData>;
type KTIType = HashMap<Key, NodeIndex>;

#[pyclass]
#[derive(Debug, Clone)]
pub struct EquivalenceLibrary {
    graph: GraphType,
    key_to_node_index: KTIType,
    rule_id: usize,
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
                graph: base.graph.clone(),
                key_to_node_index: base.key_to_node_index.clone(),
                rule_id: base.rule_id,
            }
        } else {
            Self {
                graph: GraphType::new(),
                key_to_node_index: KTIType::new(),
                rule_id: 0_usize,
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
            name: gate.name.unwrap(),
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
    pub fn get_entry(&self, _gate: GateRep) {
        todo!()
    }
}

// Rust native methods
impl EquivalenceLibrary {
    /// Create a new node if key not found
    fn set_default_node(&mut self, key: Key) -> NodeIndex {
        *self
            .key_to_node_index
            .entry(key.to_owned())
            .or_insert(self.graph.add_node(NodeData {
                key,
                equivs: vec![],
            }))
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
            name: gate.name.unwrap(),
            num_qubits: gate.num_qubits.unwrap(),
        };
        let equiv = Equivalence {
            params: gate.params,
            circuit: equivalent_circuit.to_owned(),
        };

        let target = self.set_default_node(key);
        let node = self.graph.node_weight_mut(target).unwrap();
        node.equivs.push(equiv.to_owned());

        let sources: HashSet<Key> =
            HashSet::from_iter(equivalent_circuit.data.iter().map(|inst| Key {
                name: inst.name.to_owned().unwrap(),
                num_qubits: inst.num_qubits.unwrap_or_default(),
            }));
        let edges = Vec::from_iter(sources.iter().map(|key| {
            (
                self.set_default_node(key.to_owned()),
                target,
                EdgeData {
                    index: self.rule_id as u32,
                    num_gates: sources.len(),
                    rule: equiv.to_owned(),
                    source: key.to_owned(),
                },
            )
        }));
        for edge in edges {
            self.graph.add_edge(edge.0, edge.1, edge.2);
        }
        self.rule_id += 1;
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
            name: gate.name.to_owned().unwrap(),
            num_qubits: gate.num_qubits.unwrap_or_default(),
        };
        let node_index = self.set_default_node(key);

        let graph_ind = &mut self.graph.node_weight_mut(node_index).unwrap();
        graph_ind.equivs.clear();

        let edges: Vec<EdgeIndex> = self
            .graph
            .edges_directed(node_index, rustworkx_core::petgraph::Direction::Incoming)
            .map(|x| x.id())
            .collect();
        for edge in edges {
            self.graph.remove_edge(edge);
        }
        for equiv in entry {
            self.add_equiv(gate.to_owned(), equiv.to_owned())?
        }
        Ok(())
    }

    /// Get all the equivalences for the given key
    fn get_equivalences(&self, key: &Key) -> Vec<Equivalence> {
        if let Some(key_in) = self.key_to_node_index.get(key) {
            self.graph[*key_in].equivs.to_owned()
        } else {
            vec![]
        }
    }
}

fn raise_if_param_mismatch(
    gate_params: &[Param],
    circuit_parameters: &[Param],
) -> Result<(), EquivalenceError> {
    let gate_parameters: Vec<&Param> = gate_params
        .iter()
        .filter(|x| matches!(x, Param::ParameterExpression(_)))
        .collect();
    if circuit_parameters
        .iter()
        .any(|x| gate_parameters.contains(&x))
    {
        return Err(EquivalenceError::new_err(format!(
            "Cannot add equivalence between circuit and gate \
            of different parameters. Gate params: {:?}. \
            Circuit params: {:?}.",
            gate_parameters, circuit_parameters
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
