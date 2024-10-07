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
use pyo3::types::{PyList, PyString, PyTuple};
use smallvec::{smallvec, SmallVec};
use std::borrow::Borrow;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{multiply_param, Param, StandardGate};
use qiskit_circuit::Qubit;

use rustiq_core::structures::{CliffordGate, Metric, PauliLike, PauliSet};
use rustiq_core::synthesis::pauli_network::greedy_pauli_network;

use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::prelude::StableDiGraph;
use rustworkx_core::petgraph::Incoming;

/// A Qiskit gate
pub type QiskitGate = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

/// Expands the sparse pauli string representation to the full representation.
///
/// For example: for the input `sparse_pauli = "XY", qubits = [1, 3], num_qubits = 6`,
/// the function returns `"IXIYII"`.
fn expand_pauli(sparse_pauli: String, qubits: Vec<u32>, num_qubits: usize) -> String {
    let mut v: Vec<char> = vec!["I".parse().unwrap(); num_qubits];
    for (i, q) in qubits.iter().enumerate() {
        v[*q as usize] = sparse_pauli.chars().nth(i).unwrap();
    }
    v.into_iter().collect()
}

/// Return the Qiskit's gate corresponding to the given Rustiq's Clifford gate.
fn qiskit_clifford_gate(rustiq_gate: &CliffordGate) -> QiskitGate {
    match rustiq_gate {
        CliffordGate::CNOT(i, j) => (
            StandardGate::CXGate,
            smallvec![],
            smallvec![Qubit(*i as u32), Qubit(*j as u32)],
        ),
        CliffordGate::CZ(i, j) => (
            StandardGate::CZGate,
            smallvec![],
            smallvec![Qubit(*i as u32), Qubit(*j as u32)],
        ),
        CliffordGate::H(i) => (
            StandardGate::HGate,
            smallvec![],
            smallvec![Qubit(*i as u32)],
        ),
        CliffordGate::S(i) => (
            StandardGate::SGate,
            smallvec![],
            smallvec![Qubit(*i as u32)],
        ),
        CliffordGate::Sd(i) => (
            StandardGate::SdgGate,
            smallvec![],
            smallvec![Qubit(*i as u32)],
        ),
        CliffordGate::SqrtX(i) => (
            StandardGate::SXGate,
            smallvec![],
            smallvec![Qubit(*i as u32)],
        ),
        CliffordGate::SqrtXd(i) => (
            StandardGate::SXdgGate,
            smallvec![],
            smallvec![Qubit(*i as u32)],
        ),
    }
}

/// Return the Qiskit rotation gate corresponding to the single-qubit Pauli rotation.
/// # Arguments
///
/// * py: a GIL handle, needed to negate rotation parameters in Python space.
/// * paulis: Rustiq's data structure storing pauli rotations.
/// * i: index of the single-qubit Pauli rotation.
/// * angle: - Qiskit's rotation angle.
fn qiskit_rotation_gate(py: Python, paulis: &PauliSet, i: usize, angle: &Param) -> QiskitGate {
    let (phase, pauli_str) = paulis.get(i);
    for (q, c) in pauli_str.chars().enumerate() {
        if c != 'I' {
            println!(
                "-- hit pauli {:?}, q = {:?}. c = {:?}, phase = {:?}",
                i, q, c, phase
            );
            let standard_gate = match c {
                'X' => StandardGate::RXGate,
                'Y' => StandardGate::RYGate,
                'Z' => StandardGate::RZGate,
                _ => panic!(),
            };
            let param = match phase {
                false => angle.clone(),
                true => multiply_param(angle, -1.0, py),
            };
            return (standard_gate, smallvec![param], smallvec![Qubit(q as u32)]);
        }
    }
    unreachable!()
}

/// A DAG that stores ordered Paulis, up to commutativity.
struct CommutativityDag {
    /// Rustworkx's DAG
    dag: StableDiGraph<usize, ()>,
}

impl CommutativityDag {
    /// Construct a DAG based on the commutativity relations between paulis.
    pub fn from_paulis(paulis: &PauliSet) -> Self {
        let mut dag = StableDiGraph::<usize, ()>::new();

        let node_indices: Vec<NodeIndex> = (0..paulis.len()).map(|i| dag.add_node(i)).collect();

        for i in 0..paulis.len() {
            let pauli_i = paulis.get_as_pauli(i);
            for j in i + 1..paulis.len() {
                let pauli_j = paulis.get_as_pauli(j);

                if !pauli_i.commutes(&pauli_j) {
                    dag.add_edge(node_indices[i], node_indices[j], ());
                }
            }
        }

        CommutativityDag { dag }
    }

    /// Return whether the given node is a front node (i.e. has no predecessors).
    pub fn is_front_node(&self, index: usize) -> bool {
        self.dag
            .neighbors_directed(NodeIndex::new(index), Incoming)
            .next()
            .is_none()
    }

    /// Remove node from the DAG.
    pub fn remove_node(&mut self, index: usize) {
        self.dag.remove_node(NodeIndex::new(index));
    }
}

/// Return a Qiskit circuit with Clifford gates and rotations.
///
/// The rotations are assumed to be unordered.
///
/// # Arguments
///
/// * py: a GIL handle, needed to negate rotation parameters in Python space.
/// * gates: the sequence of Rustiq's Clifford gates returned by Rustiq's
///     pauli network synthesis algorithm.
/// * paulis: Rustiq's data structure storing the pauli rotations.
/// * angles: Qiskit's rotation angles corresponding to the pauli rotations.
fn inject_rotations_unordered(
    py: Python,
    gates: &Vec<CliffordGate>,
    paulis: &PauliSet,
    angles: &[Param],
) -> Vec<QiskitGate> {
    let mut out_gates: Vec<QiskitGate> = Vec::with_capacity(gates.len() + paulis.len());

    let mut cur_paulis = paulis.clone();
    let mut hit_paulis: Vec<bool> = vec![false; cur_paulis.len()];

    // check which paulis are hit at the very start
    for i in 0..cur_paulis.len() {
        if !hit_paulis[i] && cur_paulis.support_size(i) == 1 {
            out_gates.push(qiskit_rotation_gate(py, &cur_paulis, i, &angles[i]));
            hit_paulis[i] = true;
        }
    }

    for gate in gates {
        out_gates.push(qiskit_clifford_gate(gate));

        cur_paulis.conjugate_with_gate(gate);

        // check which paulis are hit now
        for i in 0..cur_paulis.len() {
            if !hit_paulis[i] && cur_paulis.support_size(i) == 1 {
                out_gates.push(qiskit_rotation_gate(py, &cur_paulis, i, &angles[i]));
                hit_paulis[i] = true;
            }
        }
    }

    out_gates
}

/// Return a Qiskit circuit with Clifford gates and rotations.
///
/// The rotations are assumed to be ordered (up to commutativity).
///
/// # Arguments
///
/// * py: a GIL handle, needed to negate rotation parameters in Python space.
/// * gates: the sequence of Rustiq's Clifford gates returned by Rustiq's
///     pauli network synthesis algorithm.
/// * paulis: Rustiq's data structure storing the pauli rotations.
/// * angles: Qiskit's rotation angles corresponding to the pauli rotations.
fn inject_rotations_ordered(
    py: Python,
    gates: &Vec<CliffordGate>,
    paulis: &PauliSet,
    angles: &[Param],
) -> Vec<QiskitGate> {
    let mut out_gates: Vec<QiskitGate> = Vec::with_capacity(gates.len() + paulis.len());

    let mut cur_paulis = paulis.clone();
    let mut hit_paulis: Vec<bool> = vec![false; cur_paulis.len()];

    let mut dag = CommutativityDag::from_paulis(paulis);

    // check which paulis are hit at the very start
    for i in 0..cur_paulis.len() {
        if !hit_paulis[i] && cur_paulis.support_size(i) == 1 && dag.is_front_node(i) {
            out_gates.push(qiskit_rotation_gate(py, &cur_paulis, i, &angles[i]));
            hit_paulis[i] = true;
            dag.remove_node(i);
        }
    }

    for gate in gates {
        out_gates.push(qiskit_clifford_gate(gate));

        cur_paulis.conjugate_with_gate(gate);

        // check which paulis are hit now
        for i in 0..cur_paulis.len() {
            if !hit_paulis[i] && cur_paulis.support_size(i) == 1 && dag.is_front_node(i) {
                out_gates.push(qiskit_rotation_gate(py, &cur_paulis, i, &angles[i]));
                hit_paulis[i] = true;
                dag.remove_node(i);
            }
        }
    }

    out_gates
}

/// Calls Rustiq's pauli network synthesis algorithm and returns the
/// Qiskit circuit data with Clifford gates and rotations.
///
/// # Arguments
///
/// * py: a GIL handle, needed to negate rotation parameters in Python space.
/// * num_qubits: total number of qubits.
/// * pauli_network: pauli network represented in sparse format. It's a list
///     of triples such as `[("XX", [0, 3], theta), ("ZZ", [0, 1], 0.1)]`.
/// * preserve_order: whether the order of paulis should be preserved, up to
///     commutativity.
#[pyfunction]
#[pyo3(signature = (num_qubits, pauli_network, preserve_order=true))]
pub fn pauli_network_synthesis(
    py: Python,
    num_qubits: usize,
    pauli_network: &Bound<PyList>,
    preserve_order: bool,
) -> PyResult<CircuitData> {
    let mut operator_sequence: Vec<String> = Vec::new();
    let mut params: Vec<Param> = Vec::new();

    for item in pauli_network {
        let tuple = item.downcast::<PyTuple>()?;
        let inner = tuple.borrow();

        let ss: String = inner.get_item(0)?.downcast::<PyString>()?.extract()?;
        let param: Param = inner.get_item(2)?.extract()?;
        let vv = inner.get_item(1)?;
        let ww: Vec<u32> = vv
            .downcast::<PyList>()?
            .iter()
            .map(|v| v.extract())
            .collect::<PyResult<_>>()?;

        operator_sequence.push(expand_pauli(ss, ww, num_qubits));
        params.push(param);
    }

    // todo: maybe we can immediately create pauli set rather than operator_sequence first
    let mut bucket = PauliSet::from_slice(&operator_sequence);
    let circuit = greedy_pauli_network(&mut bucket, &Metric::COUNT, true, 0, false, false);

    let gates = match preserve_order {
        false => inject_rotations_unordered(py, &circuit.gates, &bucket, &params),
        true => inject_rotations_ordered(py, &circuit.gates, &bucket, &params),
    };

    CircuitData::from_standard_gates(py, num_qubits as u32, gates, Param::Float(0.0))
}

pub fn evolution(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pauli_network_synthesis, m)?)?;
    Ok(())
}
