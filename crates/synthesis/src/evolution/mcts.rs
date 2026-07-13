// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate, multiply_param, radd_param};
use qiskit_quantum_info::clifford::PauliList;

use rustworkx_core::petgraph::Direction::Outgoing;
use rustworkx_core::petgraph::Incoming;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::prelude::StableDiGraph;

use smallvec::{SmallVec, smallvec};

use std::f64::consts::SQRT_2;
use std::fmt;

use crate::clifford::greedy_synthesis::resynthesize_clifford_circuit;
use crate::evolution::EvolutionSynthesisError;
use crate::evolution::chunks::{ALL_CHUNKS, REDUCING_CHUNKS, SUPPORT_DELTA};

/// The multiplicative scaling parameter used in the MCTS algorithm.
const MCTS_PARAM: f64 = SQRT_2;

/// Sequence of standard gates constructed by the algorithm, including
/// both Clifford gates and single-qubit rotations.
type GateSequence = Vec<(StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>;

/// A particular point during Pauli network synthesis.
#[derive(Clone)]
struct PauliSynthesisState {
    /// The tableau storing Pauli rotations corresponding to this state
    /// (the initial tableau conjugated by Clifford gates leading to this state).
    /// The Paulis that have already been synthesized are stored as ``None`` in
    /// ``in_degrees``.
    tab: PauliList,

    /// Number of processed Pauli rotations.
    num_processed: usize,

    /// Sequence of gates (including Clifford and rotation gates)
    /// leading to this state.
    gate_sequence: GateSequence,

    /// Number of unprocessed predecessors for each node in the commutativity DAG
    /// (with ``None`` nodes considered as removed).
    in_degrees: Vec<Option<usize>>,
}

impl fmt::Display for PauliSynthesisState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tab = {}, num_processed = {}, in_degrees = {:?}",
            self.tab, self.num_processed, self.in_degrees
        )
    }
}

/// A node in the MCTS tree.
struct MctsNode {
    /// Number of times this node was visited across simulations.
    ni: usize,

    /// Cumulative cost (e.g. number of CX=gates) from root to terminal node
    /// across simulations visiting this node.
    qi: usize,

    /// Parent node in the MCTS tree.
    parent: Option<usize>,

    /// Child nodes in the MCTS tree.
    children: Vec<usize>,

    /// The corresponding state of the Pauli network synthesis algorithm.
    synthesis_state: PauliSynthesisState,

    /// Unexplored actions at this node. Correspond to Pauli indices that have not yet been
    /// explored by the MCTS algorithm when starting from this state.
    unexplored_actions: Vec<usize>,
}

/// Count the number of CX-gates in [GateSequence].
fn cx_count(gate_seq: &GateSequence) -> usize {
    gate_seq
        .iter()
        .filter(|(gate, _, _)| *gate == StandardGate::CX)
        .count()
}

/// Count the number of CX-gates in [GateSequence].
fn cx_count_with_swaps(gate_seq: &GateSequence) -> usize {
    // This function may also be called on the resynthesized Clifford circuit,
    // and so also considers SWAP gates.
    gate_seq
        .iter()
        .map(|(gate, _, _)| match gate {
            StandardGate::CX => 1,
            StandardGate::Swap => 3,
            _ => 0,
        })
        .sum()
}

/// Score of applying a 2-qubit Clifford circuit on a pair of qubits in the PauliList.
/// Larger scores are better.
type Score = (isize, isize);

/// Count how many times each of the 16 possible 2-qubit Paulis appear for a fixed
/// pair `(ctrl, targ)`.
fn count_active_pairs(
    synthesis_state: &PauliSynthesisState,
    ctrl: usize,
    targ: usize,
) -> [u32; 16] {
    let mut pair_counts = [0u32; 16];
    for pauli_idx in 0..synthesis_state.tab.num_paulis {
        if synthesis_state.in_degrees[pauli_idx].is_some() {
            let pair_idx = synthesis_state.tab.pauli_pair_index(pauli_idx, ctrl, targ);
            pair_counts[pair_idx] += 1;
        }
    }
    pair_counts
}

/// Computes the score of applying a Clifford circuit represented by `chunk_idx` on a pair of
/// qubits, where `pair_counts` represent the numbers of different types of 2-qubit Paulis
/// (obtained by restricting the active Pauli roations to this pair of qubits).
fn compute_score(pair_counts: &[u32; 16], chunk_idx: usize) -> Score {
    let deltas = &SUPPORT_DELTA[chunk_idx];
    let mut total_decrease: isize = 0;
    let mut num_decreased: isize = 0;

    for (pair_idx, &pair_count) in pair_counts.iter().enumerate() {
        if pair_count == 0 {
            continue;
        }
        let delta = deltas[pair_idx] as isize;
        total_decrease -= delta * (pair_count as isize);
        num_decreased += ((delta < 0) as isize) * (pair_count as isize);
    }
    (total_decrease, num_decreased)
}

/// Synthesizes a given Pauli, that is finds a sequence of Clifford gates that brings this Pauli to a single-qubit
/// Pauli rotation (note that the same sequence can bring other Paulis to single-qubit Paulis as well).
/// Updates the synthesis state in-place, adjusting the Pauli tableau (conjugating by Clifford gates) and extending
/// the gate sequence.
fn synthesize_pauli(synthesis_state: &mut PauliSynthesisState, ndx: usize) {
    let mut prev_support_size: Option<usize> = None;

    loop {
        let support_size = synthesis_state.tab.get_pauli_support_size(ndx);
        assert!(support_size >= 1);

        if let Some(old_support_size) = prev_support_size {
            assert!(support_size < old_support_size);
        }
        prev_support_size = Some(support_size);

        // We have successfully reduced this Pauli to a single-qubit rotation.
        if support_size == 1 {
            return;
        }

        // loop to cycle over all qubit indices to get all combinations of control and target
        // for now, keep the same order as in python code
        let support = synthesis_state.tab.get_pauli_support(ndx);

        let mut best_score = (isize::MIN, isize::MIN);
        let mut best_ctrl = 0;
        let mut best_targ = 0;
        let mut best_chunk_idx = 0;
        for ii in 0..support.len() {
            for jj in ii + 1..support.len() {
                let ctrl = support[ii];
                let targ = support[jj];
                // get the index 0..16
                let pair_idx = synthesis_state.tab.pauli_pair_index(ndx, ctrl, targ);

                // count the numbers of different 2-qubit Paulis on (ctrl, targ)
                let pair_counts = count_active_pairs(synthesis_state, ctrl, targ);

                // get the indices of chunks that can reduce this
                assert!(!REDUCING_CHUNKS[pair_idx].is_empty());
                for chunk_idx in REDUCING_CHUNKS[pair_idx] {
                    let score = compute_score(&pair_counts, *chunk_idx);
                    if score > best_score {
                        best_score = score;
                        best_chunk_idx = *chunk_idx;
                        best_ctrl = ctrl;
                        best_targ = targ;
                    }
                }
            }
        }

        // Apply the best chunk
        for (gate, qubits) in ALL_CHUNKS[best_chunk_idx] {
            let mapped_qubits: SmallVec<[Qubit; 2]> = qubits
                .iter()
                .map(|q| match q {
                    0 => Qubit(best_ctrl as u32),
                    1 => Qubit(best_targ as u32),
                    _ => {
                        unreachable!("can only have 0/1");
                    }
                })
                .collect();

            match gate {
                StandardGate::H => {
                    synthesis_state.tab.append_h(mapped_qubits[0].index());
                }
                StandardGate::S => {
                    synthesis_state.tab.append_s(mapped_qubits[0].index());
                }
                StandardGate::SX => {
                    synthesis_state.tab.append_sx(mapped_qubits[0].index());
                }
                StandardGate::CX => {
                    synthesis_state
                        .tab
                        .append_cx(mapped_qubits[0].index(), mapped_qubits[1].index());
                }
                _ => {
                    panic!("should only have s/sx/h/cx gates");
                }
            }
            synthesis_state
                .gate_sequence
                .push((*gate, smallvec![], mapped_qubits));
        }
    }
}

/// Updates in-degree in-place with newly processed nodes.
fn update_in_degrees(
    dag: &StableDiGraph<usize, ()>,
    in_degrees: &mut [Option<usize>],
    new_processed: &[usize],
) {
    for i in new_processed {
        for n in dag.neighbors_directed(NodeIndex::new(*i), Outgoing) {
            let degree = in_degrees[n.index()]
                .as_mut()
                .expect("the successor node should not be processed");
            assert_ne!(*degree, 0);
            *degree -= 1;
        }
        // println!("=> Marking pauli idx = {:?} as processed", *i);
        in_degrees[*i] = None;
    }
}

/// Given a DAG, and a list of in-degrees (see the algorithm for detail),
/// computes frontier nodes.
fn compute_frontier_nodes(
    dag: &StableDiGraph<usize, ()>,
    in_degrees: &[Option<usize>],
) -> Vec<usize> {
    dag.node_indices()
        .filter(|n| in_degrees[n.index()] == Some(0))
        .map(|n| n.index())
        .collect()
}

/// The main algorithm.
struct MctsAlgorithm {
    /// Total number of Pauli rotations to synthesize.
    num_paulis: usize,

    /// Paulis to be synthesized.
    paulis: PauliList,

    /// Parameters for rotation angles.
    angles: Vec<Param>,

    /// MCTS nodes (forming a tree).
    mcts_nodes: Vec<MctsNode>,

    /// Anti-commutativity DAG for Pauli rotations.
    dag: StableDiGraph<usize, ()>,

    /// Global phase of the circuit.
    global_phase: Param,
}

impl MctsAlgorithm {
    /// Creates a tree with a single root node
    fn new(paulis: &PauliList, angles: &[Param], preserve_order: bool) -> Self {
        let num_paulis = paulis.num_paulis;

        // Create anti-commutativity DAG.
        let mut dag = StableDiGraph::<usize, ()>::new();
        let node_indices: Vec<NodeIndex> = (0..num_paulis).map(|i| dag.add_node(i)).collect();

        if preserve_order {
            for i in 0..num_paulis {
                for j in i + 1..num_paulis {
                    if !paulis.commute(i, j) {
                        dag.add_edge(node_indices[i], node_indices[j], ());
                    }
                }
            }
        }

        Self {
            global_phase: Param::Float(0.0),
            num_paulis,
            paulis: paulis.clone(),
            angles: angles.to_vec(),
            mcts_nodes: vec![],
            dag,
        }
    }

    /// Adds the root MCTS node.
    /// The root includes the tableau after processing all-identity and all active single-qubit rotations
    /// in the original tableau.
    fn add_root_mcts_node(&mut self, tab: &PauliList) {
        let mut in_degrees: Vec<Option<usize>> = vec![None; self.num_paulis];
        self.dag.node_indices().for_each(|i| {
            in_degrees[i.index()] = Some(self.dag.neighbors_directed(i, Incoming).count());
        });

        let mut synthesis_state = PauliSynthesisState {
            tab: tab.clone(),
            num_processed: 0,
            gate_sequence: vec![],
            in_degrees,
        };

        self.global_phase = self.process_all_identity_paulis(&mut synthesis_state);
        self.process_synthesized_paulis(&mut synthesis_state);

        let unexplored_actions = compute_frontier_nodes(&self.dag, &synthesis_state.in_degrees)
            .iter()
            .rev()
            .cloned()
            .collect();

        let node = MctsNode {
            ni: 0,
            qi: 0,
            parent: None,
            children: vec![],
            synthesis_state,
            unexplored_actions,
        };
        self.mcts_nodes.push(node);
    }

    fn upper_confidence_bound(&self, node_idx: usize) -> f64 {
        let node = &self.mcts_nodes[node_idx];

        if node.ni == 0 {
            // it has been completely unexplored
            f64::INFINITY
        } else {
            match node.parent {
                None => {
                    // it is a root
                    -((node.qi as f64) / (node.ni as f64))
                }
                Some(parent_idx) => {
                    let parent_node = &self.mcts_nodes[parent_idx];
                    -((node.qi as f64) / (node.ni as f64))
                        + MCTS_PARAM * ((parent_node.ni as f64).ln() / (node.ni as f64)).sqrt()
                }
            }
        }
    }

    /// function to backpropagate estimated value from the terminal state up through the tree.
    fn backpropagate(&mut self, leaf: usize, value: usize) {
        let mut node = leaf;
        loop {
            self.mcts_nodes[node].ni += 1;
            self.mcts_nodes[node].qi += value;
            match self.mcts_nodes[node].parent {
                Some(parent) => node = parent,
                None => break,
            }
        }
    }

    /// The main function which runs everything.
    fn run(&mut self, num_sims: usize) -> (GateSequence, Param) {
        // Add the root state (this also removes all-identity and synthesizable 1-qubit rotations).
        self.add_root_mcts_node(&self.paulis.clone());

        // On the first iteration, runs rollout (greedy synthesis) starting at the root node.
        // For consecutive iterations (if any): each simulation either selects an existing MCTS state
        // with unexplored actions or creates a new MCTS state, and calls rollout (greedy synthesis)
        // starting at this state.
        let best_solution = std::iter::once(self.rollout_policy(0))
            .map(|solution| {
                let value = cx_count(&solution);
                (solution, value)
            })
            .chain((0..num_sims).map(|_| {
                let leaf_node_id = self.tree_policy();
                let solution = self.rollout_policy(leaf_node_id);
                let value = cx_count(&solution);
                self.backpropagate(leaf_node_id, value);
                (solution, value)
            }))
            .min_by_key(|(_solution, value)| *value)
            .map(|(solution, _value)| solution)
            .expect("we should have at least one solution");

        (best_solution, self.global_phase.clone())
    }

    /// Implement the "tree policy": find the best MCTS node to start the rollout from.
    fn tree_policy(&mut self) -> usize {
        let mut mcts_node_id = 0; // root
        loop {
            // If all the Paulis have been processed, return the current id.
            if self.mcts_nodes[mcts_node_id].synthesis_state.num_processed == self.num_paulis {
                return mcts_node_id;
            }

            // Node includes unexplored actions (Paulis that can be immediately synthesized).
            // We will create a new MCTS by synthesizing one of the unexplored Paulis and removing
            // newly synthesized rotations.
            if !self.mcts_nodes[mcts_node_id].unexplored_actions.is_empty() {
                // Choose one of the unprocessed paulis in the front layer.
                let chosen_pauli_idx = self.mcts_nodes[mcts_node_id]
                    .unexplored_actions
                    .last()
                    .unwrap();

                // Synthesize it.
                let mut synthesis_state = self.mcts_nodes[mcts_node_id].synthesis_state.clone();
                synthesize_pauli(&mut synthesis_state, *chosen_pauli_idx);
                self.process_synthesized_paulis(&mut synthesis_state);

                let unexplored_actions =
                    compute_frontier_nodes(&self.dag, &synthesis_state.in_degrees)
                        .iter()
                        .rev()
                        .cloned()
                        .collect();

                // Create the child mcts node with the result of the above synthesis.
                let child_node = MctsNode {
                    ni: 0,
                    qi: 0,
                    parent: Some(mcts_node_id),
                    children: vec![],
                    synthesis_state,
                    unexplored_actions,
                };

                // Add child to tree and remove the synthesized pauli from the set of unexplored actions.
                self.mcts_nodes.push(child_node);
                let child_node_id = self.mcts_nodes.len() - 1;
                self.mcts_nodes[mcts_node_id].children.push(child_node_id);
                self.mcts_nodes[mcts_node_id].unexplored_actions.pop();
                return child_node_id;
            };

            // All the children have been explored; choose the child with highest UCT score
            // (and proceed recursively examining this child).
            mcts_node_id = *self.mcts_nodes[mcts_node_id]
                .children
                .iter()
                .max_by(|&&a, &&b| {
                    self.upper_confidence_bound(a)
                        .partial_cmp(&self.upper_confidence_bound(b))
                        .unwrap()
                })
                .unwrap();
        }
    }

    /// Finds all-identity paulis and updates the synthesis state accordingly.
    /// Called once at the start of the algorithm.
    /// Returns the global phase update.
    fn process_all_identity_paulis(&self, synthesis_state: &mut PauliSynthesisState) -> Param {
        let mut global_phase = Param::Float(0.0);
        let mut new_processed: Vec<usize> = Vec::new();
        for pauli_idx in 0..self.num_paulis {
            if synthesis_state.tab.get_pauli_support_size(pauli_idx) == 0 {
                new_processed.push(pauli_idx);
                synthesis_state.num_processed += 1;
                global_phase =
                    radd_param(global_phase, multiply_param(&self.angles[pauli_idx], -0.5));
            }
        }
        if !new_processed.is_empty() {
            update_in_degrees(&self.dag, &mut synthesis_state.in_degrees, &new_processed);
        }
        global_phase
    }

    /// Recursively find synthesized paulis and update the synthesis state accordingly.
    fn process_synthesized_paulis(&self, state: &mut PauliSynthesisState) {
        // Runs recursively because processing one single-qubit Pauli in the front layer may enable
        // additional single-qubit Paulis in the following layers.
        loop {
            // All the Paulis have been processed.
            if state.num_processed == self.num_paulis {
                break;
            }

            // Find Paulis of weight 1 in the front layer,
            let mut new_processed: Vec<usize> = Vec::new();
            let frontier_nodes = compute_frontier_nodes(&self.dag, &state.in_degrees);
            for idx in &frontier_nodes {
                let support = state.tab.get_pauli_support(*idx);
                if support.len() == 1 {
                    new_processed.push(*idx);

                    // update number of processed nodes
                    state.num_processed += 1;

                    // update the circuit, including the single-qubit rotation gate
                    let q = support[0];
                    let angle = &self.angles[*idx];
                    match (
                        state.tab.get_pauli_x(*idx, q),
                        state.tab.get_pauli_z(*idx, q),
                        state.tab.get_pauli_phase(*idx),
                    ) {
                        (true, false, false) => state.gate_sequence.push((
                            StandardGate::RX,
                            smallvec![angle.clone()],
                            smallvec![Qubit(q as u32)],
                        )),
                        (true, false, true) => state.gate_sequence.push((
                            StandardGate::RX,
                            smallvec![multiply_param(angle, -1.0)],
                            smallvec![Qubit(q as u32)],
                        )),
                        (false, true, false) => state.gate_sequence.push((
                            StandardGate::RZ,
                            smallvec![angle.clone()],
                            smallvec![Qubit(q as u32)],
                        )),
                        (false, true, true) => state.gate_sequence.push((
                            StandardGate::RZ,
                            smallvec![multiply_param(angle, -1.0)],
                            smallvec![Qubit(q as u32)],
                        )),
                        (true, true, false) => state.gate_sequence.push((
                            StandardGate::RY,
                            smallvec![angle.clone()],
                            smallvec![Qubit(q as u32)],
                        )),
                        (true, true, true) => state.gate_sequence.push((
                            StandardGate::RY,
                            smallvec![multiply_param(angle, -1.0)],
                            smallvec![Qubit(q as u32)],
                        )),
                        _ => {
                            unreachable!("The Pauli support qubit cannot be I");
                        }
                    }
                }
            }

            if new_processed.is_empty() {
                break;
            }

            // recompute in_degrees
            update_in_degrees(&self.dag, &mut state.in_degrees, &new_processed);
        }
    }

    /// Starting from an MCTS node, implements the rollout (synthesizing all Paulis).
    /// Returns the solution.
    /// ToDo: consider calling this function greedy_synthesis.
    /// We should be able to get Rustiq implementation (for minimizing CX-count)
    /// by changing the internal scoring function.
    fn rollout_policy(&mut self, mcts_node_id: usize) -> GateSequence {
        let num_paulis = self.num_paulis;

        // We are cloning this state, since are going to update it in-place.
        let mut synthesis_state = self.mcts_nodes[mcts_node_id].synthesis_state.clone();

        if synthesis_state.num_processed == self.num_paulis {
            return synthesis_state.gate_sequence.clone();
        }

        loop {
            // Compute front nodes.
            // ToDo: consider storing frontier nodes as part of the synthesis state as well.
            let front_nodes = compute_frontier_nodes(&self.dag, &synthesis_state.in_degrees);
            assert!(!front_nodes.is_empty());

            // Find active pauli of minimum weight.
            let pauli_idx = front_nodes
                .iter()
                .min_by_key(|idx| synthesis_state.tab.get_pauli_support_size(**idx))
                .expect("The frontier cannot be empty.");

            // We should have filtered all already synthesized paulis.
            assert!(synthesis_state.tab.get_pauli_support_size(*pauli_idx) >= 2);
            synthesize_pauli(&mut synthesis_state, *pauli_idx);

            // Update the state by finding all Paulis that got synthesized.
            self.process_synthesized_paulis(&mut synthesis_state);

            // Check if all the Paulis are processed now.
            if synthesis_state.num_processed == num_paulis {
                break;
            }
        }
        synthesis_state.gate_sequence
    }
}

static ROTATION_GATES: [StandardGate; 3] = [StandardGate::RX, StandardGate::RY, StandardGate::RZ];

#[allow(clippy::too_many_arguments)]
pub fn pauli_network_mcts_inner(
    num_qubits: usize,
    paulis: Vec<String>,
    angles: Vec<Param>,
    preserve_order: bool,
    upto_clifford: bool,
    upto_phase: bool,
    num_simulations: usize,
) -> Result<CircuitData, EvolutionSynthesisError> {
    let paulis = PauliList::from_pauli_labels(num_qubits, &paulis)?;
    let mut mcts = MctsAlgorithm::new(&paulis, &angles, preserve_order);
    let (mut circuit, global_phase) = mcts.run(num_simulations); // number of simulations

    // If upto_clifford is true, we just return the above circuit. However, if upto_clifford is false, we need to
    // add the inverse clifford part of the circuit.
    if !upto_clifford {
        // Extract & inverse the Clifford part of the circuit
        let inverse_clifford_circuit: GateSequence = circuit
            .iter()
            .filter(|(gate, _params, _qubits)| !ROTATION_GATES.contains(gate))
            .rev()
            .map(|(gate, params, qubits)| {
                let (inverse_gate, inverse_params) = gate
                    .inverse(params)
                    .expect("All Clifford gates that can appear here are invertible");
                (inverse_gate, inverse_params, qubits.clone())
            })
            .collect();

        if !upto_phase {
            // If the circuit needs to be synthesized exactly, we cannot use Clifford resynthesis methods
            // since they do not preserve the global phase.
            circuit.extend(inverse_clifford_circuit);
        } else {
            //  However, if upto_phase is true, we can attempt to resynthesize the final clifford.
            let resynthesized =
                resynthesize_clifford_circuit(num_qubits, &inverse_clifford_circuit).unwrap();
            if cx_count_with_swaps(&inverse_clifford_circuit) <= cx_count_with_swaps(&resynthesized)
            {
                circuit.extend(inverse_clifford_circuit);
            } else {
                circuit.extend(resynthesized);
            }
        }
    }

    Ok(CircuitData::from_standard_gates(
        num_qubits as u32,
        circuit,
        global_phase,
    )?)
}
