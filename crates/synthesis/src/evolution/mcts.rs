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
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::{Param, StandardGate, multiply_param};
use qiskit_quantum_info::clifford::PauliList;

use rustworkx_core::petgraph::Direction::Outgoing;
use rustworkx_core::petgraph::Incoming;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::prelude::StableDiGraph;

use smallvec::SmallVec;

use std::f64::consts::SQRT_2;
use std::fmt;

/// The multiplicative scaling parameter used in the MCTS algorithm.
const MCTS_PARAM: f64 = SQRT_2;

/// Sequence of standard gates constructed by the algorithm, including
/// both Clifford gates and single-qubit rotations.
type GateSequence = Vec<(StandardGate, Vec<Param>, Vec<usize>)>;

/// A particular point during Pauli network synthesis.
#[derive(Clone)]
struct PauliSynthesisState {
    /// The tableau storing Pauli rotations corresponding to this state
    /// (the initial tableau conjugated by Clifford gates leading to this state).
    /// The Paulis that havve already been synthesized are stored as ``None`` in
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
        write!(f, "tab = {}, num_processed = {}, in_degrees = {:?}", self.tab, self.num_processed, self.in_degrees)
    }
}

/// A node in the MCTS tree.
struct MctsNode {
    /// Number of times this node was visited across simulations.
    ni: usize,

    /// Cumulative cost (e.g. number of CX=gates) from root to terminal node
    /// across simulations.
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


type Score = (isize, isize);

/// ToDo: add CX to (control, target) AFTER calling this function
/// ToDo: there might be an optimization (similar to the one in Rustiq that does not need to clone the table)
/// Alternatively, modify the table + modify back
fn identify(
    synthesis_state: &mut PauliSynthesisState,
    op_lists: &[GateSequence],
    ctrl: usize,
    targ: usize,
) -> (Score, GateSequence) {
    // println!("!!! identify called with tab = {}, ctrl = {:?}, targ = {:?}", synthesis_state.tab, ctrl, targ);

    let (best_idx, best_score) = op_lists
        .iter()
        .map(|op| {
            // ToDo: instead of copying & evolving, consider evolving back, or consider implementing rustiq-like strategy
            // that replaces evolve by a table lookup
            // let mut tab = synthesis_state.tab.clone();
            for (gate, _params, qubits) in op {
                if *gate == StandardGate::H {
                    synthesis_state.tab.append_h(qubits[0]);
                } else if *gate == StandardGate::S {
                    synthesis_state.tab.append_s(qubits[0])
                }
            }
            let x_ctrl = synthesis_state.tab.get_x(ctrl);
            let x_targ = synthesis_state.tab.get_x(targ);
            let z_ctrl = synthesis_state.tab.get_z(ctrl);
            let z_targ = synthesis_state.tab.get_z(targ);

            let num_decreased: usize = (0..synthesis_state.tab.num_paulis)
                .filter(|i| synthesis_state.in_degrees[*i].is_some())
                .map(|i| {
                    ((x_ctrl.contains(i) && x_targ.contains(i) && !z_targ.contains(i))
                        | (!x_ctrl.contains(i) && z_ctrl.contains(i) && z_targ.contains(i)))
                        as usize
                })
                .sum();

            let num_increased: usize = (0..synthesis_state.tab.num_paulis)
                .filter(|i| synthesis_state.in_degrees[*i].is_some())
                .map(|i| {
                    ((!x_ctrl.contains(i) && !z_ctrl.contains(i) && z_targ.contains(i))
                        || (x_ctrl.contains(i) && !x_targ.contains(i) && !z_targ.contains(i)))
                        as usize
                })
                .sum();
            for (gate, _params, qubits) in op.iter().rev() {
                if *gate == StandardGate::H {
                    synthesis_state.tab.append_h(qubits[0]);
                } else if *gate == StandardGate::S {
                    synthesis_state.tab.append_sdg(qubits[0])
                }
            }
            // println!("-- with op = {:?}: num_decreased = {:?}, num_increased = {:?}", op, num_decreased, num_increased);
            ((num_decreased as isize) - (num_increased as isize), num_decreased as isize)

        })
        .enumerate()
        .rev()
        .max_by_key(|(_idx, score)| *score)
        .expect("list of ops cannot be empty");

    let mut best_seq = op_lists[best_idx].clone();
    // println!("identify i = {}, j = {} returned score {:?} and seq {:?}", ctrl, targ, best_score, best_seq);
    best_seq.push((StandardGate::CX, vec![], vec![ctrl, targ]));
    (best_score, best_seq)
}

// QUESTION: DO WE COMPARE ON WHAT???! ALL SYNTHESIZED NODES?? ALL FRONT NODES???

fn compare(
    synthesis_state: &mut PauliSynthesisState,
    ndx: usize,
    ctrl: usize,
    targ: usize,
) -> (Score, GateSequence) {
    let x_ctrl = synthesis_state.tab.get_x(ctrl).contains(ndx);
    let x_targ = synthesis_state.tab.get_x(targ).contains(ndx);
    let z_ctrl = synthesis_state.tab.get_z(ctrl).contains(ndx);
    let z_targ = synthesis_state.tab.get_z(targ).contains(ndx);

    // todo: replace vecs by arrays
    let op_lists = match (x_ctrl, z_ctrl, x_targ, z_targ) {
        // I*
        (false, false, _, _) => {
            panic!("The function should not be called with ctrl=I");
        }

        // *I
        (_, _, false, false) => {
            panic!("The function should not be called with targ=I");
        }

        // XX
        (true, false, true, false) => vec![
            vec![],
            vec![(StandardGate::S, vec![], vec![ctrl])],
            vec![
                (StandardGate::H, vec![], vec![ctrl]),
                (StandardGate::H, vec![], vec![targ]),
            ],
            vec![
                (StandardGate::H, vec![], vec![ctrl]),
                (StandardGate::S, vec![], vec![targ]),
            ],
        ],

        // XY
        (true, false, true, true) => vec![
            vec![
                (StandardGate::S, vec![], vec![ctrl]),
                (StandardGate::S, vec![], vec![targ]),
            ],
            vec![(StandardGate::S, vec![], vec![targ])],
            vec![(StandardGate::H, vec![], vec![ctrl])],
            vec![
                (StandardGate::H, vec![], vec![ctrl]),
                (StandardGate::H, vec![], vec![targ]),
            ],
        ],

        // XZ
        (true, false, false, true) => vec![
            vec![
                (StandardGate::S, vec![], vec![ctrl]),
                (StandardGate::H, vec![], vec![targ]),
            ],
            vec![(StandardGate::H, vec![], vec![targ])],
            vec![(StandardGate::H, vec![], vec![ctrl])],
            vec![
                (StandardGate::H, vec![], vec![ctrl]),
                (StandardGate::S, vec![], vec![targ]),
            ],
        ],

        // YX
        (true, true, true, false) => vec![
            vec![],
            vec![(StandardGate::S, vec![], vec![ctrl])],
            vec![(StandardGate::H, vec![], vec![ctrl])],
        ],

        // YY
        (true, true, true, true) => vec![
            vec![
                (StandardGate::S, vec![], vec![ctrl]),
                (StandardGate::S, vec![], vec![targ]),
            ],
            vec![
                (StandardGate::H, vec![], vec![ctrl]),
                (StandardGate::S, vec![], vec![targ]),
            ],
            vec![(StandardGate::S, vec![], vec![targ])],
        ],

        // YZ
        (true, true, false, true) => vec![
            vec![(StandardGate::H, vec![], vec![targ])],
            vec![(StandardGate::S, vec![], vec![ctrl]), (StandardGate::H, vec![], vec![targ])],
        ],

        // ZX
        (false, true, true, false) => vec![
            vec![(StandardGate::H, vec![], vec![targ])],
            vec![(StandardGate::H, vec![], vec![ctrl])],
            vec![
                (StandardGate::S, vec![], vec![ctrl]),
                (StandardGate::H, vec![], vec![targ]),
            ],
            vec![(StandardGate::S, vec![], vec![targ])],
        ],

        // ZY
        (false, true, true, true) => vec![
            vec![],
            vec![(StandardGate::S, vec![], vec![ctrl])],
            vec![(StandardGate::H, vec![], vec![targ])],
            vec![
                (StandardGate::S, vec![], vec![ctrl]),
                (StandardGate::H, vec![], vec![targ]),
            ],
            vec![
                (StandardGate::H, vec![], vec![ctrl]),
                (StandardGate::S, vec![], vec![targ]),
            ],
        ],

        // ZZ
        (false, true, false, true) => vec![
            vec![],
            vec![
                (StandardGate::H, vec![], vec![ctrl]),
                (StandardGate::H, vec![], vec![targ]),
            ],
            vec![(StandardGate::S, vec![], vec![ctrl])],
            vec![(StandardGate::S, vec![], vec![targ])],
            vec![
                (StandardGate::S, vec![], vec![ctrl]),
                (StandardGate::S, vec![], vec![targ]),
            ],
        ],
    };

    identify(synthesis_state, &op_lists, ctrl, targ)
}

/// Synthesizes a given Pauli, that is finds a sequence of Clifford gates that brings this Pauli to a single-qubit
/// Pauli rotation (note that the same sequence can bring other Paulis to single-qubit Paulis as well).
/// Updates the synthesis state in-place, adjusting the Pauli tableau (conjugating by Clifford gates) and extending
/// the gate sequence.
fn synthesize_pauli(synthesis_state: &mut PauliSynthesisState, ndx: usize) {
    loop {
        let support_size = synthesis_state.tab.get_pauli_support_size(ndx);
        assert!(support_size >= 1);

        // We have successfully reduced this Pauli to a single-qubit rotation.
        if support_size == 1 {
            return;
        }

        // loop to cycle over all qubit indices to get all combinations of control and target
        // for now, keep the same order as in python code
        let ndx_support = synthesis_state.tab.get_pauli_support(ndx);

        let pairs = (0..ndx_support.len())
            .flat_map(|i| ((i + 1)..ndx_support.len()).flat_map(move |j| [(i, j), (j, i)]));

        let (score, gate_seq) = pairs
            .map(|(i, j)| compare(synthesis_state, ndx, ndx_support[i], ndx_support[j]))
            .rev()
            .max_by_key(|(score, _gate_seq)| *score)
            .expect("The list should not be empty");

        // println!("===> chose action: gate_sequence = {:?}, with score {:?}", gate_seq, score);


        // Apply the best operation
        for (gate, _paramas, qubits) in &gate_seq {
            match gate {
                StandardGate::H => {
                    synthesis_state.tab.append_h(qubits[0]);
                }
                StandardGate::S => {
                    synthesis_state.tab.append_s(qubits[0]);
                }
                StandardGate::CX => {
                    synthesis_state.tab.append_cx(qubits[0], qubits[1]);
                }
                _ => {
                    panic!("should only have s/h/cx gates");
                }
            }
        }
        synthesis_state.gate_sequence.extend(gate_seq);
    }
}

struct MctsAlgorithm {
    /// Total number of Pauli rotations to synthesize.
    num_paulis: usize,

    /// Paulis to be synthesized
    /// The input will probably be not of type PauliList
    paulis: PauliList,

    /// Angle params
    angles: Vec<Param>,

    /// MCTS nodes (forming a tree)
    mcts_nodes: Vec<MctsNode>,

    // list of solutions found with score
    solutions: Vec<GateSequence>,

    // ToDo: add commutativity DAG
    dag: StableDiGraph<usize, ()>,
}

impl MctsAlgorithm {
    /// Creates a tree with a single root node
    fn new(paulis: &PauliList, angles: &[Param]) -> Self {
        // Commutativity DAG
        let num_paulis = paulis.num_paulis;

        let mut dag = StableDiGraph::<usize, ()>::new();
        let node_indices: Vec<NodeIndex> = (0..num_paulis).map(|i| dag.add_node(i)).collect();
        for i in 0..num_paulis {
            for j in i + 1..num_paulis {
                if !paulis.commute(i, j) {
                    dag.add_edge(node_indices[i], node_indices[j], ());
                }
            }
        }

        Self {
            num_paulis,
            paulis: paulis.clone(),
            angles: angles.to_vec(),
            mcts_nodes: vec![],
            solutions: vec![],
            dag: dag,
        }
    }

    /// Updates in-degree in-place with newly processed nodes.
    fn update_in_degrees(&self, in_degrees: &mut [Option<usize>], new_processed: &[usize]) {
        for i in new_processed {
            for n in self.dag.neighbors_directed(NodeIndex::new(*i), Outgoing) {
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

    /// Computes frontier nodes by examining in-degrees.
    fn compute_frontier_nodes(&self, in_degrees: &[Option<usize>]) -> Vec<usize> {
        self.dag
            .node_indices()
            .filter(|n| in_degrees[n.index()] == Some(0))
            .map(|n| n.index())
            .collect()
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

        self.process_all_identity_paulis(&mut synthesis_state);
        self.process_synthesized_paulis(&mut synthesis_state);

        let unexplored_actions = self.compute_frontier_nodes(&synthesis_state.in_degrees).iter().rev().cloned().collect();
        // let unexplored_actions = self.compute_frontier_nodes(&synthesis_state.in_degrees);

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

        // if it has been completely unexplored
        if node.ni == 0 {
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

    /// function to backpropagate estimated value at terminal state up through tree
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

    /// The main function (mcts) which runs everything
    fn run(&mut self, num_sims: usize) -> GateSequence {
        // ToDo: remove all-I paulis; moving them to processed
        self.add_root_mcts_node(&self.paulis.clone());

        if true {
            // for now, just rollout from the root node
            let circuit = self.rollout_policy(0);
            self.solutions.push(circuit);
        }

        while self.mcts_nodes[0].ni < num_sims {
            let leaf_node_id = self.tree_policy();
            let solution = self.rollout_policy(leaf_node_id);

            // The full solution is obtained by taking the solution from root to leaf + rollout solutions
            // let mut solution = self.nodes[leaf_node_id].gate_sequence.clone();
            // solution.extend(new_gate_sequence);

            let value = cx_count(&solution);

            self.solutions.push(solution);
            self.backpropagate(leaf_node_id, value);
        }

        // Return best solution
        let best_solution = self
            .solutions
            .iter()
            .min_by_key(|sol| cx_count(sol))
            .expect("we should have at least one solution");

        // println!("BEST SOLUTION = {:?}", best_solution);
        best_solution.clone()
    }

    // DEBUG ONLY FUNCTION
    fn print_tree(&self) {
        println!("========");
        println!("Tree contains {} nodes", self.mcts_nodes.len());
        for (i, node) in self.mcts_nodes.iter().enumerate() {
            println!(
                "node_id = {}: ni = {}, qi = {}, unexplored_actions = {:?}, parent = {:?}, children = {:?}, state = {}, num_processed = {:?}, gate_sequence = {:?} ",
                i,
                node.ni,
                node.qi,
                node.unexplored_actions,
                node.parent,
                node.children,
                node.synthesis_state.tab,
                node.synthesis_state.num_processed,
                node.synthesis_state.gate_sequence.len()
            );
        }

        println!("========");
    }

    /// Implement the "tree policy": find the best MCTS node to find the rollout from.
    fn tree_policy(&mut self) -> usize {
        // println!("=> Starting tree policy");
        // self.print_tree();

        let mut mcts_node_id = 0; // root
        loop {
            // println!("===> Current state:");
            // println!("===> {}", self.mcts_nodes[mcts_node_id].synthesis_state);
            


            // If all the Paulis have been processed, return the current id.
            if self.mcts_nodes[mcts_node_id].synthesis_state.num_processed == self.num_paulis {
                // println!("===> No paulis remaining; id = {}", mcts_node_id);
                return mcts_node_id;
            }

            // Node includes unexplored Paulis.
            if !self.mcts_nodes[mcts_node_id].unexplored_actions.is_empty() {
                // Choose one of the unprocessed paulis in the front layer.
                let chosen_pauli_idx = self.mcts_nodes[mcts_node_id]
                    .unexplored_actions
                    .last()
                    .unwrap();
                // println!("===> Choose unexplored action; pauli id = {}", chosen_pauli_idx);

                // Synthesize it
                let mut synthesis_state = self.mcts_nodes[mcts_node_id].synthesis_state.clone();
                synthesize_pauli(&mut synthesis_state, *chosen_pauli_idx);
                self.process_synthesized_paulis(&mut synthesis_state);

                // let unexplored_actions = self.compute_frontier_nodes(&synthesis_state.in_degrees);
                let unexplored_actions = self.compute_frontier_nodes(&synthesis_state.in_degrees).iter().rev().cloned().collect();
                
                // Create the child mcts node with the result of the above synthesis.
                let child_node = MctsNode {
                    ni: 0,
                    qi: 0,
                    parent: Some(mcts_node_id),
                    children: vec![],
                    synthesis_state: synthesis_state,
                    unexplored_actions,
                };

                // add child to tree and remove the synthesized pauli from the set of unexplored frontier nodes.
                self.mcts_nodes.push(child_node);
                let child_node_id = self.mcts_nodes.len() - 1;
                self.mcts_nodes[mcts_node_id].children.push(child_node_id);
                self.mcts_nodes[mcts_node_id].unexplored_actions.pop();
                // println!("===> Adding child node with index {:?}", child_node_id);
                return child_node_id;
            };

            // All the children have been explored; choose the one with highest UCT score.

            mcts_node_id = *self.mcts_nodes[mcts_node_id]
                .children
                .iter()
                .max_by(|&&a, &&b| {
                    self.upper_confidence_bound(a)
                        .partial_cmp(&self.upper_confidence_bound(b))
                        .unwrap()
                })
                .unwrap();
            // println!("===> Considering action with max score {:?}", mcts_node_id);

        }
    }

    /// Finds all-identity paulis and updates the synthesis state accordingly.
    /// Should be called once at the start of the algorithm.
    /// TODO: FIX GLOBAL PHASE!!!
    fn process_all_identity_paulis(&self, synthesis_state: &mut PauliSynthesisState) {
        let mut new_processed: Vec<usize> = Vec::new();
        for idx in 0..self.num_paulis {
            if synthesis_state.tab.get_pauli_support_size(idx) == 0 {
                new_processed.push(idx);
                synthesis_state.num_processed += 1;
            }
        }
        if !new_processed.is_empty() {
            self.update_in_degrees(&mut synthesis_state.in_degrees, &new_processed);
        }
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
            let frontier_nodes = self.compute_frontier_nodes(&state.in_degrees);
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
                            vec![angle.clone()],
                            vec![q],
                        )),
                        (true, false, true) => state.gate_sequence.push((
                            StandardGate::RX,
                            vec![multiply_param(angle, -1.0)],
                            vec![q],
                        )),
                        (false, true, false) => state.gate_sequence.push((
                            StandardGate::RZ,
                            vec![angle.clone()],
                            vec![q],
                        )),
                        (false, true, true) => state.gate_sequence.push((
                            StandardGate::RZ,
                            vec![multiply_param(angle, -1.0)],
                            vec![q],
                        )),
                        (true, true, false) => state.gate_sequence.push((
                            StandardGate::RY,
                            vec![angle.clone()],
                            vec![q],
                        )),
                        (true, true, true) => state.gate_sequence.push((
                            StandardGate::RY,
                            vec![multiply_param(angle, -1.0)],
                            vec![q],
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
            // println!("===> new processed: {:?}", new_processed);

            // recompute in_degrees
            self.update_in_degrees(&mut state.in_degrees, &new_processed);
        }
    }

    /// Starting from an MCTS node, implements the rollout (synthesizing all Paulis).
    /// (TODO: experiment implementing this step using Rustiq.)
    /// Returns the solution.
    fn rollout_policy(&mut self, mcts_node_id: usize) -> GateSequence {
        let num_paulis = self.num_paulis;

        // println!("=> Starting Rollout:");

        // We are cloning this state, since are going to update it in-place.
        let mut synthesis_state = self.mcts_nodes[mcts_node_id].synthesis_state.clone();

        if synthesis_state.num_processed == self.num_paulis {
            // println!("===> All Paulis are processed");
            return synthesis_state.gate_sequence.clone();
        }

        loop {
            // println!("");
            // println!("ROLLOUT: remain {}, synthesis_state: {}", synthesis_state.tab.num_paulis - synthesis_state.num_processed, synthesis_state);
            // Find active Pauli of minimum weight.
            let front_nodes = self.compute_frontier_nodes(&synthesis_state.in_degrees);
            assert!(!front_nodes.is_empty());
            // println!("===> front nodes: {:?}", front_nodes);


            // Find active pauli of smallest weight
            let idx = front_nodes
                .iter()
                .min_by_key(|idx| {
                    synthesis_state.tab.get_pauli_support_size(**idx)
                })
                .expect("The frontier cannot be empty.");
            // println!("===> chose pauli id {:?}: {:?}", idx, synthesis_state.tab.to_pauli_strings()[*idx]);


            // We should have filtered all already synthesized paulis.
            assert!(synthesis_state.tab.get_pauli_support_size(*idx) >= 2);
            synthesize_pauli(&mut synthesis_state, *idx);

            // Update the state by finding all Paulis that got synthesized.
            self.process_synthesized_paulis(&mut synthesis_state);

            // Check if all the Paulis are processed now.
            if synthesis_state.num_processed == num_paulis {
                // println!("===> All Paulis are processed");
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
    upto_clifford: bool,
    upto_phase: bool,
    num_simulations: usize,
) -> Result<CircuitData, CircuitDataError> {
    // println!("=> Called pauli_network_mcts_inner with paulis = {:?} and angles = {:?}", paulis, angles);
    let paulis = PauliList::from_pauli_strings(&paulis);
    let mut mcts = MctsAlgorithm::new(&paulis, &angles);
    let mut circuit = mcts.run(num_simulations); // number of simulations

    // For now, extract & inverse the Clifford part of the circuit
    let inverse_clifford_circuit: GateSequence = circuit
        .iter()
        .filter(|(gate, _params, _qubits)| !ROTATION_GATES.contains(gate))
        .map(|(gate, params, qubits)| {
            let inverse_clifford_gate = match gate {
                StandardGate::CX => StandardGate::CX,
                StandardGate::H => StandardGate::H,
                StandardGate::S => StandardGate::Sdg,
                _ => {
                    panic!("only CX/H/S");
                }
            };
            (inverse_clifford_gate, params.clone(), qubits.clone())
        })
        .rev()
        .collect();

    circuit.extend(inverse_clifford_circuit);
    // // if the circuit needs to be synthesized exactly, we cannot use either Rustiq's
    // // or Qiskit's synthesis methods for Cliffords, since they do not necessarily preserve
    // // the global phase.
    // let resynth_clifford_method = match upto_phase {
    //     true => resynth_clifford_method,
    //     false => 0,
    // };

    // // synthesize the final Clifford
    // if !upto_clifford {
    //     let final_clifford = synthesize_final_clifford(&circuit.dagger(), resynth_clifford_method);
    //     for gate in final_clifford {
    //         gates.push(gate);
    //     }
    // }
    // (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>
    CircuitData::from_standard_gates(
        num_qubits as u32,
        circuit.into_iter().map(|(gate, params, qubits)| {
            let params: SmallVec<[Param; 3]> = params.into_iter().collect();

            let qubits: SmallVec<[Qubit; 2]> = qubits.iter().map(|q| Qubit(*q as u32)).collect();
            (gate, params, qubits)
        }),
        Param::Float(0.0),
    )
}
