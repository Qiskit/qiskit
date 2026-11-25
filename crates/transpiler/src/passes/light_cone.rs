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

use hashbrown::HashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::bit::Register;
use qiskit_circuit::classical::expr;
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit, Wire};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{OperationRef, Param, PauliProductRotation, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Block, Clbit, Qubit};
use qiskit_quantum_info::sparse_observable::{PySparseObservable, SparseObservable};
use rustworkx_core::petgraph::Direction::Outgoing;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::commutation_checker::CommutationChecker;

/// We match the default of `CommutationChecker.commute` in python. Rust requires this field and for
/// now we don't have a good way of allowing the user to set it.
const MATRIX_MAX_NUM_QUBITS: u32 = 3;

/// Something later in the circuit that an earlier operation must not disturb.
enum Item<'a> {
    /// An operation of the circuit that is in the light cone.
    Gate(&'a PackedInstruction),
    /// Observables, measurements, mid-circuit or final measurements. We associate them to the
    /// `PauliProductRotation`` with the same commutation relationships. Since they do not belong to
    /// the circuit they can not be referenced as `Gate`, so we also store the qubits they are
    /// applied to.
    Front {
        pauli: PauliProductRotation,
        qubits: Vec<Qubit>,
    },
    /// A clbit that a measured circuit returns, so that whatever writes it is kept.
    Output,
}

/// The light cone built so far, while the circuit is walked backwards.
///
/// Every wire (qubit, clbit or var) keeps the indices of the items on it, in the order they were
/// added. For every new operation we only look at the items on is wires.
/// Qubits and classical wires are checked differently. On its qubits, an operation is kept if it
/// does not commute with an item there. On its classical wires, an operation is kept if it writes
/// a wire that some item uses later, like a classically controlled operation.
struct LightCone<'a> {
    dag: &'a DAGCircuit,
    items: Vec<Item<'a>>,
    by_wire: HashMap<Wire, Vec<usize>>,
}

// Pauli helper functions

/// Since `PauliGate` does not have a Rust implementation yet, we store a Pauli `P` as a
/// `PauliProductRotation` at angle π, which is `-iP` and has the same commutation properties.
/// If PauliGate gets implemented in rust, it would make more sense to use it instead.
fn pauli(z: Vec<bool>, x: Vec<bool>) -> PauliProductRotation {
    PauliProductRotation {
        z,
        x,
        angle: Param::Float(std::f64::consts::PI),
    }
}

/// The Pauli that `op` measures, if `op` is a measurement: `Z` for a `Measure`, and its own Pauli
/// for a `PauliProductMeasurement`.
///
/// Measurements become fronts instead of gates because the [CommutationChecker] treats a `Measure`
/// as commuting with nothing on its qubits, and has no matrix for a `PauliProductMeasurement`. If
/// the checker learns to handle both, measurements could be kept as `Item::Gate` instead.
fn measured_pauli(op: &OperationRef) -> Option<PauliProductRotation> {
    match op {
        OperationRef::StandardInstruction(StandardInstruction::Measure) => {
            Some(pauli(vec![true], vec![false]))
        }
        OperationRef::PauliProductMeasurement(ppm) => Some(pauli(ppm.z.clone(), ppm.x.clone())),
        _ => None,
    }
}

/// Whether `dag` contains a measurement.
fn contains_measurement(dag: &DAGCircuit) -> bool {
    dag.op_nodes(true).any(|(_, inst)| {
        measured_pauli(&inst.op.view()).is_some()
            || dag
                .try_view_control_flow(inst)
                .is_some_and(|cf| cf.blocks().into_iter().any(contains_measurement))
    })
}

impl<'a> LightCone<'a> {
    fn new(dag: &'a DAGCircuit) -> Self {
        LightCone {
            dag,
            items: Vec::new(),
            by_wire: HashMap::new(),
        }
    }

    fn add(&mut self, item: Item<'a>, wires: impl IntoIterator<Item = Wire>) {
        for wire in wires {
            self.by_wire.entry(wire).or_default().push(self.items.len());
        }
        self.items.push(item);
    }

    /// Add a front, on its qubits and on the clbits of the measurement that made it, if any.
    fn add_front(&mut self, pauli: PauliProductRotation, qubits: Vec<Qubit>, clbits: &[Clbit]) {
        let wires: Vec<Wire> = (qubits.iter().map(|&qubit| Wire::Qubit(qubit)))
            .chain(clbits.iter().map(|&clbit| Wire::Clbit(clbit)))
            .collect();
        self.add(Item::Front { pauli, qubits }, wires);
    }

    /// Add the fronts of an observable.
    ///
    /// As in [SparseObservable], `bit_terms[i]` acts on qubit `indices[i]`. A gate only has to
    /// commute with the product of the Pauli factors of a term, so they form one front. A projector
    /// such as `|0><0| = (I + Z) / 2` instead becomes a front of its own Pauli, which is sufficient
    /// but not necessarily the minimal light cone.
    fn add_observable(&mut self, observable: &SparseObservable) {
        for term in observable.iter() {
            let mut product = pauli(vec![], vec![]);
            let mut product_qubits = vec![];
            for (bit, &index) in term.bit_terms.iter().zip(term.indices) {
                let (z, x) = (bit.has_z_component(), bit.has_x_component());
                if bit.is_projector() {
                    self.add_front(pauli(vec![z], vec![x]), vec![Qubit(index)], &[]);
                } else {
                    product.z.push(z);
                    product.x.push(x);
                    product_qubits.push(Qubit(index));
                }
            }
            if !product_qubits.is_empty() {
                self.add_front(product, product_qubits, &[]);
            }
        }
    }

    /// Keep the operation at `node`, on every wire it acts on.
    fn add_gate(&mut self, node: NodeIndex) {
        let wires: Vec<Wire> = (self.dag.dag())
            .edges_directed(node, Outgoing)
            .map(|edge| *edge.weight())
            .collect();
        self.add(Item::Gate(self.dag[node].unwrap_operation()), wires);
    }

    /// The indices of the items on `qubits`, newest first and each once.
    /// Newest first because the last item added is the one most likely not to commute.
    fn items_on(&self, qubits: &[Qubit]) -> Vec<usize> {
        let mut indices = Vec::new();
        for &qubit in qubits {
            if let Some(list) = self.by_wire.get(&Wire::Qubit(qubit)) {
                indices.extend_from_slice(list);
            }
        }
        indices.sort_unstable_by(|a, b| b.cmp(a));
        indices.dedup();
        indices
    }

    /// Whether `inst` commutes with every item on its qubits.
    fn commutes_with(
        &self,
        inst: &PackedInstruction,
        checker: &CommutationChecker,
    ) -> PyResult<bool> {
        let qargs = self.dag.get_qargs(inst.qubits);
        let cargs = self.dag.get_cargs(inst.clbits);
        for index in self.items_on(qargs) {
            let commutes = match &self.items[index] {
                Item::Gate(other) => checker.commute(
                    &inst.op.view(),
                    inst.params.as_deref(),
                    qargs,
                    cargs,
                    &other.op.view(),
                    other.params.as_deref(),
                    self.dag.get_qargs(other.qubits),
                    self.dag.get_cargs(other.clbits),
                    None,
                    MATRIX_MAX_NUM_QUBITS,
                    1.0,
                )?,
                Item::Front { pauli, qubits } => checker.commute(
                    &inst.op.view(),
                    inst.params.as_deref(),
                    qargs,
                    cargs,
                    &OperationRef::PauliProductRotation(pauli),
                    None::<&Parameters<Block>>,
                    qubits,
                    &[] as &[Clbit],
                    None,
                    MATRIX_MAX_NUM_QUBITS,
                    1.0,
                )?,
                Item::Output => unreachable!("outputs are only on clbits"),
            };
            if !commutes {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether an item is on a classical wire that the operation at `node` writes, so that
    /// something later may read the value.
    fn writes_live_wire(&self, node: NodeIndex) -> bool {
        (written_wires(self.dag, node).iter()).any(|wire| self.by_wire.contains_key(wire))
    }
}

/// The classical wires that the operation at `node` may write: the target of a `store`, and every
/// classical wire of any other operation.
fn written_wires(dag: &DAGCircuit, node: NodeIndex) -> Vec<Wire> {
    if let OperationRef::Store(store) = dag[node].unwrap_operation().op.view() {
        let mut wires = Vec::new();
        for var in store.lvalue().vars() {
            match var {
                expr::Var::Bit { bit } => wires.push(Wire::Clbit(dag.clbits().find(bit).unwrap())),
                expr::Var::Register { register, .. } => wires.extend(
                    register
                        .bits()
                        .map(|bit| Wire::Clbit(dag.clbits().find(&bit).unwrap())),
                ),
                expr::Var::Standalone { .. } => {
                    wires.push(Wire::Var(dag.vars().find(var).unwrap()))
                }
            }
        }
        return wires;
    }
    dag.dag()
        .edges_directed(node, Outgoing)
        .map(|edge| *edge.weight())
        .filter(|wire| !matches!(wire, Wire::Qubit(_)))
        .collect()
}

/// Find the operations that affect neither the measurements nor the observable.
/// The circuit is walked backwards. An operation is in the light cone if it does not commute with
/// an item on its qubits, or if it writes a classical wire that carries an item, which could affect
/// the result.
fn nodes_outside_light_cone(
    dag: &DAGCircuit,
    observable: Option<&SparseObservable>,
    checker: &CommutationChecker,
) -> PyResult<Vec<NodeIndex>> {
    let mut cone = LightCone::new(dag);
    if let Some(observable) = observable {
        // Checked before the walk, so the error does not depend on where the measurement is.
        if contains_measurement(dag) {
            return Err(PyValueError::new_err(
                "The circuit contains measurements and an observable has been given: \
                remove the observable or the measurements.",
            ));
        }
        cone.add_observable(observable);
    } else {
        // The clbits are what a measured circuit returns. Vars are not outputs.
        for clbit in 0..dag.num_clbits() {
            cone.add(Item::Output, [Wire::Clbit(Clbit::new(clbit))]);
        }
    }

    let mut outside = Vec::new();
    for node in dag.topological_op_nodes(true) {
        let inst = dag[node].unwrap_operation();
        if let Some(measured) = measured_pauli(&inst.op.view()) {
            let qubits = dag.get_qargs(inst.qubits).to_vec();
            cone.add_front(measured, qubits, dag.get_cargs(inst.clbits));
        } else if cone.commutes_with(inst, checker)? && !cone.writes_live_wire(node) {
            outside.push(node);
        } else {
            cone.add_gate(node);
        }
    }
    Ok(outside)
}

/// Remove the operations of `dag` outside the light cone.
pub fn run_light_cone(
    dag: &mut DAGCircuit,
    observable: Option<&SparseObservable>,
    checker: &CommutationChecker,
) -> PyResult<()> {
    for node in nodes_outside_light_cone(dag, observable, checker)? {
        dag.remove_op_node(node);
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "light_cone")]
pub fn py_run_light_cone(
    dag: &mut PyDAGCircuit,
    observable: Option<PyRef<PySparseObservable>>,
    commutation_checker: &CommutationChecker,
) -> PyResult<()> {
    let observable = observable.as_ref().map(|obs| obs.as_inner()).transpose()?;
    run_light_cone(dag.try_write()?, observable.as_deref(), commutation_checker)
}

pub fn light_cone_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_light_cone))?;
    Ok(())
}
