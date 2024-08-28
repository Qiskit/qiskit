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

use hashbrown::{HashMap, HashSet};
use once_cell::sync::Lazy;
use pyo3::{exceptions::PyTypeError, prelude::*};
use qiskit_circuit::imports::{CIRCUIT_TO_DAG, QUANTUM_CIRCUIT};
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::{
    circuit_data::CircuitData,
    dag_circuit::{DAGCircuit, NodeType},
    dag_node::DAGOpNode,
    operations::{Operation, Param},
};
use rustworkx_core::petgraph::graph::NodeIndex;
use smallvec::SmallVec;

// Custom types
// TODO: Remove these and use the version from `basis_search`
type BasisTransforms = Vec<(String, u32, SmallVec<[Param; 3]>, CircuitRep)>;
static CONTROL_FLOW_OP_NAMES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    ["for_loop", "while_loop", "if_else", "switch_case"]
        .into_iter()
        .collect()
});
// TODO: Remove these and use the version from `EquivalenceLibrary`

/// Representation of QuantumCircuit which the original circuit object + an
/// instance of `CircuitData`.
#[derive(Debug, Clone)]
pub struct CircuitRep {
    object: PyObject,
    pub num_qubits: usize,
    pub num_clbits: usize,
    pub data: CircuitData,
}

impl CircuitRep {
    /// Performs a shallow cloning of the structure by using `clone_ref()`.
    pub fn py_clone(&self, py: Python) -> Self {
        Self {
            object: self.object.clone_ref(py),
            num_qubits: self.num_qubits,
            num_clbits: self.num_clbits,
            data: self.data.clone(),
        }
    }
}

impl FromPyObject<'_> for CircuitRep {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if ob.is_instance(QUANTUM_CIRCUIT.get_bound(ob.py()))? {
            let data: Bound<PyAny> = ob.getattr("_data")?;
            let data_downcast: Bound<CircuitData> = data.downcast_into()?;
            let data_extract: CircuitData = data_downcast.extract()?;
            let num_qubits: usize = data_extract.num_qubits();
            let num_clbits: usize = data_extract.num_clbits();
            Ok(Self {
                object: ob.into_py(ob.py()),
                num_qubits,
                num_clbits,
                data: data_extract,
            })
        } else {
            Err(PyTypeError::new_err(
                "Provided object was not an instance of QuantumCircuit",
            ))
        }
    }
}

impl IntoPy<PyObject> for CircuitRep {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.object
    }
}

impl ToPyObject for CircuitRep {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.object.clone_ref(py)
    }
}

#[pyfunction(name = "compose_transforms")]
fn py_compose_transforms(
    _py: Python,
    _basis_transforms: BasisTransforms,
    _source_basis: HashSet<(String, u32)>,
    _source_dag: &DAGCircuit,
) -> HashMap<String, (SmallVec<[Param; 3]>, DAGCircuit)> {
    todo!()
}

fn compose_transforms(
    _basis_transforms: &BasisTransforms,
    _source_basis: &HashSet<(String, u32)>,
    _source_dag: DAGCircuit,
) -> HashMap<String, (SmallVec<[Param; 3]>, DAGCircuit)> {
    /*
    example_gates = _get_example_gates(source_dag)
    mapped_instrs = {}

    for gate_name, gate_num_qubits in source_basis:
        # Need to grab a gate instance to find num_qubits and num_params.
        # Can be removed following https://github.com/Qiskit/qiskit-terra/pull/3947 .
        example_gate = example_gates[gate_name, gate_num_qubits]
        num_params = len(example_gate.params)

        placeholder_params = ParameterVector(gate_name, num_params)
        placeholder_gate = Gate(gate_name, gate_num_qubits, list(placeholder_params))
        placeholder_gate.params = list(placeholder_params)

        dag = DAGCircuit()
        qr = QuantumRegister(gate_num_qubits)
        dag.add_qreg(qr)
        dag.apply_operation_back(placeholder_gate, qr, (), check=False)
        mapped_instrs[gate_name, gate_num_qubits] = placeholder_params, dag

    for gate_name, gate_num_qubits, equiv_params, equiv in basis_transforms:
        logger.debug(
            "Composing transform step: %s/%s %s =>\n%s",
            gate_name,
            gate_num_qubits,
            equiv_params,
            equiv,
        )

        for mapped_instr_name, (dag_params, dag) in mapped_instrs.items():
            doomed_nodes = [
                node
                for node in dag.op_nodes()
                if (node.name, node.num_qubits) == (gate_name, gate_num_qubits)
            ]

            if doomed_nodes and logger.isEnabledFor(logging.DEBUG):

                logger.debug(
                    "Updating transform for mapped instr %s %s from \n%s",
                    mapped_instr_name,
                    dag_params,
                    dag_to_circuit(dag, copy_operations=False),
                )

            for node in doomed_nodes:

                replacement = equiv.assign_parameters(dict(zip_longest(equiv_params, node.params)))

                replacement_dag = circuit_to_dag(replacement)

                dag.substitute_node_with_dag(node, replacement_dag)

            if doomed_nodes and logger.isEnabledFor(logging.DEBUG):

                logger.debug(
                    "Updated transform for mapped instr %s %s to\n%s",
                    mapped_instr_name,
                    dag_params,
                    dag_to_circuit(dag, copy_operations=False),
                )

    return mapped_instrs
    */
    todo!()
}

fn get_example_gates(
    py: Python,
    dag: &DAGCircuit,
    example_gates: Option<Box<HashMap<(String, u32), NodeIndex>>>,
) -> PyResult<Box<HashMap<(String, u32), NodeIndex>>> {
    /*
    def _get_example_gates(source_dag):
        def recurse(dag, example_gates=None):
            example_gates = example_gates or {}
            for node in dag.op_nodes():
                example_gates[(node.name, node.num_qubits)] = node
                if node.name in CONTROL_FLOW_OP_NAMES:
                    for block in node.op.blocks:
                        example_gates = recurse(circuit_to_dag(block), example_gates)
            return example_gates

        return recurse(source_dag)
    */
    let mut example_gates = example_gates.unwrap_or_default();
    for node in dag.op_nodes(true) {
        if let Some(NodeType::Operation(op)) = dag.dag.node_weight(node) {
            example_gates.insert((op.op.name().to_string(), op.op.num_qubits()), node);
            if CONTROL_FLOW_OP_NAMES.contains(op.op.name()) {
                let OperationRef::Instruction(inst) = op.op.view() else {
                    panic!("Control flow op can only be an instruction")
                };
                let inst_bound = inst.instruction.bind(py);
                let blocks = inst_bound.getattr("blocks")?;
                for block in blocks.iter()? {
                    let block_as_dag = CIRCUIT_TO_DAG.get_bound(py).call1((block?,))?;
                    let block_as_dag = block_as_dag.downcast_into::<DAGCircuit>()?.extract()?;
                    example_gates = get_example_gates(py, &block_as_dag, Some(example_gates))?;
                }
            }
        }
    }
    Ok(example_gates)
}
