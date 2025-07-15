// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use pyo3::intern;
use pyo3::prelude::*;

use crate::bit::{ShareableClbit, ShareableQubit};
use crate::circuit_data::{CircuitData, CircuitVar};
use crate::dag_circuit::DAGIdentifierInfo;
use crate::dag_circuit::{DAGCircuit, NodeType};
use crate::operations::{OperationRef, PythonOperation};
use crate::packed_instruction::PackedInstruction;

/// An extractable representation of a QuantumCircuit reserved only for
/// conversion purposes.
#[derive(Debug, Clone)]
pub struct QuantumCircuitData<'py> {
    pub data: CircuitData,
    pub name: Option<String>,
    pub metadata: Option<Bound<'py, PyAny>>,
}

impl<'py> FromPyObject<'py> for QuantumCircuitData<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let circuit_data = ob.getattr("_data")?;
        let data_borrowed = circuit_data.extract::<CircuitData>()?;
        Ok(QuantumCircuitData {
            data: data_borrowed,
            name: ob.getattr(intern!(py, "name"))?.extract()?,
            metadata: ob.getattr(intern!(py, "metadata")).ok(),
        })
    }
}

#[pyfunction(signature = (quantum_circuit, copy_operations = true, qubit_order = None, clbit_order = None))]
pub fn circuit_to_dag(
    quantum_circuit: QuantumCircuitData,
    copy_operations: bool,
    qubit_order: Option<Vec<ShareableQubit>>,
    clbit_order: Option<Vec<ShareableClbit>>,
) -> PyResult<DAGCircuit> {
    DAGCircuit::from_circuit(quantum_circuit, copy_operations, qubit_order, clbit_order)
}

#[pyfunction(signature = (dag, copy_operations = true))]
pub fn dag_to_circuit(dag: &DAGCircuit, copy_operations: bool) -> PyResult<CircuitData> {
    CircuitData::from_packed_instructions(
        dag.qubits().clone(),
        dag.clbits().clone(),
        dag.qargs_interner().clone(),
        dag.cargs_interner().clone(),
        dag.qregs_data().clone(),
        dag.cregs_data().clone(),
        dag.qubit_locations().clone(),
        dag.clbit_locations().clone(),
        dag.topological_op_nodes()?.map(|node_index| {
            let NodeType::Operation(ref instr) = dag[node_index] else {
                unreachable!(
                    "The received node from topological_op_nodes() is not an Operation node."
                )
            };
            if copy_operations {
                let op = match instr.op.view() {
                    OperationRef::Gate(gate) => {
                        Python::with_gil(|py| gate.py_deepcopy(py, None))?.into()
                    }
                    OperationRef::Instruction(instruction) => {
                        Python::with_gil(|py| instruction.py_deepcopy(py, None))?.into()
                    }
                    OperationRef::Operation(operation) => {
                        Python::with_gil(|py| operation.py_deepcopy(py, None))?.into()
                    }
                    OperationRef::StandardGate(gate) => gate.into(),
                    OperationRef::StandardInstruction(instruction) => instruction.into(),
                    OperationRef::Unitary(unitary) => unitary.clone().into(),
                };
                Ok(PackedInstruction {
                    op,
                    qubits: instr.qubits,
                    clbits: instr.clbits,
                    params: Some(Box::new(instr.params_view().iter().cloned().collect())),
                    label: instr.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                })
            } else {
                Ok(instr.clone())
            }
        }),
        dag.get_global_phase(),
        dag.identifiers() // Map and pass DAGCircuit variables and stretches to CircuitData style
            .map(|identifier| match identifier {
                DAGIdentifierInfo::Stretch(dag_stretch_info) => CircuitVar::Stretch(
                    dag.get_stretch(dag_stretch_info.get_stretch())
                        .expect("Stretch not found for the specified index")
                        .clone(),
                    dag_stretch_info.get_type().into(),
                ),
                DAGIdentifierInfo::Var(dag_var_info) => CircuitVar::Var(
                    dag.get_var(dag_var_info.get_var())
                        .expect("Var not found for the specified index")
                        .clone(),
                    dag_var_info.get_type().into(),
                ),
            })
            .collect::<Vec<CircuitVar>>(),
    )
}

pub fn converters(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(circuit_to_dag, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_circuit, m)?)?;
    Ok(())
}
