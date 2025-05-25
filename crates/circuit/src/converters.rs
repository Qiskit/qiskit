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

use crate::circuit_data::CircuitData;
use crate::classical::expr;
use crate::dag_circuit::{DAGCircuit, NodeType};
use crate::packed_instruction::PackedInstruction;

/// An extractable representation of a QuantumCircuit reserved only for
/// conversion purposes.
#[derive(Debug, Clone)]
pub struct QuantumCircuitData<'py> {
    pub data: CircuitData,
    pub name: Option<String>,
    pub metadata: Option<Bound<'py, PyAny>>,
    pub input_vars: Vec<expr::Var>,
    pub captured_vars: Vec<expr::Var>,
    pub declared_vars: Vec<expr::Var>,
    pub captured_stretches: Vec<expr::Stretch>,
    pub declared_stretches: Vec<expr::Stretch>,
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
            input_vars: ob
                .call_method0(intern!(py, "iter_input_vars"))?
                .try_iter()?
                .map(|x| x?.extract())
                .collect::<PyResult<Vec<_>>>()?,
            captured_vars: ob
                .call_method0(intern!(py, "iter_captured_vars"))?
                .try_iter()?
                .map(|x| x?.extract())
                .collect::<PyResult<Vec<_>>>()?,
            declared_vars: ob
                .call_method0(intern!(py, "iter_declared_vars"))?
                .try_iter()?
                .map(|x| x?.extract())
                .collect::<PyResult<Vec<_>>>()?,
            captured_stretches: ob
                .call_method0(intern!(py, "iter_captured_stretches"))?
                .try_iter()?
                .map(|x| x?.extract())
                .collect::<PyResult<Vec<_>>>()?,
            declared_stretches: ob
                .call_method0(intern!(py, "iter_declared_stretches"))?
                .try_iter()?
                .map(|x| x?.extract())
                .collect::<PyResult<Vec<_>>>()?,
        })
    }
}

#[pyfunction(signature = (quantum_circuit, copy_operations = true, qubit_order = None, clbit_order = None))]
pub fn circuit_to_dag(
    py: Python,
    quantum_circuit: QuantumCircuitData,
    copy_operations: bool,
    qubit_order: Option<Vec<Bound<PyAny>>>,
    clbit_order: Option<Vec<Bound<PyAny>>>,
) -> PyResult<DAGCircuit> {
    DAGCircuit::from_circuit(
        py,
        quantum_circuit,
        copy_operations,
        qubit_order,
        clbit_order,
    )
}

#[pyfunction(signature = (dag, copy_operations = true))]
pub fn dag_to_circuit(
    py: Python,
    dag: &DAGCircuit,
    copy_operations: bool,
) -> PyResult<CircuitData> {
    CircuitData::from_packed_instructions(
        py,
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
                let op = instr.op.py_deepcopy(py, None)?;
                Ok(PackedInstruction {
                    op,
                    qubits: instr.qubits,
                    clbits: instr.clbits,
                    params: Some(Box::new(
                        instr
                            .params_view()
                            .iter()
                            .map(|param| param.clone_ref(py))
                            .collect(),
                    )),
                    label: instr.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                })
            } else {
                Ok(instr.clone())
            }
        }),
        dag.get_global_phase(),
    )
}

pub fn converters(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(circuit_to_dag, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_circuit, m)?)?;
    Ok(())
}
