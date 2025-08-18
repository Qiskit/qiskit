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

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::imports::CIRCUIT_TO_DAG;
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::{PhysicalQubit, Qubit};

use crate::target::Target;

fn recurse(
    py: Python,
    dag: &DAGCircuit,
    target: &Target,
    wire_map: Option<&[Qubit]>,
) -> PyResult<Option<(String, [u32; 2])>> {
    let check_qubits = |qubits: &[Qubit]| -> bool {
        match wire_map {
            Some(wire_map) => {
                let mapped_bits = [
                    PhysicalQubit(wire_map[qubits[0].index()].0),
                    PhysicalQubit(wire_map[qubits[1].index()].0),
                ];
                target.contains_qargs(&mapped_bits)
                    || target.contains_qargs(&[mapped_bits[1], mapped_bits[0]])
            }
            None => {
                target.contains_qargs(&[PhysicalQubit(qubits[0].0), PhysicalQubit(qubits[1].0)])
                    || target
                        .contains_qargs(&[PhysicalQubit(qubits[1].0), PhysicalQubit(qubits[0].0)])
            }
        }
    };

    for (_node, inst) in dag.op_nodes(false) {
        let qubits = dag.get_qargs(inst.qubits);
        if inst.op.control_flow() {
            if let OperationRef::Instruction(py_inst) = inst.op.view() {
                let raw_blocks = py_inst.instruction.getattr(py, "blocks")?;
                let circuit_to_dag = CIRCUIT_TO_DAG.get_bound(py);
                for raw_block in raw_blocks.bind(py).try_iter()? {
                    let block_obj = raw_block?;
                    let block = block_obj
                        .getattr(intern!(py, "_data"))?
                        .downcast::<CircuitData>()?
                        .borrow();
                    let new_dag: DAGCircuit =
                        circuit_to_dag.call1((block_obj.clone(),))?.extract()?;
                    let wire_map = (0..block.num_qubits())
                        .map(|inner| {
                            let outer = qubits[inner];
                            match wire_map {
                                Some(wire_map) => wire_map[outer.index()],
                                None => outer,
                            }
                        })
                        .collect::<Vec<_>>();
                    let res = recurse(py, &new_dag, target, Some(&wire_map))?;
                    if res.is_some() {
                        return Ok(res);
                    }
                }
            }
        } else if qubits.len() == 2 && !check_qubits(qubits) {
            return Ok(Some((
                inst.op.name().to_string(),
                [qubits[0].0, qubits[1].0],
            )));
        }
    }
    Ok(None)
}

#[pyfunction]
#[pyo3(name = "check_map")]
pub fn py_run_check_map(
    py: Python,
    dag: &DAGCircuit,
    target: &Target,
) -> PyResult<Option<(String, [u32; 2])>> {
    if dag.has_control_flow() {
        recurse(py, dag, target, None)
    } else {
        Ok(run_check_map(dag, target)
            .map(|(name, qubits)| (name.to_string(), [qubits[0].0, qubits[1].0])))
    }
}

/// Check that all 2q gates are in the target
pub fn run_check_map<'a>(
    dag: &'a DAGCircuit,
    target: &Target,
) -> Option<(&'a str, [PhysicalQubit; 2])> {
    dag.op_nodes(false)
        .filter(|(_idx, inst)| inst.op.num_qubits() == 2)
        .find_map(|(_idx, inst)| {
            let qargs_raw = dag.get_qargs(inst.qubits);
            let qargs = [
                PhysicalQubit::new(qargs_raw[0].0),
                PhysicalQubit::new(qargs_raw[1].0),
            ];
            if !target.contains_qargs(&qargs) && !target.contains_qargs(&[qargs[1], qargs[0]]) {
                Some((inst.op.name(), [qargs[0], qargs[1]]))
            } else {
                None
            }
        })
}

pub fn check_map_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_check_map))?;
    Ok(())
}
