use pyo3::prelude::*;
use pyo3::types::{PySet, PyTuple};
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::{Qargs, Target};
use qiskit_circuit::imports;
use qiskit_circuit::operations::{OperationRef, PyInstruction};
use qiskit_circuit::{
    dag_circuit::{DAGCircuit, NodeType},
    operations::Operation,
    packed_instruction::PackedInstruction,
    Qubit,
};

// Handle control flow instruction, namely recurse into its circuit blocks in the analysis
fn check_gate_direction_control_flow<T>(
    py: Python,
    py_inst: &PyInstruction,
    gate_complies: &T,
) -> PyResult<bool>
where
    T: Fn(&DAGCircuit, &PackedInstruction, &Vec<Qubit>) -> bool,
{
    let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py); // TODO: Take out of the recursion
    let py_inst = py_inst.instruction.bind(py);

    let raw_blocks = py_inst.getattr("blocks")?;
    let blocks = raw_blocks.downcast::<PyTuple>()?;

    for block in blocks.iter() {
        let inner_dag: DAGCircuit = circuit_to_dag.call1((block,))?.extract()?;

        if !check_gate_direction(py, &inner_dag, gate_complies)? {
            return Ok(false);
        }
    }

    Ok(true)
}

// The main analysis pass routine.
fn check_gate_direction<T>(py: Python, dag: &DAGCircuit, gate_complies: &T) -> PyResult<bool>
where
    T: Fn(&DAGCircuit, &PackedInstruction, &Vec<Qubit>) -> bool,
{
    for node in dag.op_nodes(false) {
        let Some(NodeType::Operation(packed_inst)) = dag.dag.node_weight(node) else {
            return Ok(false);
        }; // TODO: proper error

        if let OperationRef::Instruction(py_inst) = packed_inst.op.view() {
            if py_inst.control_flow()
                && !check_gate_direction_control_flow(py, py_inst, gate_complies)?
            {
                return Ok(false);
            }
        } else {
            let op_args = dag.get_qubits(packed_inst.qubits);
            if op_args.len() == 2 && !gate_complies(dag, packed_inst, op_args) {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

// Map a qubit interned in curr_dag to its corresponding qubit entry interned in orig_dag.
// Required for checking control flow instruction which are represented in blocks (circuits)
// and converted to DAGCircuit with possibly different qargs than the original one.
fn map_qubit(py: Python, orig_dag: &DAGCircuit, curr_dag: &DAGCircuit, qubit: Qubit) -> Qubit {
    let qubit = curr_dag
        .qubits
        .get(qubit)
        .expect("Qubit in curr_dag")
        .bind(py);
    orig_dag.qubits.find(qubit).expect("Qubit in orig_dag")
}

#[pyfunction]
#[pyo3(name = "check_gate_direction_coupling")]
fn py_check_with_coupling_map(
    py: Python,
    dag: &DAGCircuit,
    coupling_edges: &Bound<PySet>,
) -> PyResult<bool> {
    let coupling_map_check =
        |curr_dag: &DAGCircuit, _: &PackedInstruction, op_args: &Vec<Qubit>| -> bool {
            coupling_edges
                .contains((
                    map_qubit(py, dag, curr_dag, op_args[0]).0,
                    map_qubit(py, dag, curr_dag, op_args[1]).0,
                ))
                .unwrap_or(false)
        };

    check_gate_direction(py, dag, &coupling_map_check)
}

#[pyfunction]
#[pyo3(name = "check_gate_direction_target")]
fn py_check_with_target(py: Python, dag: &DAGCircuit, target: &Bound<Target>) -> PyResult<bool> {
    let target = target.borrow();

    let target_check =
        |curr_dag: &DAGCircuit, inst: &PackedInstruction, op_args: &Vec<Qubit>| -> bool {
            let mut qargs = Qargs::new();

            qargs.push(PhysicalQubit::new(
                map_qubit(py, dag, curr_dag, op_args[0]).0,
            ));
            qargs.push(PhysicalQubit::new(
                map_qubit(py, dag, curr_dag, op_args[1]).0,
            ));

            target.instruction_supported(inst.op.name(), Some(&qargs))
        };

    check_gate_direction(py, dag, &target_check)
}

#[pymodule]
pub fn gate_direction(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_check_with_coupling_map))?;
    m.add_wrapped(wrap_pyfunction!(py_check_with_target))?;
    Ok(())
}
