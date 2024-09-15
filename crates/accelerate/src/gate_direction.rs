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
use pyo3::intern;
use pyo3::types::PyTuple;
use hashbrown::HashSet;
use smallvec::smallvec;
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::Target;
use qiskit_circuit::imports;
use qiskit_circuit::{
    dag_circuit::{DAGCircuit, NodeType},
    packed_instruction::PackedInstruction,
    Qubit, TupleLikeArg,
    operations::{OperationRef, StandardGate, Param, Operation},
    circuit_instruction::ExtraInstructionAttributes,
};
use crate::target_transpiler::exceptions::TranspilerError;

//#########################################################################
//              CheckGateDirection analysis pass functions
//#########################################################################


/// Check if the two-qubit gates follow the right direction with respect to the coupling map.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     coupling_edges: set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// Returns:
///     true iff all two-qubit gates comply with the coupling constraints
#[pyfunction]
#[pyo3(name = "check_gate_direction_coupling")]
fn py_check_with_coupling_map(
    py: Python,
    dag: &DAGCircuit,
    coupling_edges: HashSet<[Qubit; 2]>,
) -> PyResult<bool> {
    let coupling_map_check =
        |_: &PackedInstruction, op_args: &[Qubit]| -> bool { coupling_edges.contains(op_args) };

    check_gate_direction(py, dag, &coupling_map_check, None)
}


/// Check if the two-qubit gates follow the right direction with respect to instructions supported in the given target.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     target: the Target against which gate directionality compliance is checked
///
/// Returns:
///     true iff all two-qubit gates comply with the target's coupling constraints
#[pyfunction]
#[pyo3(name = "check_gate_direction_target")]
fn py_check_with_target(py: Python, dag: &DAGCircuit, target: &Target) -> PyResult<bool> {
    let target_check = |inst: &PackedInstruction, op_args: &[Qubit]| -> bool {
        let qargs = smallvec![
            PhysicalQubit::new(op_args[0].0),
            PhysicalQubit::new(op_args[1].0)
        ];

        target.instruction_supported(inst.op.name(), Some(&qargs))
    };

    check_gate_direction(py, dag, &target_check, None)
}

// The main routine for checking gate directionality.
//
// gate_complies: a function returning true iff the two-qubit gate direction complies with directionality constraints
//
// qubit_mapping: used for mapping the index of a given qubit within an instruction qargs vector to the corresponding qubit index of the
//  original DAGCircuit the pass was called with. This mapping is required since control flow blocks are represented by nested DAGCircuit
//  objects whose instruction qubit indices are relative to the parent DAGCircuit they reside in, thus when we recurse into nested DAGs, we need
//  to carry the mapping context relative to the original DAG.
//  When qubit_mapping is None, the identity mapping is assumed
fn check_gate_direction<T>(
    py: Python,
    dag: &DAGCircuit,
    gate_complies: &T,
    qubit_mapping: Option<&[Qubit]>,
) -> PyResult<bool>
where
    T: Fn(&PackedInstruction, &[Qubit]) -> bool,
{
    for node in dag.op_nodes(false) {
        let NodeType::Operation(packed_inst) = &dag.dag()[node] else {
            panic!("PackedInstruction is expected");
        };

        let inst_qargs = dag.get_qargs(packed_inst.qubits);

        if let OperationRef::Instruction(py_inst) = packed_inst.op.view() {
            if py_inst.control_flow() {
                let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py); // TODO: Take out of the recursion
                let py_inst = py_inst.instruction.bind(py);

                for block in py_inst.getattr("blocks")?.iter()? {
                    let inner_dag: DAGCircuit = circuit_to_dag.call1((block?,))?.extract()?;

                    let block_ok = if let Some(mapping) = qubit_mapping {
                        let mapping = inst_qargs // Create a temp mapping for the recursive call
                            .iter()
                            .map(|q| mapping[q.0 as usize])
                            .collect::<Vec<Qubit>>();

                        check_gate_direction(py, &inner_dag, gate_complies, Some(&mapping))?
                    } else {
                        check_gate_direction(py, &inner_dag, gate_complies, Some(inst_qargs))?
                    };

                    if !block_ok {
                        return Ok(false);
                    }
                }
                continue;
            }
        }

        if inst_qargs.len() == 2
            && !match qubit_mapping {
                // Check gate direction based either on a given custom mapping or the identity mapping
                Some(mapping) => gate_complies(
                    packed_inst,
                    &[
                        mapping[inst_qargs[0].0 as usize],
                        mapping[inst_qargs[1].0 as usize],
                    ],
                ),
                None => gate_complies(packed_inst, inst_qargs),
            }
        {
            return Ok(false);
        }
    }

    Ok(true)
}

//#########################################################################
//              GateDirection transformation pass functions
//#########################################################################


// type GateDirectionCheckFn<'a> = Box<dyn Fn(&DAGCircuit, &PackedInstruction, &[Qubit]) -> bool + 'a>;

// // Return a closure function that checks whether the direction of a given gate complies with the given coupling map. This is used in the
// // pass functions below
// fn coupling_direction_checker<'a>(py: &'a Python, dag: &'a DAGCircuit, coupling_edges: &'a Bound<PySet>) -> GateDirectionCheckFn<'a> {
//     Box::new(move |curr_dag: &DAGCircuit, _: &PackedInstruction, op_args: &[Qubit]| -> bool {
//         coupling_edges
//             .contains((
//                 map_qubit(&py, dag, curr_dag, op_args[0]).0,
//                 map_qubit(&py, dag, curr_dag, op_args[1]).0,
//             ))
//             .unwrap_or(false)
//         })
// }

// // Return a closure function that checks whether the direction of a given gate complies with the given target. This is used in the
// // pass functions below
// fn target_direction_checker<'a>(py: &'a Python, dag: &'a DAGCircuit, target: PyRef<'a, Target>) -> GateDirectionCheckFn<'a> {
//     Box::new(move |curr_dag: &DAGCircuit, inst: &PackedInstruction, op_args: &[Qubit]| -> bool {
//         let mut qargs = Qargs::new();

//         qargs.push(PhysicalQubit::new(
//             map_qubit(py, dag, curr_dag, op_args[0]).0,
//         ));
//         qargs.push(PhysicalQubit::new(
//             map_qubit(py, dag, curr_dag, op_args[1]).0,
//         ));

//         target.instruction_supported(inst.op.name(), Some(&qargs))
//     })
// }

///
///
///
///
///
#[pyfunction]
#[pyo3(name = "fix_gate_direction_coupling")]
fn py_fix_with_coupling_map(py: Python, dag: &DAGCircuit, coupling_edges: HashSet<[Qubit; 2]>) -> PyResult<DAGCircuit> {
    let coupling_map_check =
    |_: &PackedInstruction, op_args: &[Qubit]| -> bool { coupling_edges.contains(op_args) };


    fix_gate_direction(py, dag, coupling_map_check)
}

fn fix_gate_direction<T>(py: Python, dag: &DAGCircuit, gate_complies: T)  -> PyResult<DAGCircuit>
where T: Fn(&PackedInstruction, &[Qubit]) -> bool

{
    for node in dag.op_nodes(false) {
        let NodeType::Operation(packed_inst) = &dag.dag()[node] else {panic!("PackedInstruction is expected");
        };

        if let OperationRef::Instruction(py_inst) = packed_inst.op.view() {
            if py_inst.control_flow() {
                todo!("direction fix control flow blocks");
            }
        }

        let op_args = dag.get_qargs(packed_inst.qubits);
        if op_args.len() != 2 {continue;}

        if !gate_complies(packed_inst, op_args) {
            if !gate_complies(packed_inst, &[op_args[1], op_args[0]]) {
                return Err(TranspilerError::new_err(format!("The circuit requires a connection between physical qubits {:?}", op_args)));
            }

            if let OperationRef::Standard(std_gate) = packed_inst.op.view() {
                match std_gate {
                    StandardGate::CXGate |
                    StandardGate::CZGate |
                    StandardGate::ECRGate |
                    StandardGate::SwapGate |
                    StandardGate::RZXGate |
                    StandardGate::RXXGate |
                    StandardGate::RYYGate => todo!("Direction fix for {:?}", std_gate),
                    StandardGate::RZZGate => println!("PARAMs: {:?}", packed_inst.params),
                    _ => continue,
                }
            }
        }
    }

    Ok(dag.clone()) // TODO: avoid cloning
}

//###################################################
// Utility functions to build the replacement dags
// TODO: replace once we have fully Rust-friendly versions of QuantumRegister, DAGCircuit and ParemeterExpression

fn create_qreg<'py>(py: Python<'py>, size: u32) -> PyResult<Bound<'py, PyAny>> {
    imports::QUANTUM_REGISTER.get_bound(py).call1((size,))
}

fn qreg_bit<'py>(py: Python, qreg: &Bound<'py, PyAny>, index: u32) -> PyResult<Bound<'py, PyAny>> {
    qreg.call_method1(intern!(py, "__getitem__"), (index,))
}

fn std_gate(py: Python, gate: StandardGate) -> PyResult<Py<PyAny>> {
    gate.create_py_op(py, None, &ExtraInstructionAttributes::new(None, None, None, None))
}

fn parameterized_std_gate(py: Python, gate: StandardGate, param: Param) -> PyResult<Py<PyAny>> {
    gate.create_py_op(py, Some(&[param]), &ExtraInstructionAttributes::new(None, None, None, None))
}

fn apply_op_back(py: Python, dag: &mut DAGCircuit, op: &Py<PyAny>, qargs: &Vec<&Bound<PyAny>>) -> PyResult<()> {
    dag.py_apply_operation_back(py,
        op.bind(py).clone(),
        Some( TupleLikeArg::extract_bound( &PyTuple::new_bound(py, qargs))? ),
        None,
        false)?;

    Ok(())
}

// fn build_dag(py: Python) -> PyResult<DAGCircuit> {
//     let qreg = create_qreg(py, 2)?;
//     let new_dag = &mut DAGCircuit::new(py)?;
//     new_dag.add_qreg(py, &qreg)?;

//     let (q0, q1) = (qreg_bit(py, &qreg, 0)?, qreg_bit(py, &qreg, 0)?);

//     apply_standard_gate_back(py, new_dag, StandardGate::HGate, &vec![&q0])?;
//     apply_standard_gate_back(py, new_dag, StandardGate::CXGate, &vec![&q0, &q1])?;

//     Ok( new_dag.clone() ) // TODO: Get rid of the clone
// }

fn cx_replacement_dag(py: Python) -> PyResult<DAGCircuit> {
    let qreg = create_qreg(py, 2)?;
    let new_dag = &mut DAGCircuit::new(py)?;
    new_dag.add_qreg(py, &qreg)?;

    let (q0, q1) = (qreg_bit(py, &qreg, 0)?, qreg_bit(py, &qreg, 0)?);
    apply_op_back(py, new_dag, &std_gate(py, StandardGate::HGate)?, &vec![&q0])?;
    apply_op_back(py, new_dag, &std_gate(py, StandardGate::HGate)?, &vec![&q1])?;
    apply_op_back(py, new_dag, &std_gate(py, StandardGate::HGate)?, &vec![&q1, &q0])?;
    apply_op_back(py, new_dag, &std_gate(py, StandardGate::HGate)?, &vec![&q0])?;
    apply_op_back(py, new_dag, &std_gate(py, StandardGate::HGate)?, &vec![&q1])?;

    Ok( new_dag.clone() ) // TODO: Get rid of the clone
}


#[pymodule]
pub fn gate_direction(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_check_with_coupling_map))?;
    m.add_wrapped(wrap_pyfunction!(py_check_with_target))?;
    m.add_wrapped(wrap_pyfunction!(py_fix_with_coupling_map))?;
    Ok(())
}
