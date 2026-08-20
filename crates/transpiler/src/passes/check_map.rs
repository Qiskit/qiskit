// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::{PhysicalQubit, Qubit};

use crate::target::Target;

fn recurse<'a>(
    dag: &'a DAGCircuit,
    target: &Target,
    wire_map: Option<&[Qubit]>,
) -> Option<(&'a str, [PhysicalQubit; 2])> {
    let mapped_qargs = |qubits: &[Qubit]| -> [PhysicalQubit; 2] {
        match wire_map {
            Some(wire_map) => [
                PhysicalQubit(wire_map[qubits[0].index()].0),
                PhysicalQubit(wire_map[qubits[1].index()].0),
            ],
            None => [PhysicalQubit(qubits[0].0), PhysicalQubit(qubits[1].0)],
        }
    };

    for (_, inst) in dag.op_nodes(false) {
        let qubits = dag.get_qargs(inst.qubits);
        if let Some(control_flow) = dag.try_view_control_flow(inst) {
            for block in control_flow.blocks() {
                let wire_map = (0..block.num_qubits())
                    .map(|inner| {
                        let outer = qubits[inner];
                        match wire_map {
                            Some(wire_map) => wire_map[outer.index()],
                            None => outer,
                        }
                    })
                    .collect::<Vec<_>>();

                let res = recurse(block, target, Some(&wire_map));
                if res.is_some() {
                    return res;
                }
            }
        } else if qubits.len() == 2 {
            let qargs = mapped_qargs(qubits);
            if !target.contains_qargs(&qargs) && !target.contains_qargs(&[qargs[1], qargs[0]]) {
                return Some((inst.op.name(), qargs));
            }
        }
    }
    None
}

#[pyfunction]
#[pyo3(name = "check_map")]
pub fn py_run_check_map(dag: &DAGCircuit, target: &Target) -> PyResult<Option<(String, [u32; 2])>> {
    Ok(run_check_map(dag, target)
        .map(|(name, qubits)| (name.to_string(), [qubits[0].0, qubits[1].0])))
}

/// Check that all 2q gates are in the target
pub fn run_check_map<'a>(
    dag: &'a DAGCircuit,
    target: &Target,
) -> Option<(&'a str, [PhysicalQubit; 2])> {
    recurse(dag, target, None)
}

pub fn check_map_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_check_map))?;
    Ok(())
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::target::InstructionProperties;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::instruction::Parameters;
    use qiskit_circuit::operations::{
        ControlFlow, ControlFlowInstruction, ForCollection, Param, StandardGate,
    };
    use qiskit_circuit::packed_instruction::PackedOperation;
    use smallvec::smallvec;

    fn target_with_cx_edges(edges: &[(u32, u32)]) -> Target {
        let mut target = Target::default();
        let props = edges
            .iter()
            .map(|&(source, target)| {
                (
                    [PhysicalQubit(source), PhysicalQubit(target)].into(),
                    None::<InstructionProperties>,
                )
            })
            .collect();
        target
            .add_instruction(StandardGate::CX.into(), None, None, Some(props))
            .unwrap();
        target
    }

    fn cx_body(num_qubits: u32, control: u32, target: u32) -> CircuitData {
        CircuitData::from_packed_operations(
            num_qubits,
            0,
            [Ok((
                StandardGate::CX.into(),
                smallvec![],
                vec![Qubit(control), Qubit(target)],
                vec![],
            ))],
            Param::Float(0.0),
        )
        .unwrap()
    }

    fn add_for_loop(parent: &mut CircuitData, body: CircuitData, qargs: &[Qubit]) {
        let block = parent.add_block(body);
        let control_flow = ControlFlowInstruction {
            control_flow: ControlFlow::ForLoop {
                collection: ForCollection::List(vec![0]),
                loop_param: None,
            },
            num_qubits: qargs.len() as u32,
            num_clbits: 0,
        };
        parent
            .push_packed_operation(
                PackedOperation::from_control_flow(Box::new(control_flow)),
                Some(Parameters::Blocks(vec![block])),
                qargs,
                &[],
            )
            .unwrap();
    }

    fn circuit_to_dag(circuit: &CircuitData) -> DAGCircuit {
        DAGCircuit::from_circuit_data(circuit, false, None, None, None, None).unwrap()
    }

    #[test]
    fn detects_top_level_violation() {
        let target = target_with_cx_edges(&[(0, 1), (1, 2)]);
        let dag = circuit_to_dag(&cx_body(3, 0, 2));

        assert_eq!(
            run_check_map(&dag, &target),
            Some(("cx", [PhysicalQubit(0), PhysicalQubit(2)]))
        );
    }

    #[test]
    fn detects_control_flow_violation() {
        let target = target_with_cx_edges(&[(0, 1), (1, 2)]);
        let mut circuit = CircuitData::with_capacity(3, 0, 1, Param::Float(0.0)).unwrap();
        add_for_loop(
            &mut circuit,
            cx_body(3, 0, 2),
            &[Qubit(0), Qubit(1), Qubit(2)],
        );
        let dag = circuit_to_dag(&circuit);

        assert_eq!(
            run_check_map(&dag, &target),
            Some(("cx", [PhysicalQubit(0), PhysicalQubit(2)]))
        );
    }

    #[test]
    fn maps_control_flow_block_qubits_to_parent_qubits() {
        let target = target_with_cx_edges(&[(0, 1), (1, 2)]);
        let mut circuit = CircuitData::with_capacity(3, 0, 1, Param::Float(0.0)).unwrap();
        add_for_loop(
            &mut circuit,
            cx_body(3, 0, 2),
            &[Qubit(1), Qubit(0), Qubit(2)],
        );
        let dag = circuit_to_dag(&circuit);

        assert_eq!(run_check_map(&dag, &target), None);
    }

    #[test]
    fn composes_nested_control_flow_qubit_maps() {
        let target = target_with_cx_edges(&[(0, 1), (1, 2)]);
        let mut inner = CircuitData::with_capacity(3, 0, 1, Param::Float(0.0)).unwrap();
        add_for_loop(
            &mut inner,
            cx_body(3, 0, 2),
            &[Qubit(2), Qubit(1), Qubit(0)],
        );

        let mut outer = CircuitData::with_capacity(3, 0, 1, Param::Float(0.0)).unwrap();
        add_for_loop(&mut outer, inner, &[Qubit(2), Qubit(1), Qubit(0)]);
        let dag = circuit_to_dag(&outer);

        assert_eq!(
            run_check_map(&dag, &target),
            Some(("cx", [PhysicalQubit(0), PhysicalQubit(2)]))
        );
    }
}
