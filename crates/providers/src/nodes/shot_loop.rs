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

//! A node that runs each of a list of circuits for a number of shots, over a sweep of
//! parameterizations.

use std::sync::Arc;

use qiskit_circuit::circuit_data::CircuitData;
use thiserror::Error;

use super::{OpNodeType, QISKIT};
use crate::data_tree::{DataTree, InvalidName};
use crate::tensor::{DType, Dim, Tensor, TensorType};

/// Run each of a set of circuits for a fixed number of shots.
///
/// There is one operand per circuit for that circuit's parameter values.
/// For each circuit, one value is returned per classical register, keyed by register name.
#[derive(Clone)]
pub struct ShotLoop {
    /// One circuit per operand, shared so that cloning this node clones no circuit.
    circuits: Vec<Arc<CircuitData>>,
    shots: usize,
    output_structure: DataTree<()>,
}

impl ShotLoop {
    /// Construct a shot loop running each of `circuits` for `shots` shots.
    ///
    /// Every classical register's name must be usable as a [`Name`](crate::Name).
    pub fn new(circuits: Vec<Arc<CircuitData>>, shots: usize) -> Result<Self, ShotLoopError> {
        let mut output_structure = DataTree::with_capacity(circuits.len());
        for (index, circuit) in circuits.iter().enumerate() {
            let names = circuit
                .cregs()
                .iter()
                .map(|register| (register.name(), DataTree::Leaf(())));
            output_structure.push_branch(DataTree::mapping(names).map_err(|source| {
                ShotLoopError::RegisterName {
                    circuit: index,
                    source,
                }
            })?);
        }
        Ok(Self {
            circuits,
            shots,
            output_structure,
        })
    }

    /// The circuits this node runs.
    pub fn circuits(&self) -> &[Arc<CircuitData>] {
        &self.circuits
    }

    /// How many shots each circuit is run for.
    pub fn shots(&self) -> usize {
        self.shots
    }

    /// How this node's results are arranged.
    ///
    /// There is one entry per circuit, each keyed by classical register name.
    pub fn output_structure(&self) -> &DataTree<()> {
        &self.output_structure
    }
}

impl OpNodeType for ShotLoop {
    type Error = ShotLoopError;

    fn name(&self) -> &str {
        "shot_loop"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        self.circuits.len()
    }
    fn has_builtin_eval(&self) -> bool {
        false
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        assert_eq!(
            inputs.len(),
            self.arity(),
            "{} expects one operand per circuit",
            self.full_name()
        );
        let mut outputs = Vec::with_capacity(self.output_structure.leaf_count());
        for (index, (circuit, operand)) in self.circuits.iter().zip(inputs).enumerate() {
            let parameters = circuit.num_parameters();
            let batch =
                leading_axes(parameters, operand).ok_or_else(|| ShotLoopError::ParameterType {
                    circuit: index,
                    parameters,
                    actual: operand.clone(),
                })?;
            for register in circuit.cregs() {
                let mut shape = batch.to_vec();
                shape.push(Dim::Fixed(self.shots));
                shape.push(Dim::Fixed(register.len()));
                outputs.push(TensorType {
                    dtype: DType::Bit,
                    shape,
                });
            }
        }
        Ok(outputs)
    }
    fn eval(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        Err(ShotLoopError::NoBuiltinEval)
    }
}

/// Strip out the leading axes, assuming the last axis matches the number of parameters.
fn leading_axes(parameters: usize, operand: &TensorType) -> Option<&[Dim]> {
    let float = matches!(operand.dtype, DType::F32 | DType::F64);
    match operand.shape.split_last() {
        Some((&Dim::Fixed(values), batch)) if float && values == parameters => Some(batch),
        _ => None,
    }
}

/// Errors returned by [`ShotLoop`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShotLoopError {
    /// A classical register's name cannot name a slot of the result.
    #[error("circuit {circuit}: {source}")]
    RegisterName {
        circuit: usize,
        #[source]
        source: InvalidName,
    },

    /// A circuit's parameter values are not a floating-point tensor whose trailing axis is that
    /// circuit's parameter count.
    #[error(
        "circuit {circuit}: expected a floating-point tensor of shape [..., {parameters}], got \
         {actual}"
    )]
    ParameterType {
        circuit: usize,
        parameters: usize,
        actual: TensorType,
    },

    /// Circuits must be executed by a backend.
    #[error("a shot loop has no in-process implementation")]
    NoBuiltinEval,
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use qiskit_circuit::bit::ClassicalRegister;
    use qiskit_circuit::operations::Param;
    use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
    use qiskit_circuit::parameter::symbol_expr::Symbol;

    use super::*;
    use crate::nodes::Mean;
    use crate::program::{ProgramFunction, QuantumProgram};

    /// A circuit taking `parameters` parameters and holding `registers` as `(name, width)` pairs.
    ///
    /// A shot loop reads a circuit's parameter count and its classical registers and nothing else, so
    /// a global phase over that many symbols stands in for parameterized instructions.
    fn circuit(parameters: usize, registers: &[(&str, u32)]) -> Arc<CircuitData> {
        let symbol =
            |index| ParameterExpression::from_symbol(Symbol::standalone(format!("p{index}"), None));
        let global_phase = (0..parameters)
            .map(symbol)
            .reduce(|sum, term| sum.add(&term).unwrap())
            .map_or(Param::Float(0.0), |expr| {
                Param::ParameterExpression(Arc::new(expr))
            });

        let mut circuit = CircuitData::new(None, None, global_phase).unwrap();
        for &(name, width) in registers {
            circuit
                .add_creg(ClassicalRegister::new_owning(name, width), true)
                .unwrap();
        }
        Arc::new(circuit)
    }

    /// A `TensorType` of `dtype` over fixed axes `shape`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    /// The `Bit` type of one register's outcomes: `batch`, then `shots` by the register's width.
    fn outcomes(batch: &[Dim], shots: usize, width: usize) -> TensorType {
        let mut shape = batch.to_vec();
        shape.extend([Dim::Fixed(shots), Dim::Fixed(width)]);
        TensorType {
            dtype: DType::Bit,
            shape,
        }
    }

    #[test]
    fn test_shot_loop_full_name_and_arity() {
        let node = ShotLoop::new(vec![circuit(1, &[("c", 2)]), circuit(0, &[])], 100).unwrap();
        assert_eq!(node.full_name(), "qiskit.shot_loop");
        assert_eq!(node.arity(), 2, "one operand per circuit");
        assert_eq!(node.shots(), 100);
        assert_eq!(node.circuits().len(), 2);
        assert!(
            !node.has_builtin_eval(),
            "a backend has to run the circuits"
        );
    }

    #[test]
    fn test_output_structure_is_one_entry_per_circuit_keyed_by_register() {
        let node = ShotLoop::new(
            vec![
                circuit(0, &[("c", 2), ("meas", 3)]),
                circuit(0, &[("d", 1)]),
                circuit(0, &[]),
            ],
            100,
        )
        .unwrap();
        assert_eq!(
            node.output_structure().to_string(),
            "[[c: _, meas: _], [d: _], []]"
        );
        assert_eq!(
            node.output_structure().dotted_paths(),
            ["0.c", "0.meas", "1.d"]
        );
    }

    #[test]
    fn test_infer_output_types_are_bit_outcomes_per_register() {
        let node = ShotLoop::new(
            vec![
                circuit(2, &[("c", 2), ("meas", 3)]),
                circuit(1, &[("d", 1)]),
            ],
            100,
        )
        .unwrap();
        assert_eq!(
            node.infer_output_types(&[ty(DType::F64, &[2]), ty(DType::F64, &[1])])
                .unwrap(),
            vec![
                outcomes(&[], 100, 2),
                outcomes(&[], 100, 3),
                outcomes(&[], 100, 1),
            ],
            "one result per register, in the order the structure describes"
        );
    }

    #[test]
    fn test_infer_output_types_carries_each_operands_batch_prefix() {
        // The prefix is opaque, so any rank of it passes through, including an axis whose size is
        // only bounded. The two circuits' prefixes are independent of each other.
        let node =
            ShotLoop::new(vec![circuit(2, &[("c", 2)]), circuit(1, &[("d", 1)])], 8).unwrap();
        let bounded = Dim::Bounded { max: 4 };
        assert_eq!(
            node.infer_output_types(&[
                ty(DType::F32, &[5, 3, 2]),
                TensorType {
                    dtype: DType::F64,
                    shape: vec![bounded, Dim::Fixed(1)],
                },
            ])
            .unwrap(),
            vec![
                outcomes(&[Dim::Fixed(5), Dim::Fixed(3)], 8, 2),
                outcomes(&[bounded], 8, 1),
            ]
        );
    }

    #[test]
    fn test_infer_output_types_needs_one_value_per_parameter_even_when_there_are_none() {
        let node = ShotLoop::new(vec![circuit(0, &[("c", 1)])], 8).unwrap();
        assert_eq!(
            node.infer_output_types(&[ty(DType::F64, &[4, 0])]).unwrap(),
            vec![outcomes(&[Dim::Fixed(4)], 8, 1)],
            "the parameter axis of a circuit taking no parameters is empty, not absent"
        );
        let no_axis = ty(DType::F64, &[4]);
        assert_eq!(
            node.infer_output_types(std::slice::from_ref(&no_axis))
                .unwrap_err(),
            ShotLoopError::ParameterType {
                circuit: 0,
                parameters: 0,
                actual: no_axis,
            },
            "the trailing axis is the parameter axis, and four values are not none"
        );
    }

    #[test]
    fn test_infer_output_types_rejects_values_that_are_not_this_circuits_parameters() {
        let node = ShotLoop::new(vec![circuit(0, &[]), circuit(2, &[("c", 1)])], 8).unwrap();
        let ok = ty(DType::F64, &[0]);

        // Both the shape a circuit's values must have and the type supplied are named.
        assert_eq!(
            node.infer_output_types(&[ok.clone(), ty(DType::F64, &[3])])
                .unwrap_err()
                .to_string(),
            "circuit 1: expected a floating-point tensor of shape [..., 2], got F64[3]"
        );
        // A dtype that is not floating point, a rank too low to carry a parameter axis, and a
        // parameter axis whose size is not known are each refused the same way.
        for operand in [
            ty(DType::I64, &[2]),
            ty(DType::Bit, &[2]),
            ty(DType::F64, &[]),
            TensorType {
                dtype: DType::F64,
                shape: vec![Dim::Bounded { max: 2 }],
            },
        ] {
            assert_eq!(
                node.infer_output_types(&[ok.clone(), operand.clone()])
                    .unwrap_err(),
                ShotLoopError::ParameterType {
                    circuit: 1,
                    parameters: 2,
                    actual: operand,
                }
            );
        }
    }

    #[test]
    fn test_a_register_whose_name_cannot_name_a_slot_is_rejected() {
        // A name may contain no dot and may not be all digits, which is what makes a dotted path
        // unambiguous. A register named that way could not be addressed in the result.
        let Err(digits) = ShotLoop::new(vec![circuit(0, &[("c", 1)]), circuit(0, &[("0", 1)])], 8)
        else {
            panic!("a register named \"0\" could not be addressed in the result")
        };
        assert_eq!(
            digits.to_string(),
            "circuit 1: a name cannot consist only of digits: \"0\""
        );

        let Err(dotted) = ShotLoop::new(vec![circuit(0, &[("a.b", 1)])], 8) else {
            panic!("a register named \"a.b\" could not be addressed in the result")
        };
        assert_eq!(
            dotted.to_string(),
            "circuit 0: a name cannot contain '.': \"a.b\""
        );
    }

    #[test]
    fn test_eval_has_no_in_process_implementation() {
        let node = ShotLoop::new(vec![circuit(0, &[("c", 1)])], 8).unwrap();
        assert_eq!(
            node.eval(&[Tensor::from(&[] as &[f64])]).unwrap_err(),
            ShotLoopError::NoBuiltinEval
        );
    }

    #[test]
    fn test_outcomes_are_ordinary_values() {
        // Averaging over the shots axis is the flagship post-processing step, and it composes with a
        // shot loop's results like any other tensor.
        let mut function = ProgramFunction::new();
        let values = function.add_parameter(ty(DType::F64, &[2]));
        let node = ShotLoop::new(vec![circuit(2, &[("c", 3)])], 100).unwrap();
        let bits = function.add_node(node, &[values]).unwrap()[0];
        let mean = function.add_node(Mean::new(0), &[bits]).unwrap()[0];

        assert_eq!(function.type_of(bits), Some(&outcomes(&[], 100, 3)));
        assert_eq!(function.type_of(mean), Some(&ty(DType::F64, &[3])));
    }

    #[test]
    fn test_a_program_of_circuits_reports_its_types_and_refuses_to_evaluate() {
        let node = ShotLoop::new(
            vec![circuit(2, &[("c", 2)]), circuit(0, &[("meas", 3)])],
            100,
        )
        .unwrap();
        // A shot loop's results are structured like any other value, so the node's own structure is
        // the one a program returning those results declares.
        let output_structure = node.output_structure().clone();

        let mut function = ProgramFunction::new();
        let theta = function.add_parameter(ty(DType::F64, &[2]));
        let none = function.add_parameter(ty(DType::F64, &[0]));
        for outcome in function.add_node(node, &[theta, none]).unwrap() {
            function.add_result(outcome).unwrap();
        }
        let program = QuantumProgram::new(
            vec![function],
            DataTree::mapping([("theta", DataTree::Leaf(())), ("none", DataTree::Leaf(()))])
                .unwrap(),
            output_structure,
        )
        .unwrap();

        // Building and type-checking a program is not evaluating one: it reports every type it
        // produces without a backend.
        assert_eq!(
            program.output_types(),
            DataTree::sequence([
                DataTree::mapping([("c", DataTree::Leaf(outcomes(&[], 100, 2)))]).unwrap(),
                DataTree::mapping([("meas", DataTree::Leaf(outcomes(&[], 100, 3)))]).unwrap(),
            ])
        );
        assert!(!program.has_builtin_eval());

        let inputs = DataTree::mapping([
            ("theta", DataTree::Leaf(Tensor::from([0.5_f64, 1.5]))),
            ("none", DataTree::Leaf(Tensor::from(&[] as &[f64]))),
        ])
        .unwrap();
        let Err(err) = program.eval(inputs) else {
            panic!("a program containing a shot loop cannot be evaluated in process")
        };
        assert_eq!(
            err.to_string(),
            "@0 node 2 (qiskit.shot_loop) has no built-in implementation",
            "the node a backend is needed for is named"
        );
    }
}
