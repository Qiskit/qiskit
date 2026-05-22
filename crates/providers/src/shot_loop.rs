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

use crate::data_tree::DataTree;
use crate::program_node::{MissingCallError, ProgramNode};
use crate::tensor::{DType, DTypeLike, Dim, Tensor, TensorType};
use qiskit_circuit::circuit_data::CircuitData;

/// A program node that runs a fixed list of circuits, each for the same number of shots.
///
/// `ShotLoop` is the canonical "remote" node — it has no local execution path
/// and its [`ProgramNode::call_flat`] always returns [`MissingCallError`]. A
/// backend is expected to dispatch it to hardware (or a simulator) and produce
/// the declared output bitstrings.
///
/// # Inputs
///
/// One leaf per circuit, list-indexed in the order the circuits were given.
/// The leaf for circuit `i` is a broadcastable `F64` tensor of shape
/// `[num_parameters_i]` carrying the parameter values for that circuit.
///
/// # Outputs
///
/// One branch per circuit, list-indexed. Each branch contains one leaf per
/// classical register, keyed by the register's name. The leaf is a
/// broadcastable `Bit` tensor of shape `[shots, num_bits]` holding the
/// per-shot measurement outcomes.
pub struct ShotLoop {
    circuits: Vec<CircuitData>,
    shots: usize,
    input_types: DataTree<TensorType>,
    output_types: DataTree<TensorType>,
}

impl ShotLoop {
    /// Construct a new `ShotLoop` for the given `circuits` and `shots`.
    pub fn new(circuits: Vec<CircuitData>, shots: usize) -> Self {
        let mut input_types = DataTree::with_capacity(circuits.len());
        let mut output_types = DataTree::with_capacity(circuits.len());

        for circuit in &circuits {
            input_types.push_leaf(TensorType {
                dtype: DTypeLike::Concrete(DType::F64),
                shape: vec![Dim::Fixed(circuit.num_parameters())],
                broadcastable: true,
            });

            let cregs = circuit.cregs();
            let mut branch = DataTree::with_capacity(cregs.len());
            for creg in cregs {
                branch.insert_leaf(
                    creg.name(),
                    TensorType {
                        dtype: DTypeLike::Concrete(DType::Bit),
                        shape: vec![Dim::Fixed(shots), Dim::Fixed(creg.len())],
                        broadcastable: true,
                    },
                );
            }
            output_types.push_branch(branch);
        }

        Self {
            circuits,
            shots,
            input_types,
            output_types,
        }
    }

    /// The circuits this `ShotLoop` will run.
    pub fn circuits(&self) -> &[CircuitData] {
        &self.circuits
    }

    /// The number of shots each circuit will be run for.
    pub fn shots(&self) -> usize {
        self.shots
    }
}

impl ProgramNode for ShotLoop {
    type CallError = MissingCallError;

    fn name(&self) -> &str {
        "shot_loop"
    }

    fn namespace(&self) -> &str {
        "qiskit"
    }

    fn input_types(&self) -> &DataTree<TensorType> {
        &self.input_types
    }

    fn output_types(&self) -> &DataTree<TensorType> {
        &self.output_types
    }

    fn implements_call(&self) -> bool {
        false
    }

    fn call_flat(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        Err(MissingCallError::new(self.full_name()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qiskit_circuit::bit::ClassicalRegister;
    use qiskit_circuit::operations::Param;

    /// Build a `CircuitData` with the given classical registers (and no
    /// quantum bits or instructions). Sufficient for exercising the layout of
    /// `ShotLoop`'s input and output type trees.
    fn circuit_with_cregs(cregs: Vec<ClassicalRegister>) -> CircuitData {
        let mut circuit = CircuitData::new(None, None, Param::Float(0.0)).unwrap();
        for creg in cregs {
            circuit.add_creg(creg, true).unwrap();
        }
        circuit
    }

    #[test]
    fn test_name_and_namespace() {
        let sl = ShotLoop::new(vec![], 100);
        assert_eq!(sl.name(), "shot_loop");
        assert_eq!(sl.namespace(), "qiskit");
        assert_eq!(sl.full_name(), "qiskit.shot_loop");
    }

    #[test]
    fn test_does_not_implement_call() {
        let sl = ShotLoop::new(vec![], 100);
        assert!(!sl.implements_call());
    }

    #[test]
    fn test_call_returns_missing_call_error() {
        let sl = ShotLoop::new(vec![], 100);
        let err = sl.call_flat(&[]).unwrap_err();
        assert_eq!(err, MissingCallError::new("qiskit.shot_loop"));
    }

    #[test]
    fn test_empty_input_and_output_types() {
        // No circuits → empty input and output type trees.
        let sl = ShotLoop::new(vec![], 100);
        assert!(sl.input_types().is_empty());
        assert!(sl.output_types().is_empty());
    }

    #[test]
    fn test_input_types_shape_zero_params() {
        // A non-parametric circuit has shape [Fixed(0)] for its parameter input.
        let sl = ShotLoop::new(vec![circuit_with_cregs(vec![])], 100);

        assert_eq!(sl.input_types().len(), 1);
        let DataTree::Leaf(tt) = sl.input_types().get(0).unwrap() else {
            panic!("expected a leaf for circuit 0's parameters");
        };
        assert!(matches!(tt.dtype, DTypeLike::Concrete(DType::F64)));
        assert_eq!(tt.shape, vec![Dim::Fixed(0)]);
        assert!(tt.broadcastable);
    }

    #[test]
    fn test_output_types_register_layout() {
        // One circuit with two classical registers of different sizes.
        let circuit = circuit_with_cregs(vec![
            ClassicalRegister::new_owning("c", 3),
            ClassicalRegister::new_owning("meas", 5),
        ]);
        let sl = ShotLoop::new(vec![circuit], 1024);

        // The output is a branch keyed by circuit index, each entry is itself
        // a branch keyed by register name.
        assert_eq!(sl.output_types().len(), 1);
        let circ_branch = sl.output_types().get(0).unwrap();

        let DataTree::Leaf(c_tt) = circ_branch.get_by_str_key("c").unwrap() else {
            panic!("expected a leaf at register 'c'");
        };
        assert!(matches!(c_tt.dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(c_tt.shape, vec![Dim::Fixed(1024), Dim::Fixed(3)]);
        assert!(c_tt.broadcastable);

        let DataTree::Leaf(meas_tt) = circ_branch.get_by_str_key("meas").unwrap() else {
            panic!("expected a leaf at register 'meas'");
        };
        assert!(matches!(meas_tt.dtype, DTypeLike::Concrete(DType::Bit)));
        assert_eq!(meas_tt.shape, vec![Dim::Fixed(1024), Dim::Fixed(5)]);
        assert!(meas_tt.broadcastable);
    }

    #[test]
    fn test_multiple_circuits() {
        // Two circuits with different register layouts, addressable by index.
        let sl = ShotLoop::new(
            vec![
                circuit_with_cregs(vec![ClassicalRegister::new_owning("c", 2)]),
                circuit_with_cregs(vec![ClassicalRegister::new_owning("d", 4)]),
            ],
            42,
        );

        assert_eq!(sl.input_types().len(), 2);
        assert_eq!(sl.output_types().len(), 2);

        // Inputs: one leaf per circuit.
        for i in 0..2 {
            assert!(matches!(
                sl.input_types().get(i).unwrap(),
                DataTree::Leaf(_)
            ));
        }

        // Outputs: each circuit branch has the right register name.
        let DataTree::Leaf(tt0) = sl
            .output_types()
            .get(0)
            .unwrap()
            .get_by_str_key("c")
            .unwrap()
        else {
            panic!("expected leaf at 0.c");
        };
        assert_eq!(tt0.shape, vec![Dim::Fixed(42), Dim::Fixed(2)]);

        let DataTree::Leaf(tt1) = sl
            .output_types()
            .get(1)
            .unwrap()
            .get_by_str_key("d")
            .unwrap()
        else {
            panic!("expected leaf at 1.d");
        };
        assert_eq!(tt1.shape, vec![Dim::Fixed(42), Dim::Fixed(4)]);
    }

    #[test]
    fn test_accessors() {
        let circuit = circuit_with_cregs(vec![ClassicalRegister::new_owning("c", 1)]);
        let sl = ShotLoop::new(vec![circuit], 7);
        assert_eq!(sl.shots(), 7);
        assert_eq!(sl.circuits().len(), 1);
        assert_eq!(sl.circuits()[0].cregs().len(), 1);
    }
}
