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

//! A callable collection of functions, together with the structures that name what a caller hands
//! over and what it gets back.

use std::fmt;

use thiserror::Error;

use super::program_function::{FunctionEvalError, ProgramFunction};
use crate::data_tree::DataTree;
use crate::tensor::{Tensor, TensorType};

/// The identity of one function within a [`QuantumProgram`].
///
/// An index is meaningful only against the program that issued it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FunctionId(u32);

impl FunctionId {
    /// The id of the function at `index` in definition order.
    pub fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("a function id fits in a u32"))
    }

    /// The underlying index, for use as a dense array subscript.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for FunctionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

/// Why a set of functions and a pair of structures do not make a well-formed [`QuantumProgram`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProgramError {
    /// There is no function to enter at.
    #[error("there are no functions to enter at")]
    NoFunctions,

    /// The input structure describes a different number of values than the entry point takes.
    #[error(
        "the input structure describes {leaves} value(s) but the entry point takes {parameters} \
         parameter(s)"
    )]
    InputArity { leaves: usize, parameters: usize },

    /// The output structure describes a different number of values than the entry point produces.
    #[error(
        "the output structure describes {leaves} value(s) but the entry point produces {results} \
         result(s)"
    )]
    OutputArity { leaves: usize, results: usize },
}

/// Why [`QuantumProgram::eval`] could not produce results.
#[derive(Debug, Error)]
pub enum ProgramEvalError {
    /// The inputs are arranged differently than the program declares.
    #[error("inputs are structured {actual} but the program declares {expected}")]
    InputStructureMismatch {
        expected: Box<DataTree<()>>,
        actual: Box<DataTree<()>>,
    },

    /// The entry point failed.
    #[error(transparent)]
    Function(#[from] FunctionEvalError),
}

/// A collection of [`ProgramFunction`]s, the last of them the entry point.
///
/// A caller to `eval` provides a [`DataTree`] of tensor inputs arranged in the format prescribed by
/// `input_types`, and receives back the resulting tensors as prescribed by `output_types`.
///
/// # Example
/// ```rust
/// use qiskit_providers::nodes::Add;
/// use qiskit_providers::tensor::{DType, Dim, Tensor, TensorType};
/// use qiskit_providers::{DataTree, FunctionId, ProgramFunction, QuantumProgram};
///
/// let ty = TensorType { dtype: DType::F64, shape: vec![Dim::Fixed(1)] };
/// let mut function = ProgramFunction::new();
/// let x = function.add_parameter(ty.clone());
/// let y = function.add_parameter(ty.clone());
/// let sum = function.add_node(Add, &[x, y])?[0];
/// function.add_result(sum)?;
///
/// let leaf = || DataTree::Leaf(());
/// let program = QuantumProgram::new(
///     vec![function],
///     DataTree::mapping([("x", leaf()), ("y", leaf())])?,
///     DataTree::mapping([("sum", leaf())])?,
/// )?;
/// assert_eq!(
///     program.output_types(),
///     DataTree::mapping([("sum", DataTree::Leaf(ty))])?,
/// );
///
/// let inputs = DataTree::mapping([
///     ("x", DataTree::Leaf(Tensor::from([1.5_f64]))),
///     ("y", DataTree::Leaf(Tensor::from([2.5_f64]))),
/// ])?;
/// let outputs = program.eval(inputs)?;
/// assert_eq!(
///     outputs.get_by_str_key("sum"),
///     Some(&DataTree::Leaf(Tensor::from([4.0_f64])))
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QuantumProgram {
    /// The functions, in definition order, indexed by [`FunctionId`]. The last is the entry point.
    functions: Vec<ProgramFunction>,
    input_structure: DataTree<()>,
    output_structure: DataTree<()>,
}

impl QuantumProgram {
    /// Assemble `functions` into a program whose inputs and outputs are arranged as
    /// `input_structure` and `output_structure`.
    ///
    /// The last of `functions` is the entry point, and the structures describe its slots: a
    /// structure's leaves correspond to them by DFS order.
    pub fn new(
        functions: Vec<ProgramFunction>,
        input_structure: DataTree<()>,
        output_structure: DataTree<()>,
    ) -> Result<Self, ProgramError> {
        let Some(function) = functions.last() else {
            return Err(ProgramError::NoFunctions);
        };

        let leaves = input_structure.leaf_count();
        let parameters = function.parameters().len();
        if leaves != parameters {
            return Err(ProgramError::InputArity { leaves, parameters });
        }
        let leaves = output_structure.leaf_count();
        let results = function.results().len();
        if leaves != results {
            return Err(ProgramError::OutputArity { leaves, results });
        }

        Ok(Self {
            functions,
            input_structure,
            output_structure,
        })
    }

    /// The functions, in definition order.
    pub fn functions(&self) -> &[ProgramFunction] {
        &self.functions
    }

    /// The function `id` names, or `None` if it does not belong to this program.
    pub fn function(&self, id: FunctionId) -> Option<&ProgramFunction> {
        self.functions.get(id.index())
    }

    /// The entry point, which is the last function.
    pub fn entry(&self) -> FunctionId {
        FunctionId::from_index(self.functions.len() - 1)
    }

    /// The entry point's function, which is the one a caller of this program invokes.
    pub fn entry_function(&self) -> &ProgramFunction {
        self.functions
            .last()
            .expect("a program holds at least one function")
    }

    /// How the program's inputs are arranged and named.
    pub fn input_structure(&self) -> &DataTree<()> {
        &self.input_structure
    }

    /// How the program's outputs are arranged and named.
    ///
    /// [`DataTree::dotted_paths`] turns this into an address for each output, in output order. The
    /// structure is the thing worth keeping: a path addresses one leaf of it, and a set of paths
    /// cannot reconstruct it, because an empty branch contributes no leaves and so no path.
    pub fn output_structure(&self) -> &DataTree<()> {
        &self.output_structure
    }

    /// The declared type of every input, arranged in the input structure.
    ///
    /// Answers what a program consumes without evaluating it.
    pub fn input_types(&self) -> DataTree<TensorType> {
        arrange(
            &self.input_structure,
            self.entry_function().signature().inputs,
        )
    }

    /// The declared type of every output, arranged in the output structure.
    ///
    /// Answers what a program produces without evaluating it, which is what lets a caller check that
    /// a result is the one it wanted before paying for it.
    pub fn output_types(&self) -> DataTree<TensorType> {
        arrange(
            &self.output_structure,
            self.entry_function().signature().outputs,
        )
    }

    /// Evaluate the program on a tree of inputs, returning a tree of outputs.
    ///
    /// `inputs` must be arranged exactly as [`input_structure`](Self::input_structure) says, which is
    /// checked before anything is evaluated, and the results come back arranged as
    /// [`output_structure`](Self::output_structure) says.
    pub fn eval(&self, inputs: DataTree<Tensor>) -> Result<DataTree<Tensor>, ProgramEvalError> {
        let actual = inputs.structure();
        if actual != self.input_structure {
            return Err(ProgramEvalError::InputStructureMismatch {
                expected: Box::new(self.input_structure.clone()),
                actual: Box::new(actual),
            });
        }
        let arguments: Vec<Tensor> = inputs.into_leaves().collect();
        let results = self.entry_function().eval(&arguments)?;
        Ok(arrange(&self.output_structure, results))
    }
}

/// Arrange one value per slot into `structure`, which describes exactly that many slots.
fn arrange<T>(structure: &DataTree<()>, values: Vec<T>) -> DataTree<T> {
    structure
        .unflatten(values)
        .expect("a structure describes as many slots as the entry point it was checked against")
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::nodes::Add;
    use crate::program::Signature;
    use crate::tensor::{DType, Dim};

    /// The type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    /// A one-element `F64` tensor.
    fn one_element(value: f64) -> Tensor {
        Tensor::from([value])
    }

    /// `f(x, y) = x + y` over one-element `F64` tensors: two parameters, one result.
    fn add_function() -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let y = function.add_parameter(f64_1d(1));
        let sum = function.add_node(Add, &[x, y]).unwrap()[0];
        function.add_result(sum).unwrap();
        function
    }

    /// `f(x) = x + x` over one-element `F64` tensors: one parameter, one result.
    fn double_function() -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let doubled = function.add_node(Add, &[x, x]).unwrap()[0];
        function.add_result(doubled).unwrap();
        function
    }

    /// `[x: _, y: _]` — names for [`add_function`]'s two parameters.
    fn named_inputs() -> DataTree<()> {
        DataTree::mapping([("x", DataTree::Leaf(())), ("y", DataTree::Leaf(()))]).unwrap()
    }

    /// `[sum: _]` — a name for [`add_function`]'s one result.
    fn named_output() -> DataTree<()> {
        DataTree::mapping([("sum", DataTree::Leaf(()))]).unwrap()
    }

    /// `function` as a program whose slots are unnamed: one sequence of leaves per side.
    fn positional_program(function: ProgramFunction) -> QuantumProgram {
        let positional = |count| DataTree::sequence(std::iter::repeat_n(DataTree::Leaf(()), count));
        let inputs = positional(function.parameters().len());
        let outputs = positional(function.results().len());
        QuantumProgram::new(vec![function], inputs, outputs).unwrap()
    }

    /// [`add_function`] as a program, with the structures above.
    fn add_program() -> QuantumProgram {
        QuantumProgram::new(vec![add_function()], named_inputs(), named_output()).unwrap()
    }

    // ---------------------------------------------------------------------------
    // Evaluating through a structure
    // ---------------------------------------------------------------------------

    #[test]
    fn a_program_takes_a_tree_of_inputs_and_returns_a_tree_of_outputs() {
        let inputs = DataTree::mapping([
            ("x", DataTree::Leaf(one_element(1.5))),
            ("y", DataTree::Leaf(one_element(2.5))),
        ])
        .unwrap();

        assert_eq!(
            add_program().eval(inputs).unwrap(),
            DataTree::mapping([("sum", DataTree::Leaf(one_element(4.0)))]).unwrap(),
            "the outputs arrive in the structure the program declares"
        );
    }

    #[test]
    fn a_structures_root_may_be_anything() {
        // A bare leaf on each side describes one parameter and one result.
        let program = QuantumProgram::new(
            vec![double_function()],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        )
        .unwrap();
        assert_eq!(
            program.eval(DataTree::Leaf(one_element(3.0))).unwrap(),
            DataTree::Leaf(one_element(6.0))
        );

        // An empty branch describes no slots at all.
        let program = QuantumProgram::new(
            vec![ProgramFunction::new()],
            DataTree::new(),
            DataTree::new(),
        )
        .unwrap();
        assert_eq!(program.eval(DataTree::new()).unwrap(), DataTree::new());
    }

    #[test]
    fn a_program_whose_structures_are_unnamed_is_positional_on_both_sides() {
        let program = positional_program(add_function());

        assert_eq!(program.input_structure().to_string(), "[_, _]");
        assert_eq!(program.output_structure().to_string(), "[_]");
        assert_eq!(
            program
                .eval(DataTree::sequence([
                    DataTree::Leaf(one_element(1.0)),
                    DataTree::Leaf(one_element(10.0)),
                ]))
                .unwrap(),
            DataTree::sequence([DataTree::Leaf(one_element(11.0))])
        );
    }

    // ---------------------------------------------------------------------------
    // Naming lives in the program's structures
    // ---------------------------------------------------------------------------

    #[test]
    fn a_programs_structures_describe_its_entry_point() {
        // Two functions of different arities. The last is the entry point, and the structures
        // describe that one.
        let program = QuantumProgram::new(
            vec![double_function(), add_function()],
            named_inputs(),
            named_output(),
        )
        .unwrap();

        assert_eq!(program.functions().len(), 2);
        assert_eq!(program.entry(), FunctionId::from_index(1));
        assert_eq!(program.input_structure().to_string(), "[x: _, y: _]");
        assert_eq!(program.output_structure().to_string(), "[sum: _]");
        assert_eq!(
            program.entry_function().signature().inputs.len(),
            2,
            "the entry point is the second function, not the first"
        );
        assert_eq!(
            program
                .function(FunctionId::from_index(0))
                .unwrap()
                .signature(),
            double_function().signature(),
            "a function that is not the entry point is reachable in its own right"
        );

        // The same two structures against the same two functions in the other order describe too
        // many slots, which shows which of them they are checked against.
        let Err(err) = QuantumProgram::new(
            vec![add_function(), double_function()],
            named_inputs(),
            named_output(),
        ) else {
            panic!("two leaves cannot describe the one parameter `double_function` takes")
        };
        assert_eq!(
            err,
            ProgramError::InputArity {
                leaves: 2,
                parameters: 1
            }
        );
    }

    #[test]
    fn a_program_can_be_sent_between_threads() {
        // A job holds the program it is driving, and a job is where all concurrency lives.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QuantumProgram>();
    }

    // ---------------------------------------------------------------------------
    // Types and addresses, without evaluating
    // ---------------------------------------------------------------------------

    #[test]
    fn the_type_of_every_input_and_output_is_available_without_evaluating() {
        let program = add_program();

        assert_eq!(
            program.entry_function().signature(),
            Signature {
                inputs: vec![f64_1d(1), f64_1d(1)],
                outputs: vec![f64_1d(1)],
            }
        );
        assert_eq!(
            program.input_types(),
            DataTree::mapping([
                ("x", DataTree::Leaf(f64_1d(1))),
                ("y", DataTree::Leaf(f64_1d(1))),
            ])
            .unwrap(),
            "the declared types arrive arranged in the input structure"
        );
        assert_eq!(
            program.output_types(),
            DataTree::mapping([("sum", DataTree::Leaf(f64_1d(1)))]).unwrap()
        );
    }

    #[test]
    fn an_output_is_addressable_by_a_dotted_path_derived_from_the_structure() {
        // Three results over one parameter: the doubled value, the parameter, and the doubled value
        // again.
        let mut function = double_function();
        let doubled = function.result_values().next().unwrap();
        let x = function.parameter_values().next().unwrap();
        function.add_result(x).unwrap();
        function.add_result(doubled).unwrap();

        // `[doubled: [_, _], original: _]`, so the first two results share a branch.
        let outputs = DataTree::mapping([
            (
                "doubled",
                DataTree::sequence([DataTree::Leaf(()), DataTree::Leaf(())]),
            ),
            ("original", DataTree::Leaf(())),
        ])
        .unwrap();
        let program = QuantumProgram::new(vec![function], DataTree::Leaf(()), outputs).unwrap();

        let paths = program.output_structure().dotted_paths();
        assert_eq!(paths, ["doubled.0", "doubled.1", "original"]);

        // A path addresses a leaf of the structure, and so of any result arranged in it.
        let results = program.eval(DataTree::Leaf(one_element(2.0))).unwrap();
        for (path, expected) in paths.iter().zip([4.0, 2.0, 4.0]) {
            assert_eq!(
                results.get_by_str_key(path),
                Some(&DataTree::Leaf(one_element(expected))),
                "at {path}"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Assembly rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn an_input_structure_describing_the_wrong_number_of_slots_is_rejected() {
        let Err(err) =
            QuantumProgram::new(vec![add_function()], DataTree::Leaf(()), named_output())
        else {
            panic!("one leaf cannot describe two parameters")
        };

        assert_eq!(
            err,
            ProgramError::InputArity {
                leaves: 1,
                parameters: 2
            }
        );
        assert_eq!(
            err.to_string(),
            "the input structure describes 1 value(s) but the entry point takes 2 parameter(s)",
            "both counts are named"
        );
    }

    #[test]
    fn an_output_structure_describing_the_wrong_number_of_slots_is_rejected() {
        let Err(err) = QuantumProgram::new(vec![add_function()], named_inputs(), named_inputs())
        else {
            panic!("two leaves cannot describe one result")
        };

        assert_eq!(
            err.to_string(),
            "the output structure describes 2 value(s) but the entry point produces 1 result(s)"
        );
    }

    #[test]
    fn a_program_holding_no_functions_is_rejected() {
        // The entry point is the last function, so there has to be one.
        let Err(err) = QuantumProgram::new(Vec::new(), DataTree::new(), DataTree::new()) else {
            panic!("a program with no functions has nothing to enter at")
        };

        assert_eq!(err, ProgramError::NoFunctions);
        assert_eq!(err.to_string(), "there are no functions to enter at");
    }

    // ---------------------------------------------------------------------------
    // Evaluation rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn inputs_with_the_declared_leaf_count_but_the_wrong_structure_are_rejected() {
        // Two leaves either way, so only a structural check catches this.
        let inputs = DataTree::sequence([
            DataTree::Leaf(one_element(1.5)),
            DataTree::Leaf(one_element(2.5)),
        ]);

        let Err(err) = add_program().eval(inputs) else {
            panic!("a sequence cannot stand in for a mapping")
        };

        assert!(matches!(
            err,
            ProgramEvalError::InputStructureMismatch { .. }
        ));
        assert_eq!(
            err.to_string(),
            "inputs are structured [_, _] but the program declares [x: _, y: _]",
            "each structure is reported whole"
        );
    }

    #[test]
    fn a_mismatch_below_the_root_is_reported_as_the_whole_structure() {
        // The second input is declared nested, so what diverges is one child of the root rather than
        // the root itself.
        let program = QuantumProgram::new(
            vec![add_function()],
            DataTree::mapping([
                ("x", DataTree::Leaf(())),
                ("y", DataTree::sequence([DataTree::Leaf(())])),
            ])
            .unwrap(),
            named_output(),
        )
        .unwrap();
        let inputs = DataTree::mapping([
            ("x", DataTree::Leaf(one_element(1.5))),
            ("y", DataTree::Leaf(one_element(2.5))),
        ])
        .unwrap();

        let Err(err) = program.eval(inputs) else {
            panic!("a leaf cannot stand in for a branch of one leaf")
        };

        assert_eq!(
            err.to_string(),
            "inputs are structured [x: _, y: _] but the program declares [x: _, y: [_]]",
            "the surrounding structure is reported, not the subtree that differs"
        );
    }

    #[test]
    fn an_input_that_does_not_match_its_declared_type_is_rejected_naming_both_types() {
        let inputs = DataTree::mapping([
            ("x", DataTree::Leaf(one_element(1.5))),
            ("y", DataTree::Leaf(Tensor::from([2_i64]))),
        ])
        .unwrap();

        let Err(err) = add_program().eval(inputs) else {
            panic!("a program is monomorphic, so an I64 input is rejected")
        };

        assert!(matches!(
            err,
            ProgramEvalError::Function(FunctionEvalError::ArgumentTypeMismatch {
                parameter: 1,
                ..
            })
        ));
        assert_eq!(err.to_string(), "argument 1: expected F64[1], got I64[1]");
    }
}
