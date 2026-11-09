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

//! Partition a quantum program into functions, one per unit of dispatchable work.

use hashbrown::HashMap;
use thiserror::Error;

use super::program_function::{
    InstructionId, InstructionRole, InstructionView, ProgramFunction, Value,
};
use super::quantum_program::{FunctionId, QuantumProgram};

/// An error encountered while partitioning a quantum program.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PartitionError {
    /// The entry point already calls another function.
    #[error("the entry point calls another function at instruction {instruction}")]
    EntryPointCall { instruction: InstructionId },

    /// Two execution resources declare the same op.
    #[error("resource {first} and resource {second} both handle {name}")]
    RepeatedOpType {
        name: String,
        first: usize,
        second: usize,
    },

    /// No resource handles this operation, and Qiskit cannot evaluate it locally.
    #[error(
        "no resource handles {name} at instruction {instruction}, and it has no built-in evaluation"
    )]
    NoResource {
        name: String,
        instruction: InstructionId,
    },
}

/// One unit of dispatchable work: the instructions that become one function of the new program.
struct Unit {
    /// The execution resource that handles this unit's ops.
    resource: usize,
    /// Dependency layer used to order this unit relative to other units.
    layer: usize,
    /// Instructions included in this unit, in their original program order.
    instructions: Vec<InstructionId>,
}

/// Rewrite `program` to put the work of each execution resource in a function of its own, and
/// report the resource each of those functions belongs to.
///
/// `resources` declares, per execution resource, the op names that resource handles, as
/// [`ProgramOp::full_name`](crate::ProgramOp::full_name) renders them. An op no resource declares
/// stays in the entry point, where Qiskit evaluates it in process. Boundary instructions are matched
/// by role, not by name: a parameter stays in the entry point that declares it. One input may fan
/// out to several units.
///
/// The rewritten program holds one function per unit of dispatchable work, ordered with no unit
/// depending on a later one, and then an entry point that declares the same inputs and outputs as
/// `program`'s, calls each unit once, and performs the rest itself. Each unit can be run as a whole,
/// because no path leaving a unit leads back into it. Every other function `program` holds is
/// dropped; a partitionable entry point calls none of them.
///
/// The table gives each unit's resource, indexed by [`FunctionId::index`], and so holds one entry
/// fewer than the program has functions: the entry point is the last function and belongs to no
/// resource. It is not grouped by resource. A dependency may run from one unit of a resource to
/// another, so one resource's functions cannot be run as a batch.
///
/// This is a helper for a `BackendV3` implementation, not part of the interface one must satisfy.
///
/// # Errors
///
/// Returns an error if `program`'s entry point already contains a call instruction, if two resources
/// declare the same op, or if an op no resource declares is one Qiskit cannot evaluate in process.
pub fn partition<R, N>(
    program: &QuantumProgram,
    resources: R,
) -> Result<(QuantumProgram, Vec<usize>), PartitionError>
where
    R: IntoIterator,
    R::Item: IntoIterator<Item = N>,
    N: AsRef<str>,
{
    // Index the declared op names by resource. A name matches an op by exact equality with its
    // `full_name`, so a caller depends on no type this crate defines.
    let mut categories: HashMap<String, usize> = HashMap::new();
    for (resource, names) in resources.into_iter().enumerate() {
        for name in names {
            if let Some(first) = categories.insert(name.as_ref().to_string(), resource)
                && first != resource
            {
                return Err(PartitionError::RepeatedOpType {
                    name: name.as_ref().to_string(),
                    first,
                    second: resource,
                });
            }
        }
    }

    let entry = program.entry_function();
    if let Some(instruction) = entry
        .iter_instructions()
        .find(|instruction| instruction.role() == InstructionRole::Call)
    {
        // We disallow partitioning programs that already have multiple functions to simplify the
        // implementation. This slightly more complicated case could be implemented, if needed.
        return Err(PartitionError::EntryPointCall {
            instruction: instruction.id(),
        });
    }

    // Place each op in a layer: one past any operand of another resource, and in the same layer
    // as operands of its own, so every dependency between two resources moves strictly forward.
    // Grouping by layer and resource therefore gives units that run in layer order and that no
    // path can re-enter, and every op of one resource sharing a layer ends up in one unit
    // whether or not those ops depend on each other.
    //
    // This is a greedy heuristic aiming for a few large units rather than the fewest possible:
    // an op whose layer was pushed up does not merge with an op of its own resource lower down.
    let mut placed: Vec<Option<(usize, Option<usize>)>> = vec![None; entry.instruction_count()];
    let mut units: Vec<Unit> = Vec::new();
    let mut index: HashMap<(usize, usize), usize> = HashMap::new();
    // The ops the entry point performs itself, each with the layer it sits in.
    let mut kept: Vec<(usize, InstructionId)> = Vec::new();
    for instruction in entry.iter_instructions() {
        let InstructionView::Op(op) = instruction.view() else {
            continue;
        };
        let resource = categories.get(&op.full_name()).copied();
        if resource.is_none() && !instruction.has_builtin_eval() {
            return Err(PartitionError::NoResource {
                name: op.full_name(),
                instruction: instruction.id(),
            });
        }
        let layer = instruction
            .operands()
            .iter()
            .filter_map(|value| placed[value.instruction().index()])
            .map(|(layer, of)| if of == resource { layer } else { layer + 1 })
            .max()
            .unwrap_or(0);
        placed[instruction.id().index()] = Some((layer, resource));

        // Operations without an assigned resource are not extracted into a unit;
        // they remain as instructions in the entry-point function. They still
        // participate in dependency layering, so an operation between two operations
        // assigned to the same resource can force them into separate units.
        let Some(resource) = resource else {
            kept.push((layer, instruction.id()));
            continue;
        };
        let unit = *index.entry((layer, resource)).or_insert_with(|| {
            units.push(Unit {
                resource,
                layer,
                instructions: Vec::new(),
            });
            units.len() - 1
        });
        units[unit].instructions.push(instruction.id());
    }
    // Units and kept ops were collected in the order their first instruction appears, and
    // `sort_by_key` is stable, so the ones sharing a layer keep that order.
    units.sort_by_key(|unit| unit.layer);
    kept.sort_by_key(|&(layer, _)| layer);

    // Which unit holds each instruction. A boundary instruction and an op the entry point keeps
    // belong to none.
    let mut held: Vec<Option<usize>> = vec![None; entry.instruction_count()];
    for (index, unit) in units.iter().enumerate() {
        for &instruction in &unit.instructions {
            held[instruction.index()] = Some(index);
        }
    }

    // Collect the values crossing each unit's boundary:
    // - Inputs are produced outside the unit and consumed inside it.
    // - Outputs are produced inside the unit and consumed outside it.
    //
    // Sort and deduplicate them so each value becomes only one parameter or result,
    // even when multiple instructions use it.
    let mut inputs: Vec<Vec<Value>> = vec![Vec::new(); units.len()];
    let mut outputs: Vec<Vec<Value>> = vec![Vec::new(); units.len()];
    for instruction in entry.iter_instructions() {
        let consumer = held[instruction.id().index()];
        for &value in instruction.operands() {
            let producer = held[value.instruction().index()];
            if producer == consumer {
                continue;
            }
            if let Some(consumer) = consumer {
                inputs[consumer].push(value);
            }
            if let Some(producer) = producer {
                outputs[producer].push(value);
            }
        }
    }
    for values in inputs.iter_mut().chain(&mut outputs) {
        values.sort_unstable();
        values.dedup();
    }

    let type_of = |value| {
        entry
            .type_of(value)
            .expect("a value of the function being partitioned always exists")
            .clone()
    };

    // Create one function per unit. Each function receives its external inputs
    // as parameters. `moved` maps original values to their corresponding values
    // in the generated functions.
    let mut functions: Vec<ProgramFunction> = Vec::with_capacity(units.len() + 1);
    let mut moved: Vec<HashMap<Value, Value>> = Vec::with_capacity(units.len());
    for unit in &inputs {
        let mut function = ProgramFunction::new();
        let values = unit
            .iter()
            .map(|&value| (value, function.add_parameter(type_of(value))))
            .collect();
        functions.push(function);
        moved.push(values);
    }

    // Rebuild each operation in its assigned function, using the corresponding
    // values in that function. All operands are either produced earlier in the
    // same function or passed in as parameters.
    for instruction in entry.iter_instructions() {
        let InstructionView::Op(op) = instruction.view() else {
            continue;
        };
        let Some(unit) = held[instruction.id().index()] else {
            continue;
        };
        let operands: Vec<Value> = instruction
            .operands()
            .iter()
            .map(|value| moved[unit][value])
            .collect();
        let produced = functions[unit]
            .add_boxed_op(op.to_owned(), &operands)
            .expect("inference is monomorphic, so a carried op types as it typed before");
        moved[unit].extend(instruction.outputs().zip(produced));
    }
    for (unit, function) in functions.iter_mut().enumerate() {
        for value in &outputs[unit] {
            function
                .add_result(moved[unit][value])
                .expect("a unit returns a value produced within it");
        }
    }

    // Rebuild the entry point, declaring what the function being partitioned declared. It gets
    // one call instruction per unit, plus the ops no resource handles, carried in to be performed
    // in process. Both are emitted in layer order, so every value is available before it is used.
    // The closing `None` emits the kept ops that follow the last unit.
    let mut new_entry = ProgramFunction::new();
    let mut lifted: HashMap<Value, Value> = entry
        .parameter_values()
        .map(|value| (value, new_entry.add_parameter(type_of(value))))
        .collect();
    let mut next_kept = 0;
    for unit in units.iter().enumerate().map(Some).chain([None]) {
        let layer = unit.map_or(usize::MAX, |(_, unit)| unit.layer);
        while let Some(&(_, id)) = kept.get(next_kept).filter(|&&(at, _)| at < layer) {
            next_kept += 1;
            let instruction = entry
                .instruction(id)
                .expect("a kept op belongs to the function being partitioned");
            let InstructionView::Op(op) = instruction.view() else {
                unreachable!("only an op is kept")
            };
            let operands: Vec<Value> = instruction
                .operands()
                .iter()
                .map(|value| lifted[value])
                .collect();
            let produced = new_entry
                .add_boxed_op(op.to_owned(), &operands)
                .expect("inference is monomorphic, so a carried op types as it typed before");
            lifted.extend(instruction.outputs().zip(produced));
        }
        let Some((unit, _)) = unit else { break };
        let operands: Vec<Value> = inputs[unit].iter().map(|value| lifted[value]).collect();
        let produced = new_entry
            .add_call(
                FunctionId::from_index(unit),
                &functions[unit].signature(),
                &operands,
            )
            .expect("a call built from its callee's own signature type-checks");
        lifted.extend(outputs[unit].iter().copied().zip(produced));
    }
    for value in entry.result_values() {
        new_entry
            .add_result(lifted[&value])
            .expect("a result of the function being partitioned is a value the entry point holds");
    }

    let table: Vec<usize> = units.iter().map(|unit| unit.resource).collect();
    functions.push(new_entry);
    let partitioned = QuantumProgram::new(
        functions,
        program.input_structure().clone(),
        program.output_structure().clone(),
    )
    .expect("the entry point keeps its slots, and each call is built from its callee's signature");
    Ok((partitioned, table))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::data_tree::DataTree;
    use crate::ops::{Add, Constant, Mean, Multiply, ProgramOp};
    use crate::tensor::{DType, Dim, Tensor, TensorType};

    /// Return the type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    /// Wrap `functions` as a program whose slots are unnamed: one sequence of leaves per side.
    fn program(functions: Vec<ProgramFunction>) -> QuantumProgram {
        let entry = functions.last().expect("a program holds a function");
        let positional = |count| DataTree::sequence(std::iter::repeat_n(DataTree::Leaf(()), count));
        let inputs = positional(entry.parameters().len());
        let outputs = positional(entry.results().len());
        QuantumProgram::new(functions, inputs, outputs).unwrap()
    }

    /// Evaluate `program` on `args`, one per parameter, returning one tensor per result.
    fn eval(program: &QuantumProgram, args: impl IntoIterator<Item = Tensor>) -> Vec<Tensor> {
        let inputs = DataTree::sequence(args.into_iter().map(DataTree::Leaf));
        program.eval(inputs).unwrap().into_leaves().collect()
    }

    /// Return the type name of every instruction of `function`, in order.
    fn names(function: &ProgramFunction) -> Vec<String> {
        function
            .iter_instructions()
            .map(|instruction| instruction.full_name())
            .collect()
    }

    /// Return the ops of each unit of `program`, in order, leaving out its entry point.
    fn unit_ops(program: &QuantumProgram) -> Vec<Vec<String>> {
        let functions = program.functions();
        functions[..functions.len() - 1]
            .iter()
            .map(|function| {
                function
                    .iter_instructions()
                    .filter(|instruction| instruction.role() == InstructionRole::Op)
                    .map(|instruction| instruction.full_name())
                    .collect()
            })
            .collect()
    }

    /// Build `f(x, y) = x + y` over `len`-element `F64` tensors.
    fn add_function(len: usize) -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(len));
        let y = function.add_parameter(f64_1d(len));
        let sum = function.add_op(Add, &[x, y]).unwrap()[0];
        function.add_result(sum).unwrap();
        function
    }

    // ---------------------------------------------------------------------------
    // Grouping
    // ---------------------------------------------------------------------------

    #[test]
    fn a_program_of_one_category_becomes_one_call() {
        let original = program(vec![add_function(1)]);
        // A boundary instruction reports a type name like any other instruction, and declaring
        // those names changes nothing: a boundary instruction is matched by its role.
        let resources = [vec!["qiskit.add", "qiskit.parameter", "qiskit.result"]];
        let (partitioned, table) = partition(&original, resources).unwrap();

        assert_eq!(table, [0], "one unit, of the one resource");
        assert_eq!(
            table.len() + 1,
            partitioned.functions().len(),
            "the table describes every function but the entry point"
        );
        assert_eq!(
            names(partitioned.entry_function()),
            [
                "qiskit.parameter",
                "qiskit.parameter",
                "qiskit.call",
                "qiskit.result"
            ],
            "the entry point holds its own boundary and one call, and no arithmetic"
        );
        assert_eq!(
            names(partitioned.function(FunctionId::from_index(0)).unwrap()),
            [
                "qiskit.parameter",
                "qiskit.parameter",
                "qiskit.add",
                "qiskit.result"
            ],
            "a boundary instruction is left where it is, which matching on type names would not do"
        );

        let arguments = [Tensor::from([1.5_f64]), Tensor::from([2.5_f64])];
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn independent_instructions_of_one_category_become_one_unit() {
        // Two sums over four parameters, with no path between them.
        let mut function = ProgramFunction::new();
        let values: Vec<Value> = (0..4).map(|_| function.add_parameter(f64_1d(1))).collect();
        let first = function.add_op(Add, &[values[0], values[1]]).unwrap()[0];
        let second = function.add_op(Add, &[values[2], values[3]]).unwrap()[0];
        function.add_result(first).unwrap();
        function.add_result(second).unwrap();
        let original = program(vec![function]);

        let (partitioned, table) = partition(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(
            table,
            [0],
            "independent work of one category is one unit, and so one submission"
        );
        let unit = partitioned.function(FunctionId::from_index(0)).unwrap();
        assert_eq!(unit.parameters().len(), 4);
        assert_eq!(unit.results().len(), 2);

        let arguments = [1.0, 2.0, 30.0, 40.0].map(|x| Tensor::from([x]));
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn a_dependency_of_another_category_splits_a_category_in_two() {
        // `add` twice over a `mean` between them, so the two sums cannot share a unit. A unit the
        // mean's result re-entered could not be run whole.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let sum = function.add_op(Add, &[x, y]).unwrap()[0];
        let mean = function.add_op(Mean::new(0), &[sum]).unwrap()[0];
        let doubled = function.add_op(Add, &[mean, mean]).unwrap()[0];
        function.add_result(doubled).unwrap();
        let original = program(vec![function]);

        let (partitioned, table) = partition(&original, [["qiskit.add"], ["qiskit.mean"]]).unwrap();

        assert_eq!(
            table,
            [0, 1, 0],
            "the first resource runs twice, either side of the second"
        );
        let arguments = [Tensor::from([1.0_f64, 2.0]), Tensor::from([10.0_f64, 20.0])];
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn ops_no_resource_handles_stay_in_the_entry_point() {
        // `multiply` and `mean` are undeclared, so the entry point performs them itself rather than
        // handing them anywhere.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let sum = function.add_op(Add, &[x, y]).unwrap()[0];
        let product = function.add_op(Multiply, &[x, y]).unwrap()[0];
        let mean = function.add_op(Mean::new(0), &[y]).unwrap()[0];
        function.add_result(sum).unwrap();
        function.add_result(product).unwrap();
        function.add_result(mean).unwrap();
        let original = program(vec![function]);

        let (partitioned, table) = partition(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(table, [0], "only the declared op is handed anywhere");
        assert_eq!(
            names(partitioned.entry_function()),
            [
                "qiskit.parameter",
                "qiskit.parameter",
                "qiskit.call",
                "qiskit.multiply",
                "qiskit.mean",
                "qiskit.result",
                "qiskit.result",
                "qiskit.result"
            ],
            "the undeclared ops sit beside the call"
        );

        let arguments = [Tensor::from([1.0_f64, 2.0]), Tensor::from([10.0_f64, 20.0])];
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn one_input_may_fan_out_to_several_units() {
        // Both units read `x`, and the sum reads it twice.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let sum = function.add_op(Add, &[x, x]).unwrap()[0];
        let product = function.add_op(Multiply, &[x, x]).unwrap()[0];
        function.add_result(sum).unwrap();
        function.add_result(product).unwrap();
        let original = program(vec![function]);

        let (partitioned, table) =
            partition(&original, [["qiskit.add"], ["qiskit.multiply"]]).unwrap();

        assert_eq!(table, [0, 1]);
        for unit in 0..2 {
            assert_eq!(
                partitioned
                    .function(FunctionId::from_index(unit))
                    .unwrap()
                    .parameters()
                    .len(),
                1,
                "a value a unit reads twice is one parameter"
            );
        }
        assert_eq!(
            partitioned
                .entry_function()
                .iter_instructions()
                .filter(|instruction| instruction.role() == InstructionRole::Call)
                .count(),
            2,
            "the entry point passes its one parameter to both calls"
        );

        let arguments = [Tensor::from([3.0_f64, 4.0])];
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn local_work_runs_before_the_unit_that_reads_it() {
        // The second mean is written after the first sum and read by the second, and both sums share
        // a unit. The entry point must therefore perform both means before calling that unit, which
        // the order the original holds its instructions in would not do.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let first = function.add_op(Mean::new(0), &[x]).unwrap()[0];
        let sum = function.add_op(Add, &[first, first]).unwrap()[0];
        let second = function.add_op(Mean::new(0), &[x]).unwrap()[0];
        let total = function.add_op(Add, &[sum, second]).unwrap()[0];
        function.add_result(total).unwrap();
        let original = program(vec![function]);

        let (partitioned, table) = partition(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(table, [0], "the two sums share one unit");
        assert_eq!(
            names(partitioned.entry_function()),
            [
                "qiskit.parameter",
                "qiskit.mean",
                "qiskit.mean",
                "qiskit.call",
                "qiskit.result"
            ],
            "both means run before the call that reads them"
        );

        let arguments = [Tensor::from([3.0_f64, 4.0])];
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    // ---------------------------------------------------------------------------
    // What the partition guarantees
    // ---------------------------------------------------------------------------

    #[test]
    fn every_dependency_between_two_units_runs_forwards() {
        // Two categories interleaved over two diamonds, so a unit is entered from more than one unit
        // and left towards more than one.
        //
        // That the units run in the order the program defines them needs no assertion of its own: an
        // operand is accepted only if it already exists in the function being built, so a unit
        // reading a later unit could not have been built at all.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let sum = function.add_op(Add, &[x, y]).unwrap()[0];
        let product = function.add_op(Multiply, &[x, y]).unwrap()[0];
        let both = function.add_op(Add, &[sum, product]).unwrap()[0];
        let scaled = function.add_op(Multiply, &[sum, sum]).unwrap()[0];
        let total = function.add_op(Add, &[both, scaled]).unwrap()[0];
        function.add_result(total).unwrap();
        function.add_result(scaled).unwrap();
        let original = program(vec![function]);

        let (partitioned, table) =
            partition(&original, [["qiskit.add"], ["qiskit.multiply"]]).unwrap();

        assert_eq!(
            table,
            [0, 1, 0, 1, 0],
            "each category runs three times and twice, alternating"
        );
        assert_eq!(
            unit_ops(&partitioned),
            [
                ["qiskit.add"],
                ["qiskit.multiply"],
                ["qiskit.add"],
                ["qiskit.multiply"],
                ["qiskit.add"]
            ],
            "each unit holds ops of its own resource, and none is lost or repeated"
        );

        let first = partitioned.function(FunctionId::from_index(0)).unwrap();
        assert_eq!(
            first.parameters().len(),
            2,
            "the first unit reads both inputs"
        );
        assert_eq!(
            first.results().len(),
            1,
            "a value two later units read is returned once"
        );

        let arguments = [Tensor::from([1.0_f64, 2.0]), Tensor::from([10.0_f64, 20.0])];
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    #[test]
    fn a_unit_is_left_where_a_category_reappears_later() {
        // A chain of alternating categories, which is the worst case for the heuristic: every
        // instruction is its own unit.
        let mut function = ProgramFunction::new();
        let mut value = function.add_parameter(f64_1d(2));
        for step in 0..6 {
            value = if step % 2 == 0 {
                function.add_op(Add, &[value, value]).unwrap()[0]
            } else {
                function.add_op(Multiply, &[value, value]).unwrap()[0]
            };
        }
        function.add_result(value).unwrap();
        let original = program(vec![function]);

        let (partitioned, table) =
            partition(&original, [["qiskit.add"], ["qiskit.multiply"]]).unwrap();

        assert_eq!(
            table,
            [0, 1, 0, 1, 0, 1],
            "no two steps of the chain can share a unit"
        );
        assert_eq!(
            unit_ops(&partitioned),
            [
                ["qiskit.add"],
                ["qiskit.multiply"],
                ["qiskit.add"],
                ["qiskit.multiply"],
                ["qiskit.add"],
                ["qiskit.multiply"]
            ],
            "each unit holds the one op it can"
        );

        let arguments = [Tensor::from([1.0_f64, 2.0])];
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    // ---------------------------------------------------------------------------
    // What the rewritten program keeps
    // ---------------------------------------------------------------------------

    #[test]
    fn the_rewritten_program_computes_and_declares_the_same_thing() {
        // A constant of its own category, so its unit reads nothing at all; an instruction whose
        // result nothing reads; a parameter returned directly; and one value returned twice.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let two = function
            .add_op(Constant::new(Tensor::from([2.0_f64, 2.0])), &[])
            .unwrap()[0];
        let scaled = function.add_op(Multiply, &[x, two]).unwrap()[0];
        function.add_op(Mean::new(0), &[scaled]).unwrap();
        function.add_result(scaled).unwrap();
        function.add_result(x).unwrap();
        function.add_result(scaled).unwrap();

        let names = DataTree::mapping([
            ("scaled", DataTree::Leaf(())),
            ("original", DataTree::Leaf(())),
            ("again", DataTree::Leaf(())),
        ])
        .unwrap();
        let original =
            QuantumProgram::new(vec![function], DataTree::Leaf(()), names.clone()).unwrap();

        let resources = [vec!["qiskit.multiply"], vec!["qiskit.constant"]];
        let (partitioned, table) = partition(&original, resources).unwrap();

        assert_eq!(
            table,
            [1, 0],
            "the constant, and then the product, with the undeclared mean left where it was"
        );
        let constant = partitioned.function(FunctionId::from_index(0)).unwrap();
        assert!(
            constant.parameters().is_empty(),
            "a unit that reads nothing is called with no operands"
        );
        assert_eq!(
            partitioned.entry_function().signature(),
            original.entry_function().signature(),
            "the entry point declares what it declared, in the order it declared it"
        );
        assert_eq!(partitioned.input_structure(), &DataTree::Leaf(()));
        assert_eq!(partitioned.output_structure(), &names);

        let input = || DataTree::Leaf(Tensor::from([3.0_f64, 4.0]));
        assert_eq!(
            partitioned.eval(input()).unwrap(),
            original.eval(input()).unwrap()
        );
    }

    #[test]
    fn a_program_with_nothing_to_partition_keeps_only_its_entry_point() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        function.add_result(x).unwrap();
        let original = program(vec![function]);

        let (partitioned, table) = partition(&original, [["qiskit.add"]]).unwrap();

        assert!(table.is_empty(), "there is no unit to describe");
        assert_eq!(partitioned.functions().len(), 1);
        assert_eq!(
            eval(&partitioned, [Tensor::from([1.0_f64])]),
            [Tensor::from([1.0_f64])]
        );
    }

    #[test]
    fn a_function_the_entry_point_cannot_reach_is_dropped() {
        let original = program(vec![add_function(4), add_function(1)]);
        assert_eq!(original.functions().len(), 2);

        let (partitioned, table) = partition(&original, [["qiskit.add"]]).unwrap();

        assert_eq!(table, [0], "the one unit");
        assert_eq!(
            partitioned
                .function(FunctionId::from_index(0))
                .unwrap()
                .signature(),
            add_function(1).signature(),
            "@0 is the unit, not the function that was @0"
        );
    }

    /// An op with two results: the sum of its operands, and their difference.
    #[derive(Clone, Debug)]
    struct SumAndDifference;

    impl ProgramOp for SumAndDifference {
        type Error = std::convert::Infallible;

        fn name(&self) -> &str {
            "sum_and_difference"
        }
        fn namespace(&self) -> &str {
            "vendor"
        }
        fn arity(&self) -> usize {
            2
        }
        fn has_builtin_eval(&self) -> bool {
            true
        }
        fn infer_output_types(
            &self,
            inputs: &[TensorType],
        ) -> Result<Vec<TensorType>, Self::Error> {
            Ok(vec![inputs[0].clone(), inputs[0].clone()])
        }
        fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
            let [x, y] = args else { panic!("two operands") };
            Ok(vec![x.add_tensor(y).unwrap(), x.sub_tensor(y).unwrap()])
        }
    }

    #[test]
    fn a_result_other_than_the_first_crosses_a_unit_boundary_as_itself() {
        // Only the difference is read by the other unit, so the value that crosses is the second
        // result of its producer, and the unit returns that one alone.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let both = function.add_op(SumAndDifference, &[x, y]).unwrap();
        let scaled = function.add_op(Multiply, &[both[1], both[1]]).unwrap()[0];
        function.add_result(both[0]).unwrap();
        function.add_result(scaled).unwrap();
        let original = program(vec![function]);

        let resources = [vec!["vendor.sum_and_difference"], vec!["qiskit.multiply"]];
        let (partitioned, table) = partition(&original, resources).unwrap();

        assert_eq!(table, [0, 1]);
        let first = partitioned.function(FunctionId::from_index(0)).unwrap();
        assert_eq!(
            first.results().len(),
            2,
            "the sum is returned for the entry point and the difference for the second unit"
        );
        assert_eq!(
            partitioned
                .function(FunctionId::from_index(1))
                .unwrap()
                .signature()
                .inputs
                .len(),
            1,
            "the second unit reads one value, whatever slot it came from"
        );

        let arguments = [Tensor::from([10.0_f64, 20.0]), Tensor::from([1.0_f64, 2.0])];
        assert_eq!(
            eval(&partitioned, arguments.clone()),
            eval(&original, arguments)
        );
    }

    // ---------------------------------------------------------------------------
    // Rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn an_entry_point_that_already_calls_a_function_is_rejected() {
        // Partitioning twice would put calls into units, so the second attempt is refused rather
        // than returning a program whose units call units.
        let callee = add_function(1);
        let signature = callee.signature();
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        let y = entry.add_parameter(f64_1d(1));
        let called = entry
            .add_call(FunctionId::from_index(0), &signature, &[x, y])
            .unwrap()[0];
        entry.add_result(called).unwrap();
        let original = program(vec![callee, entry]);

        let Err(err) = partition(&original, [["qiskit.add"]]) else {
            panic!("a program holding a call is already partitioned")
        };
        let call = original
            .entry_function()
            .iter_instructions()
            .find(|instruction| instruction.role() == InstructionRole::Call)
            .expect("the entry point holds a call")
            .id();
        assert_eq!(err, PartitionError::EntryPointCall { instruction: call });
        assert_eq!(
            err.to_string(),
            "the entry point calls another function at instruction 2"
        );
    }

    #[test]
    fn two_resources_cannot_declare_one_op() {
        let original = program(vec![add_function(1)]);

        let resources = [vec!["qiskit.add"], vec!["qiskit.mean", "qiskit.add"]];
        let Err(err) = partition(&original, resources) else {
            panic!("an op belongs to one resource")
        };
        assert_eq!(
            err,
            PartitionError::RepeatedOpType {
                name: "qiskit.add".to_string(),
                first: 0,
                second: 1,
            }
        );
        assert_eq!(
            err.to_string(),
            "resource 0 and resource 1 both handle qiskit.add"
        );

        assert!(
            partition(&original, [["qiskit.add", "qiskit.add"]]).is_ok(),
            "one resource may name an op twice"
        );
    }

    /// An op that must be dispatched, since Qiskit ships no implementation of it.
    #[derive(Clone, Debug)]
    struct NeedsHardware;

    impl ProgramOp for NeedsHardware {
        type Error = std::convert::Infallible;

        fn name(&self) -> &str {
            "needs_hardware"
        }
        fn namespace(&self) -> &str {
            "vendor"
        }
        fn arity(&self) -> usize {
            1
        }
        fn has_builtin_eval(&self) -> bool {
            false
        }
        fn infer_output_types(
            &self,
            inputs: &[TensorType],
        ) -> Result<Vec<TensorType>, Self::Error> {
            Ok(vec![inputs[0].clone()])
        }
        fn eval(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
            unreachable!("this op reports no built-in implementation")
        }
    }

    #[test]
    fn an_op_left_in_the_entry_point_must_run_in_process() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let sampled = function.add_op(NeedsHardware, &[x]).unwrap()[0];
        function.add_result(sampled).unwrap();
        let original = program(vec![function]);

        let Err(err) = partition(&original, [["qiskit.add"]]) else {
            panic!("nothing declared can run this op, and neither can Qiskit")
        };
        let offending = original
            .entry_function()
            .iter_instructions()
            .find(|instruction| instruction.role() == InstructionRole::Op)
            .expect("the entry point holds the op")
            .id();
        assert_eq!(
            err,
            PartitionError::NoResource {
                name: "vendor.needs_hardware".to_string(),
                instruction: offending,
            }
        );
        assert_eq!(
            err.to_string(),
            "no resource handles vendor.needs_hardware at instruction 1, and it has no built-in \
             evaluation"
        );

        assert!(
            partition(&original, [["vendor.needs_hardware"]]).is_ok(),
            "a resource declaring it is what makes the program partitionable"
        );
    }
}
