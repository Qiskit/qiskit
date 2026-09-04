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

use std::fmt;

use thiserror::Error;

use crate::nodes::{BoxedOpNodeError, BoxedOpNodeType, OpNodeType, QISKIT, erase};
use crate::tensor::{Tensor, TensorType};

/// A position within one node's ordered operands or results.
type Slot = u16;

/// A node's position in the function that holds it.
///
/// Ids are dense, are never reused, and are only meaningful within the function that issued them.
/// They are also an evaluation order: every operand of a node is produced by a strictly lower id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u32);

impl NodeId {
    /// The underlying index, for use as a dense array subscript.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One tensor value: an output slot of the node that produces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value {
    node: NodeId,
    slot: Slot,
}

impl Value {
    /// The node that produces this value.
    pub fn node(self) -> NodeId {
        self.node
    }

    /// Which of that node's results this value is.
    pub fn slot(self) -> usize {
        self.slot as usize
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.slot == 0 {
            write!(f, "%{}", self.node)
        } else {
            write!(f, "%{}.{}", self.node, self.slot)
        }
    }
}

/// What part a node plays in its function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// A function input, supplied by the caller.
    Parameter,
    /// An operation, with a [`OpNodeType`].
    Op,
    /// A function output.
    Result,
}

/// The types a function consumes and produces, positionally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// The type of each parameter, in declaration order.
    pub inputs: Vec<TensorType>,
    /// The type of each declared result, in declaration order.
    pub outputs: Vec<TensorType>,
}

/// The evaluation content of a node.
enum NodeBody {
    Parameter,
    Op(BoxedOpNodeType),
    Result,
}

impl NodeBody {
    fn name(&self) -> &str {
        match self {
            Self::Parameter => "parameter",
            Self::Op(node) => node.name(),
            Self::Result => "result",
        }
    }

    fn namespace(&self) -> &str {
        match self {
            Self::Parameter | Self::Result => QISKIT,
            Self::Op(node) => node.namespace(),
        }
    }

    fn full_name(&self) -> String {
        match self {
            Self::Parameter | Self::Result => format!("{}.{}", self.namespace(), self.name()),
            Self::Op(node) => node.full_name(),
        }
    }

    fn role(&self) -> NodeRole {
        match self {
            Self::Parameter => NodeRole::Parameter,
            Self::Op(_) => NodeRole::Op,
            Self::Result => NodeRole::Result,
        }
    }

    fn arity(&self) -> usize {
        match self {
            Self::Parameter => 0,
            Self::Op(node) => node.arity(),
            Self::Result => 1,
        }
    }

    fn has_builtin_eval(&self) -> bool {
        match self {
            Self::Parameter | Self::Result => true,
            Self::Op(node) => node.has_builtin_eval(),
        }
    }
}

/// One node of a function: what it is, what it reads, and what it produces.
struct Node {
    body: NodeBody,
    /// The values this node consumes, in operand order. There are always [`NodeBody::arity`] of
    /// them, so a half-wired node cannot be represented, and this is the only record of how the
    /// function is connected.
    operands: Vec<Value>,
    /// The types inference produced for this node's results, one per result.
    output_types: Vec<TensorType>,
}

/// A read-only view of one node of a [`ProgramFunction`].
///
/// This holds the function as well as the node, because a node's operand types live on the nodes
/// that produce them and so cannot be read from the node alone.
#[derive(Clone, Copy)]
pub struct NodeRef<'a> {
    function: &'a ProgramFunction,
    id: NodeId,
}

impl<'a> NodeRef<'a> {
    fn node(&self) -> &'a Node {
        &self.function.nodes[self.id.index()]
    }

    /// This node's position in its function, which is its identity.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// What part this node plays: an operation, or one end of the function's boundary.
    pub fn role(&self) -> NodeRole {
        self.node().body.role()
    }

    /// This node's type name within its namespace, `add` for instance.
    pub fn name(&self) -> &'a str {
        self.node().body.name()
    }

    /// The namespace this node's type belongs to, [`QISKIT`](crate::nodes::QISKIT) for a node type
    /// Qiskit defines.
    pub fn namespace(&self) -> &'a str {
        self.node().body.namespace()
    }

    /// The name that categorizes this node and that a backend dispatches on, `qiskit.add` for
    /// instance. Allocates; [`Self::name`] and [`Self::namespace`] do not.
    pub fn full_name(&self) -> String {
        self.node().body.full_name()
    }

    /// Whether Qiskit can evaluate this node in-process.
    pub fn has_builtin_eval(&self) -> bool {
        self.node().body.has_builtin_eval()
    }

    /// The values this node consumes, in operand order.
    pub fn operands(&self) -> &'a [Value] {
        &self.node().operands
    }

    /// The type of each operand, read from the node that produces it.
    pub fn operand_types(&self) -> impl Iterator<Item = &'a TensorType> {
        let function = self.function;
        self.node()
            .operands
            .iter()
            .map(move |&value| function.type_of(value).expect("an operand always exists"))
    }

    /// The values this node produces, in result order.
    pub fn outputs(&self) -> impl Iterator<Item = Value> + 'a {
        let node = self.id;
        (0..self.node().output_types.len() as Slot).map(move |slot| Value { node, slot })
    }

    /// The type of each value this node produces, in result order.
    pub fn output_types(&self) -> &'a [TensorType] {
        &self.node().output_types
    }
}

/// Why a node or a result could not be added to a [`ProgramFunction`].
#[derive(Debug, Error)]
pub enum FunctionError {
    /// An operand names a value this function has not produced.
    #[error("operand {operand}: {value} was not produced by this function")]
    UnknownOperand { operand: usize, value: Value },

    /// A declared result names a value this function has not produced.
    #[error("{0} was not produced by this function")]
    UnknownValue(Value),

    /// The operand count does not match the node's declared [`OpNodeType::arity`].
    #[error("{full_name} takes {expected} operand(s), got {actual}")]
    OperandArity {
        full_name: String,
        expected: usize,
        actual: usize,
    },

    /// The node rejected the types of the values wired to it. This is the build-time counterpart of
    /// a run-time dtype or shape error.
    #[error("{full_name} rejected its operand types")]
    TypeError {
        full_name: String,
        #[source]
        source: BoxedOpNodeError,
    },
}

/// Why [`ProgramFunction::eval`] could not produce results.
#[derive(Debug, Error)]
pub enum FunctionEvalError {
    /// The argument count does not match the number of declared parameters.
    #[error("expected {expected} argument(s), got {actual}")]
    ArgumentArity { expected: usize, actual: usize },

    /// An argument does not satisfy the type this (monomorphic) function declares for that
    /// parameter.
    #[error("argument {parameter}: expected {expected}, got {actual}")]
    ArgumentTypeMismatch {
        parameter: usize,
        expected: TensorType,
        actual: TensorType,
    },

    /// The function contains a node Qiskit has no in-process implementation of.
    #[error("node {node} ({full_name}) has no built-in implementation")]
    NoBuiltinEval { node: NodeId, full_name: String },

    /// A node returned an error from its [`OpNodeType::eval`].
    #[error("evaluating node {node} ({full_name})")]
    NodeFailed {
        node: NodeId,
        full_name: String,
        #[source]
        source: BoxedOpNodeError,
    },

    /// A node's `eval` returned a different number of tensors than its type inference promised when
    /// it was added. This is a bug in the node; it cannot be caught statically, because nodes are
    /// stored type-erased.
    #[error("node {node} returned {actual} result(s), expected {expected}")]
    ResultArityMismatch {
        node: NodeId,
        expected: usize,
        actual: usize,
    },
}

/// A tensor dataflow of nodes.
///
/// Each node can have one of three roles:
///  * parameter: an input to the function specifying exactly one tensor demanded at call time
///  * result: an output to the function specifying one tensor the function returns
///  * op: some atomic operation represented as a [`OpNodeType`]
///
/// Each node has some number of operands, and a node can only be added when values exist in
/// the graph to assign to the operands; the first node added must have arity `0`, such as a
/// parameter or [`Constant`](crate::nodes::Constant) node. Values are specified using [`Value`]
/// which is a struct containing the index of an existing node along with an index of one of its
/// output slots.
///
/// Type compatibility of values and the operands of the nodes that act on them is checked when
/// adding the node. This implies that a `ProgramFunction` cannot be malformed by construction.
/// Also by construction, this data model is SSA compliant and the stored node order is topological
/// with respect to evaluation.
pub struct ProgramFunction {
    /// Every node, indexed by [`NodeId`].
    nodes: Vec<Node>,
    /// Indices of all parameter nodes in declaration order.
    parameters: Vec<NodeId>,
    /// Indices of all result nodes in declaration order. Two of them may read one value.
    results: Vec<NodeId>,
}

impl Default for ProgramFunction {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgramFunction {
    /// Construct a new, empty function.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            parameters: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Declare a parameter of the given type, returning its value.
    pub fn add_parameter(&mut self, ty: TensorType) -> Value {
        let id = self.push(NodeBody::Parameter, Vec::new(), vec![ty]);
        self.parameters.push(id);
        Value { node: id, slot: 0 }
    }

    /// Apply `node` to `operands`, returning the values it produces in result order.
    pub fn add_node<N>(&mut self, node: N, operands: &[Value]) -> Result<Vec<Value>, FunctionError>
    where
        N: OpNodeType + Clone + Send + Sync + 'static,
        N::Error: std::error::Error + Send + Sync + 'static,
    {
        self.add_boxed_node(erase(node), operands)
    }

    /// Apply a boxed node type, returning the values it produces in result order.
    pub fn add_boxed_node(
        &mut self,
        node: BoxedOpNodeType,
        operands: &[Value],
    ) -> Result<Vec<Value>, FunctionError> {
        if operands.len() != node.arity() {
            return Err(FunctionError::OperandArity {
                full_name: node.full_name(),
                expected: node.arity(),
                actual: operands.len(),
            });
        }

        let mut operand_types = Vec::with_capacity(operands.len());
        for (operand, &value) in operands.iter().enumerate() {
            let Some(ty) = self.type_of(value) else {
                return Err(FunctionError::UnknownOperand { operand, value });
            };
            operand_types.push(ty.clone());
        }

        let output_types =
            node.infer_output_types(&operand_types)
                .map_err(|source| FunctionError::TypeError {
                    full_name: node.full_name(),
                    source,
                })?;

        let id = self.push(NodeBody::Op(node), operands.to_vec(), output_types);
        Ok(self
            .node(id)
            .expect("the node was just pushed")
            .outputs()
            .collect())
    }

    /// Declare `value` as the next result of this function.
    pub fn add_result(&mut self, value: Value) -> Result<(), FunctionError> {
        if self.type_of(value).is_none() {
            return Err(FunctionError::UnknownValue(value));
        }
        let id = self.push(NodeBody::Result, vec![value], Vec::new());
        self.results.push(id);
        Ok(())
    }

    /// Append a node, returning the id it was given.
    fn push(
        &mut self,
        body: NodeBody,
        operands: Vec<Value>,
        output_types: Vec<TensorType>,
    ) -> NodeId {
        debug_assert_eq!(
            operands.len(),
            body.arity(),
            "a node stores exactly as many operands as its arity"
        );
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            body,
            operands,
            output_types,
        });
        id
    }

    /// The type of `value`, or `None` if it does not belong to this function.
    pub fn type_of(&self, value: Value) -> Option<&TensorType> {
        self.nodes
            .get(value.node.index())?
            .output_types
            .get(value.slot())
    }

    /// The parameter nodes, in declaration order.
    pub fn parameters(&self) -> &[NodeId] {
        &self.parameters
    }

    /// The result nodes, in declaration order.
    pub fn results(&self) -> &[NodeId] {
        &self.results
    }

    /// The value of each parameter, in declaration order.
    pub fn parameter_values(&self) -> impl Iterator<Item = Value> + '_ {
        self.parameters.iter().map(|&node| Value { node, slot: 0 })
    }

    /// The value each result returns, in declaration order.
    pub fn result_values(&self) -> impl Iterator<Item = Value> + '_ {
        self.results
            .iter()
            .map(|&node| self.nodes[node.index()].operands[0])
    }

    /// This function's type contract: the types of its parameters and of its results.
    pub fn signature(&self) -> Signature {
        Signature {
            inputs: self.value_types(self.parameter_values()),
            outputs: self.value_types(self.result_values()),
        }
    }

    /// A view of the node with the given id, or `None`.
    pub fn node(&self, id: NodeId) -> Option<NodeRef<'_>> {
        (id.index() < self.nodes.len()).then_some(NodeRef { function: self, id })
    }

    /// The number of nodes in this function, boundary nodes included.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Iterate over every node, in topological order.
    pub fn iter_nodes(&self) -> impl Iterator<Item = NodeRef<'_>> {
        (0..self.nodes.len() as u32).map(|id| NodeRef {
            function: self,
            id: NodeId(id),
        })
    }

    /// Whether Qiskit can evaluate every node of this function in-process.
    pub fn has_builtin_eval(&self) -> bool {
        self.first_without_builtin_eval().is_none()
    }

    /// Evaluate this function against `args`, one per parameter in declaration order.
    ///
    /// Walks the nodes in storage order, which is topological, over a single dense environment,
    /// releasing each intermediate once its last consumer has run.
    pub fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, FunctionEvalError> {
        // The function is monomorphic, so its declared parameter types are the only ones its nodes
        // were built for. Checking them here names the argument the caller supplied.
        self.check_argument_types(args)?;
        if let Some(node) = self.first_without_builtin_eval() {
            return Err(FunctionEvalError::NoBuiltinEval {
                node: node.id(),
                full_name: node.full_name(),
            });
        }

        let offsets = self.value_offsets();
        let last_use = self.last_use(&offsets);
        let flat = |value: Value| offsets[value.node.index()] as usize + value.slot();

        let total = *offsets.last().expect("offsets always end with the total");
        let mut env: Vec<Option<Tensor>> = vec![None; total as usize];
        let mut outputs: Vec<Option<Tensor>> = vec![None; self.results.len()];
        let mut next_result = 0;

        for (position, node) in self.nodes.iter().enumerate() {
            let id = NodeId(position as u32);
            match &node.body {
                NodeBody::Parameter => {
                    // Parameters are the sources of the dataflow: their values come from the caller,
                    // in the order the parameters were declared.
                    let parameter = self
                        .parameters
                        .iter()
                        .position(|&declared| declared == id)
                        .expect("a parameter node is always in `parameters`");
                    env[offsets[position] as usize] = Some(args[parameter].clone());
                }
                NodeBody::Op(op) => {
                    // Every operand is present: it was produced by an earlier node, and it cannot
                    // have been released, because this node's use of it is at or before its last.
                    let operands: Vec<Tensor> = node
                        .operands
                        .iter()
                        .map(|&value| {
                            env[flat(value)]
                                .clone()
                                .expect("an operand is produced before its consumer runs")
                        })
                        .collect();

                    let results =
                        op.eval(&operands)
                            .map_err(|source| FunctionEvalError::NodeFailed {
                                node: id,
                                full_name: op.full_name(),
                                source,
                            })?;
                    if results.len() != node.output_types.len() {
                        return Err(FunctionEvalError::ResultArityMismatch {
                            node: id,
                            expected: node.output_types.len(),
                            actual: results.len(),
                        });
                    }
                    for (slot, tensor) in results.into_iter().enumerate() {
                        env[offsets[position] as usize + slot] = Some(tensor);
                    }
                }
                NodeBody::Result => {
                    outputs[next_result] = env[flat(node.operands[0])].clone();
                    next_result += 1;
                }
            }

            // A result node is an ordinary final consumer, so nothing here has to make an exception
            // for a value the function returns.
            for &value in &node.operands {
                if last_use[flat(value)] == Some(position) {
                    env[flat(value)] = None;
                }
            }
            for slot in 0..node.output_types.len() {
                let index = offsets[position] as usize + slot;
                if last_use[index] == Some(position) {
                    env[index] = None;
                }
            }
        }

        Ok(outputs
            .into_iter()
            .map(|tensor| tensor.expect("every result node runs"))
            .collect())
    }

    /// The first node Qiskit has no in-process implementation of.
    ///
    /// [`Self::eval`] consults this before computing anything, so a function that needs a backend
    /// fails at the top rather than part-way through a walk that has produced intermediates.
    fn first_without_builtin_eval(&self) -> Option<NodeRef<'_>> {
        self.iter_nodes().find(|node| !node.has_builtin_eval())
    }

    /// The types of `values`, which must all belong to this function.
    fn value_types(&self, values: impl Iterator<Item = Value>) -> Vec<TensorType> {
        values
            .map(|value| {
                self.type_of(value)
                    .expect("a boundary value always exists")
                    .clone()
            })
            .collect()
    }

    /// Validate that every argument satisfies its parameter's declared type.
    fn check_argument_types(&self, args: &[Tensor]) -> Result<(), FunctionEvalError> {
        if args.len() != self.parameters.len() {
            return Err(FunctionEvalError::ArgumentArity {
                expected: self.parameters.len(),
                actual: args.len(),
            });
        }
        for (parameter, (arg, value)) in args.iter().zip(self.parameter_values()).enumerate() {
            let expected = self.type_of(value).expect("a parameter always exists");
            if !arg.matches(expected) {
                return Err(FunctionEvalError::ArgumentTypeMismatch {
                    parameter,
                    expected: expected.clone(),
                    actual: arg.tensor_type(),
                });
            }
        }
        Ok(())
    }

    /// Where each node's block of values starts in a dense environment, with the total appended.
    ///
    /// This is the flat value numbering [`Self::eval`] uses, derived when it is needed rather than
    /// stored: a node's values are already grouped by their producer, so the offsets are one prefix
    /// sum away.
    fn value_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(self.nodes.len() + 1);
        let mut total = 0;
        for node in &self.nodes {
            offsets.push(total);
            total += node.output_types.len() as u32;
        }
        offsets.push(total);
        offsets
    }

    /// For each value, the node position after which nothing needs it.
    ///
    /// Lets [`Self::eval`] release a tensor as soon as its final consumer has run, so that peak
    /// memory is the working set rather than every intermediate the walk ever produced. A value
    /// nothing goes on to consume dies where it is produced, which is what keeps an unused result
    /// from being held for the whole walk.
    fn last_use(&self, offsets: &[u32]) -> Vec<Option<usize>> {
        let total = *offsets.last().expect("offsets always end with the total");
        let mut last = vec![None; total as usize];
        for (position, node) in self.nodes.iter().enumerate() {
            for slot in 0..node.output_types.len() {
                last[offsets[position] as usize + slot] = Some(position);
            }
            // A consumer overwrites the producer's own position, which is strictly later.
            for &value in &node.operands {
                last[offsets[value.node.index()] as usize + value.slot()] = Some(position);
            }
        }
        last
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::nodes::{Add, Constant, Mean};
    use crate::tensor::{DType, Dim};

    /// The type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    // ---------------------------------------------------------------------------
    // Building and evaluating
    // ---------------------------------------------------------------------------

    #[test]
    fn a_function_adds_two_parameters_and_evaluates() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let y = function.add_parameter(f64_1d(3));
        let sum = function.add_node(Add, &[x, y]).unwrap();
        assert_eq!(sum.len(), 1);
        function.add_result(sum[0]).unwrap();

        let results = function
            .eval(&[
                Tensor::from([1.0_f64, 2.0, 3.0]),
                Tensor::from([10.0_f64, 20.0, 30.0]),
            ])
            .unwrap();

        assert_eq!(results, vec![Tensor::from([11.0_f64, 22.0, 33.0])]);
    }

    #[test]
    fn a_signature_is_available_without_evaluating() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let sum = function.add_node(Add, &[x, y]).unwrap()[0];
        let mean = function.add_node(Mean::new(0), &[sum]).unwrap()[0];
        function.add_result(mean).unwrap();

        assert_eq!(
            function.signature(),
            Signature {
                inputs: vec![f64_1d(2), f64_1d(2)],
                outputs: vec![TensorType {
                    dtype: DType::F64,
                    shape: vec![],
                }],
            }
        );
        assert_eq!(function.type_of(sum), Some(&f64_1d(2)));
    }

    #[test]
    fn parameters_and_results_are_positional() {
        // The two parameters have distinct types, so the order in which they reach the node is
        // observable in the result.
        let mut function = ProgramFunction::new();
        let short = function.add_parameter(f64_1d(1));
        let long = function.add_parameter(f64_1d(2));
        assert_eq!(
            function.parameter_values().collect::<Vec<_>>(),
            vec![short, long]
        );

        // Both results name the same value, which is allowed, and a parameter may be returned
        // directly.
        function.add_result(long).unwrap();
        function.add_result(long).unwrap();
        function.add_result(short).unwrap();
        assert_eq!(
            function.result_values().collect::<Vec<_>>(),
            vec![long, long, short]
        );

        let results = function
            .eval(&[Tensor::from([1.0_f64]), Tensor::from([2.0_f64, 3.0])])
            .unwrap();
        assert_eq!(
            results,
            vec![
                Tensor::from([2.0_f64, 3.0]),
                Tensor::from([2.0_f64, 3.0]),
                Tensor::from([1.0_f64]),
            ]
        );
    }

    #[test]
    fn a_value_may_be_consumed_more_than_once() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let doubled = function.add_node(Add, &[x, x]).unwrap()[0];
        let quadrupled = function.add_node(Add, &[doubled, doubled]).unwrap()[0];
        function.add_result(quadrupled).unwrap();

        assert_eq!(
            function.eval(&[Tensor::from([1.0_f64, 2.0])]).unwrap(),
            vec![Tensor::from([4.0_f64, 8.0])]
        );
    }

    #[test]
    fn a_node_whose_results_nothing_uses_does_not_disturb_the_rest() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        function.add_node(Mean::new(0), &[x]).unwrap();
        let sum = function.add_node(Add, &[x, x]).unwrap()[0];
        function.add_result(sum).unwrap();

        assert_eq!(
            function.eval(&[Tensor::from([1.0_f64, 2.0])]).unwrap(),
            vec![Tensor::from([2.0_f64, 4.0])]
        );
    }

    #[test]
    fn reducing_a_zero_length_axis_yields_a_non_finite_value() {
        // A zero-length axis divides by zero. Failing the whole program over it would be worse than
        // handing back a value the caller can see is not a number.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(0));
        let mean = function.add_node(Mean::new(0), &[x]).unwrap()[0];
        function.add_result(mean).unwrap();

        let results = function
            .eval(&[Tensor::from(&[] as &[f64])])
            .expect("an empty reduction is not an error");
        let [Tensor::F64(mean)] = &results[..] else {
            panic!("expected one F64 result, got {results:?}")
        };
        assert!(mean.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn a_function_can_be_sent_between_threads() {
        // A job drives evaluation and is where all concurrency lives, so it must be able to hold one.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ProgramFunction>();
    }

    #[test]
    fn a_constant_holds_one_tensor() {
        let mut function = ProgramFunction::new();
        let two = function
            .add_node(Constant::new(Tensor::from([2.0_f64, 2.0])), &[])
            .unwrap();
        assert_eq!(two.len(), 1);
        let x = function.add_parameter(f64_1d(2));
        let sum = function.add_node(Add, &[x, two[0]]).unwrap()[0];
        function.add_result(sum).unwrap();

        assert_eq!(
            function.eval(&[Tensor::from([1.0_f64, 10.0])]).unwrap(),
            vec![Tensor::from([3.0_f64, 12.0])]
        );
    }

    // ---------------------------------------------------------------------------
    // Nodes, values, and roles
    // ---------------------------------------------------------------------------

    #[test]
    fn a_node_type_name_carries_its_namespace() {
        assert_eq!(Add.full_name(), "qiskit.add");
        assert_eq!(Elsewhere.full_name(), "vendor.elsewhere");
    }

    #[test]
    fn a_function_reports_the_nodes_it_holds() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        function.add_node(Add, &[x, x]).unwrap();
        let second = function.add_node(Add, &[x, x]).unwrap()[0];
        function.add_result(second).unwrap();

        // A node has no name of its own, so the shape of a built function is pinned by counting nodes
        // per type rather than by looking one up.
        assert_eq!(
            function
                .iter_nodes()
                .filter(|node| node.full_name() == "qiskit.add")
                .count(),
            2
        );
        // Boundary nodes are nodes, so they are counted too: one parameter, two adds, one result.
        assert_eq!(function.node_count(), 4);

        let add = function
            .iter_nodes()
            .find(|node| node.name() == "add")
            .unwrap();
        assert_eq!(add.role(), NodeRole::Op);
        assert_eq!(add.operands(), &[x, x]);
        assert_eq!(add.outputs().count(), 1);
        assert!(add.has_builtin_eval());
    }

    #[test]
    fn a_functions_boundary_is_made_of_nodes() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let sum = function.add_node(Add, &[x, x]).unwrap()[0];
        function.add_result(sum).unwrap();

        let roles: Vec<NodeRole> = function.iter_nodes().map(|node| node.role()).collect();
        assert_eq!(
            roles,
            vec![NodeRole::Parameter, NodeRole::Op, NodeRole::Result]
        );

        // A boundary node is named like any other, so a consumer matching on type names sees it.
        let names: Vec<String> = function.iter_nodes().map(|node| node.full_name()).collect();
        assert_eq!(
            names,
            vec!["qiskit.parameter", "qiskit.add", "qiskit.result"]
        );

        // A parameter produces its value and consumes nothing; a result is the other way
        // round.
        let parameter = function.node(function.parameters()[0]).unwrap();
        assert!(parameter.operands().is_empty());
        assert_eq!(parameter.output_types(), &[f64_1d(2)]);

        let result = function.node(function.results()[0]).unwrap();
        assert_eq!(result.operands(), &[sum]);
        assert!(result.output_types().is_empty());
    }

    #[test]
    fn a_value_names_the_node_and_slot_that_produced_it() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let sum = function.add_node(Add, &[x, x]).unwrap()[0];

        assert_eq!(x.node(), function.parameters()[0]);
        assert_eq!(x.slot(), 0);
        assert_eq!(sum.node().index(), 1, "the add is the second node");
        assert_eq!(sum.to_string(), "%1");

        // An operand's type is read through the node that produced it rather than stored twice.
        let add = function.node(sum.node()).unwrap();
        assert_eq!(
            add.operand_types().collect::<Vec<_>>(),
            vec![&f64_1d(2), &f64_1d(2)]
        );
    }

    // ---------------------------------------------------------------------------
    // Build-time rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn the_wrong_operand_count_is_rejected() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));

        let Err(err) = function.add_node(Add, &[x]) else {
            panic!("a one-operand add is rejected")
        };
        assert_eq!(err.to_string(), "qiskit.add takes 2 operand(s), got 1");
    }

    #[test]
    fn an_unknown_operand_is_rejected_naming_its_position() {
        let mut known = ProgramFunction::new();
        let x = known.add_parameter(f64_1d(1));

        // A value is only meaningful within the function that issued it, so one from elsewhere is
        // simply unknown here.
        let mut other = ProgramFunction::new();
        other.add_parameter(f64_1d(1));
        let stranger = other.add_parameter(f64_1d(1));

        let Err(err) = known.add_node(Add, &[x, stranger]) else {
            panic!("an operand from another function is rejected")
        };
        assert_eq!(
            err.to_string(),
            "operand 1: %1 was not produced by this function"
        );
    }

    #[test]
    fn an_unknown_result_is_rejected() {
        let mut known = ProgramFunction::new();
        let mut other = ProgramFunction::new();
        let stranger = other.add_parameter(f64_1d(1));

        let Err(err) = known.add_result(stranger) else {
            panic!("a result from another function is rejected")
        };
        assert!(matches!(err, FunctionError::UnknownValue(value) if value == stranger));
    }

    #[test]
    fn a_rejected_operand_type_is_reported_when_the_node_is_added() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let y = function.add_parameter(f64_1d(4));

        let Err(err) = function.add_node(Add, &[x, y]) else {
            panic!("shapes that do not broadcast are rejected")
        };
        let FunctionError::TypeError { full_name, source } = &err else {
            panic!("expected a type error, got {err}")
        };
        assert_eq!(full_name, "qiskit.add", "the node type is named");
        assert_eq!(
            source.to_string(),
            "shapes [3] and [4] are not broadcast-compatible",
            "both operand shapes are named"
        );
    }

    #[test]
    fn a_dtype_the_node_cannot_compute_is_rejected_when_the_node_is_added() {
        // A node accepts only the dtypes its implementation covers, so a function that type-checks
        // cannot then fail on a dtype as it runs.
        let mut function = ProgramFunction::new();
        let bit = function.add_parameter(TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(2)],
        });

        let Err(err) = function.add_node(Add, &[bit, bit]) else {
            panic!("`add` has no Bit implementation, so a Bit operand is rejected")
        };
        let FunctionError::TypeError { source, .. } = &err else {
            panic!("expected a type error, got {err}")
        };
        assert_eq!(
            source.to_string(),
            "operands of dtype Bit and Bit promote to Bit, which is not supported"
        );
    }

    #[test]
    fn a_reduction_out_of_bounds_axis_is_reported_when_the_node_is_added() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));

        let Err(err) = function.add_node(Mean::new(1), &[x]) else {
            panic!("an out-of-bounds axis is rejected")
        };
        let FunctionError::TypeError { source, .. } = &err else {
            panic!("expected a type error, got {err}")
        };
        assert_eq!(
            source.to_string(),
            "axis 1 is out of bounds for tensor with 1 dimension(s)"
        );
    }

    // ---------------------------------------------------------------------------
    // Coercion
    // ---------------------------------------------------------------------------

    #[test]
    fn arithmetic_promotes_its_operand_dtypes() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let y = function.add_parameter(TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Fixed(3)],
        });
        let sum = function.add_node(Add, &[x, y]).unwrap()[0];
        function.add_result(sum).unwrap();

        assert_eq!(
            function.type_of(sum),
            Some(&f64_1d(3)),
            "F32 and F64 promote to F64"
        );
        assert_eq!(
            function
                .eval(&[
                    Tensor::from([1.0_f64, 2.0, 3.0]),
                    Tensor::from([10.0_f32, 20.0, 30.0]),
                ])
                .unwrap(),
            vec![Tensor::from([11.0_f64, 22.0, 33.0])]
        );
    }

    #[test]
    fn arithmetic_broadcasts_its_operand_shapes() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let y = function.add_parameter(f64_1d(1));
        let sum = function.add_node(Add, &[x, y]).unwrap()[0];
        function.add_result(sum).unwrap();

        assert_eq!(function.type_of(sum), Some(&f64_1d(3)));
        assert_eq!(
            function
                .eval(&[Tensor::from([1.0_f64, 2.0, 3.0]), Tensor::from([10.0_f64])])
                .unwrap(),
            vec![Tensor::from([11.0_f64, 12.0, 13.0])]
        );
    }

    #[test]
    fn averaging_a_bit_tensor_yields_a_float() {
        // The flagship post-processing step: shots come back as bits and a mean of them is a
        // probability. A global promotion rule could not express this, so it is the reduce family's.
        let mut function = ProgramFunction::new();
        let shots = function.add_parameter(TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(4)],
        });
        let mean = function.add_node(Mean::new(0), &[shots]).unwrap()[0];
        function.add_result(mean).unwrap();

        assert_eq!(
            function.type_of(mean),
            Some(&TensorType {
                dtype: DType::F64,
                shape: vec![],
            })
        );
        let results = function
            .eval(&[Tensor::from([1_u8, 0, 1, 1]).cast(DType::Bit)])
            .unwrap();
        let [Tensor::F64(mean)] = &results[..] else {
            panic!("expected one F64 result, got {results:?}")
        };
        assert_eq!(mean.iter().copied().collect::<Vec<f64>>(), vec![0.75]);
    }

    #[test]
    fn two_bounded_axes_cannot_be_combined() {
        // Nothing proves their true sizes agree, and the sizes post-selection produces are
        // exponentially unlikely to. Refusing here beats a shape error after the data was paid for.
        let bounded = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 8 }],
        };
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(bounded.clone());
        let y = function.add_parameter(bounded);

        let Err(err) = function.add_node(Add, &[x, y]) else {
            panic!("two bounded axes cannot be paired up")
        };
        let FunctionError::TypeError { source, .. } = &err else {
            panic!("expected a type error, got {err}")
        };
        assert_eq!(
            source.to_string(),
            "shape [<=8] has an axis whose size is only bounded above, where a true size is required"
        );
    }

    #[test]
    fn a_bounded_axis_broadcasts_against_a_fixed_one() {
        // Post-selected data against a per-register constant is what bounded dynamism is for.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 8 }, Dim::Fixed(2)],
        });
        let y = function.add_parameter(f64_1d(2));
        let scaled = function.add_node(Add, &[x, y]).unwrap()[0];

        assert_eq!(
            function.type_of(scaled),
            Some(&TensorType {
                dtype: DType::F64,
                shape: vec![Dim::Bounded { max: 8 }, Dim::Fixed(2)],
            })
        );
    }

    // ---------------------------------------------------------------------------
    // Evaluation-time rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn the_wrong_argument_count_is_rejected() {
        let mut function = ProgramFunction::new();
        function.add_parameter(f64_1d(1));
        function.add_parameter(f64_1d(1));

        let Err(err) = function.eval(&[Tensor::from([1.0_f64])]) else {
            panic!("one argument for two parameters is rejected")
        };
        assert_eq!(err.to_string(), "expected 2 argument(s), got 1");
    }

    #[test]
    fn an_argument_of_the_wrong_type_is_rejected_naming_both_types() {
        let mut function = ProgramFunction::new();
        function.add_parameter(f64_1d(3));

        let Err(err) = function.eval(&[Tensor::from([1.0_f64, 2.0])]) else {
            panic!("an argument of the wrong shape is rejected")
        };
        assert_eq!(
            err.to_string(),
            "argument 0: expected F64[3], got F64[2]",
            "both the declared and the supplied type are named"
        );
    }

    #[test]
    fn a_bounded_parameter_admits_any_argument_within_its_bound() {
        // A declared type constrains an argument rather than equalling it, so a bounded axis is
        // usable: an argument shorter than the bound is what a bounded axis is for.
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 4 }],
        });
        let doubled = function.add_node(Elsewhere, &[x]).unwrap()[0];
        function.add_result(doubled).unwrap();

        // `Elsewhere` has no built-in implementation, so evaluation stops at the locality check, which
        // is past the argument check under test here.
        for len in 0..=4 {
            let arg = Tensor::from(vec![1.0_f64; len].as_slice());
            assert!(
                matches!(
                    function.eval(&[arg]),
                    Err(FunctionEvalError::NoBuiltinEval { .. })
                ),
                "an argument of length {len} is within the bound of 4"
            );
        }

        let Err(err) = function.eval(&[Tensor::from([1.0_f64; 5])]) else {
            panic!("an argument past the bound is rejected")
        };
        assert_eq!(err.to_string(), "argument 0: expected F64[<=4], got F64[5]");
    }

    // ---------------------------------------------------------------------------
    // Locality
    // ---------------------------------------------------------------------------

    /// A node type defined outside the crate, in its own namespace, with no in-process
    /// implementation — which is how a backend contributes work Qiskit cannot perform itself.
    #[derive(Clone)]
    struct Elsewhere;

    /// The error [`Elsewhere`] returns when asked to evaluate itself.
    #[derive(Debug)]
    struct NoImplementation;

    impl fmt::Display for NoImplementation {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "vendor.elsewhere has no in-process implementation")
        }
    }

    impl std::error::Error for NoImplementation {}

    impl OpNodeType for Elsewhere {
        type Error = NoImplementation;

        fn name(&self) -> &str {
            "elsewhere"
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
            Err(NoImplementation)
        }
    }

    #[test]
    fn a_function_of_built_in_nodes_is_locally_evaluable() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        assert!(
            function.has_builtin_eval(),
            "an empty function has nothing that cannot run"
        );
        function.add_node(Add, &[x, x]).unwrap();
        assert!(function.has_builtin_eval());
    }

    #[test]
    fn one_node_without_a_built_in_implementation_makes_the_function_not_locally_evaluable() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let out = function.add_node(Elsewhere, &[x]).unwrap()[0];
        function.add_result(out).unwrap();

        assert!(!function.has_builtin_eval());

        let Err(err) = function.eval(&[Tensor::from([1.0_f64])]) else {
            panic!("a function with no built-in implementation cannot be evaluated")
        };
        let FunctionEvalError::NoBuiltinEval { node, full_name } = &err else {
            panic!("expected a locality error, got {err}")
        };
        assert_eq!(*node, out.node(), "the offending node is named");
        assert_eq!(full_name, "vendor.elsewhere");
    }

    #[test]
    fn a_node_defined_outside_the_crate_is_type_checked_like_any_other() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(4));
        let out = function.add_node(Elsewhere, &[x]).unwrap()[0];

        assert_eq!(function.type_of(out), Some(&f64_1d(4)));

        let Err(err) = function.add_node(Elsewhere, &[x, x]) else {
            panic!("a two-operand call to a unary node is rejected")
        };
        assert_eq!(
            err.to_string(),
            "vendor.elsewhere takes 1 operand(s), got 2"
        );
    }
}
