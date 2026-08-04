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

use anyhow;
use hashbrown::HashMap;
use std::any::{Any, TypeId};
use thiserror::Error;

pub struct PassContext {
    // The String key could be promoted to some AnalysisKey if needed.
    pub data: HashMap<String, Box<dyn Any>>,
}

impl PassContext {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

/// A pass for Qiskit's compiler framework.
///
/// This represents an atomic unit of work done by the compiler, which runs a
/// (possibly non-linear) flow of passes. This trait must be implemented by
/// passes that should run in the compiler.
pub trait Pass {
    // We store a type-erased version of this Pass, and to store the [TypeId] we need the
    // IRs here to have static lifetimes.  This is a requirement of [TypeId] which is only
    // currently defined for static types.
    type InputIR: 'static;
    type OutputIR: 'static;

    /// Run the pass.
    ///
    /// # Args
    /// * `ir` - The input IR. Takes ownership of it.
    ///
    /// # Returns
    /// * `Ok(Self::OutputIR)` - The transformed IR.
    /// * `Err(anyhow::Error)` - A type-erased [anyhow::Error].
    // TODO or use an associated error type?
    fn run(&self, ir: Self::InputIR, context: &mut PassContext) -> anyhow::Result<Self::OutputIR>;
}

/// A type-erased version of the [Pass] trait. This is required to store passes with different
/// associated types in the generic [Task::Transformation] variant.
pub trait AnyPass {
    fn input_type_id(&self) -> TypeId;
    fn output_type_id(&self) -> TypeId;
    fn run(&self, ir: Box<dyn Any>, context: &mut PassContext) -> anyhow::Result<Box<dyn Any>>;
}

impl<P: Pass> AnyPass for P {
    fn input_type_id(&self) -> TypeId {
        TypeId::of::<P::InputIR>()
    }

    fn output_type_id(&self) -> TypeId {
        TypeId::of::<P::OutputIR>()
    }

    fn run(&self, ir: Box<dyn Any>, context: &mut PassContext) -> anyhow::Result<Box<dyn Any>> {
        let ir = ir
            .downcast::<P::InputIR>()
            .expect("The pipeline construction guarantees the input IR has the correct type.");
        let out_ir = P::run(&self, *ir, context)?;
        Ok(Box::new(out_ir))
    }
}

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. It describes how work is being executed, ranging
/// from the simple execution of a single pass, over groups of passes to structured flow control,
/// such as loops. The [PassManager] stores a vector of [Task]s and executes them.
pub enum Task {
    /// A single pass.
    Transformation(Box<dyn AnyPass>),

    /// A group of tasks.
    Group(Vec<Task>),

    /// A sequence of named tasks.
    Stages(Vec<(String, Task)>),

    /// A conditional execution of a task.
    /// Takes a switch function, that takes the type-erased IR and the pass context,
    /// and returns an index to which case to run.
    Switch {
        switch: fn(&dyn Any, &PassContext) -> usize,
        cases: Vec<Task>,
    },

    /// A looped execution.
    /// Runs the body until the condition function returns false.
    Loop {
        condition: fn(&dyn Any, &PassContext) -> bool,
        body: Box<Task>, // must be boxed to define Task's size
    },
}

impl Task {
    fn io_types(&self) -> Result<(TypeId, TypeId), PassManagerError> {
        match self {
            Task::Transformation(pass) => Ok((pass.input_type_id(), pass.output_type_id())),
            Task::Group(group) => {
                // If the group has length 0, or the type is None, return None
                match (group.first(), group.last()) {
                    (Some(first), Some(last)) => Ok((first.io_types()?.0, last.io_types()?.1)),
                    _ => Err(PassManagerError::EmptyTask),
                }
            }
            Task::Loop { condition: _, body } => (*body).io_types(),
            Task::Switch { switch: _, cases } => {
                if let Some(case0) = cases.first() {
                    case0.io_types()
                } else {
                    Err(PassManagerError::EmptyTask)
                }
            }
            Task::Stages(stages) => match (stages.first(), stages.last()) {
                (Some(first), Some(last)) => Ok((first.1.io_types()?.0, last.1.io_types()?.1)),
                _ => Err(PassManagerError::EmptyTask),
            },
        }
    }
}

/// Qiskit's pass manager.
pub struct PassManager {
    // It is UNSAFE to directly mutate the task vector since we are checking that the types
    // match upon construction.
    tasks: Vec<Task>,
}

#[derive(Error, Debug)]
pub enum PassManagerError {
    #[error("Incompatible IR types")]
    IncompatibleTypes,
    #[error(transparent)]
    PassError(#[from] anyhow::Error),
    #[error("Conversion to output IR failed.")]
    FailedOutputConversion,
    #[error("Encountered an empty task.")]
    EmptyTask,
}

impl PassManager {
    pub fn new() -> Self {
        Self { tasks: vec![] }
    }

    pub fn run<IRIn, IROut>(&self, ir: IRIn) -> Result<IROut, PassManagerError>
    where
        IRIn: 'static,
        IROut: 'static,
    {
        if self.tasks.len() == 0 {
            // If there are no tasks, return the input, but cast to IROut
            return cast_box::<IROut>(Box::new(ir));
        }

        let (first, _) = self.tasks[0].io_types()?;
        if first != TypeId::of::<IROut>() {
            return Err(PassManagerError::IncompatibleTypes);
        }

        // Erase the type to pass it through the task execution
        let mut ir: Box<dyn Any> = Box::new(ir);

        // Main iteration loop over tasks
        let mut context = PassContext::new();
        for task in self.tasks.iter() {
            ir = execute_task(task, ir, &mut context)?;
        }

        cast_box::<IROut>(ir)
    }

    /// Try push a [Task] to the pass manager. Returns an error if the types are not
    /// compatible.
    pub fn try_push_task(&mut self, task: Task) -> Result<(), PassManagerError> {
        // Check that the task types are compatible, if there's an existing task and if
        // neither of the tasks are empty.
        if let Some(last_task) = self.tasks.last() {
            let (_, out_type) = last_task.io_types()?;
            let (in_type, _) = task.io_types()?;
            if in_type != out_type {
                return Err(PassManagerError::IncompatibleTypes);
            }
        }
        self.tasks.push(task);
        Ok(())
    }

    /// Try push a pass to the pass manager, which will automatically wrap into a
    /// [Task::Transformation]. Returns an error if the types are not compatible.
    pub fn try_push_pass(&mut self, pass: Box<dyn AnyPass>) -> Result<(), PassManagerError> {
        let task = Task::Transformation(pass.into());
        self.try_push_task(task)
    }

    /// Try and remove a task.
    ///
    /// Returns the removed task and verifies the types are still compatible after removal.
    ///
    /// Panics if the index is out of bounds.
    pub fn try_remove_task(&mut self, index: usize) -> Result<Task, PassManagerError> {
        let task = self.tasks.remove(index);

        if index > 0 && index < self.tasks.len() - 1 {
            let (_, before) = self.tasks[index - 1].io_types()?;
            let (after, _) = self.tasks[index + 1].io_types()?;
            if before != after {
                return Err(PassManagerError::IncompatibleTypes);
            }
        }
        Ok(task)
    }

    /// The number of tasks in the pass manager.
    ///
    /// Note that this does not count any nested tasks.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }
}

/// The task runner. This should not be called standalone, passes should be run
/// via the pass manager.
fn execute_task(
    task: &Task,
    mut ir: Box<dyn Any>,
    context: &mut PassContext,
) -> Result<Box<dyn Any>, PassManagerError> {
    let out = match task {
        Task::Transformation(pass) => pass
            .run(ir, context)
            .map_err(|e| PassManagerError::PassError(e)),
        Task::Group(tasks) => {
            for task in tasks.iter() {
                ir = execute_task(task, ir, context)?;
            }
            Ok(ir)
        }
        Task::Switch { switch, cases } => {
            let index = switch(&ir, context);
            execute_task(&cases[index], ir, context)
        }
        Task::Loop { condition, body } => {
            while condition(&ir, &context) {
                ir = execute_task(&body, ir, context)?;
            }
            Ok(ir)
        }
        Task::Stages(stages) => {
            for (_name, task) in stages.iter() {
                ir = execute_task(task, ir, context)?;
            }
            Ok(ir)
        }
    };
    out
}

/// Internal helper to cast a Box<dyn Any> to an output type.
fn cast_box<OutType>(obj: Box<dyn Any>) -> Result<OutType, PassManagerError>
where
    OutType: 'static,
{
    match obj.downcast::<OutType>() {
        Ok(out) => Ok(*out),
        Err(_) => Err(PassManagerError::FailedOutputConversion),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use qiskit_circuit::{
        bit::ShareableQubit,
        circuit_data::CircuitData,
        dag_circuit::DAGCircuit,
        instruction::Parameters,
        operations::{Param, StandardGate},
    };
    use qiskit_transpiler::passes::run_remove_identity_equiv;
    use smallvec::smallvec;

    struct RemoveIdentities {}

    impl Pass for RemoveIdentities {
        type InputIR = DAGCircuit;
        type OutputIR = DAGCircuit;

        fn run(
            &self,
            mut ir: Self::InputIR,
            _context: &mut PassContext,
        ) -> anyhow::Result<Self::OutputIR> {
            run_remove_identity_equiv(&mut ir, None, None)?;
            Ok(ir)
        }
    }

    struct CountT {}

    impl Pass for CountT {
        type InputIR = CircuitData;
        type OutputIR = CircuitData;

        fn run(
            &self,
            ir: Self::InputIR,
            context: &mut PassContext,
        ) -> anyhow::Result<Self::OutputIR> {
            let count = ir.count_ops();
            let t_count = count.get("t").unwrap_or(&0) + count.get("tdg").unwrap_or(&0);
            context
                .data
                .insert("t_count".to_string(), Box::new(t_count));
            Ok(ir)
        }
    }

    #[test]
    fn test_pass() -> Result<(), PassManagerError> {
        let pass = RemoveIdentities {};

        let mut pm = PassManager::new();
        pm.try_push_pass(Box::new(pass))?;

        let mut dag = DAGCircuit::new();
        let q0 = dag
            .add_qubit_unchecked(ShareableQubit::new_anonymous())
            .unwrap();
        dag.apply_operation_back(StandardGate::H.into(), &[q0], &[], None, None)
            .unwrap();
        dag.apply_operation_back(
            StandardGate::RX.into(),
            &[q0],
            &[],
            Some(Parameters::Params(smallvec![Param::Float(0.)])),
            None,
        )
        .unwrap();

        let out: DAGCircuit = pm.run(dag)?;
        let ops = out.count_ops(false).unwrap();
        assert_eq!(ops.get("h").map(|v| *v), Some(1));
        assert_eq!(ops.get("rx"), None);
        Ok(())
    }

    #[test]
    fn test_incompatible_types() -> Result<(), PassManagerError> {
        let pass1 = RemoveIdentities {};
        let pass2 = CountT {};

        let mut pm = PassManager::new();
        pm.try_push_pass(Box::new(pass1))?;
        let result = pm.try_push_pass(Box::new(pass2));
        assert!(matches!(result, Err(PassManagerError::IncompatibleTypes)));
        Ok(())
    }
}
