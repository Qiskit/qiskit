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

use hashbrown::{HashMap, HashSet};
use std::{
    any::{Any, TypeId},
    fmt::Debug,
};
use thiserror::Error;

/// The pass manager execution environment.
///
/// This contains data managed by the pass manager. A local handle to this is passed into
/// the passes.
#[derive(Default, Debug)]
pub struct PassManagerContext {
    /// The global, catch-all data. The local [PassContext] handles get read-only access to
    /// this data and after pass execution this global state is updated.
    data: HashMap<String, Box<dyn Any>>,
}

/// Context information provided to the passes.
#[derive(Debug)]
pub struct PassContext<'a> {
    /// A reference to the global execution environment.
    global_context: &'a PassManagerContext,

    /// Whether the pass changed the IR or not. If this is `false`, the pass manager
    /// can assume that no changes to IR have been made and potentially perform optimizations.
    pub has_changed: bool,

    /// A local cache of new data.
    updates: ContextUpdates,
}

/// A private struct representing the context updates performed. As long as this contains only
/// a HashMap, we could skip this object, but the context is supposed to contain more generic
/// information.
#[derive(Default, Debug)]
struct ContextUpdates {
    /// New values to insert into the global context.
    insertions: HashMap<String, Box<dyn Any>>,
    /// Keys to delete from the global context.
    deletions: HashSet<String>,
}

impl ContextUpdates {
    fn insert(&mut self, key: String, value: Box<dyn Any>) {
        self.deletions.remove(&key);
        self.insertions.insert(key, value);
    }

    fn delete(&mut self, key: String) {
        self.insertions.remove(&key);
        self.deletions.insert(key);
    }

    fn get(&self, key: impl AsRef<str>) -> Option<&dyn Any> {
        Some(self.insertions.get(key.as_ref())?)
    }
}

impl PassManagerContext {
    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, mut updates: ContextUpdates) {
        for (key, value) in updates.insertions.drain() {
            self.data.insert(key, value);
        }
        for key in updates.deletions.iter() {
            self.data.remove(key);
        }
    }
}

impl<'a> PassContext<'a> {
    fn spawn(global_context: &'a PassManagerContext) -> Self {
        Self {
            global_context,
            has_changed: true,
            updates: ContextUpdates::default(),
        }
    }

    fn into_updates(self) -> ContextUpdates {
        self.updates
    }

    /// Set a new entry in the pass context.
    /// Overwrites the existing value under that key, if it exists.
    pub fn set(&mut self, key: String, value: Box<dyn Any>) {
        self.updates.insert(key, value);
    }

    pub fn delete(&mut self, key: String) {
        self.updates.delete(key);
    }

    /// Get an entry, if it exists.
    ///
    /// This first queries from the local context, then the global.
    pub fn get(&self, key: impl AsRef<str>) -> Option<&dyn Any> {
        let key = key.as_ref();

        // The local registry takes precedence.
        self.updates
            .get(key)
            .or_else(|| self.global_context.data.get(key).map(|value| value as _))
    }
}

/// A pass for Qiskit's compiler framework.
///
/// This represents an atomic unit of work done by the compiler, which runs a
/// (possibly non-linear) flow of passes. This trait must be implemented by
/// passes that should run in the compiler.
pub trait Pass: Send + Sync {
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
    fn run(&self, ir: Self::InputIR, context: &mut PassContext) -> anyhow::Result<Self::OutputIR>;
}

/// A type-erased version of the [Pass] trait. This is required to store passes with different
/// associated types in the generic [Task::Transformation] variant.
pub trait AnyPass: Send + Sync {
    /// Cast the pass to Any to allow downcasting to a target type.
    fn as_any(&self) -> &dyn Any;
    /// Return the type ID of the input IR.
    fn input_type_id(&self) -> TypeId;
    /// Return the type ID of the output IR.
    fn output_type_id(&self) -> TypeId;
    /// Run the pass.
    fn run(&self, ir: Box<dyn Any>, context: &mut PassContext) -> anyhow::Result<Box<dyn Any>>;
}

impl<P: Pass + 'static> AnyPass for P {
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        let out_ir = P::run(self, *ir, context)?;
        Ok(Box::new(out_ir))
    }
}

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. It describes how work is being executed, ranging
/// from the simple execution of a single pass, over groups of passes to structured flow control,
/// such as loops. The [PassManager] stores a vector of [Task]s and executes them.
#[non_exhaustive]
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
        body: Box<Task>,
    },
}

impl Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Task::Transformation(_) => writeln!(f, "Transformation(Box<dyn AnyPass>)"),
            Task::Group(tasks) => writeln!(f, "Group({tasks:?}"),
            Task::Stages(stages) => writeln!(f, "Stages({stages:?}"),
            Task::Switch { switch, cases } => {
                writeln!(f, "Switch {{ switch: {switch:?}, cases: {cases:?} }}")
            }
            Task::Loop { condition, body } => {
                writeln!(f, "Loop {{ condition: {condition:?}, body: {body:?} }}")
            }
        }
    }
}

impl Task {
    fn io_types(&self) -> Option<(TypeId, TypeId)> {
        match self {
            Task::Transformation(pass) => Some((pass.input_type_id(), pass.output_type_id())),
            Task::Group(group) => Some((group.first()?.io_types()?.0, group.last()?.io_types()?.1)),
            Task::Loop { condition: _, body } => (*body).io_types(),
            Task::Switch { switch: _, cases } => cases.first()?.io_types(),
            Task::Stages(stages) => Some((
                stages.first()?.1.io_types()?.0,
                stages.last()?.1.io_types()?.1,
            )),
        }
    }
}

/// Hookpoint for the callback.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CallbackType {
    PostPass,
    PostTask,
    PostStage,
}

/// A (set of) callback(s) to trigger during the pass manager execution.
///
/// A callback sets hookpoints (usually a single one), upon which the functions
/// are called with the defined arguments.
///
/// This trait only requires the trigger to be implemented, the function calls are
/// all optional.
pub trait Callback {
    /// The hookpoints for when the functions are called.
    fn trigger(&self, hookpoint: &CallbackType) -> bool;

    /// The standard callback providing only the IR and pass context.
    fn ir_and_context(&self, _ir: &dyn Any, _context: &PassContext) {}

    /// A callback also providing the pass. This is only called for [CallbackType::PostPass].
    fn with_pass(&self, _pass: &dyn AnyPass, _ir: &dyn Any, _context: &PassContext) {}
}

/// Qiskit's pass manager.
#[derive(Default, Debug)]
pub struct PassManager {
    // It is UNSAFE to directly mutate the task vector since we are checking that the types
    // match upon construction, hence the tasks are private.
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
    #[error("Invalid index ({index}) for ({len}) tasks")]
    IndexError { index: usize, len: usize },
}

impl PassManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the pass manager on the input IR.
    pub fn run<IRIn, IROut>(
        &self,
        ir: IRIn,
        callback: Option<&dyn Callback>,
    ) -> Result<(IROut, PassManagerContext), PassManagerError>
    where
        IRIn: 'static,
        IROut: 'static,
    {
        let mut context = PassManagerContext::new();
        if let Some(first) = self.tasks.first() {
            // Validate the input ID of the first task matches IRIn
            let Some((first_in_id, _)) = first.io_types() else {
                return Err(PassManagerError::EmptyTask);
            };
            if first_in_id != TypeId::of::<IRIn>() {
                return Err(PassManagerError::IncompatibleTypes);
            }
        } else {
            // If there are no tasks, return the input, but cast to IROut
            let ir_out = cast_box::<IROut>(Box::new(ir))?;
            return Ok((ir_out, context));
        }

        let last = self.tasks.last().expect("There is at least 1 task now");
        let Some((_, last_out_id)) = last.io_types() else {
            return Err(PassManagerError::EmptyTask);
        };
        if last_out_id != TypeId::of::<IROut>() {
            return Err(PassManagerError::IncompatibleTypes);
        }

        // Erase the type to pass it through the task execution
        let mut ir: Box<dyn Any> = Box::new(ir);

        // Main iteration loop over tasks
        for task in self.tasks.iter() {
            let mut pass_context = PassContext::spawn(&context);
            ir = execute_task(task, ir, &mut pass_context, callback)?;
            let updates = pass_context.into_updates();
            context.update(updates);
        }

        let ir_out = cast_box::<IROut>(ir)?;
        Ok((ir_out, context))
    }

    /// The number of first-level tasks in the pass manager.
    ///
    /// Note that this does not count any nested tasks.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// Try push a [Task] to the pass manager. Returns an error if the types are not
    /// compatible.
    pub fn try_push_task(&mut self, task: Task) -> Result<(), PassManagerError> {
        // Check that the task types are compatible, if there's an existing task and if
        // neither of the tasks are empty.
        if let Some(last_task) = self.tasks.last() {
            let Some((_, out_type)) = last_task.io_types() else {
                return Err(PassManagerError::EmptyTask);
            };
            let Some((in_type, _)) = task.io_types() else {
                return Err(PassManagerError::EmptyTask);
            };
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
        let task = Task::Transformation(pass);
        self.try_push_task(task)
    }

    /// Try and remove a task.
    ///
    /// Returns the removed task and verifies the types are still compatible after removal.
    pub fn try_remove_task(&mut self, index: usize) -> Result<Task, PassManagerError> {
        if index >= self.tasks.len() {
            return Err(PassManagerError::IndexError {
                index,
                len: self.tasks.len(),
            });
        }

        if index > 0 && index < self.tasks.len() - 1 {
            let (Some((_, before)), Some((after, _))) = (
                self.tasks[index - 1].io_types(),
                self.tasks[index + 1].io_types(),
            ) else {
                return Err(PassManagerError::EmptyTask);
            };
            if before != after {
                return Err(PassManagerError::IncompatibleTypes);
            }
        }
        let task = self.tasks.remove(index);

        Ok(task)
    }

    /// Try and insert a task at an index.
    ///
    /// This pushes the task at `index` to `index + 1` and shifts all subsequent ones by one.
    /// If `index` equals [Self::num_tasks], the task is appended at the end.
    pub fn try_insert_task(&mut self, index: usize, task: Task) -> Result<(), PassManagerError> {
        if index > self.tasks.len() {
            return Err(PassManagerError::IndexError {
                index,
                len: self.tasks.len(),
            });
        }

        if !self.tasks.is_empty() {
            let Some((in_type, out_type)) = task.io_types() else {
                return Err(PassManagerError::EmptyTask);
            };
            if index > 0 {
                let Some((_, before)) = self.tasks[index - 1].io_types() else {
                    return Err(PassManagerError::EmptyTask);
                };
                if before != in_type {
                    return Err(PassManagerError::IncompatibleTypes);
                }
            }
            if index < self.tasks.len() - 1 {
                let Some((after, _)) = self.tasks[index + 1].io_types() else {
                    return Err(PassManagerError::EmptyTask);
                };
                if out_type != after {
                    return Err(PassManagerError::IncompatibleTypes);
                }
            }
        }

        self.tasks.insert(index, task);
        Ok(())
    }

    /// Get a reference to a [Task] at a given index.
    pub fn get_task(&self, index: usize) -> Option<&Task> {
        self.tasks.get(index)
    }
}

/// The task runner. This should not be called standalone, passes should be run
/// via the pass manager.
fn execute_task(
    task: &Task,
    mut ir: Box<dyn Any>,
    context: &mut PassContext,
    callback: Option<&dyn Callback>,
) -> Result<Box<dyn Any>, PassManagerError> {
    let out = match task {
        Task::Transformation(pass) => {
            let out = pass.run(ir, context).map_err(PassManagerError::PassError)?;
            if let Some(cb) = callback
                && cb.trigger(&CallbackType::PostPass)
            {
                // Note that we want to pass a reference to the box content, not cast the
                // box itself to any. Hence we deref, before passing the reference. Not doing this
                // still compiles since Box<dyn Any> itself is castable to Any, but the downcasting
                // further down the line will fail since it tries to cast Box<..> into the type.
                cb.ir_and_context(&*out, context);
                cb.with_pass(&**pass, &out, context);
            }
            Ok(out)
        }
        Task::Group(tasks) => {
            for task in tasks.iter() {
                ir = execute_task(task, ir, context, callback)?;
            }
            Ok(ir)
        }
        Task::Switch { switch, cases } => {
            let index = switch(&ir, context);
            execute_task(&cases[index], ir, context, callback)
        }
        Task::Loop { condition, body } => {
            while condition(&ir, context) {
                ir = execute_task(body, ir, context, callback)?;
            }
            Ok(ir)
        }
        Task::Stages(stages) => {
            for (_name, task) in stages.iter() {
                ir = execute_task(task, ir, context, callback)?;
                if let Some(cb) = callback
                    && cb.trigger(&CallbackType::PostStage)
                {
                    cb.ir_and_context(&*ir, context)
                }
            }
            Ok(ir)
        }
    }?;
    if let Some(cb) = callback
        && cb.trigger(&CallbackType::PostTask)
    {
        cb.ir_and_context(&*out, context)
    }
    Ok(out)
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
    use std::{cell::Cell, rc::Rc};

    use super::*;
    use qiskit_circuit::{
        Qubit,
        bit::ShareableQubit,
        circuit_data::CircuitData,
        dag_circuit::DAGCircuit,
        instruction::Parameters,
        operations::{Param, StandardGate},
    };
    use qiskit_transpiler::passes::run_remove_identity_equiv;
    use smallvec::smallvec;

    #[derive(Clone, Debug)]
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

    #[derive(Clone, Debug)]
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
            let t_count: usize = count.get("t").unwrap_or(&0) + count.get("tdg").unwrap_or(&0);
            context.set("t_count".to_string(), Box::new(t_count));
            Ok(ir)
        }
    }

    struct CheckTCount {
        expected_t_count: usize,
    }

    impl Pass for CheckTCount {
        type InputIR = CircuitData;
        type OutputIR = CircuitData;

        fn run(
            &self,
            ir: Self::InputIR,
            context: &mut PassContext,
        ) -> anyhow::Result<Self::OutputIR> {
            let Some(t_count) = context.get("t_count") else {
                return Err(anyhow::anyhow!("Missing `t_count`"));
            };
            let Some(t_count) = t_count.downcast_ref::<usize>() else {
                return Err(anyhow::anyhow!("Downcasting to usize failed"));
            };
            if *t_count != self.expected_t_count {
                return Err(anyhow::anyhow!(
                    "Expected T count of {} but got {}",
                    self.expected_t_count,
                    t_count
                ));
            }
            Ok(ir)
        }
    }

    struct DagToCircuit {}

    impl Pass for DagToCircuit {
        type InputIR = DAGCircuit;
        type OutputIR = CircuitData;

        fn run(
            &self,
            ir: Self::InputIR,
            _context: &mut PassContext,
        ) -> anyhow::Result<Self::OutputIR> {
            Ok(CircuitData::from_dag_ref(&ir)?)
        }
    }

    struct CounterCallback {
        counter: Rc<Cell<usize>>,
        hookpoint: CallbackType,
    }

    impl CounterCallback {
        fn new(hookpoint: CallbackType) -> Self {
            Self {
                counter: Rc::new(Cell::new(0)),
                hookpoint,
            }
        }
    }

    impl Callback for CounterCallback {
        fn trigger(&self, hookpoint: &CallbackType) -> bool {
            self.hookpoint.eq(hookpoint)
        }

        fn ir_and_context(&self, _ir: &dyn Any, _context: &PassContext) {
            self.counter.set(self.counter.get() + 1);
        }
    }

    #[test]
    fn test_io_types() -> Result<(), PassManagerError> {
        let dag_type = TypeId::of::<DAGCircuit>();
        let circ_type = TypeId::of::<CircuitData>();

        let make_dag_pass = || Task::Transformation(Box::new(RemoveIdentities {}));
        assert_eq!(make_dag_pass().io_types().unwrap(), (dag_type, dag_type));

        let circ_pass = Task::Transformation(Box::new(CountT {}));
        assert_eq!(circ_pass.io_types().unwrap(), (circ_type, circ_type));

        let infinity = Task::Loop {
            condition: |_, _| true,
            body: Box::new(make_dag_pass()),
        };
        assert_eq!(infinity.io_types().unwrap(), (dag_type, dag_type));

        let switch = Task::Switch {
            switch: |_, _| 0,
            cases: vec![make_dag_pass()],
        };
        assert_eq!(switch.io_types().unwrap(), (dag_type, dag_type));

        let stages = Task::Stages(vec![("one_and_only".to_string(), make_dag_pass())]);
        assert_eq!(stages.io_types().unwrap(), (dag_type, dag_type));

        let nested = Task::Stages(vec![
            ("pass".to_string(), make_dag_pass()),
            ("loop".to_string(), infinity),
            ("switch".to_string(), switch),
            ("stages".to_string(), stages),
        ]);
        assert_eq!(nested.io_types().unwrap(), (dag_type, dag_type));

        Ok(())
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

        let (out, _) = pm.run::<_, DAGCircuit>(dag, None)?;
        let ops = out.count_ops(false).unwrap();
        assert_eq!(ops.get("h"), Some(&1));
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

    #[test]
    fn test_callback() -> Result<(), PassManagerError> {
        let make_task = || Task::Transformation(Box::new(RemoveIdentities {}));

        let mut pm = PassManager::new();
        pm.try_push_task(make_task())?;
        pm.try_push_task(make_task())?;
        pm.try_push_task(make_task())?;

        pm.try_push_task(Task::Stages(
            (0..3)
                .map(|i| (format!("stage_{}", i).to_string(), make_task()))
                .collect(),
        ))?;

        pm.try_push_task(Task::Group(vec![make_task(), make_task(), make_task()]))?;

        for (hookpoint, expected_count) in [
            (CallbackType::PostPass, 9),
            (CallbackType::PostStage, 3),
            (CallbackType::PostTask, 11),
        ] {
            // The ownership story around the callback is a bit tricky: The callback is immutable (Fn,
            // not FnMut), so it cannot simply modify a local variable. To get around this, we store
            // the data as refcell and pass a clone to the callback (which takes ownership) so we can
            // later check the original refcell for the value
            let callback = CounterCallback::new(hookpoint);
            let (_, _) = pm.run::<_, DAGCircuit>(DAGCircuit::new(), Some(&callback))?;

            assert_eq!(expected_count, callback.counter.get())
        }

        Ok(())
    }

    #[test]
    fn test_insertion() -> Result<(), PassManagerError> {
        let make_dag_task = || Task::Transformation(Box::new(RemoveIdentities {}));
        let make_circ_task = || Task::Transformation(Box::new(CountT {}));

        let mut pm = PassManager::new();
        pm.try_insert_task(0, make_dag_task())?;
        pm.try_insert_task(1, make_dag_task())?;
        pm.try_insert_task(0, make_dag_task())?;

        assert!(matches!(
            pm.try_insert_task(1, make_circ_task()),
            Err(PassManagerError::IncompatibleTypes)
        ));

        Ok(())
    }

    #[test]
    fn test_removal() -> Result<(), PassManagerError> {
        let mut pm = PassManager::new();
        pm.try_push_pass(Box::new(RemoveIdentities {}))?;
        pm.try_push_pass(Box::new(RemoveIdentities {}))?;
        pm.try_push_pass(Box::new(DagToCircuit {}))?;
        pm.try_push_pass(Box::new(CountT {}))?;

        assert!(matches!(
            pm.try_remove_task(2),
            Err(PassManagerError::IncompatibleTypes)
        ));

        pm.try_remove_task(3)?;
        pm.try_remove_task(2)?;
        pm.try_remove_task(0)?;

        assert_eq!(pm.num_tasks(), 1);

        Ok(())
    }

    #[test]
    fn test_task_retrieval() -> Result<(), PassManagerError> {
        let make_task = || Task::Transformation(Box::new(RemoveIdentities {}));

        let group = Task::Group(vec![make_task(), make_task()]);
        let loop_task = Task::Loop {
            condition: |_, _| true,
            body: Box::new(make_task()),
        };
        let switch = Task::Switch {
            switch: |_, _| 0,
            cases: vec![make_task()],
        };
        let stages = Task::Stages(vec![("one_and_only".to_string(), make_task())]);

        let mut pm = PassManager::new();
        pm.try_push_task(make_task())?;
        pm.try_push_task(group)?;
        pm.try_push_task(loop_task)?;
        pm.try_push_task(switch)?;
        pm.try_push_task(stages)?;

        assert!(matches!(pm.get_task(0), Some(Task::Transformation(_))));

        if let Some(Task::Group(group)) = pm.get_task(1) {
            assert_eq!(group.len(), 2);
        } else {
            panic!("Expected a Task::Group");
        }

        assert!(matches!(pm.get_task(2), Some(Task::Loop { .. })));
        assert!(matches!(pm.get_task(3), Some(Task::Switch { .. })));

        if let Some(Task::Stages(stages)) = pm.get_task(4) {
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].0, "one_and_only".to_string());
        } else {
            panic!("Expected a Task::Stage");
        }

        Ok(())
    }

    #[test]
    fn test_pass_context() -> anyhow::Result<()> {
        let qubits: Vec<ShareableQubit> = (0..3).map(|_| ShareableQubit::new_anonymous()).collect();
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.))?;
        let num_t = 50;
        for i in 0..num_t {
            circuit.push_standard_gate(StandardGate::T, &[], &[Qubit(i % 3)])?;
            circuit.push_standard_gate(StandardGate::H, &[], &[Qubit(i % 3)])?;
        }

        let mut pm = PassManager::new();
        pm.try_push_pass(Box::new(CountT {}))?;
        pm.try_push_pass(Box::new(CheckTCount {
            expected_t_count: num_t as usize,
        }))?;

        let (_, context) = pm.run::<_, CircuitData>(circuit, None)?;
        let t_count = context
            .data
            .get("t_count")
            .expect("Failed to retrieve `t_count`")
            .downcast_ref::<usize>()
            .expect("Downcasting failed");
        assert_eq!(*t_count, num_t as usize);

        Ok(())
    }
}
