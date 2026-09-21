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
    any::{self, Any},
    fmt::Debug,
    hash, marker,
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
    pub fn set(&mut self, key: String, value: Box<dyn any::Any>) {
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
        self.updates.get(key).or_else(|| {
            self.global_context
                .data
                .get(key)
                .map(|value| value.as_ref())
        })
    }
}

/// An identifier for a type that may have additional runtime-defined components in it, such as a
/// dynamic trait implementer that comes from C or Python.
#[derive(Copy, Clone, Debug)]
pub struct DynTypeId<'a> {
    static_id: any::TypeId,
    /// The dynamic components of the type information.
    ///
    /// The payload and name are logically tied to some object that creates them.  The pointer, if
    /// used, is valid for the same lifetime as `'a`.
    dynamic: Option<(*mut (), &'a str)>,
}
impl DynTypeId<'_> {
    /// Produce a representation of a [`DynTypeId`] for a type whose implementation is completely
    /// known at Rust compile time.
    ///
    /// Use [`Self::with_dynamic`] to add dynamic context afterwards.
    pub fn of<T: 'static>() -> Self {
        Self {
            static_id: any::TypeId::of::<T>(),
            dynamic: None,
        }
    }

    /// A key object that subsets the fields to define equality and hashing.
    #[inline]
    fn compare_key(&self) -> impl Eq + hash::Hash {
        (self.static_id, self.dynamic.map(|(addr, _)| addr))
    }
}
impl<'a> DynTypeId<'a> {
    /// Set the dynamic components of the type identifier.
    ///
    /// The combination of the Rust type `T` and the address of `payload` is what uniquely defines
    /// the "dynamic type".  If you are using this object to represent a pure type from (say)
    /// Python, you might want to use the pointer to the object's Python `type`.  If you are
    /// representing a dynamic implementation of some trait coming in from C, you probably want to
    /// use a pointer to the vtable of the trait methods.
    ///
    /// Note that the `name` is purely for human inspectability and plays no part in hashing or
    /// comparisons.
    pub fn with_dynamic(self, payload: *mut (), name: &'a str) -> Self {
        Self {
            dynamic: Some((payload, name)),
            ..self
        }
    }
}
impl PartialEq for DynTypeId<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.compare_key() == other.compare_key()
    }
}
impl Eq for DynTypeId<'_> {}
impl hash::Hash for DynTypeId<'_> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.compare_key().hash(state)
    }
}

/// The base behavior for compiler passes written in first-party Rust code.
///
/// This is the component that pass authors actually need to implement.
pub trait StaticPass<In, Out = In>: Send + Sync + Sized + 'static
where
    In: Send + Sync + 'static,
    Out: Send + Sync + 'static,
{
    /// Run the pass.
    fn run(&self, ir: Box<In>, context: &mut PassContext) -> anyhow::Result<Box<Out>>;

    /// Turn this object into the full type-erased version.
    ///
    /// If the base structure implements [`StaticPass`] for more than one pair of input and output
    /// types, you might need to call this as something like
    /// ```ignore
    /// <MyImplementer as StaticPass<In, Out>>::into_pass(ob)
    /// ```
    fn into_pass(self) -> Box<dyn Pass> {
        Box::new(StaticPassOb {
            ob: self,
            phantom: marker::PhantomData,
        })
    }
}

/// Type-system wrapper object to move the `In` and `Out` IR types of `StaticPass` into a concrete
/// object.
///
/// Storing the "associated types" as separate phantom markers in this object lets us have base Rust
/// structs that implement [`StaticPass`] for more than one input/output pair.  We still need to
/// have a fully monomorphised object to hold the type parameters when we do `impl Pass for
/// SomeObject`, because otherwise that implementation would overlap for _all_ the `StaticPass`
/// implementations of the base object.
struct StaticPassOb<T, In, Out = In> {
    ob: T,
    phantom: marker::PhantomData<(In, Out)>,
}
impl<P, In, Out> Pass for StaticPassOb<P, In, Out>
where
    In: Send + Sync + 'static,
    Out: Send + Sync + 'static,
    P: StaticPass<In, Out>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn ir_id_in(&self) -> DynTypeId<'_> {
        DynTypeId::of::<In>()
    }
    fn ir_id_out(&self) -> DynTypeId<'_> {
        DynTypeId::of::<Out>()
    }
    fn name(&self) -> &str {
        any::type_name::<P>()
    }
    fn run(&self, ir: Box<dyn Any>, context: &mut PassContext) -> anyhow::Result<Box<dyn Any>> {
        let ir = ir
            .downcast::<In>()
            .map_err(|_| PassManagerError::IncompatibleTypes)?;
        self.ob.run(ir, context).map(|out| out as Box<dyn Any>)
    }
}

/// A type-erased version of the [Pass] trait. This is required to store passes with different
/// associated types in the generic [Task::Transformation] variant.
pub trait Pass: Send + Sync {
    /// Cast the pass to Any to allow downcasting to a target type.
    fn as_any(&self) -> &dyn Any;
    /// Return the type ID of the input IR.
    fn ir_id_in(&self) -> DynTypeId<'_>;
    /// Return the type ID of the output IR.
    fn ir_id_out(&self) -> DynTypeId<'_>;
    fn name(&self) -> &str;
    /// Run the pass.
    fn run(&self, ir: Box<dyn Any>, context: &mut PassContext) -> anyhow::Result<Box<dyn Any>>;
}

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. It describes how work is being executed, ranging
/// from the simple execution of a single pass, over groups of passes to structured flow control,
/// such as loops. The [PassManager] stores a vector of [Task]s and executes them.
#[non_exhaustive]
pub enum Task {
    /// A single pass.
    Transformation(Box<dyn Pass>),

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
            Task::Transformation(p) => f.debug_tuple("Transformation").field(&p.name()).finish(),
            Task::Group(tasks) => f.debug_tuple("Group").field(tasks).finish(),
            Task::Stages(stages) => f.debug_tuple("Stages").field(stages).finish(),
            Task::Switch { switch, cases } => f
                .debug_struct("Switch")
                .field("switch", switch)
                .field("cases", cases)
                .finish(),
            Task::Loop { condition, body } => f
                .debug_struct("Loop")
                .field("condition", condition)
                .field("body", body)
                .finish(),
        }
    }
}

impl Task {
    fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
        match self {
            Task::Transformation(pass) => Some([pass.ir_id_in(), pass.ir_id_out()]),
            Task::Group(group) => {
                Some([group.first()?.io_types()?[0], group.last()?.io_types()?[1]])
            }
            Task::Loop { condition: _, body } => (*body).io_types(),
            Task::Switch { switch: _, cases } => cases.first()?.io_types(),
            Task::Stages(stages) => Some([
                stages.first()?.1.io_types()?[0],
                stages.last()?.1.io_types()?[1],
            ]),
        }
    }
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
    ///
    /// This is a typed helper wrapper around [`Self::run_erased`].
    pub fn run<IRIn, IROut>(
        &self,
        ir: IRIn,
    ) -> Result<(IROut, PassManagerContext), PassManagerError>
    where
        IRIn: 'static,
        IROut: 'static,
    {
        let (ir, context) = self.run_erased(Box::new(ir))?;
        ir.downcast::<IROut>()
            .map(|ir| (*ir, context))
            .map_err(|_| PassManagerError::FailedOutputConversion)
    }

    pub fn run_erased(
        &self,
        mut ir: Box<dyn Any>,
    ) -> Result<(Box<dyn Any>, PassManagerContext), PassManagerError> {
        let mut context = PassManagerContext::new();
        for task in self.tasks.iter() {
            let mut pass_context = PassContext::spawn(&context);
            ir = execute_task(task, ir, &mut pass_context)?;
            let updates = pass_context.into_updates();
            context.update(updates);
        }
        Ok((ir, context))
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
            let Some([_, out_type]) = last_task.io_types() else {
                return Err(PassManagerError::EmptyTask);
            };
            let Some([in_type, _]) = task.io_types() else {
                return Err(PassManagerError::EmptyTask);
            };
            if in_type != out_type {
                return Err(PassManagerError::IncompatibleTypes);
            }
        }
        self.tasks.push(task);
        Ok(())
    }

    pub fn try_push_static_pass<In, Out>(
        &mut self,
        ob: impl StaticPass<In, Out>,
    ) -> Result<(), PassManagerError>
    where
        In: Send + Sync + 'static,
        Out: Send + Sync + 'static,
    {
        self.try_push_task(Task::Transformation(ob.into_pass()))
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
) -> Result<Box<dyn Any>, PassManagerError> {
    match task {
        Task::Transformation(pass) => pass.run(ir, context).map_err(PassManagerError::PassError),
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
            while condition(&ir, context) {
                ir = execute_task(body, ir, context)?;
            }
            Ok(ir)
        }
        Task::Stages(stages) => {
            for (_name, task) in stages.iter() {
                ir = execute_task(task, ir, context)?;
            }
            Ok(ir)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use qiskit_circuit::{
        Qubit,
        circuit_data::CircuitData,
        dag_circuit::DAGCircuit,
        operations::{Param, StandardGate},
    };
    use qiskit_transpiler::passes::run_remove_identity_equiv;

    #[derive(Clone, Debug)]
    struct RemoveIdentities;
    impl StaticPass<DAGCircuit> for RemoveIdentities {
        fn run(
            &self,
            mut ir: Box<DAGCircuit>,
            _context: &mut PassContext,
        ) -> anyhow::Result<Box<DAGCircuit>> {
            run_remove_identity_equiv(&mut ir, None, None)?;
            Ok(ir)
        }
    }

    #[derive(Clone, Debug)]
    struct CountT;
    impl StaticPass<CircuitData> for CountT {
        fn run(
            &self,
            ir: Box<CircuitData>,
            context: &mut PassContext,
        ) -> anyhow::Result<Box<CircuitData>> {
            let count = ir.count_ops();
            let t_count: usize = count.get("t").unwrap_or(&0) + count.get("tdg").unwrap_or(&0);
            context.set("t_count".to_string(), Box::new(t_count));
            Ok(ir)
        }
    }

    struct CheckTCount {
        expected_t_count: usize,
    }
    impl StaticPass<CircuitData> for CheckTCount {
        fn run(
            &self,
            ir: Box<CircuitData>,
            context: &mut PassContext,
        ) -> anyhow::Result<Box<CircuitData>> {
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

    #[test]
    fn test_io_types() -> Result<(), PassManagerError> {
        let dag_type = DynTypeId::of::<DAGCircuit>();
        let circ_type = DynTypeId::of::<CircuitData>();

        let make_dag_pass = || Task::Transformation(RemoveIdentities.into_pass());
        assert_eq!(make_dag_pass().io_types().unwrap(), [dag_type, dag_type]);

        let circ_pass = Task::Transformation(CountT.into_pass());
        assert_eq!(circ_pass.io_types().unwrap(), [circ_type, circ_type]);

        let infinity = Task::Loop {
            condition: |_, _| true,
            body: Box::new(make_dag_pass()),
        };
        assert_eq!(infinity.io_types().unwrap(), [dag_type, dag_type]);

        let switch = Task::Switch {
            switch: |_, _| 0,
            cases: vec![make_dag_pass()],
        };
        assert_eq!(switch.io_types().unwrap(), [dag_type, dag_type]);

        let stages = Task::Stages(vec![("one_and_only".to_string(), make_dag_pass())]);
        assert_eq!(stages.io_types().unwrap(), [dag_type, dag_type]);

        let nested = Task::Stages(vec![
            ("pass".to_string(), make_dag_pass()),
            ("loop".to_string(), infinity),
            ("switch".to_string(), switch),
            ("stages".to_string(), stages),
        ]);
        assert_eq!(nested.io_types().unwrap(), [dag_type, dag_type]);

        Ok(())
    }

    #[test]
    fn test_pass() -> Result<(), PassManagerError> {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(RemoveIdentities)?;

        let mut qc = CircuitData::with_capacity(1, 0, 2, Param::Float(0.0)).unwrap();
        qc.push_standard_gate(StandardGate::H, &[], &[Qubit(0)])
            .unwrap();
        qc.push_standard_gate(StandardGate::RX, &[Param::Float(0.0)], &[Qubit(0)])
            .unwrap();
        let dag = DAGCircuit::from_circuit_data(&qc, false, None, None, None, None).unwrap();

        let (out, _) = pm.run::<_, DAGCircuit>(dag)?;
        let ops = out.count_ops(false).unwrap();
        assert_eq!(ops.get("h"), Some(&1));
        assert_eq!(ops.get("rx"), None);
        Ok(())
    }

    #[test]
    fn test_incompatible_types() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(RemoveIdentities).unwrap();
        let result = pm.try_push_static_pass(CountT);
        assert!(matches!(result, Err(PassManagerError::IncompatibleTypes)));
    }

    #[test]
    fn test_task_retrieval() -> Result<(), PassManagerError> {
        let make_task = || Task::Transformation(RemoveIdentities.into_pass());

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
        pm.try_push_static_pass(RemoveIdentities)?;
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
        let num_t = 50;
        let mut circuit = CircuitData::with_capacity(3, 0, num_t, Param::Float(0.))?;
        for i in 0..num_t as u32 {
            circuit.push_standard_gate(StandardGate::T, &[], &[Qubit(i % 3)])?;
            circuit.push_standard_gate(StandardGate::H, &[], &[Qubit(i % 3)])?;
        }

        let mut pm = PassManager::new();
        pm.try_push_static_pass(CountT)?;
        pm.try_push_static_pass(CheckTCount {
            expected_t_count: num_t,
        })?;

        let (_, context) = pm.run::<_, CircuitData>(circuit)?;
        let t_count = context
            .data
            .get("t_count")
            .expect("Failed to retrieve `t_count`")
            .downcast_ref::<usize>()
            .expect("Downcasting failed");
        assert_eq!(*t_count, num_t);

        Ok(())
    }
}
