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

mod pass;
mod task;

use anyhow::Context;
use hashbrown::{HashMap, HashSet};
use qiskit_util::dyn_types::*;
use std::{any::Any, sync::LazyLock};

pub use pass::*;
pub use task::*;

/// The pass manager execution environment.
///
/// This contains data managed by the pass manager. A local handle to this is passed into
/// the passes.
#[derive(Default, Debug)]
pub struct PassManagerContext {
    /// The global, catch-all data. The local [PassContext] handles get read-only access to
    /// this data and after pass execution this global state is updated.
    data: HashMap<String, Box<dyn Any + Send + Sync>>,
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

    pub fn get(&self, key: impl AsRef<str>) -> Option<&(dyn Any + Send + Sync)> {
        self.data.get(key.as_ref()).map(Box::as_ref)
    }
}

/// A private struct representing the context updates performed. As long as this contains only
/// a HashMap, we could skip this object, but the context is supposed to contain more generic
/// information.
#[derive(Default, Debug)]
struct ContextUpdates {
    /// New values to insert into the global context.
    insertions: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Keys to delete from the global context.
    deletions: HashSet<String>,
}
impl ContextUpdates {
    fn insert(
        &mut self,
        key: String,
        value: Box<dyn Any + Send + Sync>,
    ) -> Option<Box<dyn Any + Send + Sync>> {
        self.deletions.remove(&key);
        self.insertions.insert(key, value)
    }

    fn delete(&mut self, key: String) -> Option<Box<dyn Any + Send + Sync>> {
        let out = self.insertions.remove(&key);
        self.deletions.insert(key);
        out
    }

    fn get(&self, key: impl AsRef<str>) -> Option<&(dyn Any + Send + Sync)> {
        self.insertions.get(key.as_ref()).map(|v| v.as_ref())
    }
}

/// Mutable context information for the pass to interact with the execution pipeline.
///
/// This object is used for two-way information transfer; passes can set flags like `ir_modified` to
/// pass information back to the executing pipeline, or can get/set items in the pipeline's context
/// to pass information between passes.
#[derive(Debug)]
pub struct PassContext<'a> {
    /// A reference to the global execution environment.
    global_context: &'a PassManagerContext,

    /// Whether the pass changed the IR or not.  This defaults to `true`, but passes may set it to
    /// `false` to indicate that they didn't modify the IR.  The pass-manager execution environment
    /// can use that information to optimize caching.
    pub ir_modified: bool,

    /// A local cache of new data.
    updates: ContextUpdates,
}

impl PassContext<'static> {
    /// Get a dummy version of ourselves for use as a default value in situations where we don't
    /// need it to be linked to anything.
    pub fn dummy() -> Self {
        static GLOBAL: LazyLock<PassManagerContext> = LazyLock::new(PassManagerContext::default);
        Self {
            global_context: &GLOBAL,
            ir_modified: true,
            updates: ContextUpdates::default(),
        }
    }
}
impl<'a> PassContext<'a> {
    fn spawn(global_context: &'a PassManagerContext) -> Self {
        Self {
            global_context,
            ir_modified: true,
            updates: ContextUpdates::default(),
        }
    }

    /// Set a new entry in the pass context.
    /// Overwrites the existing value under that key, if it exists.
    ///
    /// Returns the previous local entry, if it existed.
    pub fn set(
        &mut self,
        key: String,
        value: Box<dyn Any + Send + Sync>,
    ) -> Option<Box<dyn Any + Send + Sync>> {
        self.updates.insert(key, value)
    }

    /// Delete the given key.  Returns the corresponding local value, if any.
    pub fn delete(&mut self, key: String) -> Option<Box<dyn Any + Send + Sync>> {
        self.updates.delete(key)
    }

    /// Get an entry, if it exists.
    ///
    /// This first queries from the local context, then the global.
    pub fn get(&self, key: impl AsRef<str>) -> Option<&(dyn Any + Send + Sync)> {
        let key = key.as_ref();

        if self.updates.deletions.contains(key) {
            return None;
        }

        // The local registry takes precedence.
        self.updates.get(key).or_else(|| {
            self.global_context
                .data
                .get(key)
                .map(|value| value.as_ref())
        })
    }
}
/// Types that can be used as an IR by the [`PassManager`].
pub trait IR: DynTyped + Send + Sync + 'static {}

/// Qiskit's pass manager.
#[derive(Default, Debug)]
pub struct PassManager {
    // It is UNSAFE to directly mutate the task vector since we are checking that the types
    // match upon construction, hence the tasks are private.
    tasks: Vec<Task>,
}

impl PassManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the pass manager on the input IR, and attempt to convert it to a specific output type.
    ///
    /// This is a typed helper wrapper around [`Self::run_erased`].
    pub fn run<IRIn, IROut>(&self, ir: IRIn) -> anyhow::Result<(IROut, PassManagerContext)>
    where
        IRIn: IR,
        IROut: IR,
    {
        let (ir, context) = self.run_erased(Box::new(ir))?;
        (ir as Box<dyn Any>)
            .downcast::<IROut>()
            .map(|ir| (*ir, context))
            .map_err(|_| PassError::Conversion)
            .with_context(|| {
                format!(
                    "trying to convert output to {}",
                    DynTypeId::of::<IROut>().describe()
                )
            })
    }

    /// Run the pass manager
    pub fn run_erased(
        &self,
        mut ir: Box<dyn IR>,
    ) -> Result<(Box<dyn IR>, PassManagerContext), PassError> {
        let mut context = PassManagerContext::new();
        for task in self.tasks.iter() {
            let mut pass_context = PassContext::spawn(&context);
            ir = task.execute(ir, &mut pass_context)?;
            let PassContext { updates, .. } = pass_context;
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

    /// Get the type identifier of the input of this pipeline.
    pub fn ir_id_in(&self) -> Option<DynTypeId<'_>> {
        sequence_io_types(self.tasks.iter()).map(|[in_, _]| in_)
    }
    /// Get the type identifier of the output of this pipeline.
    pub fn ir_id_out(&self) -> Option<DynTypeId<'_>> {
        sequence_io_types(self.tasks.iter()).map(|[_, out]| out)
    }

    /// Try to push a [`Task`] onto the end of the task list.
    ///
    /// Fails, returning the same task back to the caller, if the types are incompatible.
    pub fn try_push_task(&mut self, task: Task) -> Result<(), TypeMismatch<Task>> {
        let ours = self.ir_id_out();
        let theirs = task.io_types().map(|[in_, _]| in_);
        if let Some((ours, theirs)) = ours.zip(theirs)
            && ours != theirs
        {
            let error = TypeMismatchError::new(MismatchPosition::Append, ours, theirs);
            return Err(error.reject(task));
        }
        self.tasks.push(task);
        Ok(())
    }

    pub fn try_push_static_pass<In: IR, Out: IR>(
        &mut self,
        ob: impl StaticPass<In, Out>,
    ) -> Result<(), TypeMismatch<Task>> {
        self.try_push_task(Task::transformation(ob.into_pass()))
    }

    /// Get a reference to a [Task] at a given index.
    pub fn get_task(&self, index: usize) -> Option<&Task> {
        self.tasks.get(index)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use anyhow::anyhow;
    use std::sync::{Arc, RwLock};

    #[derive(Clone, Debug)]
    struct MyUint(u32);
    static_dyn_typed!(MyUint);
    impl IR for MyUint {}

    #[derive(Debug)]
    struct MyInt(i32);
    static_dyn_typed!(MyInt);
    impl IR for MyInt {}

    struct AddOne;
    impl StaticPass<MyUint> for AddOne {
        fn run(
            &self,
            mut ir: Box<MyUint>,
            _context: &mut PassContext,
        ) -> anyhow::Result<Box<MyUint>> {
            ir.0 += 1;
            Ok(ir)
        }
    }

    struct WriteToContext;
    impl StaticPass<MyUint> for WriteToContext {
        fn run(&self, ir: Box<MyUint>, context: &mut PassContext) -> anyhow::Result<Box<MyUint>> {
            context.set("snapshot".into(), ir.clone());
            context.ir_modified = false;
            Ok(ir)
        }
    }

    struct LeakFromContext<T>(Arc<RwLock<T>>);
    impl<T: Clone + IR + 'static> StaticPass<T> for LeakFromContext<T> {
        fn run(&self, ir: Box<T>, context: &mut PassContext) -> anyhow::Result<Box<T>> {
            let ob = context
                .get("snapshot")
                .ok_or_else(|| anyhow!("object absent"))?;
            let ob = ob
                .downcast_ref::<T>()
                .ok_or_else(|| anyhow!("failed downcast"))?
                .clone();
            *self.0.write().expect("lock shouldn't be poisoned") = ob;
            Ok(ir)
        }
    }

    struct DeleteFromContext;
    impl StaticPass<MyUint> for DeleteFromContext {
        fn run(&self, ir: Box<MyUint>, context: &mut PassContext) -> anyhow::Result<Box<MyUint>> {
            context.delete("snapshot".into());
            context.ir_modified = false;
            Ok(ir)
        }
    }

    struct LowerToInt;
    impl StaticPass<MyUint, MyInt> for LowerToInt {
        fn run(&self, ir: Box<MyUint>, _context: &mut PassContext) -> anyhow::Result<Box<MyInt>> {
            let ir = MyInt(ir.0.try_into().map_err(|_| anyhow!("too big!"))?);
            Ok(Box::new(ir))
        }
    }

    #[track_caller]
    fn assert_disagreement(error: &TypeMismatchError, first: &str, second: &str) {
        assert!(error.first.contains(first), "{error}");
        assert!(error.second.contains(second), "{error}");
    }

    #[test]
    fn test_empty_child_keeps_type_checks() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(LowerToInt).unwrap();
        let group = Task::group(vec![
            Task::group(vec![]).unwrap(),
            Task::transformation(AddOne.into_pass()),
        ])
        .unwrap();
        assert!(pm.try_push_task(group).is_err());
    }

    #[test]
    fn test_pass() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(AddOne).unwrap();

        let (out, _) = pm.run::<MyUint, MyUint>(MyUint(4)).unwrap();
        assert_eq!(out.0, 8);
    }

    #[test]
    fn test_incompatible_types() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(LowerToInt).unwrap();
        let err = pm.try_push_static_pass(AddOne).unwrap_err();
        assert_disagreement(&err.error, "MyInt", "MyUint");
    }

    #[test]
    fn test_task_retrieval() {
        let make_task = || Task::transformation(AddOne.into_pass());

        let group = Task::group(vec![make_task(), make_task()]).unwrap();
        let stages = Task::stages(vec![("one_and_only".to_string(), make_task())]).unwrap();

        let mut pm = PassManager::new();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_task(group).unwrap();
        pm.try_push_task(stages).unwrap();

        assert!(matches!(
            pm.get_task(0),
            Some(Task(TaskInner::Transformation(_)))
        ));

        if let Some(Task(TaskInner::Group(group))) = pm.get_task(1) {
            assert_eq!(group.len(), 2);
        } else {
            panic!("Expected a Task::Group");
        }

        if let Some(Task(TaskInner::Stages(stages))) = pm.get_task(2) {
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].0, "one_and_only".to_string());
        } else {
            panic!("Expected a Task::Stage");
        }
    }

    #[test]
    fn test_pass_context() {
        let cell = Arc::new(RwLock::new(MyUint(0)));

        let mut pm = PassManager::new();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(WriteToContext).unwrap();
        pm.try_push_static_pass(LeakFromContext(Arc::clone(&cell)))
            .unwrap();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(WriteToContext).unwrap();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(LowerToInt).unwrap();

        let (MyInt(from_pm), context) = pm.run(MyUint(4)).unwrap();
        assert_eq!(cell.read().unwrap().0, 5);
        let MyUint(from_context) = *context.data["snapshot"].downcast_ref().unwrap();
        assert_eq!(from_context, 6);
        assert_eq!(from_pm, 7);
    }

    #[test]
    fn test_delete_context() {
        let cell = Arc::new(RwLock::new(MyUint(0)));

        let mut pm = PassManager::new();
        pm.try_push_static_pass(WriteToContext).unwrap();
        pm.try_push_static_pass(DeleteFromContext).unwrap();
        pm.try_push_static_pass(LeakFromContext(Arc::clone(&cell)))
            .unwrap();

        let e = pm.run::<_, MyUint>(MyUint(1)).unwrap_err();
        assert!(e.to_string().contains("object absent"), "{:?}", e);
    }

    #[test]
    fn test_pass_error() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(LowerToInt).unwrap();
        let e = pm.run::<MyUint, MyInt>(MyUint(u32::MAX)).unwrap_err();
        assert!(e.to_string().contains("too big!"), "{:?}", e);
    }

    #[test]
    fn test_pass_name() {
        let add = AddOne.into_pass();
        let lower = LowerToInt.into_pass();
        assert!(add.name().contains("AddOne"));
        assert!(lower.name().contains("LowerToInt"));
    }
}
