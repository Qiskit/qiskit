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

use anyhow::Context;
use hashbrown::{HashMap, HashSet};
use std::{
    any::{self, Any},
    borrow, fmt, hash,
};

pub use pass::*;

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

    pub fn get(&self, key: impl AsRef<str>) -> Option<&dyn Any> {
        self.data.get(key.as_ref()).map(Box::as_ref)
    }
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

/// An identifier for a type that may have additional runtime-defined components in it, such as a
/// dynamic trait implementer that comes from C or Python.
#[derive(Copy, Clone, Debug)]
pub struct DynTypeId<'a> {
    static_id: any::TypeId,
    static_name: &'static str,
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
            static_name: any::type_name::<T>(),
            dynamic: None,
        }
    }

    /// A key object that subsets the fields to define equality and hashing.
    #[inline]
    fn compare_key(&self) -> impl Eq + hash::Hash {
        (self.static_id, self.dynamic.map(|(addr, _)| addr))
    }

    /// Describe this type.
    pub fn describe(&self) -> borrow::Cow<'_, str> {
        match self.dynamic {
            Some((_addr, dyn_name)) => {
                borrow::Cow::Owned(format!("{}[{}]", self.static_name, dyn_name))
            }
            None => borrow::Cow::Borrowed(self.static_name),
        }
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

/// Trait for types that interact with the Qiskit dynamic-typing system ([`DynTyped`]) as static
/// Rust objects.
///
/// This trait is not object safe; use the blanket implementation of [`DynTyped`] for that.
pub trait StaticDynTyped {
    fn static_dyn_type_id() -> DynTypeId<'static>;
}
/// Declare a static Rust type as directly usable with the Qiskit dynamic-typing system.
#[macro_export]
macro_rules! static_dyn_typed {
    ($ty:ty) => {
        impl $crate::StaticDynTyped for $ty {
            fn static_dyn_type_id() -> $crate::DynTypeId<'static> {
                $crate::DynTypeId::of::<$ty>()
            }
        }
    };
}
/// Objects that can interact with Qiskit's dynamic-typing subsystem.
///
/// There are two components to the system: the static Rust type that backs the object, and any
/// additional dynamic typing on top of that.
///
/// First-class objects defined in Rust can implement [`StaticDynTyped`] and use the blanket
/// implementation that provides this object-safe variant.
pub trait DynTyped: Any {
    /// The dynamic type identifier.
    fn dyn_type_id(&self) -> DynTypeId<'_>;
}
impl<T: StaticDynTyped + 'static> DynTyped for T {
    fn dyn_type_id(&self) -> DynTypeId<'_> {
        T::static_dyn_type_id()
    }
}
/// Types that can be used as an IR by the [`PassManager`].
pub trait IR: DynTyped + Send + Sync + 'static {}

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. It describes how work is being executed, ranging
/// from the simple execution of a single pass, over groups of passes to structured flow control,
/// such as loops. The [PassManager] stores a vector of [Task]s and executes them.
#[non_exhaustive]
pub enum Task {
    // TODO Add Loop and Switch with conditions that can be set from Python/C and
    // proper error handlings that occur during the condition evaluation.
    /// A single pass.
    Transformation(Box<dyn Pass>),

    /// A group of tasks.
    Group(Vec<Task>),

    /// A sequence of named tasks.
    Stages(Vec<(String, Task)>),
}

impl fmt::Debug for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Task::Transformation(p) => f.debug_tuple("Transformation").field(&p.name()).finish(),
            Task::Group(tasks) => f.debug_tuple("Group").field(tasks).finish(),
            Task::Stages(stages) => f.debug_tuple("Stages").field(stages).finish(),
        }
    }
}

impl Task {
    fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
        // TODO: the implementation of `Task` as a `pub enum` means that nothing enforces the
        // pipeline (dynamic) type safety of `Group`, `Switch` or `Stages`; these need to be
        // enforced during construction of the `Task`.  This might motivate a swap to a structure
        // like
        //
        //      enum TaskInner {
        //          Pass(Box<dyn Pass>),
        //          Group(Vec<Task>),
        //          // ...
        //      }
        //      pub struct Task(TaskInner);
        //      impl Task {
        //          pub fn group(vals: Vec<Task>) -> Result<Self, Vec<Task>) {}
        //          // ...
        //      }
        match self {
            Task::Transformation(pass) => Some([pass.ir_id_in(), pass.ir_id_out()]),
            Task::Group(group) => {
                Some([group.first()?.io_types()?[0], group.last()?.io_types()?[1]])
            }
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
    ) -> anyhow::Result<(Box<dyn IR>, PassManagerContext)> {
        let mut context = PassManagerContext::new();
        for task in self.tasks.iter() {
            let mut pass_context = PassContext::spawn(&context);
            ir = execute_task(task, ir, &mut pass_context)?;
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
        // This might not just be `last` if the last task is an empty group or stage.
        self.tasks
            .iter()
            .find_map(|t| t.io_types().map(|[in_, _]| in_))
    }
    /// Get the type identifier of the output of this pipeline.
    pub fn ir_id_out(&self) -> Option<DynTypeId<'_>> {
        // This might not just be `last` if the last task is an empty group or stage.
        self.tasks
            .iter()
            .rev()
            .find_map(|t| t.io_types().map(|[_, out]| out))
    }

    /// Try to push a [`Task`] onto the end of the task list.
    ///
    /// Fails, returning the same task back to the caller, if the types are incompatible.
    pub fn try_push_task(&mut self, task: Task) -> Result<(), Task> {
        let ours = self.ir_id_out();
        let theirs = task.io_types().map(|[in_, _]| in_);
        if let Some((ours, theirs)) = ours.zip(theirs)
            && ours != theirs
        {
            // We may want to change the error type of this in the future to provide structured
            // information about _what_ went wrong, but in the first implementation we're just doing
            // the easy thing.
            return Err(task);
        }
        self.tasks.push(task);
        Ok(())
    }

    pub fn try_push_static_pass<In: IR, Out: IR>(
        &mut self,
        ob: impl StaticPass<In, Out>,
    ) -> Result<(), Task> {
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
    mut ir: Box<dyn IR>,
    context: &mut PassContext,
) -> Result<Box<dyn IR>, PassError> {
    match task {
        Task::Transformation(pass) => pass.run(ir, context),
        Task::Group(tasks) => {
            for task in tasks.iter() {
                ir = execute_task(task, ir, context)?;
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

    #[test]
    fn test_io_types() -> Result<(), PassError> {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        let make_uint_pass = || Task::Transformation(AddOne.into_pass());
        assert_eq!(make_uint_pass().io_types().unwrap(), [uint_ty, uint_ty]);

        let lower_pass = Task::Transformation(LowerToInt.into_pass());
        assert_eq!(lower_pass.io_types().unwrap(), [uint_ty, int_ty]);

        let stages = Task::Stages(vec![("one_and_only".to_string(), make_uint_pass())]);
        assert_eq!(stages.io_types().unwrap(), [uint_ty, uint_ty]);

        let nested = Task::Stages(vec![
            ("pass".to_string(), make_uint_pass()),
            ("stages".to_string(), stages),
        ]);
        assert_eq!(nested.io_types().unwrap(), [uint_ty, uint_ty]);

        Ok(())
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
        assert!(pm.try_push_static_pass(AddOne).is_err());
    }

    #[test]
    fn test_task_retrieval() {
        let make_task = || Task::Transformation(AddOne.into_pass());

        let group = Task::Group(vec![make_task(), make_task()]);
        let stages = Task::Stages(vec![("one_and_only".to_string(), make_task())]);

        let mut pm = PassManager::new();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_task(group).unwrap();
        pm.try_push_task(stages).unwrap();

        assert!(matches!(pm.get_task(0), Some(Task::Transformation(_))));

        if let Some(Task::Group(group)) = pm.get_task(1) {
            assert_eq!(group.len(), 2);
        } else {
            panic!("Expected a Task::Group");
        }

        if let Some(Task::Stages(stages)) = pm.get_task(2) {
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
