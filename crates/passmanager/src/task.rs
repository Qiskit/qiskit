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

use crate::pass::{Pass, PassError};
use crate::{IR, PassContext};
use qiskit_util::dyn_types::DynTypeId;

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. It describes how work is being executed, ranging
/// from the simple execution of a single pass, over pipelines of passes to structured flow control,
/// such as loops. The [`PassManager`](crate::PassManager) stores a [Pipeline] of [Task]s and executes
/// them.
#[derive(Debug)]
#[non_exhaustive]
pub enum Task {
    // TODO Add Loop and Switch with conditions that can be set from Python/C and
    // proper error handlings that occur during the condition evaluation.
    /// A single pass.
    Transformation(Box<dyn Pass>),
    /// A pipeline of tasks.
    Pipeline(Pipeline),
    /// A pipeline of named tasks.
    Stages(StagedPipeline),
}

impl Task {
    pub(crate) fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
        match self {
            Task::Transformation(pass) => Some([pass.ir_id_in(), pass.ir_id_out()]),
            Task::Pipeline(pipeline) => pipeline.io_types(),
            Task::Stages(stages) => stages.io_types(),
        }
    }

    /// Execute this task.
    ///
    /// This should not be called standalone, passes should be run via the pass manager.
    pub(crate) fn execute(
        &self,
        mut ir: Box<dyn IR>,
        context: &mut PassContext,
    ) -> Result<Box<dyn IR>, PassError> {
        // TODO We might be able to only type-check in the pass manager's execution method,
        // since the pipeline is already type-checked upon construction. For now we keep it here,
        // which is the safer choice.
        if let Some([task_in_type, _]) = self.io_types()
            && task_in_type != ir.dyn_type_id()
        {
            return Err(PassError::Conversion);
        }

        match self {
            Task::Transformation(pass) => pass.run(ir, context),
            Task::Pipeline(pipeline) => {
                for task in pipeline.iter() {
                    ir = task.execute(ir, context)?;
                }
                Ok(ir)
            }
            Task::Stages(stages) => {
                for (_name, task) in stages.iter() {
                    ir = task.execute(ir, context)?;
                }
                Ok(ir)
            }
        }
    }
}

/// A sequence of tasks to run in order.
#[derive(Default, Debug)]
pub struct Pipeline(Vec<Task>);

impl Pipeline {
    /// Build a pipeline of the provided `tasks`, fail if any adjacent IR types are mismatched.
    pub fn new(tasks: Vec<Task>) -> Result<Self, Vec<Task>> {
        if has_io_mismatch(tasks.iter()) {
            // We may want to change the error type of this in the future to provide structured
            // information about _what_ went wrong, but in the first implementation we're just doing
            // the easy thing.
            return Err(tasks);
        }
        Ok(Self(tasks))
    }

    /// Return the number of tasks in the pipeline.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get a reference to the [Task] at a given index.
    pub fn get(&self, index: usize) -> Option<&Task> {
        self.0.get(index)
    }

    /// Iterate over the tasks in execution order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Task> {
        self.0.iter()
    }

    /// Try to push a [Task] onto the end of the pipeline, failing if the input IR type is incompatible.
    pub fn try_push(&mut self, task: Task) -> Result<(), Task> {
        let ours = self.io_types().map(|[_, out]| out);
        let theirs = task.io_types().map(|[in_, _]| in_);
        if let Some((ours, theirs)) = ours.zip(theirs)
            && ours != theirs
        {
            return Err(task);
        }
        self.0.push(task);
        Ok(())
    }

    pub(crate) fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
        sequence_io_types(self.iter())
    }
}

/// A sequence of named tasks to be run in order.
#[derive(Default, Debug)]
pub struct StagedPipeline(Vec<(String, Task)>);

impl StagedPipeline {
    /// Build a pipeline of the provided `stages`, fail if any adjacent IR types are mismatched.
    pub fn new(stages: Vec<(String, Task)>) -> Result<Self, Vec<(String, Task)>> {
        if has_io_mismatch(stages.iter().map(|(_name, task)| task)) {
            return Err(stages);
        }
        Ok(Self(stages))
    }

    /// Return the number of stages in the pipeline.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return whether the pipeline contains no stages.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get a reference to the name and [Task] of the stage at a given index.
    pub fn get(&self, index: usize) -> Option<(&str, &Task)> {
        self.0.get(index).map(|(name, task)| (name.as_str(), task))
    }

    /// Iterate over the stages in execution order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&str, &Task)> {
        self.0.iter().map(|(name, task)| (name.as_str(), task))
    }

    pub(crate) fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
        sequence_io_types(self.iter().map(|(_name, task)| task))
    }
}

/// Return the input and output types of a sequence of tasks if non-empty.
fn sequence_io_types<'a>(
    tasks: impl DoubleEndedIterator<Item = &'a Task>,
) -> Option<[DynTypeId<'a>; 2]> {
    let mut typed = tasks.filter_map(|task| task.io_types());
    let first = typed.next()?;
    let last = typed.next_back().unwrap_or(first);
    Some([first[0], last[1]])
}

/// Whether any adjacent pair of tasks in a sequence has disagreeing output and input types.
fn has_io_mismatch<'a>(tasks: impl IntoIterator<Item = &'a Task>) -> bool {
    let mut last_out: Option<DynTypeId<'a>> = None;
    for task in tasks {
        let Some([in_, out]) = task.io_types() else {
            continue;
        };
        if let Some(previous) = last_out
            && previous != in_
        {
            return true;
        }
        last_out = Some(out);
    }
    false
}

#[cfg(test)]
mod test {
    use crate::pass::{PassError, StaticPass};
    use crate::{IR, PassContext, Pipeline, StagedPipeline, Task};
    use anyhow::anyhow;
    use qiskit_util::{dyn_types::DynTypeId, static_dyn_typed};

    #[derive(Clone, Debug)]
    struct MyUint(u32);
    static_dyn_typed!(MyUint);
    impl IR for MyUint {}

    #[derive(Debug)]
    struct MyInt(#[allow(dead_code)] i32);
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

        let one_stage = Task::Stages(
            StagedPipeline::new(vec![("one_and_only".to_string(), make_uint_pass())]).unwrap(),
        );
        assert_eq!(one_stage.io_types().unwrap(), [uint_ty, uint_ty]);

        let nested = StagedPipeline::new(vec![
            ("pass".to_string(), make_uint_pass()),
            ("stages".to_string(), one_stage),
        ])
        .unwrap();
        assert_eq!(nested.io_types().unwrap(), [uint_ty, uint_ty]);

        Ok(())
    }

    #[test]
    fn test_io_types_empty() {
        assert!(Pipeline::default().io_types().is_none());
        assert!(StagedPipeline::default().io_types().is_none());

        let nested_empty = Pipeline::new(vec![Task::Stages(
            StagedPipeline::new(vec![(
                "empty".to_string(),
                Task::Pipeline(Pipeline::default()),
            )])
            .unwrap(),
        )])
        .unwrap();
        assert!(nested_empty.io_types().is_none());
    }

    #[test]
    fn test_io_types_skips_empty_children() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        let empty = || Task::Pipeline(Pipeline::default());
        let lower = || Task::Transformation(LowerToInt.into_pass());

        let leading = Pipeline::new(vec![empty(), lower()]).unwrap();
        assert_eq!(leading.io_types().unwrap(), [uint_ty, int_ty]);

        let trailing = Pipeline::new(vec![lower(), empty()]).unwrap();
        assert_eq!(trailing.io_types().unwrap(), [uint_ty, int_ty]);

        let surrounded = StagedPipeline::new(vec![
            ("before".to_string(), empty()),
            ("lower".to_string(), lower()),
            ("after".to_string(), empty()),
        ])
        .unwrap();
        assert_eq!(surrounded.io_types().unwrap(), [uint_ty, int_ty]);
    }

    #[test]
    fn test_pipeline_rejects_mismatch() {
        let tasks = vec![
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ];
        assert_eq!(Pipeline::new(tasks).unwrap_err().len(), 2);
    }

    #[test]
    fn test_stages_reject_mismatch() {
        let tasks = vec![
            (
                "lower".to_string(),
                Task::Transformation(LowerToInt.into_pass()),
            ),
            ("add".to_string(), Task::Transformation(AddOne.into_pass())),
        ];
        assert_eq!(StagedPipeline::new(tasks).unwrap_err().len(), 2);
    }

    #[test]
    fn test_pipeline_rejects_across_empty() {
        let tasks = vec![
            Task::Transformation(LowerToInt.into_pass()),
            Task::Pipeline(Pipeline::default()),
            Task::Transformation(AddOne.into_pass()),
        ];
        assert!(Pipeline::new(tasks).is_err());
    }

    #[test]
    fn test_pipeline_accepts_valid_chain() {
        let tasks = vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
        ];
        let pipeline = Pipeline::new(tasks).unwrap();
        let expected = [DynTypeId::of::<MyUint>(), DynTypeId::of::<MyInt>()];
        assert_eq!(pipeline.io_types().unwrap(), expected);
    }

    #[test]
    fn test_empty_children_keep_types() {
        let tasks = vec![
            Task::Pipeline(Pipeline::default()),
            Task::Transformation(LowerToInt.into_pass()),
            Task::Stages(StagedPipeline::default()),
        ];
        let pipeline = Pipeline::new(tasks).unwrap();
        let expected = [DynTypeId::of::<MyUint>(), DynTypeId::of::<MyInt>()];
        assert_eq!(pipeline.io_types().unwrap(), expected);
    }

    #[test]
    fn test_try_push_rejects_mismatch() {
        let mut pipeline = Pipeline::new(vec![Task::Transformation(LowerToInt.into_pass())])
            .expect("a single task always matches");
        assert!(
            pipeline
                .try_push(Task::Transformation(AddOne.into_pass()))
                .is_err()
        );
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn test_try_push_accepts_match() {
        let mut pipeline = Pipeline::default();
        pipeline
            .try_push(Task::Transformation(AddOne.into_pass()))
            .unwrap();
        pipeline
            .try_push(Task::Transformation(LowerToInt.into_pass()))
            .unwrap();
        assert_eq!(pipeline.len(), 2);
        let expected = [DynTypeId::of::<MyUint>(), DynTypeId::of::<MyInt>()];
        assert_eq!(pipeline.io_types().unwrap(), expected);
    }

    #[test]
    fn test_pipeline_len_and_is_empty() {
        let empty = Pipeline::default();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let filled = Pipeline::new(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ])
        .unwrap();
        assert_eq!(filled.len(), 2);
        assert!(!filled.is_empty());
    }

    #[test]
    fn test_pipeline_get() {
        let pipeline = Pipeline::new(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Pipeline(Pipeline::default()),
        ])
        .unwrap();
        assert!(matches!(pipeline.get(0), Some(Task::Transformation(_))));
        assert!(matches!(pipeline.get(1), Some(Task::Pipeline(_))));
        assert!(pipeline.get(2).is_none());
    }

    #[test]
    fn test_pipeline_iter_is_in_order() {
        let pipeline = Pipeline::new(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
        ])
        .unwrap();
        let names: Vec<&str> = pipeline
            .iter()
            .map(|task| match task {
                Task::Transformation(pass) => pass.name(),
                _ => panic!("expected a Task::Transformation"),
            })
            .collect();
        assert_eq!(names.len(), 2);
        assert!(names[0].contains("AddOne"));
        assert!(names[1].contains("LowerToInt"));
    }

    #[test]
    fn test_pipeline_iter_is_double_ended() {
        let pipeline = Pipeline::new(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
        ])
        .unwrap();
        let last = pipeline.iter().next_back().unwrap();
        assert_eq!(last.io_types().unwrap()[1], DynTypeId::of::<MyInt>());
    }

    #[test]
    fn test_stages_len_and_is_empty() {
        let empty = StagedPipeline::default();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let filled = StagedPipeline::new(vec![(
            "only".to_string(),
            Task::Transformation(AddOne.into_pass()),
        )])
        .unwrap();
        assert_eq!(filled.len(), 1);
        assert!(!filled.is_empty());
    }

    #[test]
    fn test_stages_get() {
        let stages = StagedPipeline::new(vec![
            (
                "first".to_string(),
                Task::Transformation(AddOne.into_pass()),
            ),
            (
                "second".to_string(),
                Task::Transformation(LowerToInt.into_pass()),
            ),
        ])
        .unwrap();
        let (name, task) = stages.get(0).unwrap();
        assert_eq!(name, "first");
        assert!(matches!(task, Task::Transformation(_)));
        assert_eq!(stages.get(1).unwrap().0, "second");
        assert!(stages.get(2).is_none());
    }

    #[test]
    fn test_stages_iter_is_in_order() {
        let stages = StagedPipeline::new(vec![
            ("init".to_string(), Task::Transformation(AddOne.into_pass())),
            (
                "lower".to_string(),
                Task::Transformation(LowerToInt.into_pass()),
            ),
        ])
        .unwrap();
        let names: Vec<&str> = stages.iter().map(|(name, _task)| name).collect();
        assert_eq!(names, ["init", "lower"]);
    }

    #[test]
    fn test_stages_iter_is_double_ended() {
        let stages = StagedPipeline::new(vec![
            ("init".to_string(), Task::Transformation(AddOne.into_pass())),
            (
                "lower".to_string(),
                Task::Transformation(LowerToInt.into_pass()),
            ),
        ])
        .unwrap();
        assert_eq!(stages.iter().next_back().unwrap().0, "lower");
    }
}
