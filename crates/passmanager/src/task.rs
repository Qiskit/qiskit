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

use crate::pass::Pass;
use qiskit_util::dyn_types::DynTypeId;

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. Each task variant describes a different mode of
/// flow, ranging from the simple execution of a single pass, to lists of other tasks,
/// to structured flow control such as loops and conditionals.
///
/// It is allowable and expected for a given task instance to have incorrect IR typing up until
/// the point that it is owned by a [`PassManager`](crate::PassManager), which then assumes the
/// responsibility of validating and maintaining correct IR type flow for its lifetime. This
/// allows a [Task] to represent a partially assembled pipeline.
#[derive(Debug)]
#[non_exhaustive]
pub enum Task {
    // TODO Add Loop and Switch with conditions that can be set from Python/C and
    // proper error handlings that occur during the condition evaluation.
    /// A single pass.
    Transformation(Box<dyn Pass>),
    /// A sequence of tasks.
    Sequence(Vec<Task>),
    /// A list of named tasks.
    Stages(Vec<(String, Task)>),
}

impl Task {
    /// Return the input and output IR types, or `None` if not defined.
    pub fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
        match self {
            Task::Transformation(pass) => Some([pass.ir_id_in(), pass.ir_id_out()]),
            Task::Sequence(tasks) => sequence_io_types(tasks.iter()),
            Task::Stages(stages) => sequence_io_types(stages.iter().map(|(_name, task)| task)),
        }
    }

    /// Validate (recursively) that the IR typing of this task is self-consistent.
    pub fn validate(&self) -> Result<(), TaskTypeError> {
        match self {
            Task::Transformation(_) => Ok(()),
            Task::Sequence(tasks) => validate_sequence(
                tasks
                    .iter()
                    .enumerate()
                    .map(|(index, task)| (TaskPathSegmentRef::Index(index), task)),
            ),
            Task::Stages(stages) => validate_sequence(
                stages
                    .iter()
                    .map(|(name, task)| (TaskPathSegmentRef::Stage(name), task)),
            ),
        }
    }
}

/// Return the input and output IR types of a sequence of tasks, or `None` if no task in the
/// sequence has defined types.
pub(crate) fn sequence_io_types<'a>(
    tasks: impl DoubleEndedIterator<Item = &'a Task>,
) -> Option<[DynTypeId<'a>; 2]> {
    let mut typed = tasks.filter_map(|task| task.io_types());
    let first = typed.next()?;
    let last = typed.next_back().unwrap_or(first);
    Some([first[0], last[1]])
}

/// Validate a sequence of tasks that execute in order.
fn validate_sequence<'a>(
    tasks: impl IntoIterator<Item = (TaskPathSegmentRef<'a>, &'a Task)>,
) -> Result<(), TaskTypeError> {
    let mut last_out: Option<DynTypeId<'a>> = None;
    for (segment, task) in tasks {
        if let Err(err) = task.validate() {
            return Err(err.within(segment));
        }
        let Some([in_, out]) = task.io_types() else {
            continue;
        };
        if let Some(previous) = last_out
            && previous != in_
        {
            return Err(TaskTypeError::mismatch(previous, in_).within(segment));
        }
        last_out = Some(out);
    }
    Ok(())
}

/// One step of the path that locates a [Task] within the tree of tasks that contains it.
///
/// This may be extended if some future [Task] decides it needs a new way to index itself.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskPathSegment {
    /// The position within a [`Task::Sequence`].
    Index(usize),
    /// The name of a stage of a [`Task::Stages`].
    Stage(String),
}
impl fmt::Display for TaskPathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(index) => write!(f, "[{index}]"),
            // Single quotes, to match how Python displays a string key.
            Self::Stage(name) => write!(f, "['{}']", name.escape_debug()),
        }
    }
}

/// A [TaskPathSegment] that may borrow attributes of the task being validated.
#[derive(Clone, Copy, Debug)]
enum TaskPathSegmentRef<'a> {
    Index(usize),
    Stage(&'a str),
}
impl TaskPathSegmentRef<'_> {
    fn to_segment(self) -> TaskPathSegment {
        match self {
            Self::Index(index) => TaskPathSegment::Index(index),
            Self::Stage(name) => TaskPathSegment::Stage(name.to_owned()),
        }
    }
}

/// The IR types of two consecutive tasks disagree.
///
/// The path locates the task whose input type was not satisfied, relative to the task that was
/// validated.
#[derive(Clone, Debug, Error)]
#[error(
    "IR types mismatched: {} takes {input}, but the preceding task outputs {output}",
    DisplaySubject(&self.path)
)]
pub struct TaskTypeError {
    /// The innermost segment is stored first, the reverse of display order.
    path: Vec<TaskPathSegment>,
    output: String,
    input: String,
}
impl TaskTypeError {
    /// A mismatch at the task the error is reported against, with an empty path.
    pub(crate) fn mismatch(output: DynTypeId<'_>, input: DynTypeId<'_>) -> Self {
        Self {
            path: Vec::new(),
            output: output.short_describe().to_owned(),
            input: input.short_describe().to_owned(),
        }
    }

    /// Relocate this error to describe a task nested one level further in, under `segment`.
    fn within(mut self, segment: TaskPathSegmentRef<'_>) -> Self {
        self.path.push(segment.to_segment());
        self
    }

    /// The path to the task whose input type was not satisfied, innermost segment first.
    pub fn path(&self) -> &[TaskPathSegment] {
        &self.path
    }
}

/// Display helper for the subscripted form of a task path.
struct DisplaySubject<'a>(&'a [TaskPathSegment]);
impl fmt::Display for DisplaySubject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("the task");
        }
        f.write_str("the task at ")?;
        for segment in self.0.iter().rev() {
            segment.fmt(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::pass::StaticPass;
    use crate::{IR, PassContext, Task, TaskPathSegment};
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
    fn test_io_types() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        assert_eq!(
            Task::Transformation(AddOne.into_pass()).io_types().unwrap(),
            [uint_ty, uint_ty]
        );
        assert_eq!(
            Task::Transformation(LowerToInt.into_pass())
                .io_types()
                .unwrap(),
            [uint_ty, int_ty]
        );

        let list = Task::Sequence(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
        ]);
        assert_eq!(list.io_types().unwrap(), [uint_ty, int_ty]);

        let nested = Task::Stages(vec![
            ("add".to_string(), Task::Transformation(AddOne.into_pass())),
            ("lower".to_string(), Task::Sequence(vec![list])),
        ]);
        assert_eq!(nested.io_types().unwrap(), [uint_ty, int_ty]);
    }

    #[test]
    fn test_io_types_empty() {
        assert!(Task::Sequence(vec![]).io_types().is_none());
        assert!(Task::Stages(vec![]).io_types().is_none());

        let nested_empty = Task::Sequence(vec![Task::Stages(vec![(
            "empty".to_string(),
            Task::Sequence(vec![]),
        )])]);
        assert!(nested_empty.io_types().is_none());
    }

    #[test]
    fn test_io_types_skips_empty_children() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        let empty = || Task::Sequence(vec![]);
        let lower = || Task::Transformation(LowerToInt.into_pass());

        let leading = Task::Sequence(vec![empty(), lower()]);
        assert_eq!(leading.io_types().unwrap(), [uint_ty, int_ty]);

        let trailing = Task::Sequence(vec![lower(), empty()]);
        assert_eq!(trailing.io_types().unwrap(), [uint_ty, int_ty]);

        let surrounded = Task::Stages(vec![
            ("before".to_string(), empty()),
            ("lower".to_string(), lower()),
            ("after".to_string(), empty()),
        ]);
        assert_eq!(surrounded.io_types().unwrap(), [uint_ty, int_ty]);
    }

    #[test]
    fn test_io_types_ignores_internal_mismatch() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let list = Task::Sequence(vec![
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ]);
        assert_eq!(list.io_types().unwrap(), [uint_ty, uint_ty]);
        assert!(list.validate().is_err());
    }

    #[test]
    fn test_validate_accepts_valid_chain() {
        assert!(Task::Sequence(vec![]).validate().is_ok());
        assert!(Task::Transformation(AddOne.into_pass()).validate().is_ok());
        assert!(
            Task::Sequence(vec![
                Task::Transformation(AddOne.into_pass()),
                Task::Transformation(LowerToInt.into_pass())
            ])
            .validate()
            .is_ok()
        );
        assert!(
            Task::Stages(vec![
                ("add".to_string(), Task::Transformation(AddOne.into_pass())),
                (
                    "lower".to_string(),
                    Task::Transformation(LowerToInt.into_pass())
                )
            ])
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn test_validate_reports_index_of_mismatch() {
        let list = Task::Sequence(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ]);
        let err = list.validate().unwrap_err();
        assert_eq!(err.path(), [TaskPathSegment::Index(2)]);
        assert!(err.to_string().contains("the task at [2]"), "{err}");
        assert!(err.to_string().contains("MyUint"), "{err}");
        assert!(err.to_string().contains("MyInt"), "{err}");
    }

    #[test]
    fn test_validate_reports_stage_of_mismatch() {
        let stages = Task::Stages(vec![
            (
                "lower".to_string(),
                Task::Transformation(LowerToInt.into_pass()),
            ),
            ("add".to_string(), Task::Transformation(AddOne.into_pass())),
        ]);
        let err = stages.validate().unwrap_err();
        assert_eq!(err.path(), [TaskPathSegment::Stage("add".to_string())]);
        assert!(err.to_string().contains("the task at ['add']"), "{err}");
    }

    #[test]
    fn test_validate_reports_nested_path() {
        let inner = Task::Sequence(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ]);
        let outer = Task::Sequence(vec![Task::Stages(vec![("layout".to_string(), inner)])]);
        let err = outer.validate().unwrap_err();
        assert_eq!(
            err.path(),
            [
                TaskPathSegment::Index(2),
                TaskPathSegment::Stage("layout".to_string()),
                TaskPathSegment::Index(0),
            ]
        );
        assert!(
            err.to_string().contains("the task at [0]['layout'][2]"),
            "{err}"
        );
    }

    #[test]
    fn test_validate_rejects_across_empty() {
        let list = Task::Sequence(vec![
            Task::Transformation(LowerToInt.into_pass()),
            Task::Sequence(vec![]),
            Task::Transformation(AddOne.into_pass()),
        ]);
        let err = list.validate().unwrap_err();
        assert_eq!(err.path(), [TaskPathSegment::Index(2)]);
    }

    #[test]
    fn test_validate_accepts_across_empty() {
        let list = Task::Sequence(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Sequence(vec![]),
            Task::Transformation(LowerToInt.into_pass()),
        ]);
        assert!(list.validate().is_ok());
    }
}
