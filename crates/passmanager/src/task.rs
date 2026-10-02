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
use thiserror::Error;

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. It describes how work is being executed, ranging
/// from the simple execution of a single pass, over groups of passes to structured flow control,
/// such as loops. The [`PassManager`](crate::PassManager) stores a vector of [Task]s and executes
/// them. Adjacent passes and tasks in a pipeline must share IR types at their boundary, checked
/// dynamically at construction.
pub struct Task(pub(crate) TaskInner);

/// The kinds of [Task].
pub(crate) enum TaskInner {
    // TODO Add Loop and Switch with conditions that can be set from Python/C and
    // proper error handlings that occur during the condition evaluation.
    /// A single pass.
    Transformation(Box<dyn Pass>),

    /// A group of tasks.
    Group(Vec<Task>),

    /// A sequence of named tasks.
    Stages(Vec<(String, Task)>),
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            TaskInner::Transformation(p) => {
                f.debug_tuple("Transformation").field(&p.name()).finish()
            }
            TaskInner::Group(tasks) => f.debug_tuple("Group").field(tasks).finish(),
            TaskInner::Stages(stages) => f.debug_tuple("Stages").field(stages).finish(),
        }
    }
}

impl Task {
    /// Build a task that runs a single pass.
    pub fn transformation(pass: Box<dyn Pass>) -> Self {
        Self(TaskInner::Transformation(pass))
    }

    /// Try to build a task that runs `tasks` in order.
    pub fn group(tasks: Vec<Task>) -> Result<Self, TypeMismatch<Vec<Task>>> {
        if let Some(error) = first_io_mismatch(tasks.iter(), MismatchPosition::Group) {
            return Err(error.reject(tasks));
        }
        Ok(Self(TaskInner::Group(tasks)))
    }

    /// Try to build a task that runs named `stages` in order.
    pub fn stages(stages: Vec<(String, Task)>) -> Result<Self, TypeMismatch<Vec<(String, Task)>>> {
        let name_mismatch = |[first, second]: [usize; 2]| {
            MismatchPosition::Stages([stages[first].0.clone(), stages[second].0.clone()])
        };
        if let Some(error) =
            first_io_mismatch(stages.iter().map(|(_name, task)| task), name_mismatch)
        {
            return Err(error.reject(stages));
        }
        Ok(Self(TaskInner::Stages(stages)))
    }

    pub(crate) fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
        match &self.0 {
            TaskInner::Transformation(pass) => Some([pass.ir_id_in(), pass.ir_id_out()]),
            TaskInner::Group(group) => sequence_io_types(group.iter()),
            TaskInner::Stages(stages) => sequence_io_types(stages.iter().map(|(_name, task)| task)),
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

        match &self.0 {
            TaskInner::Transformation(pass) => pass.run(ir, context),
            TaskInner::Group(tasks) => {
                for task in tasks.iter() {
                    ir = task.execute(ir, context)?;
                }
                Ok(ir)
            }
            TaskInner::Stages(stages) => {
                for (_name, task) in stages.iter() {
                    ir = task.execute(ir, context)?;
                }
                Ok(ir)
            }
        }
    }
}

/// Return the input and output types of a sequence of tasks if non-empty.
pub(crate) fn sequence_io_types<'a>(
    tasks: impl DoubleEndedIterator<Item = &'a Task>,
) -> Option<[DynTypeId<'a>; 2]> {
    let mut typed = tasks.filter_map(|task| task.io_types());
    let first = typed.next()?;
    let last = typed.next_back().unwrap_or(first);
    Some([first[0], last[1]])
}

/// A runtime IR typing mismatch description.
#[derive(Debug)]
pub struct TypeMismatch<T> {
    /// The requirement violation.
    pub error: Box<TypeMismatchError>, // boxed to keep the `Err` variant small
    /// The rejected input.
    pub rejected: T,
}

/// Two IR types that should agree don't.
#[derive(Debug, Error)]
#[error("IR types `{first}` and `{second}` do not agree {at}")]
pub struct TypeMismatchError {
    /// The position of the mismatch.
    pub at: MismatchPosition,
    /// A description of the first type.
    pub first: String,
    /// A description of the second type.
    pub second: String,
}

impl TypeMismatchError {
    /// Describe the position and types of a mismatch.
    pub(crate) fn new(at: MismatchPosition, first: DynTypeId<'_>, second: DynTypeId<'_>) -> Self {
        Self {
            at,
            first: first.describe().into_owned(),
            second: second.describe().into_owned(),
        }
    }

    /// Attach the input that is being rejected.
    pub(crate) fn reject<T>(self, rejected: T) -> TypeMismatch<T> {
        TypeMismatch {
            error: Box::new(self),
            rejected,
        }
    }
}

/// Descriptor for the position of a dynamic IR type mismatch.
#[derive(Debug)]
#[non_exhaustive]
pub enum MismatchPosition {
    /// Between two positions in a group of tasks.
    Group([usize; 2]),
    /// Between two named stages.
    Stages([String; 2]),
    /// Between a pipeline and a task being appended to it.
    Append,
}

impl std::fmt::Display for MismatchPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MismatchPosition::Group([first, second]) => {
                write!(f, "between tasks {first} and {second}")
            }
            MismatchPosition::Stages([first, second]) => {
                write!(f, "between stages `{first}` and `{second}`")
            }
            MismatchPosition::Append => f.write_str("at the end of the pipeline"),
        }
    }
}

/// Return the position of the first two sequential tasks whose types disagree, if any.
fn first_io_mismatch<'a>(
    tasks: impl IntoIterator<Item = &'a Task>,
    position: impl FnOnce([usize; 2]) -> MismatchPosition,
) -> Option<TypeMismatchError> {
    let mut last_out: Option<(usize, DynTypeId<'a>)> = None;
    for (index, task) in tasks.into_iter().enumerate() {
        let Some([in_, out]) = task.io_types() else {
            continue;
        };
        if let Some((last_index, previous)) = last_out
            && previous != in_
        {
            return Some(TypeMismatchError::new(
                position([last_index, index]),
                previous,
                in_,
            ));
        }
        last_out = Some((index, out));
    }
    None
}

#[cfg(test)]
mod test {
    use crate::pass::{PassError, StaticPass};
    use crate::task::{MismatchPosition, Task, TypeMismatchError};
    use crate::{IR, PassContext};
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

    #[track_caller]
    fn assert_disagreement(error: &TypeMismatchError, first: &str, second: &str) {
        assert!(error.first.contains(first), "{error}");
        assert!(error.second.contains(second), "{error}");
    }

    #[test]
    fn test_io_types() -> Result<(), PassError> {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        let make_uint_pass = || Task::transformation(AddOne.into_pass());
        assert_eq!(make_uint_pass().io_types().unwrap(), [uint_ty, uint_ty]);

        let lower_pass = Task::transformation(LowerToInt.into_pass());
        assert_eq!(lower_pass.io_types().unwrap(), [uint_ty, int_ty]);

        let stages = Task::stages(vec![("one_and_only".to_string(), make_uint_pass())]).unwrap();
        assert_eq!(stages.io_types().unwrap(), [uint_ty, uint_ty]);

        let nested = Task::stages(vec![
            ("pass".to_string(), make_uint_pass()),
            ("stages".to_string(), stages),
        ])
        .unwrap();
        assert_eq!(nested.io_types().unwrap(), [uint_ty, uint_ty]);

        Ok(())
    }

    #[test]
    fn test_io_types_empty() {
        assert!(Task::group(vec![]).unwrap().io_types().is_none());
        assert!(Task::stages(vec![]).unwrap().io_types().is_none());

        let nested_empty = Task::group(vec![
            Task::stages(vec![("empty".to_string(), Task::group(vec![]).unwrap())]).unwrap(),
        ])
        .unwrap();
        assert!(nested_empty.io_types().is_none());
    }

    #[test]
    fn test_io_types_skips_empty_children() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        let empty = || Task::group(vec![]).unwrap();
        let lower = || Task::transformation(LowerToInt.into_pass());

        let leading = Task::group(vec![empty(), lower()]).unwrap();
        assert_eq!(leading.io_types().unwrap(), [uint_ty, int_ty]);

        let trailing = Task::group(vec![lower(), empty()]).unwrap();
        assert_eq!(trailing.io_types().unwrap(), [uint_ty, int_ty]);

        let surrounded = Task::stages(vec![
            ("before".to_string(), empty()),
            ("lower".to_string(), lower()),
            ("after".to_string(), empty()),
        ])
        .unwrap();
        assert_eq!(surrounded.io_types().unwrap(), [uint_ty, int_ty]);
    }

    #[test]
    fn test_group_rejects_mismatch() {
        let tasks = vec![
            Task::transformation(LowerToInt.into_pass()),
            Task::transformation(AddOne.into_pass()),
        ];
        let err = Task::group(tasks).unwrap_err();
        assert_disagreement(&err.error, "MyInt", "MyUint");
        assert_eq!(err.rejected.len(), 2);
    }

    #[test]
    fn test_stages_reject_mismatch() {
        let stages = vec![
            (
                "lower".to_string(),
                Task::transformation(LowerToInt.into_pass()),
            ),
            ("add".to_string(), Task::transformation(AddOne.into_pass())),
        ];
        let err = Task::stages(stages).unwrap_err();
        assert_disagreement(&err.error, "MyInt", "MyUint");
        assert!(
            matches!(&err.error.at, MismatchPosition::Stages([first, second])
                if first == "lower" && second == "add"),
            "{}",
            err.error
        );
        assert_eq!(err.rejected.len(), 2);
    }

    #[test]
    fn test_group_rejects_across_empty() {
        let tasks = vec![
            Task::transformation(LowerToInt.into_pass()),
            Task::group(vec![]).unwrap(),
            Task::transformation(AddOne.into_pass()),
        ];
        let err = Task::group(tasks).unwrap_err();
        assert!(
            matches!(err.error.at, MismatchPosition::Group([0, 2])), // note we correctly skip 1
            "{}",
            err.error
        );
    }

    /// Test that a reported position indexes the whole sequence, not the typed tasks.
    #[test]
    fn test_position_counts_empties() {
        let empty = || Task::group(vec![]).unwrap();
        let tasks = vec![
            empty(),
            empty(),
            Task::transformation(LowerToInt.into_pass()),
            empty(),
            Task::transformation(AddOne.into_pass()),
        ];
        let err = Task::group(tasks).unwrap_err();
        assert!(
            matches!(err.error.at, MismatchPosition::Group([2, 4])),
            "{}",
            err.error
        );
    }

    #[test]
    fn test_stage_names_skip_empties() {
        let stages = vec![
            ("pad".to_string(), Task::group(vec![]).unwrap()),
            (
                "lower".to_string(),
                Task::transformation(LowerToInt.into_pass()),
            ),
            ("gap".to_string(), Task::group(vec![]).unwrap()),
            ("add".to_string(), Task::transformation(AddOne.into_pass())),
        ];
        let err = Task::stages(stages).unwrap_err();
        assert!(
            matches!(&err.error.at, MismatchPosition::Stages([first, second])
                if first == "lower" && second == "add"),
            "{}",
            err.error
        );
    }

    #[test]
    fn test_group_accepts_valid_chain() {
        let tasks = vec![
            Task::transformation(AddOne.into_pass()),
            Task::transformation(LowerToInt.into_pass()),
        ];
        let group = Task::group(tasks).unwrap();
        let expected = [DynTypeId::of::<MyUint>(), DynTypeId::of::<MyInt>()];
        assert_eq!(group.io_types().unwrap(), expected);
    }

    #[test]
    fn test_empty_children_keep_types() {
        let tasks = vec![
            Task::group(vec![]).unwrap(),
            Task::transformation(LowerToInt.into_pass()),
            Task::stages(vec![]).unwrap(),
        ];
        let group = Task::group(tasks).unwrap();
        let expected = [DynTypeId::of::<MyUint>(), DynTypeId::of::<MyInt>()];
        assert_eq!(group.io_types().unwrap(), expected);
    }
}
