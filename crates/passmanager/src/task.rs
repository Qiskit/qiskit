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
    pub fn group(tasks: Vec<Task>) -> Result<Self, Vec<Task>> {
        if first_io_mismatch(tasks.iter()).is_some() {
            return Err(tasks);
        }
        Ok(Self(TaskInner::Group(tasks)))
    }

    /// Try to build a task that runs named `stages` in order.
    pub fn stages(stages: Vec<(String, Task)>) -> Result<Self, Vec<(String, Task)>> {
        if first_io_mismatch(stages.iter().map(|(_name, task)| task)).is_some() {
            return Err(stages);
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

/// Return the first output and input types that disagree in a sequence of tasks, if any.
fn first_io_mismatch<'a>(tasks: impl IntoIterator<Item = &'a Task>) -> Option<[DynTypeId<'a>; 2]> {
    let mut last_out: Option<DynTypeId<'a>> = None;
    for task in tasks {
        let Some([in_, out]) = task.io_types() else {
            continue;
        };
        if let Some(previous) = last_out
            && previous != in_
        {
            return Some([previous, in_]);
        }
        last_out = Some(out);
    }
    None
}

#[cfg(test)]
mod test {
    use crate::Task;
    use crate::pass::{PassError, StaticPass};
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
        assert_eq!(Task::group(tasks).unwrap_err().len(), 2);
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
        assert_eq!(Task::stages(stages).unwrap_err().len(), 2);
    }

    #[test]
    fn test_group_rejects_across_empty() {
        let tasks = vec![
            Task::transformation(LowerToInt.into_pass()),
            Task::group(vec![]).unwrap(),
            Task::transformation(AddOne.into_pass()),
        ];
        assert!(Task::group(tasks).is_err());
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
