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

use crate::pass::Pass;
use qiskit_util::dyn_types::DynTypeId;

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. It describes how work is being executed, ranging
/// from the simple execution of a single pass, over groups of passes to structured flow control,
/// such as loops. The [`PassManager`](crate::PassManager) stores a vector of [Task]s and executes
/// them.
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

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Task::Transformation(p) => f.debug_tuple("Transformation").field(&p.name()).finish(),
            Task::Group(tasks) => f.debug_tuple("Group").field(tasks).finish(),
            Task::Stages(stages) => f.debug_tuple("Stages").field(stages).finish(),
        }
    }
}

impl Task {
    pub(crate) fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
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
            Task::Group(group) => sequence_io_types(group.iter()),
            Task::Stages(stages) => sequence_io_types(stages.iter().map(|(_name, task)| task)),
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
    fn test_io_types_empty() {
        assert!(Task::Group(vec![]).io_types().is_none());
        assert!(Task::Stages(vec![]).io_types().is_none());

        let nested_empty = Task::Group(vec![Task::Stages(vec![(
            "empty".to_string(),
            Task::Group(vec![]),
        )])]);
        assert!(nested_empty.io_types().is_none());
    }

    #[test]
    fn test_io_types_skips_empty_children() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        let empty = || Task::Group(vec![]);
        let lower = || Task::Transformation(LowerToInt.into_pass());

        let leading = Task::Group(vec![empty(), lower()]);
        assert_eq!(leading.io_types().unwrap(), [uint_ty, int_ty]);

        let trailing = Task::Group(vec![lower(), empty()]);
        assert_eq!(trailing.io_types().unwrap(), [uint_ty, int_ty]);

        let surrounded = Task::Stages(vec![
            ("before".to_string(), empty()),
            ("lower".to_string(), lower()),
            ("after".to_string(), empty()),
        ]);
        assert_eq!(surrounded.io_types().unwrap(), [uint_ty, int_ty]);
    }
}
