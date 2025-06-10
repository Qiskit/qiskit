use crate::operations::{
    ControlFlow, ControlFlowView, InstructionView, OperationRef, Param, Parameters,
    StandardGateView, StandardInstruction, StandardInstructionView, UnitaryGateView,
};
use pyo3::PyObject;

/// Implemented for various instruction-like reference types.
///
/// Provides ergonomic views of an underlying instruction.
pub trait IntoInstructionView<'a> {
    /// The type of inner circuits contained within this instruction's views.
    type Block;

    /// Returns a view of the operation.
    fn view_operation(self) -> OperationRef<'a>;

    /// Returns a view of this instruction as a standard gate, if applicable.
    fn try_view_standard_gate(self) -> Option<StandardGateView<'a>>;

    /// Returns a view of this instruction as a standard instruction, if applicable.
    fn try_view_standard_instruction(self) -> Option<StandardInstructionView<'a>>;

    /// Returns a view of this instruction as a control flow instruction, if applicable.
    fn try_view_control_flow(self) -> Option<ControlFlowView<'a, Self::Block>>;

    /// Returns the old-style [Param] sequence, unless this is a control
    /// flow instruction.
    fn try_legacy_params(self) -> Option<&'a [Param]>;

    /// Returns an immutable ergonomic view of this instruction.
    #[inline]
    fn view(self) -> InstructionView<'a, Self::Block>
    where
        Self: Copy + Sized,
    {
        match self.view_operation() {
            OperationRef::ControlFlow(_) => {
                InstructionView::ControlFlow(self.try_view_control_flow().unwrap())
            }
            OperationRef::StandardGate(_) => {
                InstructionView::StandardGate(self.try_view_standard_gate().unwrap())
            }
            OperationRef::StandardInstruction(_) => {
                InstructionView::StandardInstruction(self.try_view_standard_instruction().unwrap())
            }
            OperationRef::Gate(g) => InstructionView::Gate(g),
            OperationRef::Instruction(i) => InstructionView::Instruction(i),
            OperationRef::Operation(o) => InstructionView::Operation(o),
            OperationRef::Unitary(u) => InstructionView::Unitary(UnitaryGateView(u)),
        }
    }
}

/// Represents an instruction that is directly convertible to our Python API
/// instruction type.
pub trait Instruction {
    /// Gets a reference to this instruction's operation.
    fn op(&self) -> OperationRef<'_>;

    /// Get a reference to this instruction's parameter list, if applicable.
    ///
    /// For standard gates without parameters this may be [None] or a
    /// `Some(Parameters::Param(smallvec![]))`.
    fn parameters(&self) -> Option<&Parameters<PyObject>>;

    /// Get the label for this instruction.
    fn label(&self) -> Option<&str>;
}

impl<'a, T: Instruction> IntoInstructionView<'a> for &'a T {
    type Block = PyObject;

    fn view_operation(self) -> OperationRef<'a> {
        Instruction::op(self)
    }

    fn try_view_standard_gate(self) -> Option<StandardGateView<'a>> {
        let OperationRef::StandardGate(gate) = self.op() else {
            return None;
        };
        let params = match self.parameters() {
            Some(Parameters::Params(params)) => params.as_slice(),
            None => &[],
            _ => panic!("invalid standard gate parameters"),
        };
        Some(StandardGateView(gate, params))
    }

    fn try_view_standard_instruction(self) -> Option<StandardInstructionView<'a>> {
        let OperationRef::StandardInstruction(instruction) = self.op() else {
            return None;
        };
        Some(match instruction {
            StandardInstruction::Barrier(n) => StandardInstructionView::Barrier(n),
            StandardInstruction::Delay(unit) => {
                let Some([duration]) = self.parameters().and_then(|p| match p {
                    Parameters::Params(params) => Some(params.as_slice()),
                    _ => None,
                }) else {
                    panic!("invalid delay parameters");
                };
                StandardInstructionView::Delay { duration, unit }
            }
            StandardInstruction::Measure => StandardInstructionView::Measure,
            StandardInstruction::Reset => StandardInstructionView::Reset,
        })
    }

    fn try_view_control_flow(self) -> Option<ControlFlowView<'a, Self::Block>> {
        let OperationRef::ControlFlow(control) = self.op() else {
            return None;
        };

        Some(match control {
            ControlFlow::Box { duration, .. } => {
                let Some(Parameters::Box { body }) = self.parameters() else {
                    panic!("invalid box parameters");
                };
                ControlFlowView::Box(duration.as_ref(), body)
            }
            ControlFlow::BreakLoop { .. } => ControlFlowView::BreakLoop,
            ControlFlow::ContinueLoop { .. } => ControlFlowView::ContinueLoop,
            ControlFlow::ForLoop { .. } => {
                let Some(Parameters::ForLoop {
                    indexset,
                    loop_param,
                    body,
                }) = self.parameters()
                else {
                    panic!("invalid for loop parameters");
                };
                ControlFlowView::ForLoop {
                    indexset,
                    loop_param: loop_param.as_ref(),
                    body,
                }
            }
            ControlFlow::IfElse { condition, .. } => {
                let Some(Parameters::IfElse {
                    true_body,
                    false_body,
                }) = self.parameters()
                else {
                    panic!("invalid ifelse parameters");
                };
                ControlFlowView::IfElse {
                    condition,
                    true_body,
                    false_body: false_body.as_ref(),
                }
            }
            ControlFlow::Switch {
                target, label_spec, ..
            } => {
                let cases_specifier = label_spec
                    .iter()
                    .zip(
                        self.parameters()
                            .and_then(|p| match p {
                                Parameters::Switch { cases } => Some(cases),
                                _ => None,
                            })
                            .expect("invalid switch parameters"),
                    )
                    .collect();
                ControlFlowView::Switch {
                    target,
                    cases_specifier,
                }
            }
            ControlFlow::While { condition, .. } => {
                let Some(Parameters::While { body }) = self.parameters() else {
                    panic!("invalid while parameters");
                };
                ControlFlowView::While { condition, body }
            }
        })
    }

    fn try_legacy_params(self) -> Option<&'a [Param]> {
        match self.view() {
            InstructionView::StandardGate(_)
            | InstructionView::Gate(_)
            | InstructionView::Operation(_)
            | InstructionView::Unitary(_)
            | InstructionView::Instruction(_) => Some(
                self.parameters()
                    .and_then(|p| match p {
                        Parameters::Params(p) => Some(p.as_slice()),
                        _ => panic!("expected gate parameters"),
                    })
                    .unwrap_or_default(),
            ),
            InstructionView::StandardInstruction(inst) => match inst {
                StandardInstructionView::Delay { duration, .. } => {
                    Some(std::slice::from_ref(&duration))
                }
                _ => Some(&[]),
            },
            _ => panic!(
                "legacy parameters not supported for operation {:?}",
                self.op()
            ),
        }
    }
}
