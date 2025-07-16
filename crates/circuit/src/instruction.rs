// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::circuit_data::CircuitData;
use crate::operations::{
    BoxDuration, CaseSpecifier, Condition, ControlFlow, DelayUnit, OperationRef, Param, PyGate,
    PyInstruction, PyOperation, StandardGate, StandardInstruction, Target, UnitaryGate,
};
use nalgebra::Matrix2;
use ndarray::Array2;
use num_complex::Complex64;
use pyo3::PyObject;
use smallvec::SmallVec;
use std::ops::Deref;

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

/// The parameter list of an instruction.
#[derive(Clone, Debug)]
pub enum Parameters<T> {
    Params(SmallVec<[Param; 3]>),
    Box {
        body: T,
    },
    ForLoop {
        indexset: Vec<usize>,
        loop_param: Option<PyObject>,
        body: T,
    },
    IfElse {
        true_body: T,
        false_body: Option<T>,
    },
    Switch {
        cases: Vec<T>,
    },
    While {
        body: T,
    },
}

impl<T> Parameters<T> {
    /// Replace all blocks of this parameter set, in order.
    ///
    /// Panics if `blocks` does not contain exactly the expected number of blocks
    /// for the parameter set.
    pub fn replace_blocks(&mut self, blocks: impl IntoIterator<Item = T>) {
        let mut replacements = blocks.into_iter();
        match self {
            Parameters::Params(_) => {}
            Parameters::Box { body, .. } => {
                *body = replacements.next().expect("not enough blocks");
            }
            Parameters::ForLoop { body, .. } => {
                *body = replacements.next().expect("not enough blocks");
            }
            Parameters::IfElse {
                true_body,
                false_body,
                ..
            } => {
                *true_body = replacements.next().expect("not enough blocks");
                if false_body.is_some() {
                    *false_body = Some(replacements.next().expect("not enough blocks"));
                }
            }
            Parameters::Switch { cases, .. } => {
                for case in cases {
                    *case = replacements.next().expect("not enough blocks");
                }
            }
            Parameters::While { body, .. } => {
                *body = replacements.next().expect("not enough blocks");
            }
        }
        if replacements.next().is_some() {
            panic!("too many blocks");
        }
    }
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
                    .map(|p| match p {
                        Parameters::Params(p) => p.as_slice(),
                        _ => panic!("expected gate parameters"),
                    })
                    .unwrap_or_default(),
            ),
            InstructionView::StandardInstruction(inst) => match inst {
                StandardInstructionView::Delay { duration, .. } => {
                    Some(std::slice::from_ref(duration))
                }
                _ => Some(&[]),
            },
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum InstructionView<'a, T> {
    ControlFlow(ControlFlowView<'a, T>),
    StandardGate(StandardGateView<'a>),
    StandardInstruction(StandardInstructionView<'a>),
    Gate(&'a PyGate),
    Instruction(&'a PyInstruction),
    Operation(&'a PyOperation),
    Unitary(UnitaryGateView<'a>),
}

impl<'a, T> InstructionView<'a, T> {
    pub fn try_matrix(&self) -> Option<Array2<Complex64>> {
        match self {
            InstructionView::StandardGate(g) => g.matrix(),
            InstructionView::Gate(g) => g.matrix(),
            InstructionView::Unitary(u) => u.matrix(),
            _ => None,
        }
    }

    /// Returns a static matrix for 1-qubit gates. Will return `None` when the gate is not 1-qubit.
    #[inline]
    pub fn try_matrix_as_static_1q(&self) -> Option<[[Complex64; 2]; 2]> {
        match self {
            Self::StandardGate(standard) => standard.matrix_as_static_1q(),
            Self::Gate(gate) => gate.matrix_as_static_1q(),
            Self::Unitary(unitary) => unitary.matrix_as_static_1q(),
            _ => None,
        }
    }

    pub fn try_matrix_as_nalgebra_1q(&self) -> Option<Matrix2<Complex64>> {
        match self {
            InstructionView::Unitary(u) => u.matrix_as_nalgebra_1q(),
            // default implementation
            _ => self
                .try_matrix_as_static_1q()
                .map(|arr| Matrix2::new(arr[0][0], arr[0][1], arr[1][0], arr[1][1])),
        }
    }

    pub fn try_definition(&self) -> Option<CircuitData> {
        match self {
            InstructionView::StandardGate(g) => g.definition(),
            InstructionView::Gate(g) => g.definition(),
            InstructionView::Instruction(i) => i.definition(),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ControlFlowView<'a, T> {
    Box(Option<&'a BoxDuration>, &'a T),
    BreakLoop,
    ContinueLoop,
    ForLoop {
        indexset: &'a [usize],
        loop_param: Option<&'a PyObject>,
        body: &'a T,
    },
    IfElse {
        condition: &'a Condition,
        true_body: &'a T,
        false_body: Option<&'a T>,
    },
    Switch {
        target: &'a Target,
        cases_specifier: Vec<(&'a Vec<CaseSpecifier>, &'a T)>,
    },
    While {
        condition: &'a Condition,
        body: &'a T,
    },
}

impl<'a, T> ControlFlowView<'a, T> {
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = &'a T> {
        match self {
            ControlFlowView::Box(_, body) => vec![*body],
            ControlFlowView::BreakLoop => vec![],
            ControlFlowView::ContinueLoop => vec![],
            ControlFlowView::ForLoop { body, .. } => vec![*body],
            ControlFlowView::IfElse {
                true_body,
                false_body,
                ..
            } => {
                if let Some(false_body) = false_body {
                    vec![*true_body, *false_body]
                } else {
                    vec![*true_body]
                }
            }
            ControlFlowView::Switch {
                cases_specifier, ..
            } => cases_specifier.iter().map(|(_, block)| *block).collect(),
            ControlFlowView::While { body, .. } => vec![*body],
        }
        .into_iter()
    }
}

#[derive(Clone, Debug)]
pub enum StandardInstructionView<'a> {
    Barrier(u32),
    Delay {
        duration: &'a Param,
        unit: DelayUnit,
    },
    Measure,
    Reset,
}

#[derive(Clone, Debug)]
pub struct StandardGateView<'a>(pub StandardGate, pub &'a [Param]);

impl<'a> StandardGateView<'a> {
    #[inline]
    pub fn gate(&self) -> &StandardGate {
        &self.0
    }

    #[inline]
    pub fn params(&self) -> &'a [Param] {
        self.1
    }

    #[inline]
    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        self.0.matrix(self.1)
    }

    #[inline]
    fn matrix_as_static_1q(&self) -> Option<[[Complex64; 2]; 2]> {
        self.0.matrix_as_static_1q(self.1)
    }

    #[inline]
    pub fn definition(&self) -> Option<CircuitData> {
        self.0.definition(self.1)
    }

    #[inline]
    pub fn inverse(&self) -> Option<(StandardGate, SmallVec<[Param; 3]>)> {
        self.0.inverse(self.1)
    }
}

#[derive(Clone, Debug)]
pub struct UnitaryGateView<'a>(pub &'a UnitaryGate);

impl Deref for UnitaryGateView<'_> {
    type Target = UnitaryGate;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
