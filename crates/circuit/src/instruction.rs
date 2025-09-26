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

use crate::operations::{BoxDuration, CaseSpecifier, Condition, OperationRef, Param, SwitchTarget};
use ndarray::Array2;
use num_complex::Complex64;
use pyo3::PyObject;
use smallvec::SmallVec;

/// Represents an instruction that is directly convertible to our Python API
/// instruction type (i.e. owns its `params` data and label).
///
/// It's implemented by our unpacked instruction types like
/// [CircuitInstruction] and [OperationFromPython] which own all the data they
/// need to be converted back to a Python instance.
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

    /// Get a slice view onto the contained parameters.
    ///
    /// Panics if the instruction does not support params.
    #[inline]
    fn params_view(&self) -> &[Param] {
        self.parameters()
            .map(|p| match p {
                Parameters::Params(p) => p.as_slice(),
                _ => panic!("expected parameters"),
            })
            .unwrap_or_default()
    }

    /// Get a slice view onto the contained blocks.
    ///
    /// Panics if the instruction does not support blocks.
    #[inline]
    fn blocks_view(&self) -> &[PyObject] {
        self.parameters()
            .map(|p| match p {
                Parameters::Blocks(b) => b.as_slice(),
                _ => panic!("expected blocks"),
            })
            .unwrap_or_default()
    }

    fn try_matrix(&self) -> Option<Array2<Complex64>> {
        match self.op() {
            OperationRef::StandardGate(g) => g.matrix(self.params_view()),
            OperationRef::Gate(g) => g.matrix(),
            OperationRef::Unitary(u) => u.matrix(),
            _ => None,
        }
    }
}

/// The parameter list of an instruction.
#[derive(Clone, Debug)]
pub enum Parameters<T> {
    Params(SmallVec<[Param; 3]>),
    Blocks(Vec<T>),
}

impl<T> Parameters<T> {
    /// Get the number of parameters in this parameter list.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Parameters::Params(params) => params.len(),
            Parameters::Blocks(blocks) => blocks.len(),
        }
    }

    /// Check if the parameter list is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unwraps the parameter list as as slice of [Param]s.
    ///
    /// Panics if this is not a params list.
    #[inline]
    pub fn unwrap_params(&self) -> &[Param] {
        match self {
            Parameters::Params(params) => params.as_slice(),
            Parameters::Blocks(_) => panic!("expected params, got blocks"),
        }
    }

    /// Unwraps the parameter list as a slice of blocks.
    ///
    /// Panics if this is not a block list.
    #[inline]
    pub fn unwrap_blocks(&self) -> &[T] {
        match self {
            Parameters::Params(_) => panic!("expected params, got blocks"),
            Parameters::Blocks(blocks) => blocks.as_slice(),
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
        target: &'a SwitchTarget,
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
