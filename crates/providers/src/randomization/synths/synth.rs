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

use std::any::Any;
use std::fmt::Debug;

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::Param;

use crate::randomization::flags::BitFlags;
use crate::randomization::virtual_type::VirtualType;
use crate::tensor::Dim;
use crate::{FunctionError, ProgramFunction};

/// A way to express virtual gates as a parameterized template circuit.
///
/// A template is on one factor, which is a group of qubits. Each factor gets its own copy of the
/// template, and the parameter values of every copy come from one program function of the virtual
/// gates.
///
/// The virtual types a synth supports are [`Self::virtual_types`], and the factor sizes it
/// supports are those with a [`Self::num_params`].
pub trait Synth: Any + Debug + Send + Sync {
    /// Return the name of this synth, `rzrx` for instance.
    fn name(&self) -> &str;

    /// The virtual types this synth can express.
    fn virtual_types(&self) -> BitFlags<VirtualType>;

    /// Whether this synth can express `ty`.
    fn accepts(&self, ty: VirtualType) -> bool {
        self.virtual_types().contains(ty)
    }

    /// The number of parameters of the template on `num_qubits` qubits, or `None` if this synth
    /// has no template of that size.
    fn num_params(&self, num_qubits: usize) -> Option<usize>;

    /// Append the template on `qubits` to `circuit`, with `params` as its parameters.
    ///
    /// # Panics
    ///
    /// May panic if `params.len()` is not the [`Self::num_params`] of `qubits.len()`.
    fn make_template(
        &self,
        circuit: &mut CircuitData,
        qubits: &[Qubit],
        params: &[Param],
    ) -> Result<(), CircuitDataError>;

    /// Return a program function from virtual gates of type `ty` on `num_qubits` qubits to the
    /// values of their template parameters.
    ///
    /// The function has one parameter and one result. The parameter has dtype `ty.dtype()` and
    /// shape `[...leading, ...gate]`, where `gate` is the shape of
    /// [`ty.tensor_type(num_qubits)`](VirtualType::tensor_type). The result has dtype `F64` and
    /// shape `[...leading, num_params]`. The last leading axis usually indexes the factors.
    ///
    /// Building the function can fail, for instance when a leading axis is [`Dim::Bounded`].
    ///
    /// # Panics
    ///
    /// May panic if this synth does not accept `ty`, or if `ty` or this synth has no gate on
    /// `num_qubits` qubits.
    fn program_function(
        &self,
        ty: VirtualType,
        num_qubits: usize,
        leading: &[Dim],
    ) -> Result<ProgramFunction, FunctionError>;
}
