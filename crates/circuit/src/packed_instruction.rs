// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use ndarray::Array2;
use num_complex::Complex64;
use smallvec::SmallVec;

use crate::circuit_data::CircuitData;
use crate::imports::{get_std_gate_class, BARRIER, DEEPCOPY, DELAY, MEASURE, RESET, UNITARY_GATE};
use crate::interner::Interned;
use crate::operations::{
    Operation, OperationRef, Param, PyGate, PyInstruction, PyOperation, StandardGate,
    StandardInstruction, UnitaryGate,
};
use crate::{Clbit, Qubit};

/// The logical discriminant of `PackedOperation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum PackedOperationType {
    // It's important that the `StandardGate` item is 0, so that zeroing out a `PackedOperation`
    // will make it appear as a standard gate, which will never allow accidental dangling-pointer
    // dereferencing.
    StandardGate = 0,
    StandardInstruction = 1,
    PyGate = 2,
    PyInstruction = 3,
    PyOperation = 4,
    UnitaryGate = 5,
}

unsafe impl ::bytemuck::CheckedBitPattern for PackedOperationType {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 6
    }
}
unsafe impl ::bytemuck::NoUninit for PackedOperationType {}

/// A bit-packed `OperationType` enumeration.
///
/// This is logically equivalent to:
///
/// ```rust
/// enum Operation {
///     StandardGate(StandardGate),
///     StandardInstruction(StandardInstruction),
///     Gate(Box<PyGate>),
///     Instruction(Box<PyInstruction>),
///     Operation(Box<PyOperation>),
///     UnitaryGate(Box<UnitaryGate>),
/// }
/// ```
///
/// including all ownership semantics, except it bit-packs the enumeration into a `u64`.
///
/// The lowest three bits of this `u64` is always the discriminant and identifies which of the
/// above variants the field contains (and thus the layout required to decode it).
/// This works even for pointer variants (like `PyGate`) on 64-bit systems, which would normally
/// span the entire `u64`, since pointers on these systems have a natural alignment of 8 (and thus
/// their lowest three bits are always 0). This lets us store the discriminant within the address
/// and mask it out before reinterpreting it as a pointer.
///
/// The layouts for each variant are described as follows, written out as a 64-bit binary integer.
/// `x` marks padding bits with undefined values.
///
/// ```text
/// StandardGate:
/// 0b_xxxxxxxx_xxxxxxxx_xxxxxxxx_xxxxxxxx_xxxxxxxx_xxxxxxxx_xxxxxSSS_SSSSS000
///                                                               |-------||-|
///                                                                   |     |
///                      Standard gate, stored inline as a u8. -------+     +-- Discriminant.
///
/// StandardInstruction:
/// 0b_DDDDDDDD_DDDDDDDD_DDDDDDDD_DDDDDDDD_xxxxxxxx_xxxxxxxx_SSSSSSSS_xxxxx001
///    |---------------------------------|                   |------|      |-|
///                    |                                        |           |
///                    +-- An optional 32 bit immediate value.  |           |
///         Standard instruction type, stored inline as a u8. --+           +-- Discriminant.
///
///     Optional immediate value:
///     Depending on the variant of the standard instruction type, a 32 bit
///     inline value may be present. Currently, this is used to store the
///     number of qubits in a Barrier and the unit of a Delay.
///
/// Gate, Instruction, Operation:
/// 0b_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPP011
///    |-----------------------------------------------------------------||-|
///                                   |                                    |
///    The high 62 bits of the pointer.  Because of alignment, the low 3   |   Discriminant of the
///    bits of the full 64 bits are guaranteed to be zero so we can        +-- enumeration.  This
///    retrieve the "full" pointer by taking the whole `u64` and zeroing       is 0b011, which means
///    the low 3 bits, letting us store the discriminant in there at other     that this points to
///    times.                                                                  a `PyInstruction`.
/// ```
///
/// # Construction
///
/// From Rust space, build this type using one of the `from_*` methods, depending on which
/// implementer of `Operation` you have.  `StandardGate` and `StandardInstruction` have
/// implementations of `Into` for this.
///
/// From Python space, use the supplied `FromPyObject`.
///
/// # Safety
///
/// `PackedOperation` asserts ownership over its contained pointer (if it contains one).  This
/// has the following requirements:
///
/// * The pointer must be managed by a `Box` using the global allocator.
/// * The pointed-to data must match the type of the discriminant used to store it.
/// * `PackedOperation` must take care to forward implementations of `Clone` and `Drop` to the
///   contained pointer.
#[derive(Debug)]
pub struct PackedOperation(u64);

/// A private module to encapsulate the encoding of [StandardGate].
mod standard_gate {
    use crate::operations::StandardGate;
    use crate::packed_instruction::{PackedOperation, PackedOperationType};
    use bitfield_struct::bitfield;

    /// The packed layout of a standard gate, as a bitfield.
    ///
    /// NOTE: this _looks_ like a named struct, but the `bitfield` attribute macro
    /// turns it into a transparent wrapper around a `u64`.
    #[bitfield(u64)]
    struct StandardGateBits {
        #[bits(3)]
        discriminant: u8,
        #[bits(8)]
        standard_gate: u8,
        #[bits(53)]
        _pad1: u64,
    }

    impl From<StandardGate> for PackedOperation {
        fn from(value: StandardGate) -> Self {
            Self(
                StandardGateBits::new()
                    .with_discriminant(bytemuck::cast(PackedOperationType::StandardGate))
                    .with_standard_gate(bytemuck::cast(value))
                    .into_bits(),
            )
        }
    }

    impl TryFrom<&PackedOperation> for StandardGate {
        type Error = &'static str;

        fn try_from(value: &PackedOperation) -> Result<Self, Self::Error> {
            match value.discriminant() {
                PackedOperationType::StandardGate => {
                    let bits = StandardGateBits::from(value.0);
                    Ok(bytemuck::checked::cast(bits.standard_gate()))
                }
                _ => Err("not a standard gate!"),
            }
        }
    }
}

/// A private module to encapsulate the encoding of [StandardInstruction].
mod standard_instruction {
    use crate::operations::{StandardInstruction, StandardInstructionType};
    use crate::packed_instruction::{PackedOperation, PackedOperationType};
    use bitfield_struct::bitfield;

    /// The packed layout of a standard instruction, as a bitfield.
    ///
    /// NOTE: this _looks_ like a named struct, but the `bitfield` attribute macro
    /// turns it into a transparent wrapper around a `u64`.
    #[bitfield(u64)]
    struct StandardInstructionBits {
        #[bits(3)]
        discriminant: u8,
        #[bits(5)]
        _pad0: u8,
        #[bits(8)]
        standard_instruction: u8,
        #[bits(16)]
        _pad1: u32,
        #[bits(32)]
        payload: u32,
    }

    impl From<StandardInstruction> for PackedOperation {
        fn from(value: StandardInstruction) -> Self {
            let packed = StandardInstructionBits::new()
                .with_discriminant(bytemuck::cast(PackedOperationType::StandardInstruction));
            Self(
                match value {
                    StandardInstruction::Barrier(bits) => packed
                        .with_standard_instruction(bytemuck::cast(StandardInstructionType::Barrier))
                        .with_payload(bits),
                    StandardInstruction::Delay(unit) => packed
                        .with_standard_instruction(bytemuck::cast(StandardInstructionType::Delay))
                        .with_payload(unit as u32),
                    StandardInstruction::Measure => packed.with_standard_instruction(
                        bytemuck::cast(StandardInstructionType::Measure),
                    ),
                    StandardInstruction::Reset => packed
                        .with_standard_instruction(bytemuck::cast(StandardInstructionType::Reset)),
                }
                .into_bits(),
            )
        }
    }

    impl TryFrom<&PackedOperation> for StandardInstruction {
        type Error = &'static str;

        fn try_from(value: &PackedOperation) -> Result<Self, Self::Error> {
            match value.discriminant() {
                PackedOperationType::StandardInstruction => {
                    let bits = StandardInstructionBits::from_bits(value.0);
                    let ty: StandardInstructionType =
                        bytemuck::checked::cast(bits.standard_instruction());
                    Ok(match ty {
                        StandardInstructionType::Barrier => {
                            StandardInstruction::Barrier(bits.payload())
                        }
                        StandardInstructionType::Delay => StandardInstruction::Delay(
                            bytemuck::checked::cast(bits.payload() as u8),
                        ),
                        StandardInstructionType::Measure => StandardInstruction::Measure,
                        StandardInstructionType::Reset => StandardInstruction::Reset,
                    })
                }
                _ => Err("not a standard instruction!"),
            }
        }
    }
}

/// A private module to encapsulate the encoding of pointer types.
mod pointer {
    use crate::operations::{PyGate, PyInstruction, PyOperation, UnitaryGate};
    use crate::packed_instruction::{PackedOperation, PackedOperationType};
    use std::ptr::NonNull;

    const POINTER_MASK: u64 = !PackedOperation::DISCRIMINANT_MASK;

    /// Used to associate a supported pointer type (e.g. PyGate) with a [PackedOperationType] and
    /// a drop implementation.
    ///
    /// Note: this is public only within this file for use by [PackedOperation]'s [Drop] impl.
    pub trait PackablePointer: Sized {
        const OPERATION_TYPE: PackedOperationType;

        /// Drops `op` as this pointer type.
        fn drop_packed(op: &mut PackedOperation) {
            // This should only ever be called from PackedOperation's Drop impl after the
            // operation's type has already been validated, but this is defensive just
            // to 100% ensure that our `Drop` implementation doesn't panic.
            let Some(pointer) = try_pointer::<Self>(op) else {
                return;
            };

            // SAFETY: `PackedOperation` asserts ownership over its contents, and the contained
            // pointer can only be null if we were already dropped.  We set our discriminant to mark
            // ourselves as plain old data immediately just as a defensive measure.
            let boxed = unsafe { Box::from_raw(pointer.as_ptr()) };
            op.0 = PackedOperationType::StandardGate as u64;
            ::std::mem::drop(boxed);
        }
    }

    #[inline]
    fn try_pointer<T: PackablePointer>(value: &PackedOperation) -> Option<NonNull<T>> {
        if value.discriminant() == T::OPERATION_TYPE {
            let ptr = (value.0 & POINTER_MASK) as *mut ();
            // SAFETY: `PackedOperation` can only be constructed from a pointer via `Box`, which
            // is always non-null (except in the case that we're partway through a `Drop`).
            Some(unsafe { NonNull::new_unchecked(ptr) }.cast::<T>())
        } else {
            None
        }
    }

    macro_rules! impl_packable_pointer {
        ($type:ty, $operation_type:expr) => {
            impl PackablePointer for $type {
                const OPERATION_TYPE: PackedOperationType = $operation_type;
            }

            impl From<$type> for PackedOperation {
                #[inline]
                fn from(value: $type) -> Self {
                    Box::new(value).into()
                }
            }

            // Supports reference conversion (e.g. &PackedOperation => &PyGate).
            impl<'a> TryFrom<&'a PackedOperation> for &'a $type {
                type Error = &'static str;

                fn try_from(value: &'a PackedOperation) -> Result<Self, Self::Error> {
                    try_pointer(value)
                        .map(|ptr| unsafe { ptr.as_ref() })
                        .ok_or(concat!("not a(n) ", stringify!($type), " pointer!"))
                }
            }

            impl From<Box<$type>> for PackedOperation {
                fn from(value: Box<$type>) -> Self {
                    let discriminant = $operation_type as u64;
                    let ptr = NonNull::from(Box::leak(value)).cast::<()>();
                    let addr = ptr.as_ptr() as u64;
                    assert!((addr & PackedOperation::DISCRIMINANT_MASK == 0));
                    Self(discriminant | addr)
                }
            }
        };
    }

    impl_packable_pointer!(PyGate, PackedOperationType::PyGate);
    impl_packable_pointer!(PyInstruction, PackedOperationType::PyInstruction);
    impl_packable_pointer!(PyOperation, PackedOperationType::PyOperation);
    impl_packable_pointer!(UnitaryGate, PackedOperationType::UnitaryGate);
}

impl PackedOperation {
    const DISCRIMINANT_MASK: u64 = 0b111;

    #[inline]
    fn discriminant(&self) -> PackedOperationType {
        bytemuck::checked::cast((self.0 & Self::DISCRIMINANT_MASK) as u8)
    }

    /// Get the contained `StandardGate`.
    ///
    /// **Panics** if this `PackedOperation` doesn't contain a `StandardGate`; see
    /// `try_standard_gate`.
    #[inline]
    pub fn standard_gate(&self) -> StandardGate {
        self.try_into()
            .expect("the caller is responsible for knowing the correct type")
    }

    /// Get the contained `StandardGate`, if any.
    #[inline]
    pub fn try_standard_gate(&self) -> Option<StandardGate> {
        self.try_into().ok()
    }

    /// Get the contained `StandardInstruction`.
    ///
    /// **Panics** if this `PackedOperation` doesn't contain a `StandardInstruction`; see
    /// `try_standard_instruction`.
    #[inline]
    pub fn standard_instruction(&self) -> StandardInstruction {
        self.try_into()
            .expect("the caller is responsible for knowing the correct type")
    }

    /// Get the contained `StandardInstruction`, if any.
    #[inline]
    pub fn try_standard_instruction(&self) -> Option<StandardInstruction> {
        self.try_into().ok()
    }

    /// Get a safe view onto the packed data within, without assuming ownership.
    #[inline]
    pub fn view(&self) -> OperationRef {
        match self.discriminant() {
            PackedOperationType::StandardGate => OperationRef::StandardGate(self.standard_gate()),
            PackedOperationType::StandardInstruction => {
                OperationRef::StandardInstruction(self.standard_instruction())
            }
            PackedOperationType::PyGate => OperationRef::Gate(self.try_into().unwrap()),
            PackedOperationType::PyInstruction => {
                OperationRef::Instruction(self.try_into().unwrap())
            }
            PackedOperationType::PyOperation => OperationRef::Operation(self.try_into().unwrap()),
            PackedOperationType::UnitaryGate => OperationRef::Unitary(self.try_into().unwrap()),
        }
    }

    /// Create a `PackedOperation` from a `StandardGate`.
    #[inline]
    pub fn from_standard_gate(standard: StandardGate) -> Self {
        standard.into()
    }

    /// Create a `PackedOperation` from a `StandardInstruction`.
    #[inline]
    pub fn from_standard_instruction(instruction: StandardInstruction) -> Self {
        instruction.into()
    }

    /// Construct a new `PackedOperation` from an owned heap-allocated `PyGate`.
    #[inline]
    pub fn from_gate(gate: Box<PyGate>) -> Self {
        gate.into()
    }

    /// Construct a new `PackedOperation` from an owned heap-allocated `PyInstruction`.
    #[inline]
    pub fn from_instruction(instruction: Box<PyInstruction>) -> Self {
        instruction.into()
    }

    /// Construct a new `PackedOperation` from an owned heap-allocated `PyOperation`.
    #[inline]
    pub fn from_operation(operation: Box<PyOperation>) -> Self {
        operation.into()
    }

    pub fn from_unitary(unitary: Box<UnitaryGate>) -> Self {
        unitary.into()
    }

    /// Check equality of the operation, including Python-space checks, if appropriate.
    pub fn py_eq(&self, py: Python, other: &PackedOperation) -> PyResult<bool> {
        match (self.view(), other.view()) {
            (OperationRef::StandardGate(left), OperationRef::StandardGate(right)) => {
                Ok(left == right)
            }
            (OperationRef::StandardInstruction(left), OperationRef::StandardInstruction(right)) => {
                Ok(left == right)
            }
            (OperationRef::Gate(left), OperationRef::Gate(right)) => {
                left.gate.bind(py).eq(&right.gate)
            }
            (OperationRef::Instruction(left), OperationRef::Instruction(right)) => {
                left.instruction.bind(py).eq(&right.instruction)
            }
            (OperationRef::Operation(left), OperationRef::Operation(right)) => {
                left.operation.bind(py).eq(&right.operation)
            }
            (OperationRef::Unitary(left), OperationRef::Unitary(right)) => Ok(left == right),
            _ => Ok(false),
        }
    }

    /// Copy this operation, including a Python-space deep copy, if required.
    pub fn py_deepcopy<'py>(
        &self,
        py: Python<'py>,
        memo: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Self> {
        let deepcopy = DEEPCOPY.get_bound(py);
        match self.view() {
            OperationRef::StandardGate(standard) => Ok(standard.into()),
            OperationRef::StandardInstruction(instruction) => {
                Ok(Self::from_standard_instruction(instruction))
            }
            OperationRef::Gate(gate) => Ok(PyGate {
                gate: deepcopy.call1((&gate.gate, memo))?.unbind(),
                qubits: gate.qubits,
                clbits: gate.clbits,
                params: gate.params,
                op_name: gate.op_name.clone(),
            }
            .into()),
            OperationRef::Instruction(instruction) => Ok(PyInstruction {
                instruction: deepcopy.call1((&instruction.instruction, memo))?.unbind(),
                qubits: instruction.qubits,
                clbits: instruction.clbits,
                params: instruction.params,
                control_flow: instruction.control_flow,
                op_name: instruction.op_name.clone(),
            }
            .into()),
            OperationRef::Operation(operation) => Ok(PyOperation {
                operation: deepcopy.call1((&operation.operation, memo))?.unbind(),
                qubits: operation.qubits,
                clbits: operation.clbits,
                params: operation.params,
                op_name: operation.op_name.clone(),
            }
            .into()),
            OperationRef::Unitary(unitary) => Ok(unitary.clone().into()),
        }
    }

    /// Copy this operation, including a Python-space call to `copy` on the `Operation` subclass, if
    /// any.
    pub fn py_copy(&self, py: Python) -> PyResult<Self> {
        let copy_attr = intern!(py, "copy");
        match self.view() {
            OperationRef::StandardGate(standard) => Ok(standard.into()),
            OperationRef::StandardInstruction(instruction) => {
                Ok(Self::from_standard_instruction(instruction))
            }
            OperationRef::Gate(gate) => Ok(Box::new(PyGate {
                gate: gate.gate.call_method0(py, copy_attr)?,
                qubits: gate.qubits,
                clbits: gate.clbits,
                params: gate.params,
                op_name: gate.op_name.clone(),
            })
            .into()),
            OperationRef::Instruction(instruction) => Ok(Box::new(PyInstruction {
                instruction: instruction.instruction.call_method0(py, copy_attr)?,
                qubits: instruction.qubits,
                clbits: instruction.clbits,
                params: instruction.params,
                control_flow: instruction.control_flow,
                op_name: instruction.op_name.clone(),
            })
            .into()),
            OperationRef::Operation(operation) => Ok(Box::new(PyOperation {
                operation: operation.operation.call_method0(py, copy_attr)?,
                qubits: operation.qubits,
                clbits: operation.clbits,
                params: operation.params,
                op_name: operation.op_name.clone(),
            })
            .into()),
            OperationRef::Unitary(unitary) => Ok(unitary.clone().into()),
        }
    }

    /// Whether the Python class that we would use to represent the inner `Operation` object in
    /// Python space would be an instance of the given Python type.  This does not construct the
    /// Python-space `Operator` instance if it can be avoided (i.e. for standard gates).
    pub fn py_op_is_instance(&self, py_type: &Bound<PyType>) -> PyResult<bool> {
        let py = py_type.py();
        let py_op = match self.view() {
            OperationRef::StandardGate(standard) => {
                return get_std_gate_class(py, standard)?
                    .bind(py)
                    .downcast::<PyType>()?
                    .is_subclass(py_type)
            }
            OperationRef::StandardInstruction(standard) => {
                return match standard {
                    StandardInstruction::Barrier(_) => BARRIER
                        .get_bound(py)
                        .downcast::<PyType>()?
                        .is_subclass(py_type),
                    StandardInstruction::Delay(_) => DELAY
                        .get_bound(py)
                        .downcast::<PyType>()?
                        .is_subclass(py_type),
                    StandardInstruction::Measure => MEASURE
                        .get_bound(py)
                        .downcast::<PyType>()?
                        .is_subclass(py_type),
                    StandardInstruction::Reset => RESET
                        .get_bound(py)
                        .downcast::<PyType>()?
                        .is_subclass(py_type),
                }
            }
            OperationRef::Gate(gate) => gate.gate.bind(py),
            OperationRef::Instruction(instruction) => instruction.instruction.bind(py),
            OperationRef::Operation(operation) => operation.operation.bind(py),
            OperationRef::Unitary(_) => {
                return UNITARY_GATE
                    .get_bound(py)
                    .downcast::<PyType>()?
                    .is_subclass(py_type);
            }
        };
        py_op.is_instance(py_type)
    }
}

impl Operation for PackedOperation {
    fn name(&self) -> &str {
        let view = self.view();
        let name = match view {
            OperationRef::StandardGate(ref standard) => standard.name(),
            OperationRef::StandardInstruction(ref instruction) => instruction.name(),
            OperationRef::Gate(gate) => gate.name(),
            OperationRef::Instruction(instruction) => instruction.name(),
            OperationRef::Operation(operation) => operation.name(),
            OperationRef::Unitary(unitary) => unitary.name(),
        };
        // SAFETY: all of the inner parts of the view are owned by `self`, so it's valid for us to
        // forcibly reborrowing up to our own lifetime. We avoid using `<OperationRef as Operation>`
        // just to avoid a further _potential_ unsafeness, were its implementation to start doing
        // something weird with the lifetimes.  `str::from_utf8_unchecked` and
        // `slice::from_raw_parts` are both trivially safe because they're being called on immediate
        // values from a validated `str`.
        unsafe {
            ::std::str::from_utf8_unchecked(::std::slice::from_raw_parts(name.as_ptr(), name.len()))
        }
    }
    #[inline]
    fn num_qubits(&self) -> u32 {
        self.view().num_qubits()
    }
    #[inline]
    fn num_clbits(&self) -> u32 {
        self.view().num_clbits()
    }
    #[inline]
    fn num_params(&self) -> u32 {
        self.view().num_params()
    }
    #[inline]
    fn control_flow(&self) -> bool {
        self.view().control_flow()
    }
    #[inline]
    fn blocks(&self) -> Vec<CircuitData> {
        self.view().blocks()
    }
    #[inline]
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        self.view().matrix(params)
    }
    #[inline]
    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        self.view().definition(params)
    }
    #[inline]
    fn standard_gate(&self) -> Option<StandardGate> {
        self.view().standard_gate()
    }
    #[inline]
    fn directive(&self) -> bool {
        self.view().directive()
    }
}

impl Clone for PackedOperation {
    fn clone(&self) -> Self {
        match self.view() {
            OperationRef::StandardGate(standard) => Self::from_standard_gate(standard),
            OperationRef::StandardInstruction(instruction) => {
                Self::from_standard_instruction(instruction)
            }
            OperationRef::Gate(gate) => Self::from_gate(Box::new(gate.to_owned())),
            OperationRef::Instruction(instruction) => {
                Self::from_instruction(Box::new(instruction.to_owned()))
            }
            OperationRef::Operation(operation) => {
                Self::from_operation(Box::new(operation.to_owned()))
            }
            OperationRef::Unitary(unitary) => Self::from_unitary(Box::new(unitary.clone())),
        }
    }
}

impl Drop for PackedOperation {
    fn drop(&mut self) {
        use crate::packed_instruction::pointer::PackablePointer;
        match self.discriminant() {
            PackedOperationType::StandardGate | PackedOperationType::StandardInstruction => (),
            PackedOperationType::PyGate => PyGate::drop_packed(self),
            PackedOperationType::PyInstruction => PyInstruction::drop_packed(self),
            PackedOperationType::PyOperation => PyOperation::drop_packed(self),
            PackedOperationType::UnitaryGate => UnitaryGate::drop_packed(self),
        }
    }
}

/// The data-at-rest compressed storage format for a circuit instruction.
///
/// Much of the actual data of a `PackedInstruction` is stored in the `CircuitData` (or
/// DAG-equivalent) context objects, and the `PackedInstruction` itself just contains handles to
/// that data.  Components of the `PackedInstruction` can be unpacked individually by passing the
/// `CircuitData` object to the relevant getter method.  Many `PackedInstruction`s may contain
/// handles to the same data within a `CircuitData` objects; we are re-using what we can.
///
/// A `PackedInstruction` in general cannot be safely mutated outside the context of its
/// `CircuitData`, because the majority of the data is not actually stored here.
#[derive(Clone, Debug)]
pub struct PackedInstruction {
    pub op: PackedOperation,
    /// The index under which the interner has stored `qubits`.
    pub qubits: Interned<[Qubit]>,
    /// The index under which the interner has stored `clbits`.
    pub clbits: Interned<[Clbit]>,
    pub params: Option<Box<SmallVec<[Param; 3]>>>,
    pub label: Option<Box<String>>,

    #[cfg(feature = "cache_pygates")]
    /// This is hidden in a `OnceLock` because it's just an on-demand cache; we don't create this
    /// unless asked for it.  A `OnceLock` of a non-null pointer type (like `Py<T>`) is the same
    /// size as a pointer and there are no runtime checks on access beyond the initialisation check,
    /// which is a simple null-pointer check.
    ///
    /// WARNING: remember that `OnceLock`'s `get_or_init` method is no-reentrant, so the initialiser
    /// must not yield the GIL to Python space.  We avoid using `GILOnceCell` here because it
    /// requires the GIL to even `get` (of course!), which makes implementing `Clone` hard for us.
    /// We can revisit once we're on PyO3 0.22+ and have been able to disable its `py-clone`
    /// feature.
    pub py_op: OnceLock<Py<PyAny>>,
}

impl PackedInstruction {
    /// Access the standard gate in this `PackedInstruction`, if it is one.  If the instruction
    /// refers to a Python-space object, `None` is returned.
    #[inline]
    pub fn standard_gate(&self) -> Option<StandardGate> {
        self.op.try_standard_gate()
    }

    /// Get a slice view onto the contained parameters.
    #[inline]
    pub fn params_view(&self) -> &[Param] {
        self.params
            .as_deref()
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
    }

    /// Get a mutable slice view onto the contained parameters.
    #[inline]
    pub fn params_mut(&mut self) -> &mut [Param] {
        self.params
            .as_deref_mut()
            .map(SmallVec::as_mut_slice)
            .unwrap_or(&mut [])
    }

    /// Does this instruction contain any compile-time symbolic `ParameterExpression`s?
    pub fn is_parameterized(&self) -> bool {
        self.params_view()
            .iter()
            .any(|x| matches!(x, Param::ParameterExpression(_)))
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_ref().map(|label| label.as_str())
    }

    /// Build a reference to the Python-space operation object (the `Gate`, etc) packed into this
    /// instruction.  This may construct the reference if the `PackedInstruction` is a standard
    /// gate or instruction with no already stored operation.
    ///
    /// A standard-gate or standard-instruction operation object returned by this function is
    /// disconnected from the containing circuit; updates to its parameters, label, duration, unit
    /// and condition will not be propagated back.
    pub fn unpack_py_op(&self, py: Python) -> PyResult<Py<PyAny>> {
        let unpack = || -> PyResult<Py<PyAny>> {
            match self.op.view() {
                OperationRef::StandardGate(standard) => standard.create_py_op(
                    py,
                    self.params.as_deref().map(SmallVec::as_slice),
                    self.label.as_ref().map(|x| x.as_str()),
                ),
                OperationRef::StandardInstruction(instruction) => instruction.create_py_op(
                    py,
                    self.params.as_deref().map(SmallVec::as_slice),
                    self.label.as_ref().map(|x| x.as_str()),
                ),
                OperationRef::Gate(gate) => Ok(gate.gate.clone_ref(py)),
                OperationRef::Instruction(instruction) => Ok(instruction.instruction.clone_ref(py)),
                OperationRef::Operation(operation) => Ok(operation.operation.clone_ref(py)),
                OperationRef::Unitary(unitary) => {
                    unitary.create_py_op(py, self.label.as_ref().map(|x| x.as_str()))
                }
            }
        };

        // `OnceLock::get_or_init` and the non-stabilised `get_or_try_init`, which would otherwise
        // be nice here are both non-reentrant.  This is a problem if the init yields control to the
        // Python interpreter as this one does, since that can allow CPython to freeze the thread
        // and for another to attempt the initialisation.
        #[cfg(feature = "cache_pygates")]
        {
            if let Some(ob) = self.py_op.get() {
                return Ok(ob.clone_ref(py));
            }
        }
        let out = unpack()?;
        #[cfg(feature = "cache_pygates")]
        {
            // The unpacking operation can cause a thread pause and concurrency, since it can call
            // interpreted Python code for a standard gate, so we need to take care that some other
            // Python thread might have populated the cache before we do.
            let _ = self.py_op.set(out.clone_ref(py));
        }
        Ok(out)
    }

    /// Check equality of the operation, including Python-space checks, if appropriate.
    pub fn py_op_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        match (self.op.view(), other.op.view()) {
            (OperationRef::StandardGate(left), OperationRef::StandardGate(right)) => {
                Ok(left == right)
            }
            (OperationRef::StandardInstruction(left), OperationRef::StandardInstruction(right)) => {
                Ok(left == right)
            }
            (OperationRef::Gate(left), OperationRef::Gate(right)) => {
                left.gate.bind(py).eq(&right.gate)
            }
            (OperationRef::Instruction(left), OperationRef::Instruction(right)) => {
                left.instruction.bind(py).eq(&right.instruction)
            }
            (OperationRef::Operation(left), OperationRef::Operation(right)) => {
                left.operation.bind(py).eq(&right.operation)
            }
            // Handle the case we end up with a pygate for a standard gate
            // this typically only happens if it's a ControlledGate in python
            // and we have mutable state set.
            (OperationRef::StandardGate(_left), OperationRef::Gate(right)) => {
                self.unpack_py_op(py)?.bind(py).eq(&right.gate)
            }
            (OperationRef::Gate(left), OperationRef::StandardGate(_right)) => {
                other.unpack_py_op(py)?.bind(py).eq(&left.gate)
            }
            // Handle the case we end up with a pyinstruction for a standard instruction
            (OperationRef::StandardInstruction(_left), OperationRef::Instruction(right)) => {
                self.unpack_py_op(py)?.bind(py).eq(&right.instruction)
            }
            (OperationRef::Instruction(left), OperationRef::StandardInstruction(_right)) => {
                other.unpack_py_op(py)?.bind(py).eq(&left.instruction)
            }
            _ => Ok(false),
        }
    }
}
