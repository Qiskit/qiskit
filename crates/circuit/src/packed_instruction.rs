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

use crate::circuit_data::CircuitData;
use crate::imports::{
    BARRIER, BOX_OP, BREAK_LOOP_OP, CONTINUE_LOOP_OP, DELAY, FOR_LOOP_OP, IF_ELSE_OP, MEASURE,
    PAULI_PRODUCT_MEASUREMENT, RESET, SWITCH_CASE_OP, UNITARY_GATE, WHILE_LOOP_OP,
    get_std_gate_class,
};
use crate::instruction::Parameters;
use crate::interner::Interned;
use crate::operations::{
    ControlFlow, ControlFlowInstruction, Operation, OperationRef, Param, PauliProductMeasurement,
    PyGate, PyInstruction, PyOperation, PythonOperation, StandardGate, StandardInstruction,
    UnitaryGate,
};
use crate::{Block, Clbit, Qubit};
use hashbrown::HashMap;
use nalgebra::Matrix2;
use ndarray::Array2;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use smallvec::SmallVec;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

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
    PauliProductMeasurement = 6,
    ControlFlow = 7,
}

unsafe impl ::bytemuck::CheckedBitPattern for PackedOperationType {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 8
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
///     PauliProductMeasurement(Box<PauliProductMeasurement>),
///     ControlFlow(Box<ControlFlowInstruction>),
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
    use crate::operations::{
        ControlFlowInstruction, PauliProductMeasurement, PyGate, PyInstruction, PyOperation,
        UnitaryGate,
    };
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
    impl_packable_pointer!(
        PauliProductMeasurement,
        PackedOperationType::PauliProductMeasurement
    );
    impl_packable_pointer!(ControlFlowInstruction, PackedOperationType::ControlFlow);
}

impl PackedOperation {
    const DISCRIMINANT_MASK: u64 = 0b111;

    #[inline]
    fn discriminant(&self) -> PackedOperationType {
        bytemuck::checked::cast((self.0 & Self::DISCRIMINANT_MASK) as u8)
    }

    /// Get the contained `ControlFlowInstruction`, if any.
    pub fn control_flow(&self) -> &ControlFlowInstruction {
        self.try_into()
            .expect("the caller is responsible for knowing the correct type")
    }

    /// Get the contained `ControlFlowInstruction`.
    ///
    /// **Panics** if this `PackedOperation` doesn't contain a `ControlFlowInstruction`; see
    /// `try_control_flow`.
    pub fn try_control_flow(&self) -> Option<&ControlFlowInstruction> {
        self.try_into().ok()
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
    pub fn view(&self) -> OperationRef<'_> {
        match self.discriminant() {
            PackedOperationType::ControlFlow => OperationRef::ControlFlow(self.try_into().unwrap()),
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
            PackedOperationType::PauliProductMeasurement => {
                OperationRef::PauliProductMeasurement(self.try_into().unwrap())
            }
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

    /// Construct a new `PackedOperation` from an owned heap-allocated `UnitaryGate`.
    pub fn from_unitary(unitary: Box<UnitaryGate>) -> Self {
        unitary.into()
    }

    /// Construct a new `PackedOperation` from an owned heap-allocated `ControlFlowInstruction`.
    #[inline]
    pub fn from_control_flow(control_flow: Box<ControlFlowInstruction>) -> Self {
        control_flow.into()
    }

    /// Construct a new `PackedOperation` from an owned heap-allocated `PauliProductMeasurement`.
    #[inline]
    pub fn from_ppm(ppm: Box<PauliProductMeasurement>) -> Self {
        ppm.into()
    }

    /// Check equality of the operation, including Python-space checks, if appropriate.
    pub fn py_eq(&self, py: Python, other: &PackedOperation) -> PyResult<bool> {
        match (self.view(), other.view()) {
            (OperationRef::ControlFlow(left), OperationRef::ControlFlow(right)) => {
                left.py_eq(py, right)
            }
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
            (
                OperationRef::PauliProductMeasurement(left),
                OperationRef::PauliProductMeasurement(right),
            ) => Ok(left == right),
            _ => Ok(false),
        }
    }

    /// Whether the Python class that we would use to represent the inner `Operation` object in
    /// Python space would be an instance of the given Python type.  This does not construct the
    /// Python-space `Operator` instance if it can be avoided (i.e. for standard gates).
    pub fn py_op_is_instance(&self, py_type: &Bound<PyType>) -> PyResult<bool> {
        let py = py_type.py();
        let py_op = match self.view() {
            OperationRef::ControlFlow(control_flow) => {
                return match &control_flow.control_flow {
                    ControlFlow::Box { .. } => {
                        BOX_OP.get_bound(py).cast::<PyType>()?.is_subclass(py_type)
                    }
                    ControlFlow::BreakLoop => BREAK_LOOP_OP
                        .get_bound(py)
                        .cast::<PyType>()?
                        .is_subclass(py_type),
                    ControlFlow::ContinueLoop => CONTINUE_LOOP_OP
                        .get_bound(py)
                        .cast::<PyType>()?
                        .is_subclass(py_type),
                    ControlFlow::ForLoop { .. } => FOR_LOOP_OP
                        .get_bound(py)
                        .cast::<PyType>()?
                        .is_subclass(py_type),
                    ControlFlow::IfElse { .. } => IF_ELSE_OP
                        .get_bound(py)
                        .cast::<PyType>()?
                        .is_subclass(py_type),
                    ControlFlow::Switch { .. } => SWITCH_CASE_OP
                        .get_bound(py)
                        .cast::<PyType>()?
                        .is_subclass(py_type),
                    ControlFlow::While { .. } => WHILE_LOOP_OP
                        .get_bound(py)
                        .cast::<PyType>()?
                        .is_subclass(py_type),
                };
            }
            OperationRef::StandardGate(standard) => {
                return get_std_gate_class(py, standard)?
                    .bind(py)
                    .cast::<PyType>()?
                    .is_subclass(py_type);
            }
            OperationRef::StandardInstruction(standard) => {
                return match standard {
                    StandardInstruction::Barrier(_) => {
                        BARRIER.get_bound(py).cast::<PyType>()?.is_subclass(py_type)
                    }
                    StandardInstruction::Delay(_) => {
                        DELAY.get_bound(py).cast::<PyType>()?.is_subclass(py_type)
                    }
                    StandardInstruction::Measure => {
                        MEASURE.get_bound(py).cast::<PyType>()?.is_subclass(py_type)
                    }
                    StandardInstruction::Reset => {
                        RESET.get_bound(py).cast::<PyType>()?.is_subclass(py_type)
                    }
                };
            }
            OperationRef::Gate(gate) => gate.gate.bind(py),
            OperationRef::Instruction(instruction) => instruction.instruction.bind(py),
            OperationRef::Operation(operation) => operation.operation.bind(py),
            OperationRef::Unitary(_) => {
                return UNITARY_GATE
                    .get_bound(py)
                    .cast::<PyType>()?
                    .is_subclass(py_type);
            }
            OperationRef::PauliProductMeasurement(_) => {
                return PAULI_PRODUCT_MEASUREMENT
                    .get_bound(py)
                    .cast::<PyType>()?
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
            OperationRef::ControlFlow(control_flow) => control_flow.name(),
            OperationRef::StandardGate(ref standard) => standard.name(),
            OperationRef::StandardInstruction(ref instruction) => instruction.name(),
            OperationRef::Gate(gate) => gate.name(),
            OperationRef::Instruction(instruction) => instruction.name(),
            OperationRef::Operation(operation) => operation.name(),
            OperationRef::Unitary(unitary) => unitary.name(),
            OperationRef::PauliProductMeasurement(ppm) => ppm.name(),
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
    fn directive(&self) -> bool {
        self.view().directive()
    }
}

impl Clone for PackedOperation {
    fn clone(&self) -> Self {
        match self.view() {
            OperationRef::ControlFlow(control_flow) => {
                Self::from_control_flow(Box::new(control_flow.clone()))
            }
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
            OperationRef::PauliProductMeasurement(ppm) => Self::from_ppm(Box::new(ppm.clone())),
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
            PackedOperationType::PauliProductMeasurement => {
                PauliProductMeasurement::drop_packed(self)
            }
            PackedOperationType::ControlFlow => ControlFlowInstruction::drop_packed(self),
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
    pub params: Option<Box<Parameters<Block>>>,
    pub label: Option<Box<String>>,

    #[cfg(feature = "cache_pygates")]
    /// This is hidden in a `OnceLock` because it's just an on-demand cache; we don't create this
    /// unless asked for it.  A `OnceLock` of a non-null pointer type (like `Py<T>`) is the same
    /// size as a pointer and there are no runtime checks on access beyond the initialisation check,
    /// which is a simple null-pointer check.
    ///
    /// WARNING: remember that `OnceLock`'s `get_or_init` method is no-reentrant, so the initialiser
    /// must not yield the GIL to Python space.  We avoid using `PyOnceLock` here because it
    /// requires the GIL to even `get` (of course!), which makes implementing `Clone` hard for us.
    /// We can revisit once we're on PyO3 0.22+ and have been able to disable its `py-clone`
    /// feature.
    pub py_op: OnceLock<Py<PyAny>>,
}

impl PackedInstruction {
    /// Pack a [StandardGate] into a complete instruction.
    pub fn from_standard_gate(
        gate: StandardGate,
        params: Option<Box<SmallVec<[Param; 3]>>>,
        qubits: Interned<[Qubit]>,
    ) -> Self {
        Self {
            op: gate.into(),
            qubits,
            clbits: Default::default(),
            params: params.map(|params| Box::new(Parameters::Params(*params))),
            label: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        }
    }

    /// Pack a [ControlFlowInstruction] operation with blocks into a complete instruction.
    pub fn from_control_flow(
        control_flow: ControlFlowInstruction,
        blocks: Vec<Block>,
        qubits: Interned<[Qubit]>,
        clbits: Interned<[Clbit]>,
        label: Option<String>,
    ) -> Self {
        Self {
            op: control_flow.into(),
            qubits,
            clbits,
            params: Some(Box::new(Parameters::Blocks(blocks))),
            label: label.map(Box::new),
            #[cfg(feature = "cache_pygates")]
            py_op: Default::default(),
        }
    }

    /// Get a slice view onto the contained parameters.
    #[inline]
    pub fn params_view(&self) -> &[Param] {
        self.params
            .as_deref()
            .and_then(|p| match p {
                Parameters::Params(p) => Some(p.as_slice()),
                Parameters::Blocks(_) => None,
            })
            .unwrap_or_default()
    }

    /// Get a mutable slice view onto the contained parameters.
    #[inline]
    pub fn params_mut(&mut self) -> &mut [Param] {
        self.params
            .as_deref_mut()
            .and_then(|p| match p {
                Parameters::Params(p) => Some(p.as_mut_slice()),
                Parameters::Blocks(_) => None,
            })
            .unwrap_or_default()
    }

    /// Get a slice view onto the contained blocks.
    #[inline]
    pub fn blocks_view(&self) -> &[Block] {
        self.params
            .as_deref()
            .and_then(|p| match p {
                Parameters::Blocks(b) => Some(b.as_slice()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Get a clone of this instruction with the blocks (if any) remapped to new indices.
    ///
    /// You probably don't want to use this directly; use `BlockMapper::map_instruction` instead,
    /// which remembers the blocks it's already encountered.
    pub fn map_blocks(&self, mut map: impl FnMut(Block) -> Block) -> Self {
        let params = match self.params.as_deref() {
            Some(Parameters::Params(_)) | None => self.params.clone(),
            Some(Parameters::Blocks(blocks)) => Some(Box::new(Parameters::Blocks(
                blocks.iter().map(|b| map(*b)).collect(),
            ))),
        };
        Self {
            op: self.op.clone(),
            qubits: self.qubits,
            clbits: self.clbits,
            params,
            label: self.label.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: self.py_op.clone(),
        }
    }

    /// Does this instruction contain any compile-time symbolic `ParameterExpression`s?
    pub fn is_parameterized(&self) -> bool {
        self.params.as_deref().is_some_and(|p| match p {
            Parameters::Params(p) => p.iter().any(|x| matches!(x, Param::ParameterExpression(_))),
            Parameters::Blocks(_) => false,
        })
    }

    pub fn py_deepcopy_inplace<'py>(
        &mut self,
        py: Python<'py>,
        memo: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        match self.op.view() {
            OperationRef::Gate(gate) => self.op = gate.py_deepcopy(py, memo)?.into(),
            OperationRef::Instruction(inst) => self.op = inst.py_deepcopy(py, memo)?.into(),
            OperationRef::Operation(op) => self.op = op.py_deepcopy(py, memo)?.into(),
            _ => (),
        };
        if let Some(Parameters::Params(params)) = self.params.as_deref_mut() {
            for param in params {
                *param = param.py_deepcopy(py, memo)?;
            }
        }
        #[cfg(feature = "cache_pygates")]
        self.py_op.take();

        Ok(())
    }

    pub fn try_matrix(&self) -> Option<Array2<Complex64>> {
        match self.op.view() {
            OperationRef::StandardGate(g) => g.matrix(self.params_view()),
            OperationRef::Gate(g) => g.matrix(),
            OperationRef::Unitary(u) => u.matrix(),
            _ => None,
        }
    }

    /// Returns a static matrix for 1-qubit gates. Will return `None` when the gate is not 1-qubit.
    #[inline]
    pub fn try_matrix_as_static_1q(&self) -> Option<[[Complex64; 2]; 2]> {
        match self.op.view() {
            OperationRef::StandardGate(standard) => {
                standard.matrix_as_static_1q(self.params_view())
            }
            OperationRef::Gate(gate) => gate.matrix_as_static_1q(),
            OperationRef::Unitary(unitary) => unitary.matrix_as_static_1q(),
            _ => None,
        }
    }

    pub fn try_matrix_as_nalgebra_1q(&self) -> Option<Matrix2<Complex64>> {
        match self.op.view() {
            OperationRef::Unitary(u) => u.matrix_as_nalgebra_1q(),
            // default implementation
            _ => self
                .try_matrix_as_static_1q()
                .map(|arr| Matrix2::new(arr[0][0], arr[0][1], arr[1][0], arr[1][1])),
        }
    }

    pub fn try_definition(&self) -> Option<CircuitData> {
        match self.op.view() {
            OperationRef::StandardGate(g) => g.definition(self.params_view()),
            OperationRef::Gate(g) => g.definition(),
            OperationRef::Instruction(i) => i.definition(),
            _ => None,
        }
    }
}

/// Helper "memory" struct for mapping `PackedInstruction`s to have different blocks in another
/// circuit.
///
/// Typically you construct this, then repeatedly call `map_instruction`.
#[derive(Clone, Debug, Default)]
pub struct BlockMapper(HashMap<Block, Block>);
impl BlockMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a clone of a `PackedInstruction`, remapping the blocks inside to new values.
    ///
    /// This remembers any `Block`s previously seen by this struct, and only calls `add_block` the
    /// first time each block is encountered.
    pub fn map_instruction(
        &mut self,
        inst: &PackedInstruction,
        mut add_block: impl FnMut(Block) -> Block,
    ) -> PackedInstruction {
        inst.map_blocks(|b| *self.0.entry(b).or_insert_with(|| add_block(b)))
    }

    /// Get a clone of a `Parameters<Block>`, remapping the blocks inside to new values.
    ///
    /// This remembers any `Block`s previously seen by this struct, and only calls `add_block` the
    /// first time each block is encountered.
    pub fn map_params(
        &mut self,
        params: &Parameters<Block>,
        mut add_block: impl FnMut(Block) -> Block,
    ) -> Parameters<Block> {
        match params {
            Parameters::Params(_) => params.clone(),
            Parameters::Blocks(blocks) => Parameters::Blocks(
                blocks
                    .iter()
                    .cloned()
                    .map(|b| *self.0.entry(b).or_insert_with(|| add_block(b)))
                    .collect(),
            ),
        }
    }
}
