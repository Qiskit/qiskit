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
use std::cell::OnceCell;
use std::fmt;
use std::ptr::NonNull;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use ndarray::Array2;
use num_complex::Complex64;
use smallvec::SmallVec;

use crate::circuit_data::CircuitData;
use crate::circuit_instruction::ExtraInstructionAttributes;
use crate::imports::{get_std_gate_class, BARRIER, DEEPCOPY, DELAY, MEASURE, RESET};
use crate::interner::Interned;
use crate::operations::{
    DelayUnit, Operation, OperationRef, Param, PyGate, PyInstruction, PyOperation, StandardGate,
    StandardInstruction, StandardInstructionType,
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
    PyGatePointer = 2,
    PyInstructionPointer = 3,
    PyOperationPointer = 4,
    // Remember to update PackedOperationType::is_valid_bit_pattern below
    // if you add or remove this enum's variants!
}

unsafe impl ::bytemuck::CheckedBitPattern for PackedOperationType {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 5
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
/// }
/// ```
///
/// including all ownership semantics, except it bit-packs the enumeration into just 64 bits.
///
/// These bits are wrapped in a union with two fields, `data` and `pointer`. The `data` field
/// provides a view of the bits as a struct containing two `u32` fields, `lo` and `hi`.
/// The `lo` field always contains the enum's discriminant, which is stored in its lowest 3 bits
/// (and thus, this enum MUST NOT be made to exceed 8 variants). For standard gates and
/// instructions, the `lo` field also encodes an enumeration using its other bits to further
/// discriminate its type. The `hi` field is used to store an optional 32 bit payload for standard
/// instructions.
///
/// For pointer variants like `PyGate`, the `pointer` field provides a view of the bits as a
/// pointer. On 64-bit systems, this pointer would normally span the entire type, but since these
/// pointers are guaranteed to be align(8), we take advantage of the lowest 3 bits always being 0
/// to store our discriminant. As such, the lowest 3 bits must always be 0'ed before the value of
/// `pointer` is reinterpreted. On a 32-bit system, `pointer` will only span the 32 bits
/// corresponding to `data.hi`, and can thus be interpreted as a pointer without manipulation.
///
///
/// This is the logical memory layout of `PackedOperation` written out as two 32 bit binary integer.
/// `x` marks padding bits with undefined values, `S` is the bits that make up a `StandardGate` or
/// `StandardInstructionType`, `D` is the data payload of a standard instruction, and `P` is bits
/// that make up part of a pointer.
///
/// ```text
/// Standard gate:
///
///         hi: 0b_xxxxxxxx_xxxxxxxx_xxxxxxxx_xxxxxxxx
///         lo: 0b_xxxxxxxx_xxxxxxxx_xxxxxSSS_SSSSS000
///                                       |-------||-|
///                                           |     |
///                 Standard gate, as a u8. --+     +-- Discriminant.
///
/// Standard instruction:
///
///         hi: 0b_DDDDDDDD_DDDDDDDD_DDDDDDDD_DDDDDDDD <--- optional payload
///         lo: 0b_xxxxxxxx_xxxxxxxx_xxxxxSSS_SSSSS001
///                                       |-------||-|
///                                           |     |
///          Standard instruction, as a u8. --+     +-- Discriminant.
///
///
///     Optional payload:
///     Depending on the variant of the standard instruction type, a 32 bit
///     data payload may be present in the hi bits. Currently, this is used to
///     store the number of qubits in a Barrier and the unit of a Delay.
///
/// Pointer (64-bit system):
///
///         hi: 0b_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP <--- upper 32 bits of pointer
///         lo: 0b_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPP001
///                |------------------------------||-|
///                                |                |
///     lower 32 bits of pointer --+                +-- Discriminant, 0b011 means `PyInstruction`.
///
///   Or, read via the `pointer` field:
///
///   pointer:
///   0b_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPP011
///      |-----------------------------------------------------------------||-|
///                                     |                                    |
///      The high 62 bits of the pointer.  Because of alignment, the low 3   |   Discriminant of the
///      bits of the full 64 bits are guaranteed to be zero so we can        +-- enumeration.  This
///      retrieve the "full" pointer by taking the whole `usize` and zeroing     is 0b011, which
///      the low 3 bits, letting us store the discriminant in there at other     means that this
///      times.                                                                  points to a
///                                                                              `PyInstruction`.
/// Pointer (32-bit system):
///
///         hi: 0b_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP <--- the entire pointer
///         lo: 0b_xxxxxxxx_xxxxxxxx_xxxxxxxx_xxxxx011
///                                                |-|
///                                                 |
///    Discriminant, 0b011 means `PyInstruction`. --+
///
///   Or, read via the `pointer` field:
///
///   pointer:
///   0b_PPPPPPPP_PPPPPPPP_PPPPPPPP_PPPPPPPP
///      |---------------------------------|
///                      |
///                      +-- Because `pointer` is a usize, this is just 32 bits. It is made to line
///                          up with the `hi` field, so that it doesn't collide with the
///                          discriminant stored in `lo`.
/// ```
///
/// To deal with target endianess, the order of the `hi` and `lo` fields are swapped at compile
/// time. For a 64 bit little endian machine, `lo` comes before `hi`. For 64 bit big endian and all
/// 32-bit systems (regardless of endian) `hi` comes before `lo`, since on a 32-bit system, this
/// aligns the `pointer` field (a `usize`) with the `hi` field, to avoid clobbering the
/// discriminant.
///
/// Also note the alignment of this union. On 64-bit systems, its alignment will be 8 because the
/// `pointer` field (a `usize`) will be 8 bytes. On a 32-bit system, it becomes 4, since `pointer`
/// will be 4 bytes, and the `lo` / `hi` fields in `data` are also 4 bytes each. This is intentional
/// to avoid waste.
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
#[repr(C)]
pub union PackedOperation {
    data: LoHi,
    pointer: usize,
}

impl fmt::Debug for PackedOperation {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("PackedOperation")
            .field("lo", unsafe { &self.data.lo })
            .field("hi", unsafe { &self.data.hi })
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
#[cfg(all(target_endian = "little", target_pointer_width = "64"))]
struct LoHi {
    lo: u32,
    hi: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[cfg(any(target_endian = "big", target_pointer_width = "32"))]
struct LoHi {
    hi: u32,
    lo: u32,
}

impl PackedOperation {
    /// The bits representing the `PackedOperationType` discriminant.  This can be used to mask out
    /// the discriminant, and defines the rest of the bit shifting.
    const DISCRIMINANT_MASK: u32 = 0b111;
    /// The number of bits used to store the discriminant metadata.
    const DISCRIMINANT_BITS: u32 = Self::DISCRIMINANT_MASK.count_ones();
    /// A bitmask that masks out only the standard gate information.  This should always have the
    /// same effect as `POINTER_MASK` because the high bits should be 0 for a `StandardGate`, but
    /// this is defensive against us adding further metadata on `StandardGate` later.  After
    /// masking, the resulting integer still needs shifting downwards by the width of the
    /// discriminant, `DISCRIMINANT_BITS`.
    const STANDARD_GATE_MASK: u32 = (u8::MAX as u32) << Self::DISCRIMINANT_BITS;
    /// A bitmask that masks out only the standard instruction type.  After masking, the result must
    /// shifted downwards by `DISCRIMINANT_BITS`.
    const STANDARD_INSTRUCTION_MASK: u32 = (u8::MAX as u32) << Self::DISCRIMINANT_BITS;
    /// For 64-bit machines only, a bitmask that retrieves a 64-bit pointer from the full width of
    /// the `PackedOperation` by zeroing its low bits, which store the discriminant.
    const POINTER_MASK: usize = usize::MAX ^ (Self::DISCRIMINANT_MASK as usize);

    /// Extract the discriminant of the operation.
    #[inline]
    fn discriminant(&self) -> PackedOperationType {
        ::bytemuck::checked::cast((unsafe { self.data.lo } & Self::DISCRIMINANT_MASK) as u8)
    }

    /// Get the contained pointer to the `PyGate`/`PyInstruction`/`PyOperation` that this object
    /// contains.
    ///
    /// **Panics** if the object represents a standard gate; see `try_pointer`.
    #[inline]
    fn pointer(&self) -> NonNull<()> {
        self.try_pointer()
            .expect("the caller is responsible for knowing the correct type")
    }

    /// Get the contained pointer to the `PyGate`/`PyInstruction`/`PyOperation` that
    /// this object contains.
    ///
    /// Returns `None` if the object represents anything else.
    #[inline]
    #[cfg(target_pointer_width = "64")]
    fn try_pointer(&self) -> Option<NonNull<()>> {
        match self.discriminant() {
            PackedOperationType::StandardGate | PackedOperationType::StandardInstruction => None,
            PackedOperationType::PyGatePointer
            | PackedOperationType::PyInstructionPointer
            | PackedOperationType::PyOperationPointer => {
                #[cfg(target_pointer_width = "64")]
                let ptr = { unsafe { (self.pointer & Self::POINTER_MASK) as *mut () } };
                #[cfg(target_pointer_width = "32")]
                let ptr = { unsafe { self.data as *mut () } };
                // SAFETY: `PackedOperation` can only be constructed from a pointer via `Box`, which
                // is always non-null (except in the case that we're partway through a `Drop`).
                Some(unsafe { NonNull::new_unchecked(ptr) })
            }
        }
    }

    /// Get the contained `StandardGate`.
    ///
    /// **Panics** if this `PackedOperation` doesn't contain a `StandardGate`; see
    /// `try_standard_gate`.
    #[inline]
    pub fn standard_gate(&self) -> StandardGate {
        self.try_standard_gate()
            .expect("the caller is responsible for knowing the correct type")
    }

    /// Get the contained `StandardGate`, if any.
    #[inline]
    pub fn try_standard_gate(&self) -> Option<StandardGate> {
        match self.discriminant() {
            PackedOperationType::StandardGate => ::bytemuck::checked::try_cast(
                ((unsafe { self.data.lo } & Self::STANDARD_GATE_MASK) >> Self::DISCRIMINANT_BITS)
                    as u8,
            )
            .ok(),
            _ => None,
        }
    }

    /// Get the contained `StandardInstruction`.
    ///
    /// **Panics** if this `PackedOperation` doesn't contain a `StandardInstruction`; see
    /// `try_standard_instruction`.
    #[inline]
    pub fn standard_instruction(&self) -> StandardInstruction {
        self.try_standard_instruction()
            .expect("the caller is responsible for knowing the correct type")
    }

    /// Get the contained `StandardInstruction`, if any.
    #[inline]
    pub fn try_standard_instruction(&self) -> Option<StandardInstruction> {
        match self.discriminant() {
            PackedOperationType::StandardInstruction => {
                let standard_type: StandardInstructionType = ::bytemuck::checked::cast(
                    ((unsafe { self.data.lo } & Self::STANDARD_INSTRUCTION_MASK)
                        >> Self::DISCRIMINANT_BITS) as u8,
                );
                match standard_type {
                    StandardInstructionType::Barrier => {
                        let num_qubits = unsafe { self.data.hi };
                        Some(StandardInstruction::Barrier(num_qubits as usize))
                    }
                    StandardInstructionType::Delay => {
                        let unit: DelayUnit =
                            ::bytemuck::checked::cast(unsafe { self.data.hi } as u8);
                        Some(StandardInstruction::Delay(unit))
                    }
                    StandardInstructionType::Measure => Some(StandardInstruction::Measure),
                    StandardInstructionType::Reset => Some(StandardInstruction::Reset),
                }
            }
            _ => None,
        }
    }

    /// Get a safe view onto the packed data within, without assuming ownership.
    #[inline]
    pub fn view(&self) -> OperationRef {
        match self.discriminant() {
            PackedOperationType::StandardGate => OperationRef::Standard(self.standard_gate()),
            PackedOperationType::StandardInstruction => {
                OperationRef::StandardInstruction(self.standard_instruction())
            }
            PackedOperationType::PyGatePointer => {
                let ptr = self.pointer().cast::<PyGate>();
                OperationRef::Gate(unsafe { ptr.as_ref() })
            }
            PackedOperationType::PyInstructionPointer => {
                let ptr = self.pointer().cast::<PyInstruction>();
                OperationRef::Instruction(unsafe { ptr.as_ref() })
            }
            PackedOperationType::PyOperationPointer => {
                let ptr = self.pointer().cast::<PyOperation>();
                OperationRef::Operation(unsafe { ptr.as_ref() })
            }
        }
    }

    /// Create a `PackedOperation` from a `StandardGate`.
    #[inline]
    pub fn from_standard(standard: StandardGate) -> Self {
        Self {
            data: LoHi {
                lo: (standard as u32) << Self::DISCRIMINANT_BITS,
                hi: 0,
            },
        }
    }

    /// Create a `PackedOperation` from a `StandardInstruction`.
    pub fn from_standard_instruction(instruction: StandardInstruction) -> Self {
        Self {
            data: match instruction {
                StandardInstruction::Barrier(num_qubits) => {
                    LoHi {
                        lo: ((StandardInstructionType::Barrier as u32) << Self::DISCRIMINANT_BITS) | PackedOperationType::StandardInstruction as u32,
                        hi: num_qubits.try_into().expect(
                            "The PackedOperation representation currently requires barrier size to be <= 32 bits."
                        ),
                    }
                }
                StandardInstruction::Delay(unit) => {
                    LoHi {
                        lo: ((StandardInstructionType::Delay as u32) << Self::DISCRIMINANT_BITS) | PackedOperationType::StandardInstruction as u32,
                        hi: unit as u32,
                    }
                }
                StandardInstruction::Measure => {
                    LoHi {
                        lo: ((StandardInstructionType::Measure as u32) << Self::DISCRIMINANT_BITS) | PackedOperationType::StandardInstruction as u32,
                        hi: 0,
                    }
                }
                StandardInstruction::Reset => {
                    LoHi {
                        lo: ((StandardInstructionType::Reset as u32) << Self::DISCRIMINANT_BITS) | PackedOperationType::StandardInstruction as u32,
                        hi: 0,
                    }
                }
            }
        }
    }

    /// Create a `PackedOperation` given a raw pointer to the inner type.
    ///
    /// **Panics** if the given `discriminant` does not correspond to a pointer type.
    ///
    /// SAFETY: the inner pointer must have come from an owning `Box` in the global allocator, whose
    /// type matches that indicated by the discriminant.  The returned `PackedOperation` takes
    /// ownership of the pointed-to data.
    #[inline]
    unsafe fn from_owned_raw_pointer(
        discriminant: PackedOperationType,
        value: NonNull<()>,
    ) -> Self {
        if !matches!(
            discriminant,
            PackedOperationType::PyGatePointer
                | PackedOperationType::PyInstructionPointer
                | PackedOperationType::PyOperationPointer
        ) {
            panic!("given non-pointer discriminant during pointer-type construction");
        }
        let addr = value.as_ptr() as usize;

        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(addr & (Self::DISCRIMINANT_MASK as usize), 0);
            Self {
                pointer: addr | (discriminant as usize),
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            Self {
                data: LoHi {
                    lo: discriminant as u32,
                    hi: addr as u32,
                },
            }
        };
    }

    /// Construct a new `PackedOperation` from an owned heap-allocated `PyGate`.
    pub fn from_gate(gate: Box<PyGate>) -> Self {
        let ptr = NonNull::from(Box::leak(gate)).cast::<()>();
        // SAFETY: the `ptr` comes directly from a owning `Box` of the correct type.
        unsafe { Self::from_owned_raw_pointer(PackedOperationType::PyGatePointer, ptr) }
    }

    /// Construct a new `PackedOperation` from an owned heap-allocated `PyInstruction`.
    pub fn from_instruction(instruction: Box<PyInstruction>) -> Self {
        let ptr = NonNull::from(Box::leak(instruction)).cast::<()>();
        // SAFETY: the `ptr` comes directly from a owning `Box` of the correct type.
        unsafe { Self::from_owned_raw_pointer(PackedOperationType::PyInstructionPointer, ptr) }
    }

    /// Construct a new `PackedOperation` from an owned heap-allocated `PyOperation`.
    pub fn from_operation(operation: Box<PyOperation>) -> Self {
        let ptr = NonNull::from(Box::leak(operation)).cast::<()>();
        // SAFETY: the `ptr` comes directly from a owning `Box` of the correct type.
        unsafe { Self::from_owned_raw_pointer(PackedOperationType::PyOperationPointer, ptr) }
    }

    /// Check equality of the operation, including Python-space checks, if appropriate.
    pub fn py_eq(&self, py: Python, other: &PackedOperation) -> PyResult<bool> {
        match (self.view(), other.view()) {
            (OperationRef::Standard(left), OperationRef::Standard(right)) => Ok(left == right),
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
            OperationRef::Standard(standard) => Ok(standard.into()),
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
        }
    }

    /// Copy this operation, including a Python-space call to `copy` on the `Operation` subclass, if
    /// any.
    pub fn py_copy(&self, py: Python) -> PyResult<Self> {
        let copy_attr = intern!(py, "copy");
        match self.view() {
            OperationRef::Standard(standard) => Ok(standard.into()),
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
        }
    }

    /// Whether the Python class that we would use to represent the inner `Operation` object in
    /// Python space would be an instance of the given Python type.  This does not construct the
    /// Python-space `Operator` instance if it can be avoided (i.e. for standard gates).
    pub fn py_op_is_instance(&self, py_type: &Bound<PyType>) -> PyResult<bool> {
        let py = py_type.py();
        let py_op = match self.view() {
            OperationRef::Standard(standard) => {
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
        };
        py_op.is_instance(py_type)
    }
}

impl Operation for PackedOperation {
    fn name(&self) -> &str {
        let view = self.view();
        let name = match view {
            OperationRef::Standard(ref standard) => standard.name(),
            OperationRef::StandardInstruction(ref instruction) => instruction.name(),
            OperationRef::Gate(gate) => gate.name(),
            OperationRef::Instruction(instruction) => instruction.name(),
            OperationRef::Operation(operation) => operation.name(),
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

impl From<StandardGate> for PackedOperation {
    #[inline]
    fn from(value: StandardGate) -> Self {
        Self::from_standard(value)
    }
}

impl From<StandardInstruction> for PackedOperation {
    #[inline]
    fn from(value: StandardInstruction) -> Self {
        Self::from_standard_instruction(value)
    }
}

macro_rules! impl_packed_operation_from_py {
    ($type:ty, $constructor:path) => {
        impl From<$type> for PackedOperation {
            #[inline]
            fn from(value: $type) -> Self {
                $constructor(Box::new(value))
            }
        }

        impl From<Box<$type>> for PackedOperation {
            #[inline]
            fn from(value: Box<$type>) -> Self {
                $constructor(value)
            }
        }
    };
}
impl_packed_operation_from_py!(PyGate, PackedOperation::from_gate);
impl_packed_operation_from_py!(PyInstruction, PackedOperation::from_instruction);
impl_packed_operation_from_py!(PyOperation, PackedOperation::from_operation);

impl Clone for PackedOperation {
    fn clone(&self) -> Self {
        match self.view() {
            OperationRef::Standard(standard) => Self::from_standard(standard),
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
        }
    }
}
impl Drop for PackedOperation {
    fn drop(&mut self) {
        fn drop_pointer_as<T>(slf: &mut PackedOperation) {
            // This should only ever be called when the pointer is valid, but this is defensive just
            // to 100% ensure that our `Drop` implementation doesn't panic.
            let Some(pointer) = slf.try_pointer() else {
                return;
            };
            // SAFETY: `PackedOperation` asserts ownership over its contents, and the contained
            // pointer can only be null if we were already dropped.  We set our discriminant to mark
            // ourselves as plain old data immediately just as a defensive measure.
            let boxed = unsafe { Box::from_raw(pointer.cast::<T>().as_ptr()) };
            slf.data.lo = PackedOperationType::StandardGate as u32;
            ::std::mem::drop(boxed);
        }

        match self.discriminant() {
            PackedOperationType::StandardGate | PackedOperationType::StandardInstruction => (),
            PackedOperationType::PyGatePointer => drop_pointer_as::<PyGate>(self),
            PackedOperationType::PyInstructionPointer => drop_pointer_as::<PyInstruction>(self),
            PackedOperationType::PyOperationPointer => drop_pointer_as::<PyOperation>(self),
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
    pub extra_attrs: ExtraInstructionAttributes,

    #[cfg(feature = "cache_pygates")]
    /// This is hidden in a `OnceCell` because it's just an on-demand cache; we don't create this
    /// unless asked for it.  A `OnceCell` of a non-null pointer type (like `Py<T>`) is the same
    /// size as a pointer and there are no runtime checks on access beyond the initialisation check,
    /// which is a simple null-pointer check.
    ///
    /// WARNING: remember that `OnceCell`'s `get_or_init` method is no-reentrant, so the initialiser
    /// must not yield the GIL to Python space.  We avoid using `GILOnceCell` here because it
    /// requires the GIL to even `get` (of course!), which makes implementing `Clone` hard for us.
    /// We can revisit once we're on PyO3 0.22+ and have been able to disable its `py-clone`
    /// feature.
    pub py_op: OnceCell<Py<PyAny>>,
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
    pub fn condition(&self) -> Option<&Py<PyAny>> {
        self.extra_attrs.condition()
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.extra_attrs.label()
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
                OperationRef::Standard(standard) => standard.create_py_op(
                    py,
                    self.params.as_deref().map(SmallVec::as_slice),
                    &self.extra_attrs,
                ),
                OperationRef::StandardInstruction(instruction) => instruction.create_py_op(
                    py,
                    self.params.as_deref().map(SmallVec::as_slice),
                    &self.extra_attrs,
                ),
                OperationRef::Gate(gate) => Ok(gate.gate.clone_ref(py)),
                OperationRef::Instruction(instruction) => Ok(instruction.instruction.clone_ref(py)),
                OperationRef::Operation(operation) => Ok(operation.operation.clone_ref(py)),
            }
        };

        // `OnceCell::get_or_init` and the non-stabilised `get_or_try_init`, which would otherwise
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
            (OperationRef::Standard(left), OperationRef::Standard(right)) => Ok(left == right),
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
            (OperationRef::Standard(_left), OperationRef::Gate(right)) => {
                self.unpack_py_op(py)?.bind(py).eq(&right.gate)
            }
            (OperationRef::Gate(left), OperationRef::Standard(_right)) => {
                other.unpack_py_op(py)?.bind(py).eq(&left.gate)
            }
            _ => Ok(false),
        }
    }
}
