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

use core::slice;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    num::NonZero,
    ptr::{null, null_mut},
    sync::Arc,
};

use qiskit_circuit::{
    circuit_data::CircuitData,
    operations::{BoxedCustomOperation, CustomOperation, Operation, Param},
};

use crate::pointers::const_ptr_as_ref;

/// Represents a quantum operation fully defined in C.
///
/// This operation object contains the minimal functionality an object
/// should adhere to in order operate on a ``QkCircuit``.
///
/// Any object that can be implemented using ``QkCustomOperation`` will be
/// dynamically dispatched to be added to the circuit. In other words,
/// the circuit is unaware of the type of object it is accepting, but
/// it will work with it as long as it has the functionality expected
/// from any operation.
///
/// To achieve this, an operation is defined by two parts:
/// - The original pointer to the operation struct.
/// - The pointer to a vtable with the function slots that define
///   the functionality of this operation. See ``qk_custom_operation_vtable_new``
///   for more details.
///
/// Here's a quick example of what that looks like:
///
/// ```c
///
/// // Define an operation with a single attribute.
/// struct foo_gate {
///     uint32_t num_qubits;
/// }
///
/// // Represents the name of the operation.
/// const char *FOO_NAME = "foo";
///
/// // Design the required methods for the vtable.
///
/// const char *foo_name(const void *gate) {
///     // Cast void to original pointer.
///     struct foo_gate *_self = (struct foo_gate *)gate;
///     // Cast once more to consume it
///     (void)_self;
///     return FOO_NAME;
/// }
/// uint32_t foo_num_qubits(const void *gate) {
///     struct foo_gate *self = (struct foo_gate *)gate;
///     // Used stored attirbute as return value.
///     return self->num_qubits;
/// }
/// // Use same logic below for required methods that have
/// // fixed values.
/// uint32_t foo_num_clbits(const void *gate) {
///     struct foo_gate *self = (struct foo_gate *)gate;
///     (void)_self;
///     return 0;
/// }
/// // Implement all required methods.
///
/// // Build list of entries for the vtable (at least 7 required entries)
/// QkCustomOpVTableEntry entries[7] = {
///     {.slot = 0, .func = foo_name},
///     {.slot = 1, .func = foo_num_qubits},
///     {.slot = 2, .func = foo_num_clbits},
///     // ...
///     // End with sentinel value
///     {.slot = -1, .func = NULL},
/// };
///
/// // Create a vtable
/// QkCustomOpVTable *foo_vtable = qk_custom_operation_vtable_new(entries);
///
/// // Declare a sample instance
/// struct foo_gate foo_3q = {
///     .num_qubits = 3,
/// };
///
/// // Create the custom operation
/// QkCustomOperation foo_3q_custom = {
///     .orig = &foo_3q,
///     .v_table = foo_vtable,
/// };
///
/// // Add to a circuit
/// QkCircuit *circuit = qk_circuit_new(3, 0);
/// uint32_t qubits[3] = {0, 1, 2};
///
/// qk_circuit_add_custom_operation(circuit, foo_3q_custom, qubits, NULL, NULL);
/// ```
///
/// # Safety:
///
/// This struct contains raw pointers, which are not [`Send`] or [`Sync`].
///
/// It falls on the responsability of the implementors to ensure that the
/// data enclosed in the operation can:
/// - Be accessed safely by multiple threads concurrently.
/// - Be immutably borrowed by other threads without causing race conditions.
/// - Be preserved throughout the runtime of the program.
///
/// Failure to comply with these conditions may result in undefined behavior.
#[derive(Debug, Clone)]
struct CustomOp {
    /// A pointer to the original gate.
    orig: *mut (),
    /// A pointer to a vtable designed for the original gate.
    v_table: *const CustomOpVTable,
}

impl PartialEq for CustomOp {
    fn eq(&self, other: &Self) -> bool {
        (unsafe { ((&*self.v_table).eq)(self.orig, other.orig) }) && self.v_table == other.v_table
    }
}

unsafe impl Send for CustomOp {}
unsafe impl Sync for CustomOp {}

impl Operation for CustomOp {
    fn name(&self) -> &str {
        let name = unsafe { ((&*self.v_table).name)(self.orig) };
        // Safety violation on lifetimes of the name here
        // Document the lifetime bounds here, these pointers must only be borrowed.
        // C should not mutate origin while Rust is accessing it.
        let name_parsed = unsafe { CStr::from_ptr(name) };
        name_parsed
            .to_str()
            .expect("Expected a 'UTF-8' formatted string.")
    }

    fn num_qubits(&self) -> u32 {
        unsafe { (({ &*self.v_table }).num_qubits)(self.orig) }
    }

    fn num_clbits(&self) -> u32 {
        unsafe { (({ &*self.v_table }).num_clbits)(self.orig) }
    }

    fn num_params(&self) -> u32 {
        unsafe { (({ &*self.v_table }).num_params)(self.orig) }
    }

    fn directive(&self) -> bool {
        unsafe { (({ &*self.v_table }).directive)(self.orig) }
    }
}

impl CustomOperation for CustomOp {
    fn is_unitary(&self) -> bool {
        unsafe { (({ &*self.v_table }).is_unitary)(self.orig) }
    }

    fn num_ctrl_qubits(&self) -> Option<std::num::NonZero<u32>> {
        let num_ctrl_qubits = unsafe { (({ &*self.v_table }).num_ctrl_qubits)(self.orig) };
        match num_ctrl_qubits {
            0 => None,
            _ => Some(NonZero::new(num_ctrl_qubits).unwrap()),
        }
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        let params: Vec<*const Param> = params.iter().map(|obj| obj as *const Param).collect();
        let definition = unsafe { (({ &*self.v_table }).definition)(self.orig, params.as_ptr()) };
        if definition.is_null() {
            return None;
        }
        let circ = unsafe { Box::from_raw(definition) };
        Some(*circ)
    }

    fn label(&self) -> Option<&str> {
        unsafe { CStr::from_ptr((({ &*self.v_table }).label)(self.orig)) }
            .to_str()
            .ok()
    }
}

/// Represents a vtable containing all the function pointers
/// pertaining to the methods associated with the instance of
/// [``CustomOp``] coming from C.
///
/// All methods provided require a void pointer representing
/// the original instance to be passed as an argument, which
/// is always packed together with the vtable in [`CustomOp`].
/// An implementor is expected to provide the pointers to
/// the following required methods for implementing the [`CustomOperation`]
/// trait:
///
/// * ``name(*const ())`` -> ``*const c_char``,
/// * ``num_qubits(*const ())`` -> ``u32``,
/// * ``num_clbits(*const ())`` -> ``u32``,
/// * ``num_params(*const ())`` -> ``u32``,
/// * ``directive(*const ())`` -> ``bool``,
/// * ``is_unitary(*const ())`` -> ``bool``,
///
/// There are also functional methods that are optional but
/// implementors are expected to provide.
///
/// * ``num_ctrl_qubits(*const ())`` -> ``u32``,
/// * ``label(*const ())`` ->  ``*const c_char``,
/// * ``definition(*const (), *const Param)`` -> ``*mut CircuitData``,
/// * ``eq(*const (), *const ())`` -> ``bool``, to compare two operations of the same kind.
#[derive(Debug, Clone)]
// #[repr(C)]
pub struct CustomOpVTable {
    pub name: unsafe extern "C" fn(*const ()) -> *const c_char,
    pub num_qubits: unsafe extern "C" fn(*const ()) -> u32,
    pub num_clbits: unsafe extern "C" fn(*const ()) -> u32,
    pub num_params: unsafe extern "C" fn(*const ()) -> u32,
    pub directive: unsafe extern "C" fn(*const ()) -> bool,
    pub is_unitary: unsafe extern "C" fn(*const ()) -> bool,
    pub num_ctrl_qubits: unsafe extern "C" fn(*const ()) -> u32,
    pub label: unsafe extern "C" fn(*const ()) -> *const c_char,
    pub definition: unsafe extern "C" fn(*const (), *const *const Param) -> *mut CircuitData,
    pub eq: unsafe extern "C" fn(*const (), *const ()) -> bool,
}

extern "C" fn default_num_ctrl_qubits(_slf: *const ()) -> u32 {
    // extern C
    0
}

extern "C" fn default_label(_slf: *const ()) -> *const c_char {
    null()
}

extern "C" fn default_definition(
    _slf: *const (),
    _params: *const *const Param,
) -> *mut CircuitData {
    null_mut()
}

extern "C" fn default_eq(slf: *const (), other: *const ()) -> bool {
    slf.eq(&other)
}

impl TryFrom<CustomOpVtablePartial> for CustomOpVTable {
    type Error = CustomOpMethod;

    fn try_from(value: CustomOpVtablePartial) -> Result<Self, Self::Error> {
        use CustomOpMethod::*;
        Ok(Self {
            name: value.name.ok_or(Name)?,
            num_qubits: value.num_qubits.ok_or(NumQubits)?,
            num_clbits: value.num_clbits.ok_or(NumClbits)?,
            num_params: value.num_params.ok_or(NumParams)?,
            directive: value.directive.ok_or(Directive)?,
            is_unitary: value.is_unitary.ok_or(IsUnitary)?,
            num_ctrl_qubits: value.num_ctrl_qubits.unwrap_or(default_num_ctrl_qubits),
            label: value.label.unwrap_or(default_label),
            definition: value.definition.unwrap_or(default_definition),
            // eq: value.eq.ok_or(Eq)?,
            eq: value.eq.unwrap_or(default_eq),
        })
    }
}

/// A partial implementation of [`CustomOpVtable`] as referred
/// to by its namesake.
///
/// This implementation is only used during construction of the
/// vtable and should always be converted to a [`CustomOpVtable`].
/// The conversion will fail if any of the required methods listed
/// in the documentation are not provided, and an error code with
/// the first missing slot's [``CustomOpMethod``] index will be provided.
#[derive(Debug, Clone, Default)]
pub struct CustomOpVtablePartial {
    pub name: Option<unsafe extern "C" fn(*const ()) -> *const c_char>,
    pub num_qubits: Option<unsafe extern "C" fn(*const ()) -> u32>,
    pub num_clbits: Option<unsafe extern "C" fn(*const ()) -> u32>,
    pub num_params: Option<unsafe extern "C" fn(*const ()) -> u32>,
    pub directive: Option<unsafe extern "C" fn(*const ()) -> bool>,
    pub is_unitary: Option<unsafe extern "C" fn(*const ()) -> bool>,
    pub num_ctrl_qubits: Option<unsafe extern "C" fn(*const ()) -> u32>,
    pub label: Option<unsafe extern "C" fn(*const ()) -> *const c_char>,
    pub definition:
        Option<unsafe extern "C" fn(*const (), *const *const Param) -> *mut CircuitData>,
    pub eq: Option<unsafe extern "C" fn(*const (), *const ()) -> bool>,
}

/// Represents the Vtable index of a `CustomOperation` coming from the
/// C domain.
///
/// Each named index refers to a required/optional method of the `Operation``
/// and `CustomOperation` traits.
#[repr(u32)]
#[derive(Debug)]
pub enum CustomOpMethod {
    Name = 0,
    NumQubits = 1,
    NumClbits = 2,
    NumParams = 3,
    Directive = 4,
    IsUnitary = 5,
    NumCtrlQubits = 6,
    Label = 7,
    Definition = 8,
    Eq = 9,
}

impl TryFrom<u32> for CustomOpMethod {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        use CustomOpMethod::*;
        let ret = match value {
            0 => Name,
            1 => NumQubits,
            2 => NumClbits,
            3 => NumParams,
            4 => Directive,
            5 => IsUnitary,
            6 => NumCtrlQubits,
            7 => Label,
            8 => Definition,
            9 => Eq,
            _ => return Err(value),
        };
        Ok(ret)
    }
}

/// Represents an entry in a ``QkCustomOpVTable``.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CustomOpVTableEntry {
    /// The slot index.
    slot: u32,
    /// A function pointer for the operation to use as a method.
    func: *const ::std::ffi::c_void,
}

impl CustomOpVTableEntry {
    pub const SENTINEL: Self = Self {
        slot: u32::MAX,
        func: ::std::ptr::null(),
    };
}

/// @ingroup QkCustomOperation
/// Builds a ``QkCustomOperation`` based on a quantum operation fully
/// defined in C.
///
/// Here's a quick example of what that looks like:
///
/// ```c
///
/// // Define an operation with a single attribute.
/// struct foo_gate {
///     uint32_t num_qubits;
/// }
///
/// // Implement all required methods
/// uint32_t foo_num_qubits(const void *gate) {
///     struct foo_gate *self = (struct foo_gate *)gate;
///     // Used stored attirbute as return value.
///     return self->num_qubits;
/// }
///
/// // Build list of entries for the vtable (at least 7 required entries)
/// QkCustomOpVTableEntry entries[7] = {
///     {.slot = 1, .func = foo_num_qubits},
///     // ...
///     // End with sentinel value
///     {.slot = -1, .func = NULL},
/// };
///
/// // Create a vtable
/// QkCustomOpVTable *foo_vtable = qk_custom_operation_vtable_new(entries);
///
/// // Declare a sample instance
/// struct foo_gate foo_3q = {
///     .num_qubits = 3,
/// };
///
/// // Create the custom operation
/// QkCustomOperation foo_3q_custom = qk_custom_operation_new(&foo_3q, foo_vtable);
/// ```
///
/// @param operation A pointer to the operation struct.
/// @param v_table A pointer to a correctly constructed v_table designed to
/// work with the data of the struct `operation` points to.
///
/// @return A pointer to ``QkCustomOperation``.
///
/// # Safety
///
/// It falls on the responsibility of the implementors to ensure that the
/// data enclosed in the `operation` struct can:
/// - Be accessed safely by multiple threads concurrently.
/// - Be immutably borrowed by other threads without causing race conditions.
/// - Be preserved throughout the lifetime of the operation.
///
/// Behavior is undefined if the provided `v_table` pointer is null or non-alligned.
///
/// Failure to comply with these conditions may result in undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_new(
    operation: *mut (),
    v_table: *const CustomOpVTable,
) -> *mut BoxedCustomOperation {
    let as_custom_op = CustomOp {
        orig: operation,
        v_table,
    };

    Box::into_raw(Box::new(BoxedCustomOperation::from(as_custom_op)))
}

/// @ingroup QkCustomOperation
/// Builds a ``QkCustomOpVTable`` based on a list of ``QkCustomOpVTableEntry``
/// instances.
///
/// The vtable is built from a collection of slots that hold an index and a
/// pointer to a function of the correct argument and return types.
///
/// Refer to the following table to identify the correct slots.
///
/// | Slot                 | Arg(s) type                    | Return type   | Index | Required |
/// |----------------------|--------------------------------|---------------|-------|----------|
/// | ``name``             | `const void *`                 | `char *`      |   0   |    Yes   |
/// | ``num_qubits``       | `const void *`                 | `uint32_t`    |   1   |    Yes   |
/// | ``num_clbits``       | `const void *`                 | `uint32_t`    |   2   |    Yes   |
/// | ``num_params``       | `const void *`                 | `uint32_t`    |   3   |    Yes   |
/// | ``directive``        | `const void *`                 | `bool`        |   4   |    Yes   |
/// | ``is_unitary``       | `const void *`                 | `bool`        |   5   |    Yes   |
/// | ``num_ctrl_qubits``  | `const void *`                 | `uint32_t`    |   6   |    No    |
/// | ``label``            | `const void *`                 | `char *`      |   7   |    No    |
/// | ``definition``       | `const void *`, `QkParam **`   | `QkCircuit *` |   8   |    No    |
/// | ``eq``               | `const void *`, `const void *` | `bool`        |   9   |    No    |
///
/// Each function will be seen as a `void` pointer to Rust and will be transmuted
/// to a function pointer of the correct signature.
///
/// If a required slot is not received, the vtable will not be constructed
/// and this function will return a `NULL` pointer. If an optional slot is not
/// included, the vtable will still be built and its slots will point to default
/// implementations of the said method(s).
///
/// If a slot does not have a valid index (other than the sentinel value), the provided
/// function pointer will be ignored. This ensures that if any non-required methods are
/// added or removed from the chart above, the program should still be able to run
/// without issues.
///
/// Every list of slots should be delimited by a sentinel valued
/// ``QkCustomOpVTableEntry`` at the end. The sentinel should look as follows:
///
/// ```c
/// QkCustomOpVTableEntry sentinel = {.slot = -1, .func = NULL};
/// ```
///
/// This function will stop reading any slots located after the sentinel is found.
///
/// @param slots A pointer to a list of entries delimited by an entry with
/// a sentinel value.
///
/// @return A pointer to a constructed vtable or a null pointer if any
/// required entries are absent.
///
/// # Safety
///
/// Behavior is undefined if a list of entries without delimiting sentinel
/// value are provided.
///
/// Undefined behavior may also happen during transmutation if the provided
/// function pointer does not have the correct signature.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_vtable_new(
    mut slots: *const CustomOpVTableEntry,
) -> *const CustomOpVTable {
    let mut vtable = CustomOpVtablePartial::default();
    let mut slot = unsafe { slots.read() };
    while slot.slot != u32::MAX {
        use CustomOpMethod::*;
        match CustomOpMethod::try_from(slot.slot) {
            Ok(Name) => {
                if vtable.name.is_some() {
                    panic!("Name slot has already been set.")
                }
                vtable.name = Some(unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(*const ()) -> *const c_char,
                    >(slot.func)
                })
            }
            Ok(NumQubits) => {
                if vtable.num_qubits.is_some() {
                    panic!("NumQubits slot has already been set.")
                }
                vtable.num_qubits = Some(unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const ()) -> u32>(
                        slot.func,
                    )
                })
            }
            Ok(NumClbits) => {
                if vtable.num_clbits.is_some() {
                    panic!("NumClbits slot has already been set.")
                }
                vtable.num_clbits = Some(unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const ()) -> u32>(
                        slot.func,
                    )
                })
            }
            Ok(NumParams) => {
                if vtable.num_params.is_some() {
                    panic!("NumParams slot has already been set.")
                }
                vtable.num_params = Some(unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const ()) -> u32>(
                        slot.func,
                    )
                })
            }
            Ok(Directive) => {
                if vtable.directive.is_some() {
                    panic!("Directive slot has already been set.")
                }
                vtable.directive = Some(unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const ()) -> bool>(
                        slot.func,
                    )
                })
            }
            Ok(IsUnitary) => {
                if vtable.is_unitary.is_some() {
                    panic!("IsUnitary slot has already been set.")
                }
                vtable.is_unitary = Some(unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const ()) -> bool>(
                        slot.func,
                    )
                })
            }
            Ok(NumCtrlQubits) => {
                if vtable.num_ctrl_qubits.is_some() {
                    panic!("NumCtrlQubits slot has already been set.")
                }
                vtable.num_ctrl_qubits = Some(unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const ()) -> u32>(
                        slot.func,
                    )
                })
            }
            Ok(Label) => {
                if vtable.label.is_some() {
                    panic!("Label slot has already been set.")
                }
                vtable.label = Some(unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(*const ()) -> *const c_char,
                    >(slot.func)
                })
            }
            Ok(Definition) => {
                if vtable.definition.is_some() {
                    panic!("Name slot has already been set.")
                }
                vtable.definition = Some(unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(*const (), *const *const Param) -> *mut CircuitData,
                    >(slot.func)
                })
            }
            Ok(Eq) => {
                if vtable.eq.is_some() {
                    panic!("Name slot has already been set.")
                }
                vtable.eq = Some(unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(*const (), *const ()) -> bool,
                    >(slot.func)
                })
            }
            Err(_) => {
                continue;
            }
        }
        slots = unsafe { slots.add(1) };
        slot = unsafe { slots.read() };
    }
    CustomOpVTable::try_from(vtable)
        .map(|x| Arc::into_raw(Arc::new(x)))
        .unwrap_or(std::ptr::null())
}

/// @ingroup QkCustomOperation
///
/// Returns the name of an instance of ``QkCustomOperation``.
///
/// This method is guaranteed to return a string containing the operation name
/// as it is a required method for any defined operation.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return The instruction's name.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_name(
    inst: *const BoxedCustomOperation,
) -> *const c_char {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    if let Some(as_custom_op) = borrowed_inst.downcast_ref::<CustomOp>() {
        // Use vtable directly to avoid converting
        unsafe { ((&*as_custom_op.v_table).name)(as_custom_op.orig) }
    } else {
        CString::new(borrowed_inst.name())
            .expect("Operation name should not contain null bytes")
            .into_raw()
    }
}

/// @ingroup QkCustomOperation
///
/// Returns the number of qubits an instance of ``QkCustomOperation`` can operate on.
///
/// This method is guaranteed to return a number of qubits or 0, as it is a
/// required method for any defined operation.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return The number of classical bits the operation supports.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_num_qubits(inst: *const BoxedCustomOperation) -> u32 {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    borrowed_inst.num_qubits()
}

/// @ingroup QkCustomOperation
///
/// Returns the number of classical bits (clbits) an instance of ``QkCustomOperation`` can operate with.
///
/// This method is guaranteed to return a number of clbits or 0, as it is a
/// required method for any defined operation.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return The number of classical bits the operation supports.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_num_clbits(inst: *const BoxedCustomOperation) -> u32 {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    borrowed_inst.num_clbits()
}

/// @ingroup QkCustomOperation
///
/// Returns the number of parameters an instance of ``QkCustomOperation`` can operate with.
///
/// This method is guaranteed to return a number of parameters or 0, as it is a
/// required method for any defined operation.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return The number of parameters this operation supports.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_num_params(inst: *const BoxedCustomOperation) -> u32 {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    borrowed_inst.num_params()
}

/// @ingroup QkCustomOperation
///
/// Checks whether an instance of ``QkCustomOperation`` is a directive or not.
///
/// Directives are operations to the quantum stack meant to be interpreted by
/// the backed or the transpiler.
///
/// This method is guaranteed to return a boolean as it is a required method
/// for any defined operation.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return `true` if this instruction is a directive, otherwise `false`.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_directive(inst: *const BoxedCustomOperation) -> bool {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    borrowed_inst.directive()
}

/// @ingroup QkCustomOperation
///
/// Checks whether an instance of ``QkCustomOperation`` is a unitary operation or not.
///
/// A unitary operation is represented by a unitary matrix which is a complex square
/// invertible matrix.
///
/// Unitary operations (or gates) operate exclusively on quantum resources
/// and therefore should always have ``qk_custom_operation_num_clbits`` return ``0``
/// and they cannot be directives.
///
/// This method is guaranteed to return a boolean as it is a required method
/// for any defined operation.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return `true` if the instruction is defined as unitary
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_is_unitary(inst: *const BoxedCustomOperation) -> bool {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    borrowed_inst.is_unitary()
}

/// @ingroup QkCustomOperation
///
/// Returns the number of control qubits supported by this ``QkCustomOperation``
/// instance, if it is a controlled operation.
///
/// This method is not required for every ``QkCustomOperation`` definition. Therefoere,
/// it will return ``0`` by default unless otherwise specified.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return The number of supported control qubits, otherwise ``0``.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_num_ctrl_qubits(
    inst: *const BoxedCustomOperation,
) -> u32 {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    if let Some(number) = borrowed_inst.num_ctrl_qubits() {
        number.into()
    } else {
        0
    }
}

/// @ingroup QkCustomOperation
///
/// Returns the label of an instance of ``QkCustomOperation`` .
///
/// This method is not required for every ``QkCustomOperation`` definition. Therefoere,
/// it may return a null pointer instead of a string.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return The instruction's label, if defined, otherwise `NULL`.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_label(
    inst: *const BoxedCustomOperation,
) -> *const c_char {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    if let Some(as_custom_op) = borrowed_inst.downcast_ref::<CustomOp>() {
        // Use vtable directly to avoid converting
        unsafe { ((&*as_custom_op.v_table).label)(as_custom_op.orig) }
    } else {
        if let Some(label) = borrowed_inst.label() {
            CString::new(label)
                .expect("Label should not contain null bytes")
                .into_raw()
        } else {
            null()
        }
    }
}

/// @ingroup QkCustomOperation
///
/// Returns the definition of an instance of `QkCustomOperation` if the correct
/// parameters are provided.
///
/// When an operation is structurally complex, it may be broken down into a `QkCircuit`
/// made of other operations that perform the same transformations and result in the
/// same state. This is what we call the gate's deifnition.
///
/// This method is not required for every ``QkCustomOperation``. Therefoere,
/// it may return a null pointer instead of a Circuit.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
/// @param params A pointer to an array of `QkParam` pointers.
///
/// @return The instruction's definition if it was defined and the correct parameters are passed,
/// otherwise, a `NULL` pointer.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_definition(
    inst: *const BoxedCustomOperation,
    params: *const *const Param,
) -> *mut CircuitData {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };

    if let Some(as_custom_op) = borrowed_inst.downcast_ref::<CustomOp>() {
        // Use vtable directly to avoid converting
        unsafe { ((&*as_custom_op.v_table).definition)(as_custom_op.orig, params) }
    } else {
        let parsed_params: Vec<Param> =
            unsafe { slice::from_raw_parts(params, borrowed_inst.num_params() as usize) }
                .iter()
                .map(|&ptr| unsafe { const_ptr_as_ref(ptr) }.clone())
                .collect();

        match borrowed_inst.definition(&parsed_params) {
            Some(circ) => Box::into_raw(Box::new(circ)),
            None => null_mut(),
        }
    }
}

/// @ingroup QkCustomOperation
///
/// Compares two different instances of ``QkCustomOperation``.
///
/// If the user defined a method to compare between instances, it will be used
/// to perform this comparison. Otherwise, the comparison will be based on the
/// memory addresses passed on.
///
/// By default, this method will try to downcast the original pointer to its
/// type of origin and use the provided `eq` method to compare between the two.
/// If it's unable to downcast, it will return ``false``.
///
/// This method is not required for every ``QkCustomOperation``. Therefoere,
/// it may perform comparison via memory addresses.
///
/// @param inst A pointer to the ``QkCustomOperation``  instance.
/// @param other A pointer to another ``QkCustomOperation``  instance to compare.
///
/// @return Whether these instructions are the same.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_eq(
    inst: *const BoxedCustomOperation,
    other: *const BoxedCustomOperation,
) -> bool {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };
    let borrowed_other = unsafe { const_ptr_as_ref(other) };

    **borrowed_inst == **borrowed_other
}

/// @ingroup QkCustomOperation
///
/// Returns the `type_id` discriminant for this ``QkCustomOperation`` if it
/// originates from C. Otherwise it returns ``UINT64_MAX``.
///
/// If the user plans on casting the original pointer back to its original
/// type for additional functionality, the user must keep track of the ``type_id``
/// of the operation in question.
///
/// In this case the `type_id` will match the memory address of the operation's
/// `QkCustomOpVTable vtable` as the same v-table should always be used with every
/// instance of the same operation.
///
/// This method should only work with gates defined in C. For any other case the return
/// value will always be ``UINT64_MAX``.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return The operation's `type_id` discriminant.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_type_id(inst: *const BoxedCustomOperation) -> u64 {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };
    let Some(op): Option<&CustomOp> = borrowed_inst.downcast_ref() else {
        return u64::MAX;
    };

    op.v_table as u64
}

/// @ingroup QkCustomOperation
///
/// Returns the original pointer to the operation enclosed within.
///
/// Users are expected to use ``qk_custom_operation_type_id`` to discriminate the object
/// based on its ``type_id``.
///
/// This method should only work with gates defined in C. For any other case the return
/// value will always be ``NULL``.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// @return The operation's original raw pointer.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is null or unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_raw(inst: *const BoxedCustomOperation) -> *const () {
    let borrowed_inst = unsafe { const_ptr_as_ref(inst) };
    let Some(op): Option<&CustomOp> = borrowed_inst.downcast_ref() else {
        return null();
    };

    op.orig.cast_const()
}

/// @ingroup QkCustomOperation
///
/// Frees the memory space allocated for a ``QkCustomOperation`` if not consumed by
/// a ``QkCircuit`` or a ``QkDAG``.
///
/// When a user creates a ``QkCustomOperation`` that just happens to never get consumed
/// by any `circuit` representation. The memory address should be freed using this method.
///
/// In this case that the ``QkCustomOperation`` instance was retrieved using
/// ``qk_circuit_get_custom_operation`` or ``qk_dag_get_custom_operation``, the user does
/// not need to call this method as the memory will be freed once ``qk_circuit_free`` or
/// ``qk_dag_free`` is called.
///
/// @param inst A pointer to the ``QkCustomOperation`` instance.
///
/// # Safety
///
/// Behavior is undefined if the `inst` pointer is unaligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_free(inst: *mut BoxedCustomOperation) {
    if !inst.is_null() {
        if !inst.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(inst);
        }
    }
}
