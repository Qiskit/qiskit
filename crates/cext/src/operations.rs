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

use std::{
    ffi::{CStr, c_char, c_void},
    num::NonZero,
    ptr::{null, null_mut},
};

use qiskit_circuit::{
    circuit_data::CircuitData,
    operations::{CustomOperation, Operation, Param},
};

use crate::pointers::check_ptr;

/// Represents a quantum operation fully defined in C.
///
/// This operation object contains the minimal functionality an object
/// should adhere to in order operate on a ``QkCircuit``.
///
/// Any object that can be implemented using ``QkCustomOp`` will be
/// dynamically dispatched to be added to the circuit. In other words,
/// the circuit is unaware of the type of object it is accepting, but
/// it will work with it as long as it has the functionality expected
/// from any operation.
///
/// To achieve this, an operation is defined by two parts:
/// - The original pointer to the operation struct.
/// - The pointer to a vtable with the function slots that define
///   the functionality of this operation. See ``qk_custom_op_new_vtable``
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
/// QkCustomOpVtable *foo_vtable = qk_custom_op_new_vtable(entries);
///
/// // Declare a sample instance
/// struct foo_gate foo_3q = {
///     .num_qubits = 3,
/// };
///
/// // Create the custom operation
/// QkCustomOp foo_3q_custom = {
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
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CustomOp {
    /// A pointer to the original gate.
    orig: *mut (),
    /// A pointer to a vtable designed for the original gate.
    v_table: *mut CustomOpVtable,
}

impl CustomOp {
    /// Returns false if any of the pointers stored are null or unaligned
    pub fn is_valid(&self) -> bool {
        check_ptr(self.orig).is_ok_and(|_| check_ptr(self.v_table).is_ok())
    }
}

impl PartialEq for CustomOp {
    fn eq(&self, other: &Self) -> bool {
        ((unsafe { &*self.v_table }).eq)(self.orig, other.orig) && self.v_table == other.v_table
    }
}

unsafe impl Send for CustomOp {}
unsafe impl Sync for CustomOp {}

impl Operation for CustomOp {
    fn name(&self) -> &str {
        let name = ((unsafe { &*self.v_table }).name)(self.orig);
        let name_parsed = unsafe { CStr::from_ptr(name) };
        name_parsed
            .to_str()
            .expect("Expected a 'UTF-8' formatted string.")
    }

    fn num_qubits(&self) -> u32 {
        ((unsafe { &*self.v_table }).num_qubits)(self.orig)
    }

    fn num_clbits(&self) -> u32 {
        ((unsafe { &*self.v_table }).num_clbits)(self.orig)
    }

    fn num_params(&self) -> u32 {
        ((unsafe { &*self.v_table }).num_params)(self.orig)
    }

    fn directive(&self) -> bool {
        ((unsafe { &*self.v_table }).directive)(self.orig)
    }
}

impl CustomOperation for CustomOp {
    fn is_unitary(&self) -> bool {
        ((unsafe { &*self.v_table }).is_unitary)(self.orig)
    }

    fn num_ctrl_qubits(&self) -> Option<std::num::NonZero<u32>> {
        let num_ctrl_qubits = ((unsafe { &*self.v_table }).num_ctrl_qubits)(self.orig);
        match num_ctrl_qubits {
            0 => None,
            _ => Some(NonZero::new(num_ctrl_qubits).unwrap()),
        }
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        let definition = ((unsafe { &*self.v_table }).definition)(self.orig, params.as_ptr());
        if definition.is_null() {
            return None;
        }
        let circ = unsafe { Box::from_raw(*definition) };
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
/// * ``definition(*const (), *const Param)`` -> ``*mut *mut CircuitData``,
/// * ``eq(*const (), *const ())`` -> ``bool``, to compare two operations of the same kind.
#[derive(Debug, Clone)]
pub struct CustomOpVtable {
    pub name: fn(*const ()) -> *const c_char,
    pub num_qubits: fn(*const ()) -> u32,
    pub num_clbits: fn(*const ()) -> u32,
    pub num_params: fn(*const ()) -> u32,
    pub directive: fn(*const ()) -> bool,
    pub is_unitary: fn(*const ()) -> bool,
    pub num_ctrl_qubits: fn(*const ()) -> u32,
    pub label: fn(*const ()) -> *const c_char,
    pub definition: fn(*const (), *const Param) -> *mut *mut CircuitData,
    pub eq: fn(*const (), *const ()) -> bool,
}

fn default_num_ctrl_qubits(_slf: *const ()) -> u32 {
    0
}

fn default_label(_slf: *const ()) -> *const c_char {
    null()
}

fn default_definition(_slf: *const (), _params: *const Param) -> *mut *mut CircuitData {
    null_mut()
}

fn default_eq(slf: *const (), other: *const ()) -> bool {
    slf.eq(&other)
}

impl TryFrom<CustomOpVtablePartial> for CustomOpVtable {
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
    pub name: Option<fn(*const ()) -> *const c_char>,
    pub num_qubits: Option<fn(*const ()) -> u32>,
    pub num_clbits: Option<fn(*const ()) -> u32>,
    pub num_params: Option<fn(*const ()) -> u32>,
    pub directive: Option<fn(*const ()) -> bool>,
    pub is_unitary: Option<fn(*const ()) -> bool>,
    pub num_ctrl_qubits: Option<fn(*const ()) -> u32>,
    pub label: Option<fn(*const ()) -> *const c_char>,
    pub definition: Option<fn(*const (), *const Param) -> *mut *mut CircuitData>,
    pub eq: Option<fn(*const (), *const ()) -> bool>,
}

/// Represents the Vtable index of a `CustomOperation` coming from the
/// C domain.
///
/// Each named index refers to a required/optional method of the `Operation``
/// and `CustomOperation` traits.
#[repr(u32)]
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

/// @ingroup QkCustomOp
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
/// | ``definition``       | `const void *`, `QkParam *`    | `QkCircuit *` |   8   |    No    |
/// | ``eq``               | `const void *`, `const void *` | `bool`        |   9   |    No    |
///
/// If a required slot is not received, the vtable will not be constructed
/// and this function will return a `NULL` pointer. If an optional slot is not
/// included, the vtable will still be built and its slots will point to default
/// implementations of the said method(s).
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_op_new_vtable(
    mut slots: *const CustomOpVTableEntry,
) -> *mut CustomOpVtable {
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
                    std::mem::transmute::<*const c_void, fn(*const ()) -> *const c_char>(slot.func)
                })
            }
            Ok(NumQubits) => {
                if vtable.num_qubits.is_some() {
                    panic!("NumQubits slot has already been set.")
                }
                vtable.num_qubits = Some(unsafe {
                    std::mem::transmute::<*const c_void, fn(*const ()) -> u32>(slot.func)
                })
            }
            Ok(NumClbits) => {
                if vtable.num_clbits.is_some() {
                    panic!("NumClbits slot has already been set.")
                }
                vtable.num_clbits = Some(unsafe {
                    std::mem::transmute::<*const c_void, fn(*const ()) -> u32>(slot.func)
                })
            }
            Ok(NumParams) => {
                if vtable.num_params.is_some() {
                    panic!("NumParams slot has already been set.")
                }
                vtable.num_params = Some(unsafe {
                    std::mem::transmute::<*const c_void, fn(*const ()) -> u32>(slot.func)
                })
            }
            Ok(Directive) => {
                if vtable.directive.is_some() {
                    panic!("Directive slot has already been set.")
                }
                vtable.directive = Some(unsafe {
                    std::mem::transmute::<*const c_void, fn(*const ()) -> bool>(slot.func)
                })
            }
            Ok(IsUnitary) => {
                if vtable.is_unitary.is_some() {
                    panic!("IsUnitary slot has already been set.")
                }
                vtable.is_unitary = Some(unsafe {
                    std::mem::transmute::<*const c_void, fn(*const ()) -> bool>(slot.func)
                })
            }
            Ok(NumCtrlQubits) => {
                if vtable.num_ctrl_qubits.is_some() {
                    panic!("NumCtrlQubits slot has already been set.")
                }
                vtable.num_ctrl_qubits = Some(unsafe {
                    std::mem::transmute::<*const c_void, fn(*const ()) -> u32>(slot.func)
                })
            }
            Ok(Label) => {
                if vtable.label.is_some() {
                    panic!("Label slot has already been set.")
                }
                vtable.label = Some(unsafe {
                    std::mem::transmute::<*const c_void, fn(*const ()) -> *const c_char>(slot.func)
                })
            }
            Ok(Definition) => {
                if vtable.definition.is_some() {
                    panic!("Name slot has already been set.")
                }
                vtable.definition = Some(unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        fn(*const (), *const Param) -> *mut *mut CircuitData,
                    >(slot.func)
                })
            }
            Ok(Eq) => {
                if vtable.eq.is_some() {
                    panic!("Name slot has already been set.")
                }
                vtable.eq = Some(unsafe {
                    std::mem::transmute::<*const c_void, fn(*const (), *const ()) -> bool>(
                        slot.func,
                    )
                })
            }
            Err(e) => panic!("Expected valid slot, obtained {}", e),
        }
        slots = unsafe { slots.add(1) };
        slot = unsafe { slots.read() };
    }
    CustomOpVtable::try_from(vtable)
        .map(|x| Box::into_raw(Box::new(x)))
        .unwrap_or(std::ptr::null_mut())
}
