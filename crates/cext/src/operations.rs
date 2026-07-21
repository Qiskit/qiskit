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

/// Represents an Operation fully defined in C.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CustomOp {
    orig: *mut (),
    v_table: *mut CustomOpVtable,
}

impl PartialEq for CustomOp {
    fn eq(&self, other: &Self) -> bool {
        ((unsafe { &*self.v_table }).eq)(self.orig, other.orig) && self.v_table == other.v_table
    }
}

/// SAFETY: TODO
unsafe impl Send for CustomOp {}
/// SAFETY: TODO
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

/// DOCS: TODO
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

/// TODO: Docs
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

impl CustomOpVtablePartial {
    pub const DEFAULT: CustomOpVtablePartial = CustomOpVtablePartial {
        name: None,
        num_qubits: None,
        num_clbits: None,
        num_params: None,
        directive: None,
        is_unitary: None,
        num_ctrl_qubits: None,
        label: None,
        definition: None,
        eq: None,
    };
}

/// TODO: Docs
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

/// TODO: Docs
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CustomOpVTableEntry {
    slot: u32,
    func: *const ::std::ffi::c_void,
}

impl CustomOpVTableEntry {
    pub const SENTINEL: Self = Self {
        slot: u32::MAX,
        func: ::std::ptr::null(),
    };
}

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
