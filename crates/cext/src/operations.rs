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
    ffi::{CStr, c_char},
    num::NonZero,
};

use num_complex::Complex64;
use qiskit_circuit::{
    circuit_data::CircuitData,
    operations::{CustomOperation, Operation, Param},
};

/// DOCS: TODO
#[repr(C)]
#[derive(Debug, Clone)]
pub struct QkOperation {
    orig: *mut (),
    v_table: *mut QkOperationVtable,
}

/// SAFETY: TODO
unsafe impl Send for QkOperation {}
/// SAFETY: TODO
unsafe impl Sync for QkOperation {}

impl Operation for QkOperation {
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

impl PartialEq for QkOperation {
    fn eq(&self, _other: &Self) -> bool {
        // ((unsafe { &*self.v_table }).eq)(other.orig)
        todo!()
    }
}

// impl Clone for QkOperation {
//     fn clone(&self) -> Self {
//         Self { orig: (((unsafe {&*self.v_table}).clone))(self.orig), v_table: self.v_table.clone() }
//     }
// }

impl CustomOperation for QkOperation {
    fn is_unitary(&self) -> bool {
        todo!()
    }

    fn num_ctrl_qubits(&self) -> Option<std::num::NonZero<u32>> {
        let num_ctrl_qubits = ((unsafe { &*self.v_table }).num_ctrl_qubits)(self.orig);
        match num_ctrl_qubits {
            0 => None,
            _ => Some(NonZero::new(num_ctrl_qubits).unwrap()),
        }
    }

    fn is_controlled_gate(&self) -> bool {
        ((unsafe { &*self.v_table }).is_controlled)(self.orig)
    }

    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        todo!()
    }

    fn inverse(
        &self,
        _params: &[Param],
    ) -> Option<(
        qiskit_circuit::packed_instruction::PackedOperation,
        smallvec::SmallVec<[Param; 3]>,
    )> {
        todo!()
    }

    fn label(&self) -> Option<&str> {
        todo!()
    }

    fn matrix(&self, _params: &[Param]) -> Option<ndarray::prelude::Array2<Complex64>> {
        todo!()
    }
}

/// DOCS: TODO
#[derive(Debug, Clone)]
#[repr(C)]
pub struct QkOperationVtable {
    pub name: fn(*const ()) -> *const i8,
    pub num_qubits: fn(*const ()) -> u32,
    pub num_clbits: fn(*const ()) -> u32,
    pub num_params: fn(*const ()) -> u32,
    pub directive: fn(*const ()) -> bool,
    pub num_ctrl_qubits: fn(*const ()) -> u32,
    pub is_controlled: fn(*const ()) -> bool,
    pub is_unitary: fn(*const ()) -> bool,
    pub label: fn(*const ()) -> *mut c_char,
    pub definition: fn(*const (), *mut *mut Param) -> *mut *mut CircuitData,
    pub matrix: fn(*const (), *mut *mut Param) -> *mut Complex64,
    // pub eq: fn(*const ()) -> bool,
    // pub clone: fn(*const ()) -> *mut (),
}

/// DOCS: TODO
#[derive(Debug, Clone)]
#[repr(C)]
pub struct QkOperationVtablePartial {
    pub name: Option<fn(*const ()) -> *const i8>,
    pub num_qubits: Option<fn(*const ()) -> u32>,
    pub num_clbits: Option<fn(*const ()) -> u32>,
    pub num_params: Option<fn(*const ()) -> u32>,
    pub directive: Option<fn(*const ()) -> bool>,
    pub num_ctrl_qubits: Option<fn(*const ()) -> u32>,
    pub is_controlled: Option<fn(*const ()) -> bool>,
    pub is_unitary: Option<fn(*const ()) -> bool>,
    pub label: Option<fn(*const ()) -> *mut c_char>,
    pub definition: Option<fn(*const (), *mut *mut Param) -> *mut *mut CircuitData>,
    pub matrix: Option<fn(*const (), *mut *mut Param) -> *mut Complex64>,
    // pub eq: Option<fn(*const ()) -> bool>,
    // pub clone: Option<fn(*const ()) -> *mut ()>,
}

/// TODO: Docs
#[repr(u32)]
pub enum QkCustomOperationMethod {
    Name = 0,
    NumQubits = 1,
    NumClbits = 2,
    NumParams = 3,
    Directive = 4,
    NumCtrlQubits = 5,
    IsControlled = 6,
    IsUnitary = 7,
    Label = 8,
    Definition = 9,
    Matrix = 10,
}

impl TryFrom<u32> for QkCustomOperationMethod {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        use QkCustomOperationMethod::*;
        let ret = match value {
            0 => Name,
            1 => NumQubits,
            2 => NumClbits,
            3 => NumParams,
            4 => Directive,
            5 => NumCtrlQubits,
            6 => IsControlled,
            7 => IsUnitary,
            8 => Label,
            9 => Definition,
            10 => Matrix,
            _ => return Err(value),
        };
        Ok(ret)
    }
}
