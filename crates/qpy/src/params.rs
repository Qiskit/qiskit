use std::str::FromStr;

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
use binrw::BinWrite;
use qiskit_circuit::operations::Param;

pub mod tags {
    pub const INTEGER: u8 = b'i';
    pub const FLOAT: u8 = b'f';
    pub const COMPLEX: u8 = b'c';
    pub const CASE_DEFAULT: u8 = b'd';
    pub const REGISTER: u8 = b'R';
    pub const NUMPY_OBJ: u8 = b'n';
    pub const PARAMETER: u8 = b'p';
    pub const PARAMETER_VECTOR: u8 = b'v';
    pub const PARAMETER_EXPRESSION: u8 = b'e';
    pub const STRING: u8 = b's';
    pub const NULL: u8 = b'z';
    pub const EXPRESSION: u8 = b'x';
    pub const MODIFIER: u8 = b'm';
}

#[derive(BinWrite)]
#[brw(big)]
pub struct SerializableParam {
    type_key: u8,
    data_len: u64,
    data: Vec<u8>
}

pub fn param_to_serializable(param: &Param) -> SerializableParam {
    match param {
        Param::Float(val) => SerializableParam { type_key: (tags::FLOAT), data_len: (8), data: (val.to_le_bytes().to_vec()) },
        _ => SerializableParam { type_key: (tags::NULL), data_len: (0), data: (Vec::new()) }
    }
}

