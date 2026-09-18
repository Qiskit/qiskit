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

//! The element type of a tensor.

use std::fmt;

/// The possible data types for a Tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    C128, // complex
    C64,
    F64, // real
    F32,
    I64, // signed integer
    I32,
    I16,
    I8,
    U64, // unsigned integer
    U32,
    U16,
    U8,
    Bit, // bool
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string_repr = match self {
            DType::C128 => "C128",
            DType::C64 => "C64",
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U64 => "U64",
            DType::U32 => "U32",
            DType::U16 => "U16",
            DType::U8 => "U8",
            DType::Bit => "Bit",
        };
        write!(f, "{string_repr}")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dtype_display() {
        use DType::*;
        let cases = [
            (C128, "C128"),
            (C64, "C64"),
            (F64, "F64"),
            (F32, "F32"),
            (I64, "I64"),
            (I32, "I32"),
            (I16, "I16"),
            (I8, "I8"),
            (U64, "U64"),
            (U32, "U32"),
            (U16, "U16"),
            (U8, "U8"),
            (Bit, "Bit"),
        ];
        let mut fails = vec![];
        for (dtype, expected) in cases {
            let got = format!("{dtype}");
            if got != expected {
                fails.push((dtype, expected, got));
            }
        }
        assert_eq!(fails, [], "DType Display mismatches: {fails:?}");
    }
}
