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

use binrw::binrw;

// Easy string storage when the string size is u16.
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(crate) struct StringU16Pack {
    #[br(temp)]
    #[bw(try_calc = u16::try_from(value.len()))]
    size: u16,

    #[br(count = size, try_map = String::from_utf8)]
    #[bw(map = |value| value.as_bytes())]
    pub value: String,
}

// Packing for Complex64 numbers to avoid serializing them into bytes.
#[binrw]
#[brw(big)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Complex64Pack {
    pub re: f64,
    pub im: f64,
}

// For matrices occurring in quantum computing, the most common sizes are 2x2 and 4x4.
// It's more compact to handle this case separately.
#[binrw]
#[brw(big)]
#[derive(Clone, Debug)]
pub(crate) enum Complex64MatrixPack {
    #[brw(magic = b'o')]
    OneQubit([Complex64Pack; 4]),
    #[brw(magic = b't')]
    TwoQubit([Complex64Pack; 16]),
    #[brw(magic = b'g')]
    GeneralMatrix(Complex64GeneralMatrixPack),
}

// A struct for general-size matrix.
#[binrw]
#[brw(big)]
#[derive(Clone, Debug)]
pub(crate) struct Complex64GeneralMatrixPack {
    rows: u32,
    cols: u32,

    #[bw(ignore)]
    #[br(
        temp,
        try_calc = rows
            .checked_mul(cols)
            .and_then(|count| usize::try_from(count).ok())
            .ok_or("matrix dimensions exceed the supported size")
    )]
    count: usize,

    #[br(count = count)]
    values: Vec<Complex64Pack>,
}
