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

//! Simple bitwise manipulations.

/// The least significant bits in each byte of a `u64`.
pub const U64_LSBS: u64 = 0x0101_0101_0101_0101;

/// Spread each bit of a byte into the Lsb of the corresponding byte of a `u64`.
pub const fn spread(x: u8) -> u64 {
    let x = x as u64;
    // Move the top-four bits up by four bytes (minus four bits, so they're at the bottom).
    let x = (x | x << 28) & 0x0000_000F_0000_000F;
    // Move both sets of top-two bits up two bytes.
    let x = (x | x << 14) & 0x0003_0003_0003_0003;
    // ... and the top bits.
    (x | x << 7) & U64_LSBS
}

/// Copy the byte into each byte of a `u64`.
pub const fn splat(x: u8) -> u64 {
    x as u64 * U64_LSBS
}
