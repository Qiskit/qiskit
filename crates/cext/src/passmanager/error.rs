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

use crate::pointers::{ExposesOwnedPointers, expose_by_box};
use std::ffi::{CStr, c_char};

// TODO: docs
pub struct Error(pub anyhow::Error);
const _: () = unsafe { expose_by_box!(Error) };

// TODO: docs
/// Create a new error message.
///
/// @param msg A borrowed pointer to a nul-terminated string.
///
/// # Safety
///
/// `msg` must point to nul-terminated bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_error_new(msg: *const c_char) -> *mut Error {
    // SAFETY: per documentation, `msg` points to nul-terminated bytes.
    let msg = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    Error(anyhow::Error::msg(msg)).into_leaked()
}
