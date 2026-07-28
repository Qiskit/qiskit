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

use std::ptr;

use crate::pointers::mut_ptr_as_ref;
use num_bigint::BigUint;

/// An owned copy of an arbitrary-size unsigned integer.
///
/// The value is represented as little-endian bytes in `data`; that is, the least significant byte
/// is first.  A zero value has `num_bytes == 0` and `data == NULL`.
///
/// The memory for `data` is allocated by functions that return `QkBigUint`, and must be freed by
/// `qk_biguint_clear`.
#[repr(C)]
pub struct CBigUint {
    /// The little-endian bytes of the integer.
    pub(crate) data: *const u8,
    /// The number of bytes pointed to by `data`.
    pub(crate) num_bytes: usize,
}

pub(crate) fn biguint_to_c(value: &BigUint) -> CBigUint {
    let bytes = value.to_bytes_le().into_boxed_slice();
    CBigUint {
        num_bytes: bytes.len(),
        data: if bytes.is_empty() {
            ptr::null()
        } else {
            Box::into_raw(bytes) as *const u8
        },
    }
}

pub(crate) unsafe fn clear_biguint(value: &mut CBigUint) {
    if !value.data.is_null() && value.num_bytes > 0 {
        drop(unsafe {
            Box::from_raw(ptr::slice_from_raw_parts_mut(
                value.data as *mut u8,
                value.num_bytes,
            ))
        });
    }
    value.data = ptr::null();
    value.num_bytes = 0;
}

/// @ingroup QkBigUint
/// Clear a `QkBigUint` struct.
///
/// This function must be called to free the memory allocated by functions that
/// return `QkBigUint`. After calling this function, the data pointer in the
/// struct will be set to null and the byte count will be set to zero.
///
/// @param value A pointer to the `QkBigUint` struct to clear.
///
/// # Example
/// ```c
/// QkBigUint value = qk_value_big_uint(expr_value);
/// // Use the bytes...
/// qk_biguint_clear(&value);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``value`` is not a valid pointer to a `QkBigUint`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_biguint_clear(value: *mut CBigUint) {
    // SAFETY: Per documentation, value is a valid pointer to a CBigUint.
    let value = unsafe { mut_ptr_as_ref(value) };
    unsafe { clear_biguint(value) };
}
