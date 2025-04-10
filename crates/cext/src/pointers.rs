// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::exit_codes::CInputError;

/// Check the pointer is not null and is aligned.
pub(crate) fn check_ptr<T>(ptr: *const T) -> Result<(), CInputError> {
    if ptr.is_null() {
        return Err(CInputError::NullPointerError);
    };
    if !ptr.is_aligned() {
        return Err(CInputError::AlignmentError);
    };
    Ok(())
}

/// Casts a const pointer to a reference. Panics is the pointer is null or not aligned.
///
/// # Safety
///
/// This function requires ``ptr`` to be point to an initialized object of type ``T``.
/// While the resulting reference exists, the memory pointed to must not be mutated.
pub(crate) unsafe fn const_ptr_as_ref<'a, T>(ptr: *const T) -> &'a T {
    check_ptr(ptr).unwrap();
    let as_ref = unsafe { ptr.as_ref() };
    as_ref.unwrap() // we know the pointer is not null, hence we can safely unwrap
}

/// Casts a mut pointer to a mut reference. Panics is the pointer is null or not aligned.
///
/// # Safety
///
/// This function requires ``ptr`` to be point to an initialized object of type ``T``.
/// While the resulting reference exists, the memory pointed to must not be accessed otherwise.
pub(crate) unsafe fn mut_ptr_as_ref<'a, T>(ptr: *mut T) -> &'a mut T {
    check_ptr(ptr).unwrap();
    let as_mut_ref = unsafe { ptr.as_mut() };
    as_mut_ref.unwrap() // we know the pointer is not null, hence we can safely unwrap
}
