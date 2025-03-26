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

use super::sparse_observable::check_ptr;
use qiskit_accelerate::nlayout::{NLayout, PhysicalQubit};

/// @ingroup QkLayout
/// Create a layout object from a virtual-to-physical qubit map.
///
/// @param virt_to_phys A pointer to an array of length ``len``, defining the mapping
///     from virtual to physical qubits. A null-pointer is not allowed.
/// @param len The length of the above array.
///
/// @return The layout.
///
/// # Example
///
///     uint32_t virt_to_phys[5] = {0, 2, 3, 4, 1};
///     QkLayout layout = qk_layout_new(virt_to_phys, 5);
///
/// # Safety
///
/// Behavior is undefined if ``virt_to_phys`` is not a non-null pointer to an array of ``uint32_t``,
/// readable for ``len`` elements.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_layout_new(virt_to_phys: *const u32, len: usize) -> *mut NLayout {
    check_ptr(virt_to_phys).unwrap();

    // SAFETY: The pointer is non-null and aligned and the user promises it is readable
    // for the required length.
    let data = unsafe { ::std::slice::from_raw_parts(virt_to_phys, len) };

    let vector = data.iter().map(|q| PhysicalQubit(*q)).collect();
    let layout = NLayout::from_virtual_to_physical(vector);

    Box::into_raw(Box::new(layout))
}

/// @ingroup QkLayout
/// Free the layout object.
///
/// @param obs A pointer to the layout to free.
///
/// # Example
///
///     uint32_t virt_to_phys[5] = {0, 2, 3, 4, 1};
///     QkLayout layout = qk_layout_new(virt_to_phys, 5);
///     qk_layout_free(layout);
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not either null or a valid pointer to a
/// [NLayout].
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_layout_free(layout: *mut NLayout) {
    if !layout.is_null() {
        if !layout.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(layout);
        }
    }
}
