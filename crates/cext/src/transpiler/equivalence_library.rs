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

use qiskit_transpiler::equivalence::EquivalenceLibrary;
use qiskit_transpiler::standard_equivalence_library::generate_standard_equivalence_library;

/// @ingroup QkEquivalenceLibrary
/// Construct a new ``QkEquivalenceLibrary`` populated with Qiskit's standard
/// equivalences between gates.
///
/// @return A pointer to the new ``QkEquivalenceLibrary``.
///
/// # Example
///
/// ```c
///     QkEquivalenceLibrary *lib = qk_equivalence_library_new_standard();
///     qk_equivalence_library_free(lib);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_equivalence_library_new_standard() -> *mut EquivalenceLibrary {
    Box::into_raw(Box::new(generate_standard_equivalence_library()))
}

/// @ingroup QkEquivalenceLibrary
/// Free the ``QkEquivalenceLibrary``.
///
/// @param library A pointer to the ``QkEquivalenceLibrary`` to free.
///
/// # Example
///
/// ```c
///     QkEquivalenceLibrary *lib = qk_equivalence_library_new_standard();
///     qk_equivalence_library_free(lib);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``library`` is not a valid, non-null pointer to a
/// ``QkEquivalenceLibrary``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_equivalence_library_free(library: *mut EquivalenceLibrary) {
    if !library.is_null() {
        if !library.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(library);
        }
    }
}
