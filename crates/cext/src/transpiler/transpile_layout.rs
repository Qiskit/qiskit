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

use crate::pointers::const_ptr_as_ref;
use qiskit_transpiler::transpile_layout::TranspileLayout;

/// @ingroup QkTranspileLayout
/// Return the number of qubits in the input circuit to the transpiler.
///
/// @param layout A pointer to the ``QkTranspileLayout``.
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpile_layout_num_input_qubits(
    layout: *const TranspileLayout,
) -> u32 {
    let layout = unsafe { const_ptr_as_ref(layout) };
    layout.num_input_qubits()
}

/// @ingroup QkTranspileLayout
/// Return the number of qubits in the output circuit from the transpiler.
///
/// @param layout A pointer to the ``QkTranspileLayout``.
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpile_layout_num_output_qubits(
    layout: *const TranspileLayout,
) -> u32 {
    let layout = unsafe { const_ptr_as_ref(layout) };
    layout.num_output_qubits()
}

/// A layout array object that represent the initial layout, routing permutation
/// or final layout depending on which function was used to generate it.
#[repr(C)]
pub struct QkLayoutArray {
    /// The array size
    size: usize,
    /// A pointer to an array of ``uint32_t`` representing the physical qubit
    /// indices.
    array: *mut u32,
}

/// @ingroup QkTranspileLayout
/// Return the initial layout of a ``QkTranspileLayout``.
///
/// @param layout A pointer to the ``QkTranspileLayout``.
/// @param filter_ancillas If set to true the output array will not include any indicies for any
/// ancillas added by the transpiler.
///
/// @returns An QkLayoutArray object which is a size and pointer to an array of the initial layout. This
/// will need to be passed to ``qk_layout_array_free()`` to free the array allocation. A
/// QkLayoutArray with a size 0 and null pointer for the array is returned if there is no initial layout
/// present in the provided ``QkTranspileLayout``.
///
/// # Safety
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpile_layout_initial_layout(
    layout: *const TranspileLayout,
    filter_ancillas: bool,
) -> QkLayoutArray {
    let layout = unsafe { const_ptr_as_ref(layout) };
    let initial_layout = layout.initial_layout(filter_ancillas);
    if let Some(initial_layout) = initial_layout {
        let mut box_slice = initial_layout.into_boxed_slice();
        let array = box_slice.as_mut_ptr() as *mut u32;
        let size = box_slice.len();
        let _ = Box::into_raw(box_slice);
        QkLayoutArray { size, array }
    } else {
        QkLayoutArray {
            size: 0,
            array: std::ptr::null_mut(),
        }
    }
}

/// @ingroup QkTranspileLayout
/// Return the routing permutation of a ``QkTranspileLayout``
///
/// @param layout A pointer to the ``QkTranspileLayout``.
///
/// @returns A QkLayoutArray object which is a size and pointer to an array of the routing
/// permutation. This will need to be passed to ``qk_layout_array_free()`` to free the array allocation.
///
/// # Safety
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpile_layout_output_permutation(
    layout: *const TranspileLayout,
) -> QkLayoutArray {
    let layout = unsafe { const_ptr_as_ref(layout) };
    let permutation = layout.explicit_output_permutation().to_vec();
    let mut box_slice = permutation.into_boxed_slice();
    let array = box_slice.as_mut_ptr() as *mut u32;
    let size = box_slice.len();
    let _ = Box::into_raw(box_slice);
    QkLayoutArray { size, array }
}

/// @ingroup QkTranspileLayout
/// Return the final layout of a ``QkTranspileLayout``
///
/// @param layout A pointer to the ``QkTranspileLayout``.
/// @param filter_ancillas If set to true the output array will not include any indicies for any
/// ancillas added by the transpiler.
///
/// @returns A QkLayoutArray object which is a size and pointer to an array of the final layout. This
/// will need to be passed to ``qk_layout_array_free()`` to free the array allocation.
///
/// # Safety
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpile_layout_final_layout(
    layout: *const TranspileLayout,
    filter_ancillas: bool,
) -> QkLayoutArray {
    let layout = unsafe { const_ptr_as_ref(layout) };
    let final_layout = layout.final_index_layout(filter_ancillas);
    let mut box_slice = final_layout.into_boxed_slice();
    let array = box_slice.as_mut_ptr() as *mut u32;
    let size = box_slice.len();
    let _ = Box::into_raw(box_slice);
    QkLayoutArray { size, array }
}

/// @ingroup QkTranspileLayout
/// Free a QkLayoutArray
///
/// This function will free the memory allocation for an array pointed to by a ``QkLayoutArray``.
///
/// @param layout_array The QkLayoutArray object to free the array allocation of.
///
/// # Safety
///
/// Behavior is undefined if ``layout_array`` is not an QkLayoutArray object created by either
/// ``qk_transpile_layout_final_layout()``, ``qk_transpile_layout_output_permutation()``, or
/// ``qk_transpile_layout_initial_layout()``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_layout_array_free(layout_array: QkLayoutArray) {
    if layout_array.size == 0 && layout_array.array.is_null() {
        return;
    }
    unsafe {
        let data = std::slice::from_raw_parts_mut(layout_array.array, layout_array.size);
        let _: Box<[u32]> = Box::from_raw(data);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use qiskit_circuit::bit::ShareableQubit;
    use qiskit_circuit::nlayout::{NLayout, PhysicalQubit};
    use qiskit_circuit::Qubit;
    use qiskit_transpiler::transpile_layout::TranspileLayout;

    #[test]
    fn test_final_layout() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(routing_permutation),
            input_qubits,
            3,
        );
        let result = layout.final_index_layout(true);
        let expected = vec![PhysicalQubit(3), PhysicalQubit(5), PhysicalQubit(2)];
        assert_eq!(expected, result);
        let array_result = unsafe { qk_transpile_layout_final_layout(&layout, true) };
        let slice = unsafe { std::slice::from_raw_parts(array_result.array, array_result.size) };
        let expected = expected.into_iter().map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(expected.as_slice(), slice);
        unsafe {
            qk_layout_array_free(array_result);
        }
    }

    #[test]
    fn test_initial_layout() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(routing_permutation),
            input_qubits,
            3,
        );
        let expected: Vec<u32> = [PhysicalQubit(9), PhysicalQubit(4), PhysicalQubit(0)]
            .into_iter()
            .map(|x| x.0)
            .collect();
        let array_result = unsafe { qk_transpile_layout_initial_layout(&layout, true) };
        let slice = unsafe { std::slice::from_raw_parts(array_result.array, array_result.size) };
        assert_eq!(expected.as_slice(), slice);
        unsafe { qk_layout_array_free(array_result) };
    }

    #[test]
    fn test_routing_permutation() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let expected = routing_permutation.iter().map(|x| x.0).collect::<Vec<_>>();
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(routing_permutation),
            input_qubits,
            3,
        );
        let array_result = unsafe { qk_transpile_layout_output_permutation(&layout) };
        let slice = unsafe { std::slice::from_raw_parts(array_result.array, array_result.size) };
        assert_eq!(expected.as_slice(), slice);
        unsafe { qk_layout_array_free(array_result) };
    }

    #[test]
    fn test_input_num_qubits() {
        let initial_layout_vec = (0..256).rev().map(PhysicalQubit::new).collect();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let input_qubits = vec![ShareableQubit::new_anonymous(); 256];
        let layout = TranspileLayout::new(Some(initial_layout), None, input_qubits, 3);
        unsafe {
            assert_eq!(qk_transpile_layout_num_input_qubits(&layout), 3);
        }
    }

    #[test]
    fn test_output_num_qubits() {
        let initial_layout_vec = (0..256).rev().map(PhysicalQubit::new).collect();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let input_qubits = vec![ShareableQubit::new_anonymous(); 256];
        let layout = TranspileLayout::new(Some(initial_layout), None, input_qubits, 3);
        unsafe {
            assert_eq!(qk_transpile_layout_num_output_qubits(&layout), 256);
        }
    }
}
