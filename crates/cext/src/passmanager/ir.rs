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

use std::any::Any;
use std::ffi::{CStr, c_char, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::{Arc, LazyLock};

use super::VtableEntry;
use crate::dyn_types::*;
use crate::pointers::{ExposesOwnedPointers, expose_by_box};
use qiskit_circuit::{circuit_data::CircuitData, dag_circuit::DAGCircuit};
use qiskit_passmanager::IR;
use qiskit_util::dyn_types::*;

/// The implementation of the [`IR`] trait behavior for types defined dynamically by C.
///
/// See [`CIr`] for the full wrapped up object, including the data pointer.
#[derive(Debug)]
struct IrVtable {
    /// Human-readable name of the IR.
    ///
    /// Primarily for debugging purposes.
    name: String,
    // TODO: delete.
}

/// C-API wrapper for IR objects where a C extension defined the [`IR`] trait implementation
/// dynamically.
///
/// This is specifically for wrapping only types that originate from C.  Other [`IR`] implementers
/// (such as [`DAGCircuit`]) are passed directly using their [`StaticIrExposer`] implementation.
///
/// # Safety
///
/// The `this` pointer must be safe to send and share between threads, and all `vtable` functions
/// must be callable from any thread..  Of particular note: it must be valid to call the
/// [`IrVtable::delete`] function from any thread.
struct CIr {
    /// Data pointer to the IR.
    this: *mut c_void,
    /// Implementation vtable of the [`IR`] trait.
    vtable: Arc<IrVtable>,
}
// SAFETY: per struct documentation, `this` and `vtable` can act from any thread.
unsafe impl Send for CIr {}
// SAFETY: per struct documentation, `this` and `vtable` can act from any thread.
unsafe impl Sync for CIr {}
impl CIr {
    /// Get the full dynamic type for a particular [`IrVtable`], if it were wrapped in this type.
    fn dyn_type_for_vtable(vtable: &IrVtable) -> DynTypeId<'_> {
        DynTypeId::of::<Self>().with_dynamic(ptr::from_ref(vtable).cast_mut().cast(), &vtable.name)
    }
}
impl DynTyped for CIr {
    fn dyn_type_id(&self) -> DynTypeId<'_> {
        Self::dyn_type_for_vtable(&self.vtable)
    }
}
impl IR for CIr {}

/// @ingroup pass-manager
/// Enumeration of the different built-in C API types that are usable directly as IRs.
///
/// These are the valid inputs to `qk_ir_handle_new`.
#[derive(Clone, Copy, derive_more::TryFrom, Debug)]
#[try_from(repr)]
#[repr(u32)]
pub enum IrBuiltin {
    /// The `QkCircuit` object.
    Circuit = 0,
    /// The `QkDag` object.
    Dag = 1,
}

/// @ingroup pass-manager
/// A handle object representing an "IR type".
///
/// These objects represent only the IR behavior; they must be combined with a data pointer of the
/// correct type to produce a complete IR object.  The type handle alone is needed to define passes
/// and pass managers.
///
/// Most functions that accept one of these borrow the argument.  Once you no longer need the handle
/// any more, you should call `qk_pass_ir_handle_free` to release your reference to the type object.
pub struct IrHandle(pub(super) Arc<dyn DynTraitExposer<dyn IR>>);
/// SAFETY: `IrHandle` is always exposed and freed via `Box`.
const _: () = unsafe { expose_by_box!(IrHandle) };

/// @ingroup pass-manager
/// Define the behavior of a new "IR" type.
///
/// This function is called to define the behavioral component of a new IR type.  The result of this
/// function can then be combined with a data pointer using `qk_pass_ir_new`.
///
/// If you want the handle for a built-in Qiskit type, see `qk_pass_ir_handle_builtin`.
///
/// @param name A human-readable name for the IR type.
/// @param table A table of the defined methods for the IR, terminated by an entry using `-1` as
///     its slot id.  This may be `NULL` if there are no entries in the table.  See
///     `QkIrVtableSlot` for the allowed values and function-pointer signatures.
/// @return An owned handle to the IR type object.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `name` is a nul-terminated string.
/// * `table` is a null pointer or aligned and points to a list of valid slot entries terminated by
///   an entry with `.slot = -1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_ir_handle_new(
    name: *const c_char,
    #[expect(unused_variables)] table: *const VtableEntry,
) -> *mut IrHandle {
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    let vtable = Arc::new(IrVtable { name });
    IrHandle(Arc::new(CIrExposer(vtable))).into_leaked()
}

/// @ingroup pass-manager
/// Get the IR "type object" for a built-in Qiskit type.
///
/// @param ty An identifier for the desired type.  See `QkIrBuiltin` for the allowed values.
/// @return An owned handle to the IR type object.
#[unsafe(no_mangle)]
pub extern "C" fn qk_ir_handle_builtin(ty: u32) -> *mut IrHandle {
    match IrBuiltin::try_from(ty) {
        Ok(IrBuiltin::Circuit) => {
            static CIRCUIT: LazyLock<Arc<dyn DynTraitExposer<dyn IR>>> =
                LazyLock::new(|| Arc::new(StaticIrExposer(PhantomData::<CircuitData>)));
            IrHandle(Arc::clone(&CIRCUIT)).into_leaked()
        }
        Ok(IrBuiltin::Dag) => {
            static DAG: LazyLock<Arc<dyn DynTraitExposer<dyn IR>>> =
                LazyLock::new(|| Arc::new(StaticIrExposer(PhantomData::<DAGCircuit>)));
            IrHandle(Arc::clone(&DAG)).into_leaked()
        }
        Err(_) => ptr::null_mut(),
    }
}

/// @ingroup pass-manager
/// Free an owned IR handle.
///
/// This can be called on your own owned handle objects once you are done with the handle.  Calling
/// free on your own handle does not invalidate any extant `QkIr` objects; they own their own
/// handles.
///
/// Does nothing if the pointer is null.
///
/// @param handle The pointer to free.
///
/// # Safety
///
/// Behavior is undefined if `handle` is not null and does not point to a valid owned `QkIrHandle`
/// object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_ir_handle_free(handle: *mut IrHandle) {
    _ = (!handle.is_null()).then(|| unsafe { IrHandle::steal(handle) })
}

// Add a trait exposer for all the Rust built-in types that implemented `IR` statically.
make_static_trait_exposer!(struct StaticIrExposer<T> for dyn IR);

/// A trait exposer for implementations of [`IR`] that were defined dynamically from C.
struct CIrExposer(Arc<IrVtable>);
unsafe impl DynTraitExposer<dyn IR> for CIrExposer {
    fn object_dyn_type_id(&self) -> DynTypeId<'_> {
        CIr::dyn_type_for_vtable(&self.0)
    }
    fn leak(&self, ob: Box<dyn IR>) -> *mut c_void {
        (ob as Box<dyn Any>)
            .downcast::<CIr>()
            .expect("called should ensure correct type")
            .this
    }
    unsafe fn steal(&self, ptr: *mut c_void) -> Box<dyn IR> {
        // TODO: there is a performance optimisation possible in the `CPass` logic, where we re-use
        // an existing `Box<CIr>` allocation if both the input and output IR types use it as the
        // backing dynamic type.  That optimisation probably extends to general exposure/leakers,
        // but let's leave it for the first implementation.

        // SAFETY: constructing the `CIr` implies that `ptr` is the correct type for our `vtable`.
        // Per documentation, the caller was responsible for ensuring that.
        Box::new(CIr {
            this: ptr,
            vtable: Arc::clone(&self.0),
        })
    }
}

// TODO: `qk_pass_ir_new`
