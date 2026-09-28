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

use std::{
    any::Any,
    ffi::{CStr, c_char, c_void},
    marker::PhantomData,
    mem, ptr,
    sync::{Arc, LazyLock},
};

use crate::ExitCode;
use crate::dyn_types::*;
use crate::pointers::{
    ExposesOwnedPointers, arc_clone_from_raw, const_ptr_as_ref, expose_by_arc, expose_by_box,
    mut_ptr_as_ref,
};
use qiskit_circuit::{circuit_data::CircuitData, dag_circuit::DAGCircuit};
use qiskit_passmanager::{IR, Pass, PassContext, PassError, PassManager, Task};
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
    Circuit,
    /// The `QkDag` object.
    Dag,
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
pub struct IrHandle(Arc<dyn DynTraitExposer<dyn IR>>);
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

/// @ingroup pass-manager
/// Temporary documentation to be deleted in rebase.
#[repr(C)]
pub struct VtableEntry {
    /// The "slot" of the function, or the sentinel `(uint32_t)-1` to mark the final array entry.
    ///
    /// This is typically set to some `enum` value, where the particular `enum` varies depending on
    /// which vtable you are defining.
    pub slot: u32,
    /// Any additional "flags" for the particular table entry.  These will typically be or'd (`|`)
    /// together, and the valid set of flags will be documented by the table user.
    pub flags: u32,
    /// A function pointer implementing the correct signature for the combination of the `slot` and
    /// `flags`.
    ///
    /// This can be `NULL` only in the case of `slots` being the sentinel `-1`.
    pub ptr: *mut c_void,
}

// TODO: docs
pub struct Error(anyhow::Error);
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

/// @ingroup pass-manager
/// The available methods that can be implemented by a custom pass.
///
/// These are the valid values for `QkVtableEntry::slot` and the associated function-pointer types
/// required for the `table` argument of `qk_pass_new`.
///
/// # Safety
///
/// All functions, including the destructor, must be callable from any thread.
#[derive(Clone, Copy, derive_more::TryFrom, Debug)]
#[try_from(repr)]
#[repr(u32)]
pub enum PassSlot {
    /// Run the pass on an owned IR of the correct type. *Required*.
    ///
    /// Signature:
    /// ```c
    /// void *run(void *this, void *ir, QkPassContext *context, QkError **error);
    /// ```
    ///
    /// # Implementation
    ///
    /// This implements the pass behavior.  `this` will be equal to a value passed to `qk_pass_new`
    /// with this vtable.  `ir` will point to owned data of the type specified by the vtable's
    /// `ir_in` field.  It is the responsibility of the implementer to ensure `ir` is freed, if
    /// necessary.
    ///
    /// The function must return owned data of the type specified by `ir_out`, or `NULL` if an error
    /// as occurred.
    ///
    /// If the function wants to indicate an error state, it must write a valid owned `QkError *`
    /// object into the `error` pointer, ensure the `ir` pointer is freed, and return `NULL`.
    RunOwned = 0,
    /// A destructor for the `this` argument of a pass, at the time that the pass is destructed.
    /// *Optional*.
    ///
    /// Signature:
    /// ```c
    /// void delete(void *this);
    /// ```
    Delete = 1,
}

/// @ingroup pass-manager
/// A "type object" representing the behavior of a custom pass.
///
/// These objects represent only the behavior; they must be combined with a data pointer of the
/// correct type to produce a complete pass.
///
/// These are constructed using `qk_pass_vtable_new`, and are given to `qk_pass_new`.
///
/// See `QkPassSlot` for an enumeration of the possible behavior and its semantics.
pub struct PassVtable {
    /// A human-readable name for the pass.
    name: String,
    /// A handle to the complete IR methods of the input type, including the static constructor.
    ir_in: Arc<dyn DynTraitExposer<dyn IR>>,
    /// A handle to the complete IR methods of the output type, including the static constructor.
    ir_out: Arc<dyn DynTraitExposer<dyn IR>>,
    /// Run the pass on an owned copy of the IR.  See [`PassSlot::RunOwned`].
    run_owned: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut PassContext,
        *mut *mut Error,
    ) -> *mut c_void,
    /// The destructor of the [`CPass::this`] pointer.  See [`PassSlot::Delete`].
    delete: Option<unsafe extern "C" fn(*mut c_void) -> c_void>,
}
// SAFETY: `PassVtable` is always exposed and freed via an `Arc`.
const _: () = unsafe { expose_by_arc!(PassVtable) };
/// The primary constructor of `PassVtable` from Rust.
impl TryFrom<PassVtablePartial> for PassVtable {
    /// The first slot encountered that's required but absent.
    type Error = PassSlot;

    fn try_from(partial: PassVtablePartial) -> Result<Self, Self::Error> {
        Ok(Self {
            name: partial.name,
            ir_in: partial.ir_in,
            ir_out: partial.ir_out,
            run_owned: partial.run_owned.ok_or(PassSlot::RunOwned)?,
            delete: partial.delete,
        })
    }
}

/// @ingroup pass-manager
/// Create a new "type object" for a custom pass.
///
/// Once you have the output of this function, you create instances of the pass by combining it with
/// a data pointer using `qk_pass_new`.
///
/// @param name A borrowed nul-terminated human-readable name for the pass. Used for debugging.
/// @param ir_in A borrowed handle to the IR "type object" representing the expected input type for
///     the pass.
/// @param ir_out A borrowed handle to the IR "type object" representing the expected output type
///     for the pass.
/// @param table A borrowed table of `QkVtableEntry` objects terminated by an entry with `.slot =
///     -1`.  See `QkPassSlot` for the allowed slot values and associated function-pointer types.
///     Unknown values of `slot` are ignored.
/// @return An owned type object that can lent to `qk_pass_new` to create instances of the pass.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `name` is a nul-terminated string.
/// * `ir_in` points to a valid `QkIrHandle` object.
/// * `ir_out` points to a valid `QkIrHandle` object.
/// * `table` points to contiguous, valid instances of `VtableEntry` that are all consistent entries
///   as defined by `QkPassSlot`, with no duplicate `slot` entries and terminated by `.slot = -1`.
/// * `table` contains slot items for all required slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_vtable_new(
    name: *const c_char,
    ir_in: *const IrHandle,
    ir_out: *const IrHandle,
    mut table: *const VtableEntry,
) -> *const PassVtable {
    // SAFETY: per documentation `name` is a pointer to nul-terminated `char`s.
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: per documentation, `ir_in` points to a valid `IrHandle` object.
    let ir_in = unsafe { const_ptr_as_ref(ir_in) };
    // SAFETY: per documentation, `ir_out` points to a valid `IrHandle` object.
    let ir_out = unsafe { const_ptr_as_ref(ir_out) };
    let mut partial = PassVtablePartial::new(name, Arc::clone(&ir_in.0), Arc::clone(&ir_out.0));
    loop {
        // SAFETY: per documentation, `table` is valid for reads until we see the sentinel all-ones
        // pattern in a `slot`.
        let entry = unsafe { table.read() };
        table = if entry.slot == u32::MAX {
            break;
        } else {
            table.wrapping_add(1)
        };
        let Ok(slot) = PassSlot::try_from(entry.slot) else {
            // We assume this is a slot from a later version of Qiskit.
            // TODO: add an envvar / global to turn on debug information in these cases?
            continue;
        };
        // SAFETY: per documentation, `entry,ptr` is of the expected function-pointer type and valid
        // to call, because `entry.slot` was not all-ones.
        if unsafe { partial.set(slot, entry.ptr) } {
            // This a documented UB case.
            return ptr::dangling();
        }
    }
    // SAFETY: per documentation, all required methods were set.
    (unsafe { PassVtable::try_from(partial).unwrap_unchecked() }).into_leaked()
}

// TODO: we can very likely do a bit of macro trickery to simplify the creation of vtable objects,
// taking care that the filled public one needs to be visible to `cbindgen`.  For initial
// implementation, hard-coding is good enough.
/// Internal Rust builder object for constructing `PassVtable` via its [`TryFrom`] implementation.
struct PassVtablePartial {
    name: String,
    ir_in: Arc<dyn DynTraitExposer<dyn IR>>,
    ir_out: Arc<dyn DynTraitExposer<dyn IR>>,
    run_owned: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut PassContext,
            *mut *mut Error,
        ) -> *mut c_void,
    >,
    delete: Option<unsafe extern "C" fn(*mut c_void) -> c_void>,
}
impl PassVtablePartial {
    /// Initialize the builder.
    ///
    /// After this, call [`set`](Self::set) repeatedly to fill in the slots.
    fn new(
        name: String,
        ir_in: Arc<dyn DynTraitExposer<dyn IR>>,
        ir_out: Arc<dyn DynTraitExposer<dyn IR>>,
    ) -> Self {
        Self {
            name,
            ir_in,
            ir_out,
            run_owned: None,
            delete: None,
        }
    }

    /// Set the `slot` to the corresponding function `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid function pointer of the type expected by the corresponding method in
    /// [`PassVtable`].
    unsafe fn set(&mut self, slot: PassSlot, ptr: *mut c_void) -> bool {
        // This lint suppression is because there is no actual safety provided by writing out the
        // entire type again; the safety is only guaranteed as a long-range interaction of a C
        // caller matching the correct type signature in the documentation of `PassSlot`.
        #[allow(clippy::missing_transmute_annotations)]
        match slot {
            PassSlot::RunOwned => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<*mut c_void, _>(ptr) };
                self.run_owned.replace(ptr).is_some()
            }
            PassSlot::Delete => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<*mut c_void, _>(ptr) };
                self.delete.replace(ptr).is_some()
            }
        }
    }
}

/// @ingroup pass-manager
/// A pass with custom behavior defined through the C API.
///
/// This is a complete instance of a pass whose general behavior was previously defined using
/// `qk_pass_vtable_new`.  You construct instances of this struct by calling `qk_pass_new`.
///
/// These passes can be added to instances of `QkPassManager` using `qk_passmanager_push_pass`.
///
/// # Safety
///
/// The `this` data pointer must be safe to send and share between threads.  All methods in the
/// `QkPassVtable` must be safely callable from any thread.
pub struct CPass {
    /// The data pointer provided by C, and used as the `this` parameter in all [`PassVtable`]
    /// methods.
    this: *mut c_void,
    vtable: Arc<PassVtable>,
}
impl CPass {
    /// Create a new instance of the pass.
    ///
    /// # Safety
    ///
    /// 1. `this` must point to data of the type that is expected by all "`this`" arguments in
    ///    functions in the `vtable`.
    /// 2. the data pointed to by `this` must be safe to share and send between threads.
    unsafe fn new(this: *mut c_void, vtable: Arc<PassVtable>) -> Self {
        Self { this, vtable }
    }
}
// SAFETY: per struct documentation, the `this` pointer must be safe to send between threads.
unsafe impl Send for CPass {}
// SAFETY: per struct documentation, the `this` pointer must be safe to share between threads.
unsafe impl Sync for CPass {}
impl Pass for CPass {
    fn ir_id_in(&self) -> DynTypeId<'_> {
        self.vtable.ir_in.object_dyn_type_id()
    }
    fn ir_id_out(&self) -> DynTypeId<'_> {
        self.vtable.ir_out.object_dyn_type_id()
    }
    fn name(&self) -> &str {
        &self.vtable.name
    }
    fn run(&self, ir: Box<dyn IR>, context: &mut PassContext) -> Result<Box<dyn IR>, PassError> {
        let mut error = None::<ptr::NonNull<Error>>;
        let error_ptr = (&raw mut error).cast::<*mut Error>();
        let ir_in = self.vtable.ir_in.leak(ir).cast::<c_void>();
        let ir_out = unsafe { (self.vtable.run_owned)(self.this, ir_in, context, error_ptr) };
        match error {
            Some(error) => {
                let error = unsafe { Box::from_raw(error.as_ptr()) };
                Err(PassError::Runtime(error.0))
            }
            None => Ok(unsafe { self.vtable.ir_out.steal(ir_out) }),
        }
    }
}
impl Drop for CPass {
    fn drop(&mut self) {
        if let Some(delete) = self.vtable.delete {
            // SAFETY: per documentation of `PassVtable`, if the `delete` method is set, it is valid
            // to be passed `self.this` from any thread.
            unsafe { delete(self.this) };
        }
    }
}
// SAFETY: `CPass` is only created and freed via `Box`.
const _: () = unsafe { expose_by_box!(CPass) };

// TODO: having everything exposed as `CPass` is awkward for a future world where we have handles to
// built-in passes?

/// @ingroup pass-manager
/// Create a new instance of a pass whose behavior was previously defined.
///
/// The result pass object is typically then given to `qk_passmanager_push_pass`, or a related
/// method.
///
/// @param data The owned data pointer for this `vtable`.
/// @param vtable A borrowed "type object" created by `qk_pass_vtable_new`.
/// @return An owned pass object.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `this` points to data of the correct type for the methods in `vtable`.
/// * `vtable` points to a valid instance of `QkPassVtable`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_new(data: *mut c_void, vtable: *const PassVtable) -> *mut CPass {
    // SAFETY: per documentation, `vtable` is the still-valid result of `qk_pass_vtable_new`, which
    // returns the result of `Arc::into_raw`.
    let vtable = unsafe { arc_clone_from_raw(vtable) };
    // SAFETY: per documentation, `this` points to data of the type expected by `vtable` methods.
    (unsafe { CPass::new(data, vtable) }).into_leaked()
}

/// @ingroup pass-manager
/// Free a custom pass instance.
///
/// You typically do not need to call this, because the pass-manager building functions steal
/// ownership of `QkPass` arguments.
///
/// @param pass An owned pointer to the pass to free.
///
/// # Safety
///
/// Behavior is undefined if `pass` is not either null or a valid owned pointer to a `QkPass`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_free(pass: *mut CPass) {
    _ = (!pass.is_null()).then(|| unsafe { CPass::steal(pass) });
}

// SAFETY: `PassManager` is always exposed and freed by `Box`.
const _: () = unsafe { expose_by_box!(PassManager) };

/// @ingroup pass-manager
/// Create an empty pass manager.
///
/// @return An owned, empty pass manager.
///
/// This object must be freed by the user.
#[unsafe(no_mangle)]
pub extern "C" fn qk_passmanager_new() -> *mut PassManager {
    PassManager::new().into_leaked()
}

/// @ingroup pass-manager
/// Free the pass manager.
///
/// @param pm An owned pointer to the pass manager.
///
/// # Safety
///
/// Behavior is undefined if `pm` is not either null or a valid pointer to a `QkPassManager`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_free(pm: *mut PassManager) {
    _ = (!pm.is_null()).then(|| unsafe { PassManager::steal(pm) });
}

/// @ingroup pass-manager
/// Push a pass onto the pass manager.
///
/// @param pm A borrowed pointer to the pass manager.
/// @param pass An owned pass to push.  This steals ownership of the given pass.
///
/// @return `QkExitCode_Success` if the type was added successfully, or
///     `QkExitCode_IncompatibleTypes` if the pass could not be added because its expected IR types
///     do not match the pipeline's expectation.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `pm` points to a valid `QkPassManager`
/// * `pass` points to an owned `QkPass` instance.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_push_pass(
    pm: *mut PassManager,
    pass: *mut CPass,
) -> ExitCode {
    // SAFETY: per documentation, `pm` points to a valid `PassManager`.
    let pm = unsafe { mut_ptr_as_ref(pm) };
    // SAFETY: per documentation, `pass` points to a valid owned `CPass`.
    let pass = unsafe { CPass::steal(pass) };
    pm.try_push_task(Task::Transformation(pass as Box<dyn Pass>))
        .map_err(|_| ExitCode::IncompatibleTypes)
        .err()
        .unwrap_or(ExitCode::Success)
}

// TODO: should we move the "expose" logic into the core `qiskit-passmanager` crate, and remove the
// handles from `run_simple`?  Pro: simpler signature and less chance for disagreement.  Cons: moves
// C-specific exposure code into the core; motivates exposing the dynamic-type comparison logic to
// C, for it to check safety.

/// @ingroup pass-manager
/// Run the pass manager.
///
/// `ir_in_handle` and `ir_out_handle` must match the type expectations of the pipeline.  This
/// function will return `NULL` and (optionally) set the error state if this is violated.
///
/// @param pm A borrowed pointer to the pass manager to run.
/// @param ir An owned data pointer of the type expected by `ir_in_handle`.  This function steals
///     ownership of the pointer, and will arrange for it to be destructed, even if an error occurs.
/// @param ir_in_handle A borrowed type object corresponding to the input type of the pass manager,
///     and the type of `ir`.
/// @param ir_out_handle A borrowed type object corresponding to the output type of the pass
///     manager.
/// @param[out] error If not `NULL`, then a location to write out a `QkError *` result.
///
/// @return If no fatal error occurred, then an owned data pointer of the type corresponding to
///     `ir_out_handle`.  `NULL` if a fatal error occurred.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `pm` is a valid pointer to  `QkPassManager`.
/// * `ir` is an owning data pointer of the type indicated by `ir_in_handle`.
/// * `ir_in_handle` points a valid `IrHandle`.
/// * `ir_out_handle` points a valid `IrHandle`.
/// * `error` is either null, or points to a storage location valid for a single write of one
///   `QkError *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_run_simple(
    pm: *mut PassManager,
    ir: *mut c_void,
    ir_in_handle: *const IrHandle,
    ir_out_handle: *const IrHandle,
    error: *mut *const Error,
) -> *mut c_void {
    // SAFETY: Per documentation, `pm` is non-null and valid
    let pm = unsafe { mut_ptr_as_ref(pm) };
    let ir_in_handle = unsafe { const_ptr_as_ref(ir_in_handle) };
    let ir_out_handle = unsafe { const_ptr_as_ref(ir_out_handle) };
    let ir = unsafe { ir_in_handle.0.steal(ir) };
    pm.run_erased(ir)
        .map(|(ir_out, _)| ir_out_handle.0.leak(ir_out))
        .unwrap_or_else(|e| {
            if !error.is_null() {
                let e = Error(e).into_leaked();
                unsafe { error.write(e) };
            }
            ptr::null_mut()
        })
}
