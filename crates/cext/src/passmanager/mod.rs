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

use qiskit_circuit::{circuit_data::CircuitData, dag_circuit::DAGCircuit};
use qiskit_passmanager::{
    DynTypeId, DynTyped, IR, Pass, PassContext, PassError, PassManager, StaticDynTyped, Task,
};

use std::{
    any::Any,
    ffi::{CStr, c_char, c_void},
    marker::PhantomData,
    mem, ptr,
    sync::{Arc, LazyLock},
};

use crate::{
    ExitCode,
    pointers::{
        ExposesOwnedPointers, arc_clone_from_raw, const_ptr_as_ref, expose_by_arc, expose_by_box,
        mut_ptr_as_ref,
    },
};

unsafe trait IRExposer: Send + Sync + 'static {
    fn ir_dyn_type_id(&self) -> DynTypeId<'_>;
    fn leak(&self, ob: Box<dyn IR>) -> *mut c_void;
    unsafe fn steal(&self, ptr: *mut c_void) -> Box<dyn IR>;
}

struct StaticIRExposer<T>(PhantomData<T>);
unsafe impl<T> IRExposer for StaticIRExposer<T>
where
    T: IR + StaticDynTyped + ExposesOwnedPointers<Owner = Box<T>>,
{
    fn ir_dyn_type_id(&self) -> DynTypeId<'_> {
        T::static_dyn_type_id()
    }
    fn leak(&self, ob: Box<dyn IR>) -> *mut c_void {
        let typed = (ob as Box<dyn Any>)
            .downcast::<T>()
            .expect("called should ensure correct type");
        T::leak(typed).cast()
    }
    unsafe fn steal(&self, ptr: *mut c_void) -> Box<dyn IR> {
        (unsafe { T::steal(ptr.cast()) }) as Box<dyn IR>
    }
}

#[derive(Debug)]
struct IRVTable {
    name: String,
    // ... we don't have any instance methods on `IR` yet.
}

struct CIr {
    ir: *mut c_void,
    vtable: Arc<IRVTable>,
}
unsafe impl Send for CIr {}
unsafe impl Sync for CIr {}
impl CIr {
    fn dyn_type_for_vtable(vtable: &IRVTable) -> DynTypeId<'_> {
        DynTypeId::of::<Self>().with_dynamic(ptr::from_ref(vtable).cast_mut().cast(), &vtable.name)
    }
}
impl DynTyped for CIr {
    fn dyn_type_id(&self) -> DynTypeId<'_> {
        Self::dyn_type_for_vtable(&self.vtable)
    }
}
impl IR for CIr {}

pub struct IRHandle(Arc<dyn IRExposer>);
const _: () = unsafe { expose_by_box!(IRHandle) };

#[derive(Clone, Copy, derive_more::TryFrom, Debug)]
#[try_from(repr)]
#[repr(u32)]
pub enum BuiltinIR {
    Circuit,
    Dag,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_ir_handle_new(
    name: *const c_char,
    #[expect(unused_variables)] table: *const VTableEntry,
) -> *mut IRHandle {
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    let vtable = Arc::new(IRVTable { name });
    IRHandle(Arc::new(CIrExposer(vtable))).into_leaked()
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_pass_ir_handle_builtin(ty: u32) -> *mut IRHandle {
    match BuiltinIR::try_from(ty) {
        Ok(BuiltinIR::Circuit) => {
            static CIRCUIT: LazyLock<Arc<dyn IRExposer>> =
                LazyLock::new(|| Arc::new(StaticIRExposer(PhantomData::<CircuitData>)));
            IRHandle(Arc::clone(&CIRCUIT)).into_leaked()
        }
        Ok(BuiltinIR::Dag) => {
            static DAG: LazyLock<Arc<dyn IRExposer>> = LazyLock::new(|| {
                Arc::new(StaticIRExposer(PhantomData::<DAGCircuit>)) as Arc<dyn IRExposer>
            });
            IRHandle(Arc::clone(&DAG)).into_leaked()
        }
        Err(_) => ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_ir_handle_free(handle: *mut IRHandle) {
    _ = (!handle.is_null()).then(|| unsafe { IRHandle::steal(handle) })
}

struct CIrExposer(Arc<IRVTable>);
unsafe impl IRExposer for CIrExposer {
    fn ir_dyn_type_id(&self) -> DynTypeId<'_> {
        CIr::dyn_type_for_vtable(&self.0)
    }
    fn leak(&self, ob: Box<dyn IR>) -> *mut c_void {
        (ob as Box<dyn Any>)
            .downcast::<CIr>()
            .expect("called should ensure correct type")
            .ir
    }
    unsafe fn steal(&self, ptr: *mut c_void) -> Box<dyn IR> {
        // TODO: there is a performance optimisation possible in the `CPass` logic, where we re-use
        // an existing `Box<CIr>` allocation if both the input and output IR types use it as the
        // backing dynamic type.  That optimisation probably extends to general exposure/leakers,
        // but let's leave it for the first implementation.

        // SAFETY: constructing the `CIr` implies that `ptr` is the correct type for our `vtable`.
        // Per documentation, the caller was responsible for ensuring that.
        Box::new(CIr {
            ir: ptr,
            vtable: Arc::clone(&self.0),
        })
    }
}

// TODO: docs
/// @ingroup QkPassManager
/// An entry in a vtable for defining objects with custom behavior.
#[repr(C)]
pub struct VTableEntry {
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

/// @ingroup QkError
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

// TODO: docs
#[derive(Clone, Copy, derive_more::TryFrom, Debug)]
#[try_from(repr)]
#[repr(u32)]
pub enum PassSlot {
    /// Run the pass on an IR of the correct type. *Required*.
    ///
    /// Signature:
    /// ```c
    /// void *run(void *this, void *ir, QkPassContext *context, QkError **error);
    /// ```
    ///
    /// # Implementation
    ///
    /// This implements the pass behavior.  `this` will be equal to a value passed to `qk_pass_new`
    /// with this vtable.  `ir` will point to data of the type specified by the vtable's `ir_in`
    /// field.
    // TODO: ownership of `ir`?
    Run = 0,
    /// A destructor for the `this` argument of a pass, at the time that the pass is destructed.
    /// *Optional*.
    ///
    /// Signature:
    /// ```c
    /// void delete(void *this);
    /// ```
    ///
    /// # Safety
    ///
    /// This must be safe to call from any thread.
    Delete = 1,
}

// TODO: docs
pub struct PassVTable {
    name: String,
    ir_in: Arc<dyn IRExposer>,
    ir_out: Arc<dyn IRExposer>,
    run: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut PassContext,
        *mut *mut Error,
    ) -> *mut c_void,
    delete: Option<unsafe extern "C" fn(*mut c_void) -> c_void>,
}
// TODO: doc.
const _: () = unsafe { expose_by_arc!(PassVTable) };
impl TryFrom<PassVTablePartial> for PassVTable {
    /// The first slot encountered that's required but absent.
    type Error = PassSlot;

    fn try_from(partial: PassVTablePartial) -> Result<Self, Self::Error> {
        Ok(Self {
            name: partial.name,
            ir_in: partial.ir_in,
            ir_out: partial.ir_out,
            run: partial.run.ok_or(PassSlot::Run)?,
            delete: partial.delete,
        })
    }
}

// TODO: docs
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_vtable_new(
    name: *const c_char,
    ir_in: *const IRHandle,
    ir_out: *const IRHandle,
    mut table: *const VTableEntry,
) -> *const PassVTable {
    // SAFETY: per documentation `name` is a pointer to nul-terminated `char`s.
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: per documentation, `ir_in` is the valid output of an `IRVTable` constructor.
    let ir_in = unsafe { const_ptr_as_ref(ir_in) };
    // SAFETY: per documentation, `ir_out` is the valid output of an `IRVTable` constructor.
    let ir_out = unsafe { const_ptr_as_ref(ir_out) };
    let mut partial = PassVTablePartial::new(name, Arc::clone(&ir_in.0), Arc::clone(&ir_out.0));
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
        unsafe { partial.set(slot, entry.ptr) };
    }
    PassVTable::try_from(partial).unwrap().into_leaked()
}

// TODO: we can very likely do a bit of macro trickery to simplify the creation of vtable objects,
// taking care that the filled public one needs to be visible to `cbindgen`.  For initial
// implementation, hard-coding is good enough.
struct PassVTablePartial {
    name: String,
    ir_in: Arc<dyn IRExposer>,
    ir_out: Arc<dyn IRExposer>,
    run: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut PassContext,
            *mut *mut Error,
        ) -> *mut c_void,
    >,
    delete: Option<unsafe extern "C" fn(*mut c_void) -> c_void>,
}
impl PassVTablePartial {
    fn new(name: String, ir_in: Arc<dyn IRExposer>, ir_out: Arc<dyn IRExposer>) -> Self {
        Self {
            name,
            ir_in,
            ir_out,
            run: None,
            delete: None,
        }
    }

    /// Set the `slot` to the corresponding function `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid function pointer of the type expected by the corresponding method in
    /// [`PassVTable`].
    unsafe fn set(&mut self, slot: PassSlot, ptr: *mut c_void) -> bool {
        match slot {
            PassSlot::Run => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<*mut c_void, _>(ptr) };
                self.run.replace(ptr).is_some()
            }
            PassSlot::Delete => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<*mut c_void, _>(ptr) };
                self.delete.replace(ptr).is_some()
            }
        }
    }
}

// TODO: docs
pub struct CPass {
    /// The pass object, aka `self`.
    this: *mut c_void,
    vtable: Arc<PassVTable>,
}
impl CPass {
    /// Create a new instance of the pass.
    ///
    /// # Safety
    ///
    /// 1. `this` must point to data of the type that is expected by all "`this`" arguments in
    ///    functions in the `vtable`.
    /// 2. the data pointed to by `this` must be safe to share and send between threads.
    unsafe fn new(this: *mut c_void, vtable: Arc<PassVTable>) -> Self {
        Self { this, vtable }
    }
}
// SAFETY: per struct documentation, the `this` pointer must be safe to send between threads.
unsafe impl Send for CPass {}
// SAFETY: per struct documentation, the `this` pointer must be safe to share between threads.
unsafe impl Sync for CPass {}
impl Pass for CPass {
    fn ir_id_in(&self) -> DynTypeId<'_> {
        self.vtable.ir_in.ir_dyn_type_id()
    }
    fn ir_id_out(&self) -> DynTypeId<'_> {
        self.vtable.ir_out.ir_dyn_type_id()
    }
    fn name(&self) -> &str {
        &self.vtable.name
    }
    fn run(&self, ir: Box<dyn IR>, context: &mut PassContext) -> Result<Box<dyn IR>, PassError> {
        let mut error = None::<ptr::NonNull<Error>>;
        let error_ptr = (&raw mut error).cast::<*mut Error>();
        let ir_in = self.vtable.ir_in.leak(ir).cast::<c_void>();
        let ir_out = unsafe { (self.vtable.run)(self.this, ir_in, context, error_ptr) };
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
            // SAFETY: per documentation of `PassVTable`, if the `delete` method is set, it is valid
            // to be passed `self.this` from any thread.
            unsafe { delete(self.this) };
        }
    }
}
const _: () = unsafe { expose_by_box!(CPass) };

// TODO: having everything exposed as `CPass` is awkward for a future world where we have handles to
// built-in passes?

// TODO: docs
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_new(this: *mut c_void, vtable: *const PassVTable) -> *mut CPass {
    // SAFETY: per documentation, `vtable` is the still-valid result of `qk_pass_vtable_new`, which
    // returns the result of `Arc::into_raw`.
    let vtable = unsafe { arc_clone_from_raw(vtable) };
    // SAFETY: per documentation, `this` points to data of the type expected by `vtable` methods.
    (unsafe { CPass::new(this, vtable) }).into_leaked()
}

/// @ingroup QkPassManager
/// Free the pass. Note that :c:func:`qk_passmanager_push_pass` consumes passes and they must not
/// be freed manually anymore.
///
/// @param pass A pointer to the pass to free.
///
/// # Safety
///
/// Behavior is undefined if ``pass`` is not either null or a valid pointer to a ``QkPass``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_free(pass: *mut CPass) {
    _ = (!pass.is_null()).then(|| unsafe { CPass::steal(pass) });
}

const _: () = unsafe { expose_by_box!(PassManager) };

/// @ingroup QkPassManager
/// Create an empty pass manager.
///
/// This object must be freed by the user.
#[unsafe(no_mangle)]
pub extern "C" fn qk_passmanager_new() -> *mut PassManager {
    PassManager::new().into_leaked()
}

/// @ingroup QkPassManager
/// Free the pass manager.
///
/// @param pm A pointer to the pass manager to free.
///
/// # Safety
///
/// Behavior is undefined if ``pm`` is not either null or a valid pointer to a ``QkPassManager``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_free(pm: *mut PassManager) {
    _ = (!pm.is_null()).then(|| unsafe { PassManager::steal(pm) });
}

/// @ingroup QkPassManager
/// Push a pass onto the pass manager.
///
/// @param pm The pass manager.
/// @param pass The pass to push. This consumes the `QkPass` object, which must not be freed after.
///
/// @return ``QkExitCode_Success`` upon succesful push, else an exit code explaining the failure.
///
/// # Safety
///
/// Behavior is undefined in ``pm`` is not a non-null, valid pointer to a ``QkPassManager`` or
/// ``pass`` is not a non-null, valid pointer to a ``QkPass``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_push_pass(
    pm: *mut PassManager,
    pass: *mut CPass,
) -> ExitCode {
    // SAFETY: per documentation the pointer is non-null and valid
    let pm = unsafe { mut_ptr_as_ref(pm) };
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

/// @ingroup QkPassManager
/// Run the pass manager.
///
/// @param pm A pointer to the pass manager to run.
/// @param ir A `void *` to the IR to run.
/// @param result A pointer to a `QkPassManagerResult` object to write the results into.
///
/// @return An exit code describing the error if the compilation failed and the result pointers
/// are set to `NULL`.
///
/// # Safety
///
/// Behavior is undefined if
///
/// * `pm` not a valid, non-null pointer to a `QkPassManager`, or
/// * `callback` is not either null or a valid pointer to a `QkCallback`, or
/// * `result` is not a valid, non-null pointer to a `QkPassManagerResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_run_simple(
    pm: *mut PassManager,
    ir: *mut c_void,
    ir_in_handle: *const IRHandle,
    ir_out_handle: *const IRHandle,
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
