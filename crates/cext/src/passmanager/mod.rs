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

mod error;
mod ir;
mod pass;
// `qiskit-passmanager` is responsible for all the PM infrastructure, including passes and IRs, so
// its corresponding `cext` module mirrors it.
#[expect(clippy::module_inception)]
mod passmanager;
mod predicate;

use std::ffi::c_void;

pub use error::*;
pub use ir::*;
pub use pass::*;
pub use passmanager::*;
pub use predicate::*;

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
