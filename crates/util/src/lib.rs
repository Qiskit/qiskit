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

use std::env;

pub mod dyn_types;
#[cfg(feature = "py")]
pub mod py;

/// Set the global allocator centrally for both `cext` and `pyext`, so that Cargo feature
/// unification detects that the `global_mimalloc` features on both of them can be safely combined.
/// This is useful for the `--all-features` flag in things like Clippy.
#[cfg(feature = "global_mimalloc")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A common type alias to use IndexMap with Qiskit's default hasher all
/// internal crate usage of IndexMap should use this instead of the indexmap
/// crate directly.
#[allow(clippy::disallowed_types)]
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, foldhash::fast::RandomState>;

/// A common type alias to use IndexSet with Qiskit's default hasher all
/// internal crate usage of IndexSet should use this instead of the indexmap
/// crate directly.
#[allow(clippy::disallowed_types)]
pub type IndexSet<V> = indexmap::IndexSet<V, foldhash::fast::RandomState>;

pub mod complex {
    use num_complex::Complex64;

    // This is a very conservative version of an abbreviation for constructing new Complex64.
    // A couple of alternatives to this function are
    // `c64<T: Into<f64>, V: Into<f64>>(re: T, im: V) -> Complex64`
    // Disadvantages are:
    //  1. Some people don't like that this allows things like `c64(1, 0)`. Presumably,
    //     they prefer a more explicit construction.
    //  2. This will not work in `const` and `static` constructs.
    // Another alternative is
    //   macro_rules! c64 {
    //       ($re: expr, $im: expr $(,)*) => {
    //           Complex64::new($re as f64, $im as f64)
    //       };
    // Advantages: This allows things like `c64!(1, 2.0)`, including in
    // `static` and `const` constructs.
    //  Disadvantages:
    //  1. Three characters `c64!` rather than two `c64`.
    //  2. Some people prefer the opposite of the advantages, i.e. more explicitness.
    /// Create a new [`Complex<f64>`]
    #[inline(always)]
    pub const fn c64(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    // Use prefix `C_` to distinguish from real, for example
    pub const C_ZERO: Complex64 = c64(0., 0.);
    pub const C_ONE: Complex64 = c64(1., 0.);
    pub const C_M_ONE: Complex64 = c64(-1., 0.);
    pub const IM: Complex64 = c64(0., 1.);
    pub const M_IM: Complex64 = c64(0., -1.);
}

pub mod alias {
    use num_complex::Complex64;

    pub type GateArray0Q = [[Complex64; 1]; 1];
    pub type GateArray1Q = [[Complex64; 2]; 2];
    pub type GateArray2Q = [[Complex64; 4]; 4];
    pub type GateArray3Q = [[Complex64; 8]; 8];
    pub type GateArray4Q = [[Complex64; 16]; 16];
}

pub mod ptr {
    #[cfg(target_pointer_width = "64")]
    mod impl_ {
        use std::ptr;

        /// A wrapper around a pointer that guarantees a 64-bit size, even on 32-bit platforms,
        /// while maintaining the actual `*mut T` type internally for Miri provenance tracking.
        /// This type does not remember why type was stored; it is equivalent to a `union`, not an
        /// `enum`.  It is up to the user of this type to track the information, typically by using
        /// pointer-tagging in the least-significant bits.
        ///
        /// On 64-bit platforms, the `u64` form is a simple union with the pointer type.  On 32-bit
        /// platforms, the internal pointer type is only 32 bits, so there's extra internal padding
        /// that's only used in the `u64`-storage form.
        ///
        /// These are abstracted away for users with the [`as_u64`](Self::as_u64) and
        /// [`as_ptr`](Self::as_ptr) methods, along with their corresponding `from_*` methods.
        #[derive(Clone, Copy, Debug)]
        #[repr(transparent)]
        pub struct Ptr64<T>(*mut T);
        impl<T> Ptr64<T> {
            /// View the data as a pointer.
            #[inline]
            pub fn as_ptr(self) -> *mut T {
                self.0
            }
            #[inline]
            pub fn as_u64(self) -> u64 {
                self.0
                    .addr()
                    .try_into()
                    .expect("usize is 64 bits on this platform")
            }
            #[inline]
            pub fn from_ptr(ptr: *mut T) -> Self {
                Self(ptr)
            }
            #[inline]
            pub fn from_u64(val: u64) -> Self {
                Self(ptr::without_provenance_mut(
                    val.try_into().expect("usize is 64 bits"),
                ))
            }
        }
    }
    #[cfg(target_pointer_width = "32")]
    mod impl_ {
        use std::ptr;

        // The documentation isn't duplicated here because, in practice, we'll always be developing
        // and building documentation on 64-bit platforms.

        #[derive(Clone, Copy, Debug)]
        pub struct Ptr64<T> {
            pad: u32,
            ptr: *mut T,
        }
        impl<T> Ptr64<T> {
            #[inline]
            pub fn as_ptr(self) -> *mut T {
                self.ptr
            }
            #[inline]
            pub fn as_u64(self) -> u64 {
                (u64::from(self.pad) << 32)
                    | u64::try_from(self.ptr.addr()).expect("usize is 32 bits")
            }
            #[inline]
            pub fn from_ptr(ptr: *mut T) -> Self {
                Self { pad: 0, ptr }
            }
            #[inline]
            pub fn from_u64(val: u64) -> Self {
                Self {
                    pad: (val >> 32) as u32,
                    // Rust numeric casting rules guarantee truncation to the least-significant bits.
                    // https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.numeric.int-truncation
                    ptr: ptr::without_provenance_mut(val as usize),
                }
            }
        }
    }
    #[cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))]
    compile_error! { "Qiskit only supports 32- and 64-bit pointer widths." }

    pub use impl_::*;
    impl<T> From<u64> for Ptr64<T> {
        fn from(val: u64) -> Self {
            Self::from_u64(val)
        }
    }
    impl<T> From<*mut T> for Ptr64<T> {
        fn from(val: *mut T) -> Self {
            Self::from_ptr(val)
        }
    }
}

#[inline]
pub fn getenv_use_multiple_threads() -> bool {
    let parallel_context = env::var("QISKIT_IN_PARALLEL")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    let force_threads = env::var("QISKIT_FORCE_THREADS")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    !parallel_context || force_threads
}
