# Contributing: C API

The C API is designed for...

- python extension modules written in C.
- wrapper libraries like [Qiskit.jl](https://github.com/Qiskit/Qiskit.jl) and
  [qiskit-cpp](https://github.com/Qiskit/qiskit-cpp)
- high-performance computing workloads.
- use qiskit without linking `libpython`

See the [tutorial](#tutorial) if this is your first time extending the C API.

## Guidelines

### Structures

There are 2 ways we can define a `struct` in a C header file. Firstly, we can fully define the
`struct`, meaning callers can access its members and allocate it themselves. These structures are
called **transparent**. Secondly, we can forward-declare the `struct` without defining it. Since
the actual definition lives in the source code, callers *cannot* access its members or allocate it
themselves. These structures are called **opaque**. `cbindgen` generates transparent structures
when the layout is `repr(C)` and opaque structures for other layouts. 

Use opaque structures when...

- runtime invariant(s) need enforcement.
- complex behavior needs encapsulation.
- fields are likely to change.

```rust
pub struct Qubit {
    a: Complex64, 
    b: Complex64,
}
```

*Note that `repr(Rust)` is implicit.*

Use transparent structures when...

- dealing with plain data without complex behavior.
- fields are unlikely to change.

```rust
#[repr(C)]
pub struct QubitArray {
    pub data: **mut Qubit,
    pub len: usize,
}
```

*The `pub` keyword has no affect across FFI. Though, it's best practice to include `pub` for
documentation purposes.*

### Enumerations

In short, C enumerations are named integer constants, and nothing more. `cbindgen` creates mappings
from flat Rust `enum` types with `repr(C)` and `repr(u*)` layouts. In Qiskit, we use `repr(u*)`
fixed-width enumerations to maintain consistency across compilers. Because
`sizeof(enum t) == sizeof(int)` is the common case on modern platforms, reach for `repr(u32)`
first.

```rust
#[repr(u32)]
pub enum QubitType {
    Unknown = 0,
    Physical = 1,
    Logical = 2,
}
```

*Explicit integer values discourage accidental breaking changes.*

### Error Handling

Choose a sentinel value when the function fails trivially. The correct sentinel value, if any, is
domain dependent. `-1` and `0` are poor sentinels if the values have additional meaning. `-1`
works for functions that return an array index. While rare, `0` works if the underlying domain
value is `NonZeroU32`. Loose integer conversions are idiomatic in C, so prefer returning signed
integer types like `int32_t` and `ptrdiff_t` when the domain value is unsigned and small. Note that
`NAN` is problematic because `NAN != NAN`.

| Type           | Sentinel    |
| -------------- | ----------- |
| `T *`          | `NULL`      |
| `enum t`       | `T_UNKNOWN` |
| `int32_t`      | `-1`        |

```rust
extern "C" fn qk_matrix_type(matrix: *const Matrix) -> MatrixType {
    if matrix.is_null() {
        MatrixType::Unknown
    } else {
        let matrix = unsafe { &*matrix };
        matrix.kind.into()
    }
}
```

Return `ExitCode` for non-trivial functions with multiple failure points. These functions often
write to an `out` parameter for the success case.

```rust
extern "C" fn qk_obs_matrix(
    obs: *const QkObs,
    out: *mut *mut Matrix,
) -> ExitCode {
    let obs = unsafe { &*obs };

    match obs.to_matrix() {
        Ok(matrix) => {
            // ...
            ExitCode::Success
        },
        Err(error) => {
            error.into()
        },
    }
}
```

Create static, human-readable error messages for `ExitCode` variants.

```rust
extern "C" fn qk_exit_code_str(code: ExitCode) -> *const c_char {
    let s = match code {
        ExitCode::Success => c"success",
        ExitCode::NullPointerError => c"unexpected null pointer",
        ExitCode::QubitSum => c"qubit component sum not 1",
        _ => c"unhandled error",
    };

    s.as_ptr().cast()
}
```

### Fixed-width Integers

Use fixed-width integers in transparent structures and function signatures. Avoid imprecise-width
integers such as `c_int` and `c_long`. `cbindgen` generates headers using the fixed-width integer
types found in `inttypes.h`. For example, `i32` is mapped to `int32_t`.

### Versioning & Backwards Compatibility

C and Python share the same version number, following [SemVer](https://semver.org/). Breaking
changes are prohibited without incrementing the major version number.

Backwards compatible changes include...

- creating public functions, types, macros, constants, and headers.
- adding enum variant(s) without changing integer values.
- adding bit flags without changing the previous meanings.
- changing the size and layout of opaque structs.
- relaxing preconditions.
- fixing bugs that bring behavior in line with the documentation.

Breaking changes include...

- removing or renaming public functions, types, macros, constants, and headers.
- changing transparent struct fields.
- tightening preconditions.
- changing function signatures (including `const`-ness changes).

### Naming Conventions

Function names centered around types follow the `qk_<type>_<verb>` format. 

- `qk_obs_new`
- `qk_obs_compose`
- `qk_circuit_add_register`
- `qk_circuit_free`

Plain function names are more flexible.

- `qk_foo`
- `qk_str_free`

### Memory Management

Memory allocated by Qiskit should be free'd by Qiskit, including plain arrays and strings.
`qiskit.h` provides `qk_str_free` for leaked `CString` pointers. 

### Platform-specific Features

It's best practice to avoid compiler and platform dependent features because callers could be
targeting multiple platforms. For example, `__int128_t` and `_Complex double` are potentially
relevant in the quantum domain, but those types are only available in GCC. In this case, we would
be better off using the platform independent complex number utilities defined in `complex.h` since
C99. 

*See [this PR](https://github.com/Qiskit/qiskit/pull/17011) for details.*

### Testing

**TODO:** Discuss testing.

### Documentation

**TODO:** Explain Sphinx and link relevant documentation.

#### Functions

**TODO:** Discuss documentation requirements for functions.

#### Structures

**TODO:** Discuss documentation requirements for transparent and opaque structures.

#### Safety

`unsafe` enables actions that are otherwise prohibited in safe Rust. In writing FFI code, we often
dereference raw pointers and call unsafe functions within an `unsafe` block. Such code is unsafe
because the compiler cannot check the validitiy of the operation statically. As the programmer, you
make some assumption(s) about why the potentially unsafe operation is, in fact, safe. Those
assumptions should be documented in a **safety comment** so that future programmers understand the
assumptions you've made.

```rust
unsafe extern "C" fn qk_circuit_free(circuit: *mut Circuit) {
    if !circuit.is_null() {
        // SAFETY: `circuit` is non-null in this block.
        let circuit = unsafe { Box::from_raw(circuit) };
        drop(circuit)
    }
}
```

### Python Extensions

This section needs some more thought. Qiskit provides functions for extracting a C-native pointer
from a `PyObject *`. The working theory is that, in sequential programming contexts, only 1 thread
will access the pointer at a given instant. We assume this is true for both GIL-attached and free-
threaded Python. Thus, a simple function that returns a data model pointer given a `PyObject *`
should be sufficient.

## Tutorial

This section serves as a step-by-step guide to extending the C API.

Let's try exposing the hypothetical `Qubit` model to C! This model represents a single qubit using
2 complex amplitudes, `a` and `b`. `Qubit::new` ensures that `a` and `b` form a normalized
statevector. `Qubit::measure` collapses the statevector such that consecutive measurements are
effectively deterministic. Note that this module includes unit tests as well.

```rust
// crates/quantum_info/src/qubit.rs

use num_complex::Complex64;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("not normalized")]
pub struct QubitError;

#[derive(Debug, Clone)]
pub struct Qubit {
    a: Complex64,
    b: Complex64,
}

impl Qubit {
    pub fn new(a: Complex64, b: Complex64) -> Result<Self, QubitError> {
        const TOL: f64 = 1e-12;
        let len_sqr = a.norm_sqr() + b.norm_sqr();

        if (len_sqr - 1.0).abs() <= TOL {
            Ok(Self { a, b })
        } else {
            Err(QubitError)
        }
    }

    pub fn measure(&mut self) -> u32 {
        let sqr_a = self.a.norm_sqr();
        let sqr_b = self.b.norm_sqr();
        let p_zero = sqr_a / (sqr_a + sqr_b);

        if p_zero > rand::random() {
            self.a = Complex64::ONE;
            self.b = Complex64::ZERO;
            0
        } else {
            self.a = Complex64::ZERO;
            self.b = Complex64::ONE;
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_normalized() {
        let a = Complex64::new(0.50, 0.0);
        let b = Complex64::new(0.25, 0.0);

        let result = Qubit::new(a, b);
        assert!(matches!(result, Err(QubitError)));
    }

    #[test]
    fn test_measure_collapsed() {
        let a = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let b = Complex64::new(0.0, 1.0 / 2.0_f64.sqrt());
        let mut qubit = Qubit::new(a, b).expect("is normalized");

        let first = qubit.measure();
        for _ in 0..10 {
            assert_eq!(first, qubit.measure());
        }
    }
}
```

A Qiskit user wants to create and measure a simulated qubit in C. Let's create a `qubit` module
in the `cext` crate. We'll write a `new`, `measure`, and `free` function. Also, we'll include
documentation and safety comments. These functions have a single purpose: map the `Qubit` API to
something usable in C. The `Qubit` model defines and tests the core behavior. Note that `Qubit`
will appear as `QkQubit` in the header file generated by `cbindgen`.

```rust
// crates/cext/src/qubit.rs 

use std::ptr;
use num_complex::Complex64;
use quantum_info::qubit::Qubit;

/// @ingroup OkQubit
///  
/// Create a simulated ``QkQubit`` with 2 amplitudes, `a` and `b`.
///
/// @param a A non-null pointer to the amplitude of the 0 component.
/// @param b A non-null pointer to the amplitude of the 1 component.
///
/// @return Returns a pointer to ``QkQubit`` with a copy of `a` and `b`. The caller is responsible
///         for calling ``qk_qubit_free``. Returns `NULL` if the statevector is not normalized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qubit_new(a: *const Complex64, b: *const Complex64) -> *mut Qubit {
    // SAFETY: `a` is documented as non-null.
    let a = unsafe { *a };
    // SAFETY: `b` is documented as non-null. 
    let b = unsafe { *b };

    if let Ok(qubit) = Qubit::new(a, b) {
        Box::into_raw(Box::new(qubit))
    } else {
        ptr::null_mut()
    }
}

/// @ingroup OkQubit
///
/// Measure the ``QkQubit``, collapsing its state to 0 or 1 based on the probability calculated
/// from its components. 
///
/// @param qubit A non-null pointer to the ``QkQubit``.
///
/// @return Returns an integer, `0` or `1`, representing the collapsed state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qubit_measure(qubit: *mut Qubit) -> u32 {
    // SAFETY: `qubit` is documented as non-null.
    let qubit = unsafe { &mut *qubit };
    qubit.measure()
}

/// @ingroup OkQubit
///
/// Cleanup the ``QkQubit`` or nothing if `qubit` is `NULL`.
///
/// @param qubit A pointer to the ``QkQubit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qubit_free(qubit: *mut Qubit) {
    if !qubit.is_null() {
        // SAFETY: `qubit` is non-null in this block.
        let qubit = unsafe { Box::from_raw(qubit) };
        drop(qubit)
    }
}
```

There's some light ceremony around exposing `cext` functions when Python is in the picture. Don't
worry about the exact reasoning for now. Find or create the corresponding export list for your API
in `crates/cext-vtable/src/lib.rs`. Add your newly written functions to the list using the
`export_fn!` macro. Don't change the index of existing functions! Order matters! 

```rust
mod qubit {
    use crate::impl_::prelude::*;
    use qiskit_cext::qubit::*;

    // tip: the `leaves` function accepts a `reserve` parameter. it's difficult to change after
    // it's set. make some space for new functions!
    pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::leaves(30, || {
        vec![
            export_fn!(qk_qubit_new),
            export_fn!(qk_qubit_free),
            export_fn!(qk_qubit_measure),
        ]
    });
}
```

Our C functions need tests. There's no need to test the core bevavior covered by the unit tests in
`crates/quantum_info/qubit.rs`. Instead, we'll focus on the new behaviors introduced by the C API.
An obvious new behavior is that `qk_qubit_new` returns `NULL` if `Qubit::new` returns `Err`. The
C function is responsible for this mapping, so we'll write a C API test for it. When introducing
a new module, you'll create corresponding test file in `test/c`.

```c
// test/c/test_qubit.c

#include <complex.h>
#include "common.h"

/**
 * Test if "not normalized" error is mapped to `NULL`.
 */
static int test_not_normalized() {
    complex a = { .re = 0.25, .im = 0.0 };
    complex b = { .re = 0.50, .im = 0.0 };
    QkQubit *qubit = qk_qubit_new(&a, &b);

    int result = EqualityError;
    if qubit == NULL
        result = Ok;

    qk_qubit_free(qubit);
    return result;
}
```

