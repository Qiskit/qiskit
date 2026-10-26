# Contributing: C API

Qiskit is fundamentally a Python library, though it has evolved significantly since its early days.
As quantum computing becomes mainstream, researchers and developers increasingly have more diverse
workflows. The C API makes Qiskit more accessible in non-Python ecosystems.

The C API is useful for...

- wrapper libraries like [Qiskit.jl](https://github.com/Qiskit/Qiskit.jl) and
  [qiskit-cpp](https://github.com/Qiskit/qiskit-cpp).
- projects with high-performance computing requirements.
- writing python extension modules.

This document defines guidelines for C API development, helping address stability, consistency, and
familiarity concerns. Examples are written from a Rust and `cbindgen` perspective. If this is your
first time extending the C API, see the [tutorial](cext-tutorial.md).

## Structs

C structs are either transparent or opaque. A transparent struct is fully defined in the header
file, allowing callers to access members and allocate instances themselves. An opaque struct is
only forward-declared; callers cannot access members or allocate instances. `cbindgen` generates
transparent structs when the layout is `repr(C)`, otherwise opaque.

*Don't conflate transparency with Rust's `repr(transparent)`, which has a totally different
meaning.*

Use opaque structs when...

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

Use transparent structs when...

- encoding plain data without complex behavior.
- fields are unlikely to change.

```rust
#[repr(C)]
pub struct QubitArray {
    pub data: *mut *mut Qubit,
    pub len: usize,
}
```

*The `pub` keyword has no effect across FFI. Though, it's best practice to include `pub` for
documentation purposes.*

## Enums

C enums are named integer constants. Qiskit uses fixed-width enums with `repr(u*)` for consistency
across compilers. Since `sizeof(enum t) == sizeof(int)` on most modern platforms, `repr(u32)` is a
solid default. Use `repr(u8)` for space-critical enums within tightly packed structures. Explicit
integer values discourage accidental breaking changes.

```rust
#[repr(u32)]
pub enum QubitType {
    Unknown = 0,
    Physical = 1,
    Logical = 2,
}
```

## Integers

Prefer fixed-width integers like `i32` and `u64`. Avoid imprecise-width integers like `c_int` and
`c_long`. `cbindgen` generates headers using fixed-width integers found in `inttypes.h`. For
example, `i32` is mapped to `int32_t`.

## Functions

### Naming

Functions are often grouped around types. Grouped functions follow the `qk_<group>_<verb>` format.

- `qk_obs_new`
- `qk_obs_compose`
- `qk_circuit_add_register`
- `qk_circuit_free`

Plain function names are more flexible.

- `qk_foo`
- `qk_str_free`

Qiskit uses standard names for memory management functions.

- `qk_foo_new` - Create a (usually opaque) `struct`, allocated on the heap by Qiskit.
- `qk_foo_free` - Clean up a (usually opaque) `struct`, allocated on the heap by Qiskit.
- `qk_bar_init` - Initialize a `struct` pre-allocated by the caller on the stack or heap.
- `qk_bar_clear` - Clean up an initialized `struct` pre-allocated by the caller.
- `qk_zap_copy` - Create a deep copy.

## Errors

### Sentinels

Choose a sentinel value when functions fail trivially. Sentinels must fall outside of the return
value's domain. `-1` and `0` are poor sentinels if either value has additional meaning. `-1` is
suitable for index-returning functioons because indicies are non-negative. `0` only works when the
domain excludes zero. `NAN` is problematic because `NAN != NAN`. Constants like `u64::MAX` require
callers to cast `-1` or include `limits.h` to handle errors without compiler warnings. Prefer
returning signed types for positive domains unlikely to exceed $2^{64}/2$.

**Sentinel Examples**

| Type (Rust) | Type (C)       | Sentinel    | Domain |
| ----------- | -------------- | ----------- | ------ |
| `*mut T`    | `T *`          | `NULL`      |        |
| `enum T`    | `enum t`       | `T_UNKNOWN` |        |
| `isize`     | `intptr_t`     | `-1`        | `>= 0` |
| `u64`       | `uint64_t`     | `0`         | `>= 1` |

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

### Error Codes

Return `ExitCode` for non-trivial functions with multiple failure points. These functions often
write to an `out` parameter for the success case. Given an `ExitCode`, the callers should have
access to a human readable error message.

```rust
extern "C" fn qk_obs_matrix(
    obs: *const QkObs,
    out: *mut *mut Matrix,
) -> ExitCode {
    let obs = unsafe { &*obs };

    match obs.to_matrix() {
        Ok(matrix) => {
            *out = Box::into_raw(Box::new(matrix));
            ExitCode::Success
        },
        Err(error) => {
            *out = ptr::null_mut();
            error.into()
        },
    }
}
```

### Panic

You can almost always do better than panicking inside a function called from C. We sometimes see
`panic!` used to uphold invariants that would otherwise be UB. This is a mistake because `panic!`
itself is UB inside code called from C.

```rust
fn read_foo(foo: *mut Foo) {
    // Don't do this!
    if foo.is_null() {
        panic!("foo is null");
    } 
}

fn read_bar(bar: *mut Bar) {
    // Simply dereference if you've documented that `bar` isn't null.
    let foo = &*foo;
}
```

*The helper functions `const_ptr_as_ref` and `mut_ptr_as_ref` are unnecessary. Neither are more
safe than dereferencing with `&*p` and `&mut *p`.*

You could be panicking without realizing it. For example, indexing with `a[i]` syntax is a checked
operation, panicking in the out of bounds case. Consider using a sentinel, exit code, or
`a.get_unchecked(i)` instaed. Be sure to uphold the expectations of functions that can panic!

## Versioning & Backwards Compatibility

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
- if declared exhausitive, adding an enum variant.
- changing transparent struct fields.
- tightening preconditions.
- changing function signatures (including `const`-ness changes).

## Platform-specific Features

It's best practice to avoid compiler and platform dependent features because callers could be
targeting multiple platforms. For example, `__int128_t` and `_Complex double` are potentially
relevant in the quantum domain, but those types are only available in GCC. In this case, we would
be better off using the platform independent complex number utilities defined in `complex.h` since
C99. 

*See [this PR](https://github.com/Qiskit/qiskit/pull/17011) for details.*

## Documentation

The documentation pipeline uses Doxygen, Breathe, and Sphinx, orchestrated by the `tox -e docs`
command.

### Doxygen

C API documentation is generated from outer doc comments written with
[Doxygen](https://www.doxygen.nl/manual/commands.html) syntax.

```rust
/// @ingroup QkFoo
///
/// Describe what the function does. Then, describe memory management responsibilities.
///
/// @param arg1 Describe `arg1`, including the contract (e.g. non-null).
/// @param arg2 Describe `arg2`, including the contract (e.g. non-null).
/// 
/// @return Describe the meaning of each return value, if any. Omit this section if the function
///         returns `void`.
#[unsafe(no_mangle)]
pub extern "C" fn qk_foo(arg1: *mut c_void, arg2: *mut c_void) -> i32;
```

*The optional `@ingroup` command groups related items.*

### Groups

C API documentation is organized into groups, defined in `docs/cdoc/index.h` with the `@defgroup`
command. The `@ingroup` command assigns items to a group. Structs exposed directly from the data
model must be manually added to an `.rst` file.

### Ownership

C syntax cannot express ownership semantics, so the ownership contract must be documented
explicitly. Phrases like "takes ownership" or "consumes" aren't meaningful to every reader. Instead
of, or alongside, those phrases, state what the caller _must not_ do for each outcome.

In the example below, `qk_target_add_instruction` takes ownership of `entry` only when the
instruction is successfully added to the target. The `entry` parameter is typed `*mut` because the
function may take ownership of it.

```rust
/// @ingroup QkTarget
/// 
/// Append an instruction to the ``QkTarget`` through a ``QkTargetEntry``. The caller must not use
/// or free `entry` if `QkExitCode_Success` is returned.
/// 
/// @param target A non-null pointer to the ``QkTarget``.
/// @param entry A non-null pointer to the ``QkTargetEntry``.
///
/// @return Returns `QkExitCode_Success` if the instruction is added.
extern "C" fn qk_target_add_instruction(target: *mut Target, entry: *mut TargetEntry) -> ExitCode;
```

### Safety

`unsafe` enables operations that are otherwise prohibited in safe Rust. When writing FFI code, we
often dereference raw pointers or call unsafe functions within an `unsafe` block. These operations
are unsafe because the compiler cannot verify their validity statically; as the programmer, you are
asserting that you've checked the necessary conditions yourself. Document those conditions in a
**safety comment** so that future readers understand the assumptions you made.

```rust
unsafe extern "C" fn qk_circuit_free(circuit: *mut Circuit) {
    if !circuit.is_null() {
        // SAFETY: `circuit` is non-null in this block.
        let circuit = unsafe { Box::from_raw(circuit) };
        drop(circuit)
    }
}
```

### Release Notes

Link C objects in release notes using Sphinx syntax in the `c` domain.

```
:c:struct:`QkFoo`
:c:type:`QkFoo`
:c:enum:`QkFoo`
:c:func:`qk_foo`
```

## Python Extensions

This section needs some more thought. Qiskit provides functions for extracting a C-native pointer
from a `PyObject *`. The working theory is that, in sequential programming contexts, only 1 thread
will access the pointer at a given instant. We assume this is true for both GIL-attached and free-
threaded Python. Thus, a simple function that returns a data model pointer given a `PyObject *`
should be sufficient.
