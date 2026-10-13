# Contributing: C API

Historically, Qiskit was first and foremost a Python library. This fact is still true in some
sense; though, Qiskit has evolved significantly since its early days. As Quantum becomes
mainstream, researchers and developers alike want to use Qiskit in new and exciting ways. The C
API provides maximum flexibility for developers, enabling some of these use cases.

The C API is used for...

- wrapper libraries like [Qiskit.jl](https://github.com/Qiskit/Qiskit.jl) and
  [qiskit-cpp](https://github.com/Qiskit/qiskit-cpp).
- projects with high-performance computing (HPC) requirements.
- python extension modules.

This document defines guidelines for C API development, helping address stability, consistency, and
familiarity concerns. Examples are written from a Rust and `cbindgen` perspective. See the
[tutorial](https://www.example.com/) if this is your first time extending the C API.

## Structures

There are 2 ways we can declare a `struct` in a C header file. Firstly, we can declare and define
the `struct`, meaning callers can access its members and allocate it themselves. These structures
are called **transparent**. Secondly, we can forward-declare the `struct` without defining it.
Because definition lives elsewhere, callers *cannot* access its members or allocate it themselves.
These structures are called **opaque**. `cbindgen` generates transparent structures when the layout
is `repr(C)` and opaque structures for other layouts. 

*Don't conflate transparency with Rust's `repr(transparent)`, which has a totally different
meaning.*

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

## Fixed-width Enumerations

In short, C enumerations are named integer constants, and nothing more. `cbindgen` creates mappings
from flat Rust `enum` types with `repr(C)` and `repr(u*)` layouts. In Qiskit, we use `repr(u*)`
fixed-width enumerations to maintain consistency across compilers. Because
`sizeof(enum t) == sizeof(int)` is the common case on modern platforms, `repr(u32)` is a reasonable
default. Consider `repr(u8)` when the `enum` is a member of a tightly packed structure and
performance is critical.

```rust
#[repr(u32)]
pub enum QubitType {
    Unknown = 0,
    Physical = 1,
    Logical = 2,
}
```

*Explicit integer values discourage accidental breaking changes.*

## Fixed-width Integers

Prefer fixed-width integers like `i32` and `u64`. Avoid imprecise-width integers like `c_int` and
`c_long`. `cbindgen` generates headers using the fixed-width integers found in `inttypes.h`.
For example, `i32` is mapped to `int32_t`.

## Function Signatures

Function names centered around types follow the `qk_<type>_<verb>` format. 

- `qk_obs_new`
- `qk_obs_compose`
- `qk_circuit_add_register`
- `qk_circuit_free`

Plain function names are more flexible.

- `qk_foo`
- `qk_str_free`

## Error Handling

### Sentinels

Choose a sentinel value when the function fails trivially. The correct sentinel value, if any, is
domain dependent. `-1` and `0` are poor sentinels if the values have additional meaning. `-1`
works for functions that return an array index. While rare, `0` works if the underlying domain
value is `NonZeroU32`. Loose integer conversions are idiomatic in C, so prefer returning signed
integer types like `int32_t` and `ptrdiff_t` when the domain value is unsigned and small. `NAN` is
problematic because `NAN != NAN`.

| Type           | Sentinel    | Domain |
| -------------- | ----------- | ------ |
| `T *`          | `NULL`      |        |
| `enum t`       | `T_UNKNOWN` |        |
| `ptrdiff_t`    | `-1`        | `>= 0` |
| `uint32_t`     | `0`         | `>= 1` |

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

### Exit Codes

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

**TODO:** Explain when it's okay to panic, if ever.

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

**TODO:** Introduce Sphinx & Syntax

### Functions

**TODO:** Discuss documentation requirements for functions.

### Structures

**TODO:** Discuss documentation requirements for transparent and opaque structures.

### Ownership

C syntax cannot express ownership semantics, so, the ownership contract must be documented. Some
programmers may not understand the impact of ownership on their memory management responsibility.
Instead of writing "takes ownership" or "consumes", state what the caller _must not_ do after each
outcome. The following function takes ownership of `entry` if it's added to the target. Note that
the `entry` pointer is `*mut` because the internals potentially take ownership.

```rust
/// Append an instruction to the ``QkTarget`` through a ``QkTargetEntry``.
/// 
/// @param target A non-null pointer to the ``QkTarget``.
/// @param entry A non-null pointer to the ``QkTargetEntry``.
///
/// @return Returns `QkExitCode_Success` if the instruction was added. The caller must not use
///         or free `entry` after the instruction is added successfully.
extern "C" fn qk_target_add_instruction(target: *mut Target, entry: *mut TargetEntry);
```

### Safety

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

## Python Extensions

This section needs some more thought. Qiskit provides functions for extracting a C-native pointer
from a `PyObject *`. The working theory is that, in sequential programming contexts, only 1 thread
will access the pointer at a given instant. We assume this is true for both GIL-attached and free-
threaded Python. Thus, a simple function that returns a data model pointer given a `PyObject *`
should be sufficient.
