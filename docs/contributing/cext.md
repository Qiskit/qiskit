# Contributing: C API

Historically, Qiskit was first and foremost a Python library. This fact is still true in some
sense; though, Qiskit has evolved significantly since its early days. As Quantum becomes
mainstream, researchers and developers alike want to use Qiskit in new and exciting ways. The C API
makes Qiskit more flexible, enabling additional use cases.

The C API is used for...

- wrapper libraries like [Qiskit.jl](https://github.com/Qiskit/Qiskit.jl) and
  [qiskit-cpp](https://github.com/Qiskit/qiskit-cpp).
- projects with high-performance computing (HPC) requirements.
- python extension modules.

This document defines guidelines for C API development, helping address stability, consistency, and
familiarity concerns. Examples are written from a Rust and `cbindgen` perspective. If this is your
first time extending the C API, see the
[tutorial](https://github.com/Qiskit/qiskit/blob/main/docs/contributing/cext-tutorial.md) .

## Structures

There are 2 ways we can declare a `struct` in a C header file. Firstly, we can declare and define
the `struct`, meaning callers can access its members and allocate it themselves. These structures
are called **transparent**. Secondly, we can forward-declare the `struct` without defining it.
Because definition lives elsewhere, callers *cannot* access its members or allocate it themselves.
These structures are called **opaque**. `cbindgen` generates a transparent `struct` when the layout
is `repr(C)`. Otherwise, a `struct` is opaque.

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

- encoding plain data without complex behavior.
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

## Function Naming

Function names centered around types follow the `qk_<type>_<verb>_<detail>` format. The `<detail>`
part is optional.

- `qk_obs_new`
- `qk_obs_compose`
- `qk_circuit_add_register`
- `qk_circuit_free`

Plain function names are more flexible.

- `qk_foo`
- `qk_str_free`

### Naming Constructor & Destructors

- `qk_foo_new` - Create (usually opaque) `struct`, allocated on the heap by Qiskit.
- `qk_foo_free` - Clean up a (usually opaque) `struct`, allocated on the heap by Qiskit.
- `qk_bar_init` - Initialize a `struct` pre-allocated by the caller on the stack or heap.
- `qk_bar_destroy` - Clean up an initialized `struct`.
- `qk_zap_copy` - Create a deep copy.

## Error Handling

### Sentinels

Choose a sentinel value when the function fails trivially. The correct sentinel value, if any, is
domain dependent. `-1` and `0` are poor sentinels if either value has additional meaning. `-1` is
common for functions that return an index. While rare, `0` works if the underlying domain value is
non-zero. `NAN` is problematic because `NAN != NAN`. The constants `u32::MAX`, `u64::MAX`, and the
like are used in the C API today; however, remember that the caller must cast `-1` or include
`limits.h` for comparisons without a compiler warning. Unlike comparisons, integer conversions are
ergonomic in C, so prefer returning signed integer types like `int32_t` and `intptr_t` when the
domain value is positive and smaller than $2^{64}/2$.

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

### Panicking

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

*The helper functions `const_ptr_as_ref` and `mut_ptr_as_ref` make exactly this mistake. Neither
are more safe than dereferencing plainly with `&*p` and `&mut *p`.*

You may be panicking without realizing it. For example, indexing with `a[i]` syntax is a checked
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
Instead or alongside writing "takes ownership" or "consumes", state what the caller _must not_ do
for each outcome. The following function takes ownership of `entry` if it's added to the target.
Note that the `entry` pointer is `*mut` because the internals potentially take ownership.

```rust
/// Append an instruction to the ``QkTarget`` through a ``QkTargetEntry``.
/// 
/// @param target A non-null pointer to the ``QkTarget``.
/// @param entry A non-null pointer to the ``QkTargetEntry``.
///
/// @return Returns `QkExitCode_Success` if the instruction is added. The caller must not use
///         or free `entry` if the instruction is added.
extern "C" fn qk_target_add_instruction(target: *mut Target, entry: *mut TargetEntry) -> ExitCode;
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
