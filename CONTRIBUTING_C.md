# Contributing: C API

TODO: What is the C API for?

1. Python Extensions
2. HPC
3. language-agnostic API

***

## Tutorial

TODO: Write step-by-step instructions, including any boilerplate:

1. Exposing an opaque structure.
2. Writing `extern "C"` functions.
3. Extending `ExitCode` with message.

***

## Guidelines

### Structures

There are 2 ways we can define a `struct` in a C header file. Firstly, we can fully define the
`struct`, meaning callers can access its members and allocate it themselves. These structures are
called **transparent**. Secondly, we can forward-declare the `struct` without defining it. Since
the actual definition lives in the source code, callers *cannot* access its members or allocate it
themselves. These structures are called **opaque**. `cbindgen` generates transparent structures
when the layout is `repr(C)` and opaque structures for other layouts. 

**Use opaque structures when...**

1. runtime invariant(s) need enforcement.
2. complex behavior needs encapsulation.
3. fields are likely to change.

```rust
pub struct Qubit {
    a: Complex64, 
    b: Complex64,
}
```

*Note that `repr(Rust)` is implicit.*

**Use transparent structures when...**

1. dealing with plain, old data without complex behavior.
2. fields are unlikely to change.

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
from flat Rust `enum` types with `repr(C)` and `repr(u*)` layouts.

```rust
#[repr(C)]
pub enum QubitType {
    Physical = 0,
    Logical = 1,
}
```

*Assign integer values explicitly to discourage breaking changes.*

**Debate: Fixed-width Layout**

There are several points to consider surrounding the fixed-width `enum` debate:

1. `qiskit.h` already defines fixed-width layout for `enum` types.

2. The [Rustonomicon](https://doc.rust-lang.org/nomicon/other-reprs.html?highlight=bindgen#reprc)
states that `repr(C)` is the correct layout for any type passed through the FFI boundary.

3. The C standard, and thus `repr(C)`, guarentees that enumeration *constants* have `sizeof(int)`
width. For example, if `enum t` defines `T_UNKNOWN`, then `sizeof(T_UNKNOWN)` and `sizeof(int)` are
equal. `sizeof(enum t)` is implementation defined. However, on modern Linux, Mac, and Windows,
`sizeof(enum t)` and `sizeof(int)` are equal nonetheless. 

4. `cbindgen` handles `repr(u*)` by generating a fixed-width integer `typedef` with the same name
as the `enum` type, bloating the header file.

Because `libqiskit` is user-space, applications library, `repr(C)` is likely the most maintainable
and semantically correct layout for `enum`. If we ultimately stick with fixed-width, developers
should use `repr(u32)` because it aligns most closely with the compiler and platform defaults for
C enumerations.

### Platform-specific Features

Avoid compiler and platform dependent features. Users could be targeting multiple platforms. For
example, `__int128_t` and `_Complex double` could be relevant to Qiskit, but those types are only
available in GCC. Converging on a particular minimum C standard helps mitigate this problem. For
example, C99 offers platform independent complex numbers in `complex.h`.

### Memory Management

Memory allocated by Qiskit should be free'd by Qiskit, including plain arrays and strings. For
example, `qiskit.h` provides `qk_str_free` for leaked `CString` pointers. Future APIs might need
ownership of some caller allocated memory. In that case, we could introduce `qk_malloc` and
`qk_free` functions that use the same allocator as Qiskit.

### Python Extensions

Extension developers should be able to extract C API native pointers from a `PyObject *`. In
sequential programming contexts, no more than one thread will access the native pointer at a given
instant. This is true for both GIL and free-threaded Python. A simple function that returns a raw
pointer into the data model should be sufficient.

*\* See [this PR](https://github.com/Qiskit/qiskit/pull/17011) for details.*

### VTables

This section requires further team discussion.

### Versioning & Backwards Compatibility

C and Python share the same version number, following the [SemVer](https://semver.org/)
specification. Breaking changes are prohibited without incrementing the major version number.

Changes **allowed** without incrementing the major version:

- creating public functions, types, macros, constants, or headers
- adding enum variants without changing integer values
- adding bit flags without changing the meaning of existing bits
- changing size and layout of opaque structs
- changing size and layout of transparent structs with a size and version field
- relaxing preconditions
- bug fixes to bring behavior in line with documentation

Changes **prohibited** without incrementing the major version:

- removing or renaming public functions, types, macros, constants, or headers
- changing transparent struct fields
- tightening preconditions
- changing function signatures
- adding `const` to function signatures (breaks function pointers)

TODO: What should we do about existing API that strays from these guidelines?

## General

### Target Audience

This section requires further team discussion.

### Ownership Model

This section requires further team discussion.

### Error Handling

The error model should be simple and consistent.

1. Use sentinel values when the function fails trivially.
2. Return `ExitCode` for non-trivial functions with multiple failure points. These functions often
   mutate an `out` parameter for the success case.

**Example: Sentinel Values**

| Type           | Sentinel    |
| -------------- | ----------- |
| `T *`          | `NULL`      |
| `enum t`       | `T_UNKNOWN` |
| `int`          | `-1`        |
| `unsigned int` | `0`         |

*\* `-1` and `0` are poor sentinels if the values have additional meaning. For example, `-1` is
fine for functions that return an index because it's an invariant.*

Each `ExitCode` should have a static, human-readable error message.

**Example: Error Messages**

```c
const char *qk_exit_code_str(enum qk_exit_code code) {
    switch (result) {
    case QK_EXIT_CODE_OK:
        return "success";
    case QK_EXIT_CODE_INPUT:
        return "invalid input";
    case QK_EXIT_CODE_NULL:
        return "unexpected null pointer";
    default:
        return "unhandled error";
  }
}
```

### Naming Standards

Case conventions are enforced by `clippy` and `cbindgen`.

#### Functions

Function names centered around types follow the `qk_<type>_<verb>` format. 

**Example: Typed Functions**
- `qk_obs_new`
- `qk_obs_compose`
- `qk_circuit_add_register`
- `qk_circuit_free`

Plain function names are more flexible.

**Example: Plain Functions**
- `qk_malloc`
- `qk_free`
- `qk_strdup`

### Thread-safety

This section requires further team discussion.

### Documentation

This section requires further team discussion.

