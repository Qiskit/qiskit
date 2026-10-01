# C API Guidelines

This document serves as a starting point for the C API contributing guidelines. All sections
are open for changes and discussion.

## ABI Stability

### Memory Layout & Transparency

Model parity will naturally gravitate towards opaqueness. As a rule of thumb, use opaque,
`repr(Rust)` types when the model defines invariant(s) or special behavior. Use transparent,
`repr(C)` types for "plain, old structs" without behavior. As with any rule of thumb, there are
exceptions, particularly for performance critical scenarios. Such exceptions should be proven
necessary with measurements.

**Example: Transparent Types**

```c
struct qk_duration {
  uint64_t time;
  enum qk_duration_unit unit;
} 
```

```c
struct qk_circuit_array {
  QkCircuit **data;
  size_t len;
}
```

### Fixed-width Enums

`repr(C)` guarentees a fixed width for enums, normally `sizeof(int)`. The
[Rustonomicon](https://doc.rust-lang.org/nomicon/other-reprs.html?highlight=bindgen#reprc)
states that `repr(C)` is correct for *any type* passed through the FFI boundary. `cbindgen`
will correctly handle `repr(u*)` with a `typedef`, but `repr(C)` better explains intent.
 
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

