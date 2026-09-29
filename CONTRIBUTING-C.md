# C API Guidelines

This document serves as a starting point for the C API contributing guidelines. All sections are open for changes and discussion.

## ABI Stability

### Memory Layout & Transparency

Model parity will naturally gravitate towards opaqueness. As a rule of thumb, use opaque types when there are invariant(s) and special behavior defined by the model. Use transparent types when dealing with "plain, old structs" without behavior. As with any rule of thumb, there are exceptions, particularly for performance critical scenarios. Such exceptions should be proven necessary with measurements. `#[repr(C)]` is the correct representation for anything crossing the FFI boundary. Note that opaque types are behind FFI-safe pointers.

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

The C standard says that `enum` is an integer. It makes no guarentees about integer width or representation. According to the [Rustonomicon](https://doc.rust-lang.org/nomicon/other-reprs.html?highlight=bindgen#alternative-representations), `#[repr(C)]` is the idiomatic, correct representation for any type, *including enums*, crossing an FFI boundary.

Historically, we've used `#[repr(u8)]` for FFI enums. If backwards compatibility is a concern, consider that new variants are potentially breaking regardless of width or representation. For example, `-Wall` warns when `switch` statements don't cover every variant. Programmers may avoid the `default` label because they *want* this warning. With `-Werror`enabled, you've broken compilation without bumping the major version.

### Platform-specific Features

Avoid compiler and platform dependent features. Portability is important for users with multiple target platforms. For example, `__int128_t` and `_Complex double` could be relevant to Qiskit, but those types are only available in GCC. Converging on a particular minimum C standard helps mitigate this issue. For example, C99 offers complex number utilities in `complex.h` which is entirely platform independent.

1. We should converge on a particular minumum C standard for our ABI. C99 is a solid candidate because it defines complex number and fixed-width integer types.

### Memory Management

Memory allocated by Qiskit should be free'd by Qiskit, including plain arrays and strings. For example, `qiskit.h` provides `qk_str_free` for leaked `CString` pointers. Future APIs might need ownership of some caller allocated memory. In that case, we could introduce generic `qk_malloc` and `qk_free` functions that use the Qiskit allocator.

### Python Extensions

Extension developers should be able to extract C API native pointers from a `PyObject *`. In a sequential programming context, no more than one thread will access the native pointer at a given instant. This is true for both GIL and free-threaded Python. A simple function that returns a raw pointer into the data model should be sufficient.

1. I strongly suspect that the above is true. See [this](https://github.com/Qiskit/qiskit/pull/17011) PR for details.

2. We should explore how popular Python libraries backed by C handle this. For example, it's unlikely that `numpy` would indiscriminately wrap every data structure in a mutex.

### Virtual Tables

This section requires further team discussion.

1. Jake, can you describe your requirements for the "slots" pattern?

### Versioning & Compatibility

This section requires further team discussion.

1. Given Qiskit's current architecture, can we reasonably version the Python, C, and internal Rust API seperately? SemVer?

2. Marking identifiers with `_v1`, `_v2`, .., `_vN` is a viable strategy. Though, my gut says that REST-like versioning in a shared library should be used conservatively.

## General

### Target Audience

This section requires further team discussion.

### Ownership Model

This section requires further team discussion.

### Error Handling

The initial error model should be simple. Use sentinel values where applicable when the function fails trivially. Return `ExitCode` and provide an `out` parameter for non-trivial error cases with multiple failure points. Each `ExitCode` should have a fixed, human-readable error message.

**Example: Sentinel Values**

| Type           | Sentinel    |
| -------------- | ----------- |
| `T *`          | `NULL`      |
| `enum t`       | `T_UNKNOWN` |
| `int`          | `-1`        |

**Example: Simple Message API**

```c
const char *qk_exit_code_str(enum qk_exit_code code) {
	switch (result) {
  case QK_EXIT_CODE_OK:
    return "success";
  case QK_EXIT_CODE_INPUT:
    return "invalid input";
  case QK_EXIT_CODE_NULL:
    return "unexpected null pointer"; default: return "unhandled error";
  }
}
```

1. Will `ExitCode` grow too large? If so, we should consider seperate errors for each subsystem. While subsystem specific errors introduce *some* duplication, it becomes far more clear which errors can actually occur within that particular subsystem.

2. We can introduce a rich error model when the need arises. I anticipate that the simple model covers 99% of cases. In the future, extending the simple model is trivial compared to changing an existing, complex model.


### Naming & Formatting

For the most part, `cbindgen` will handle identifier formatting and case conventions automatically.
#### Functions

Functions centered around types follow the `qk_<type>_<verb>_<opt>` format. Plain function names are more flexible. All functions in `qiskit.h` should begin with the `qk_` prefix.

**Example: Plain Functions**
- `qk_malloc`
- `qk_free`
- `qk_strdup`

**Example: Typed Functions**
- `qk_obs_new`
- `qk_obs_compose`
- `qk_circuit_add_register`
- `qk_circuit_free`

### Thread-safety

This section requires further team discussion.

### Documentation

This section requires further team discussion.

