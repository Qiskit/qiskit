# Contributing: C API

The C API is designed for...

- python extension modules written in C.
- wrapper libraries like [Qiskit.jl](https://github.com/Qiskit/Qiskit.jl) and
  [qiskit-cpp](https://github.com/Qiskit/qiskit-cpp).
- high-performance computing workloads.

## Tutorial

This section serves as a step-by-step guide to extending the C API. Check out the
[guidelines](#guidelines) for detailed information. 

Lets try exposing the hypothetical `Qubit` model to C! This model represents a single qubit using 2
complex amplitudes, `a` and `b`. `Qubit::new` ensures that `a` and `b` form a normalized
statevector. `Qubit::measure` collapses the statevector such that consecutive measurements are
effectively deterministic.

```rust
// crates/quantum_info/src/qubit.rs

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
```

**TO BE CONTINUED**

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
extern "C" fn qk_qubit_type(qubit: *const Qubit) -> QubitType {
    if qubit.is_null() {
        QubitType::Unknown
    } else {
        let qubit = unsafe { &*qubit };
        qubit.kind.into()
    }
}
```

Return `ExitCode` for non-trivial functions with multiple failure points. These functions often
write to an `out` parameter for the success case.

```rust
extern "C" fn qk_qubit_new(
    a: *const Complex64,
    b: *const Complex64,
    out: *mut *mut Qubit
) -> ExitCode {
    if a.is_null() || b.is_null() || out.is_null() {
        return ExitCode::NullPointerError;
    }

    match Qubit::new(*a, *b) {
        Ok(qubit) => {
            out.write(Box::new(qubit).into_raw());
            ExitCode::Success
        },
        Err(QubitError::Sum) => {
            ExitCode::QubitSum
        },
        Err(_) => {
            ExitCode::Unknown
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

### Python Extensions

This section needs some more thought. Qiskit provides functions for extracting a C-native pointer
from a `PyObject *`. The working theory is that, in sequential programming contexts, only 1 thread
will access the pointer at a given instant. We assume this is true for both GIL-attached and free-
threaded Python. Thus, a simple function that returns a data model pointer given a `PyObject *`
should be sufficient.

*See [this PR](https://github.com/Qiskit/qiskit/pull/17011) for details.*

