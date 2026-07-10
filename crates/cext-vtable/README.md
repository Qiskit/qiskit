# `qiskit-cext-vtable`

This crate defines the machinery to specify ABI-stable vtables of function pointers, and provides
concrete vtables for the functions within `qiskit-cext`.

## Why this crate exists

The `_accelerate` object that ships with Qiskit already exposes C API symbols with public `qk_*`
names, but **distributable compiled Python extension modules cannot rely on the linker** to resolve
those symbols at build time or runtime. Instead, Qiskit publishes the C API through **vtable
function-pointer tables** exposed as `PyCapsule` objects when `import qiskit` runs.

Each consumer receives a base pointer to a table. Individual functions are accessed at **fixed slot
offsets** from that base. The offsets are defined statically in this crate so that extension
modules built against one Qiskit minor release keep working with later patch releases in the same
major version.

This crate is separate from `qiskit-cext` because language-binding generators and build scripts
often **cannot compile against `qiskit-cext`**. For example, the `pyext` build script would
otherwise trigger a second full compilation of Qiskit and require linking `libpython` just to run.
With the `addr` Cargo feature disabled, this crate depends only on function **names** and slot
positions, which is sufficient for header generation and linting.

## Top-level vtables

Three independent vtables are exported through `qiskit._accelerate.capi` as `PyCapsule` objects:

| Rust static | PyCapsule name | Typical contents |
|-------------|----------------|------------------|
| `FUNCTIONS_CIRCUIT` | `QK_FFI_CIRCUIT` | circuits, DAGs, parameters, classical expressions, circuit library |
| `FUNCTIONS_TRANSPILE` | `QK_FFI_TRANSPILE` | transpiler entry points, targets, transpiler passes |
| `FUNCTIONS_QI` | `QK_FFI_QI` | quantum-info objects such as sparse observables |

Using several vtables (instead of one global table) leaves more room to append new functions at the
end of each table without reorganising unrelated areas.

## `ExportedFunctions` structure

An [`ExportedFunctions`](src/impl_.rs) value describes a **reservation** of slots in a vtable.
At runtime the reservation is flat: the hierarchical nesting is a **code-organisation tool** that
keeps related functions together and makes it easier to assign future slots locally.

Each `ExportedFunctions` value is either:

* **Leaves** — `ExportedFunctions::leaves(reserve, || vec![...])` defines a contiguous run of
  function-pointer slots. `reserve` must be **at least** the number of entries in the vector; it is
  fine (and encouraged) to reserve extra space for future additions within that group.
* **Children** — `ExportedFunctions::empty().add_child(offset, &child)` composes sub-tables. The
  `offset` is the slot index where the child begins. Children must be added in **strictly
  increasing offset order**, and `offset` must be **at least** the current total reservation length.
  You cannot fill in holes left for expansion.

The compile-time `len` field on each node ensures that reservations fit together; mismatched offsets
or over-full leaves cause **compile-time failures** in this crate.

### Recommended module layout

The module tree in [`src/lib.rs`](src/lib.rs) should largely mirror `qiskit-cext/src`. If `cext`
uses a single file for a small set of functions, you may inline the corresponding vtable module
here. The goal is locality: a developer adding `qk_foo_*` in `cext` should find the matching
`export_fn!` entries nearby.

Nested `static FUNCTIONS: ExportedFunctions` values are grouped under submodule-specific names
(for example `transpiler::target::FUNCTIONS`).

## Adding a new `qk_*` function

When you add a new `pub extern "C" fn` to `qiskit-cext`, you must also assign it a vtable slot in
this crate.

1. **Implement the function** in the appropriate `qiskit-cext` module.
2. **Add an `export_fn!` entry** in the matching module here, inside the correct
   `ExportedFunctions::leaves` vector.
3. **Choose the right leaf group**. Prefer appending to an existing leaf list whose `reserve` still
   has free space. If a leaf group is full, either increase its `reserve` (without moving existing
   entries) or add a new sibling child at the **end** of the parent table.
4. **Run the slots linter**:

   ```bash
   cargo run -p qiskit-bindgen-cli -- lint-slots -c crates/cext
   ```

   This checks that every exported `extern "C"` function in `qiskit-cext` appears exactly once in
   the vtables (unless explicitly exempted) and that there are no duplicate slot assignments.

5. **Run the C API tests** (`make ctest`) and Rust tests as usual.

### `export_fn!` macro

Inside a `leaves` closure, each slot is produced by `export_fn!`:

```rust
export_fn!(qk_circuit_new),
export_fn!(qk_circuit_to_python, feature = "python_binding"),
```

The path must resolve to a function declared like:

```rust
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_my_function(...) -> ... { ... }
```

Feature-gated functions use the `feature = "..."` form so the slot is omitted when the feature is
disabled.

### Functions that are not exported

Some `extern "C"` functions are intentionally absent from the vtables (for example symbols
deprecated before vtables existed). Mark them in `qiskit-cext` with a `cbindgen` annotation:

```rust
/// cbindgen:qk-vtable-rules=[no-export]
pub extern "C" fn qk_internal_helper() { ... }
```

See [`crates/bindgen-cli/README.md`](../bindgen-cli/README.md) for other rules such as
`allow-duplicate`.

### Expanding a full leaf group

If a `leaves` reservation is exhausted, append a new sibling child at the end of the parent table
rather than reordering existing slots. For example, if `circuit::FUNCTIONS` is full:

```rust
mod circuit {
    pub static FUNCTIONS: ExportedFunctions = /* existing entries */;
    pub static FUNCTIONS_2: ExportedFunctions = ExportedFunctions::leaves(50, || vec![]);
}

pub static FUNCTIONS_CIRCUIT: ExportedFunctions = ExportedFunctions::leaves(5, || {
    vec![impl_::export_fn!(qiskit_cext::qk_api_version)]
})
.add_child(5, &circuit::FUNCTIONS)
// ... other children ...
.add_child(355, &circuit::FUNCTIONS_2);  // new space at the end
```

The numeric child offsets are part of the public ABI; new children must use offsets **after** the
current reservation.

## ABI stability rules

> **Warning:** The slot index of a function is public ABI within a Qiskit major version.

* **Do not reorder** existing slots or change which function occupies a slot.
* **Do not remove** a filled slot within a major version.
* **You may append** new functions at the end of a leaf reservation or as a new child at the end of
  a parent table.
* **You may increase** a leaf's `reserve` value to leave more space in that group, provided existing
  slot indices are unchanged.

Within a major version, Qiskit aims for **forwards compatibility** of pre-built extension
modules: a module compiled against Qiskit 2.N should keep working with 2.(N+M). Slot positions
and function signatures must remain stable for that to hold. Breaking slot layout or changing a
slot's function type is reserved for major releases.

CI enforces slot stability across minor and patch releases using `capi_slots.txt` at the
repository root:

```bash
cargo run -p qiskit-bindgen-cli -- show-slots > capi_slots.txt
cargo run -p qiskit-bindgen-cli -- check-abi capi_slots.txt
```

Maintainers refresh `capi_slots.txt` at release boundaries; see `MAINTAINING.md`.

## The `addr` feature

| Feature | Depends on `qiskit-cext` | Provides |
|---------|--------------------------|----------|
| `addr` enabled | yes | function names **and** pointer addresses |
| `addr` disabled | no | function names and slot indices only |

`qiskit-pyext` enables `addr` when building the runtime extension so the vtables contain real
function pointers for `PyCapsule` export.

Build scripts (for example `pyext/build.rs`) depend on `qiskit-cext-vtable` **without** `addr`. They
iterate slot listings to generate preprocessor macros that resolve `qk_*` calls to pointer offsets
from the capsule base, without compiling all of Qiskit.

## How the pieces connect

```text
qiskit-cext                extern "C" fn implementations
       │
       ▼
qiskit-cext-vtable         static slot reservations (this crate)
       │
       ├─► pyext (addr on)  PyCapsule vtables at import time
       │
       └─► pyext build.rs   generated offset-based headers (addr off)
              │
              └─► qiskit-bindgen-cli lint-slots / check-abi
```

For more detail on header installation and Python-extension bindings, see
[`crates/bindgen-cli/README.md`](../bindgen-cli/README.md) and
[`crates/cext/README.md`](../cext/README.md).

## Local `rustdoc`

This crate has a clean `rustdoc` build. Run:

```bash
cargo doc -p qiskit-cext-vtable --open
```

The [`ExportedFunctions`](src/impl_.rs) and [`export_fn!`](src/impl_.rs) definitions include
additional API-level documentation.
