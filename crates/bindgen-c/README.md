# `qiskit-bindgen-c`

A toolkit binary for working with the generated header files from the C API.

You can run this either by doing
```bash
cargo run -p qiskit-bindgen-c
```

Running the command without arguments will print out its help message, which explains what the
various commands are.

This is in a separate crate to avoid pulling in unnecessary dependencies to `bindgen` itself.


## Installing the standalone header files

Use the `install` subcommand, such as

```bash
cargo run -p qiskit-bindgen-c -- install -c crates/cext -o dist/c/include
```

The `-c` (`--cext-path`) argument specifies the location of the `cext` crate source tree for the
internal calls to `cbindgen`.  The `-o` (`--output-path`) argument specifies where to place the
files.

## Linting the current vtable slots

Use the `lint-slots` subcommand, such as

```bash
cargo run -p qiskit-bindgen-c -- lint-slots -c crates/cext
```

This checks various coherence properties between the declared `extern "C"` functions in
`qiskit-cext` and the vtable slots layout specified in `qiskit-cext-vtable`, such as checking that
all functions have a slot and there are no duplicates; in other words, that each exported function
is referenced exactly once.

Note that this command does not test for ABI compatibility between different Qiskit versions.

You can annotate the docstring for an `extern "C"` function in `qiskit-cext` to exempt it from these
linter rules, using the `cbindgen` annotations mechanism with the key `qk-vtable-rules`.  This
looks like:

```rust
/// Build an empty circuit.
///
/// @return A new, owned circuit.
/// cbindgen:qk-vtable-rules=[no-export]
pub extern "C" fn qk_circuit_empty() -> *mut QkCircuit { /* ... */ }
```

The magic is in the `/// cbindgen:qk-vtable-rules` line.  `cbindgen` will strip this line out of the
generated documentation. The available "rules" within the list are:

- `no-export`: assert that the function will not be in the slots list (such as for functions
  deprecated before the introduction of the vtables).
- `allow-duplicate`: allow functions to be in more than one slot (if, for example, we want to
  export a function in more than one vtable).

You can specify more than one attribute by separating them with commas.
