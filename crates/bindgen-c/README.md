# `qiskit-bindgen-c`

A simple wrapper program around the logic in `qiskit-bindgen`.

This is just expected to be called as
```bash
cargo run -p qiskit-bindgen-c -- crates/cext dist/c/include
```
or similar, as part of the build script.

This is in a separate crate to avoid pulling in unnecessary dependencies to `bindgen` itself.
