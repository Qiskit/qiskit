# `qiskit-bindgen-c`

A simple wrapper program around the logic in `qiskit-bindgen`.

This is just expected to be called as
```bash
cargo run -p qiskit-bindgen-c -- crates/cext dist/c/include
```
or similar, as part of the build script.  The two positional arguments are the location of the
`cext` crate, and the place to install the header files.  The `qiskit-bindgen-c` binary should
also provide its own help documentation if using the `--help` argument.

This is in a separate crate to avoid pulling in unnecessary dependencies to `bindgen` itself.
