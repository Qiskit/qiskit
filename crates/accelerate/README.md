# `qiskit._accelerate`

This crate provides a bits-and-pieces Python extension module for small, self-contained functions
that are used by the main Python-space components to accelerate certain tasks.  If you're trying to
speed up one particular Python function by replacing its innards with a Rust one, this is the best
place to put the code.  This is _usually_ the right place to put Rust/Python interfacing code.

The crate is made accessible as a private submodule, `qiskit._accelerate`.  There are submodules
within that (largely matching the structure of the Rust code) mostly for grouping similar functions.

Some examples of when it might be more appropriate to start a new crate instead of using the
ready-made solution of `qiskit._accelerate`:

* The feature you are developing will have a large amount of domain-specific Rust code and is a
  large self-contained module.  If it reasonably works in a single Rust file, you probably just want
  to put it here.

* The Rust code is for re-use within other Qiskit crates and maintainability of the code will be
  helped by using the crate system to provide API boundaries between the different sections.

* You want to start writing your own procedural macros.
