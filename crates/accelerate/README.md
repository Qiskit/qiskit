# `qiskit-accelerate`

This crate provides a bits-and-pieces Rust libary for small, self-contained functions
that are used by the main Python-space components to accelerate certain tasks.  If you're trying to
speed up one particular Python function by replacing its innards with a Rust one, this is the best
place to put the code.  This is _usually_ the right place to put new Rust/Python code.

The `qiskit-pyext` crate is what actually builds the C extension modules.  Modules in here should define
themselves has being submodules of `qiskit._accelerate`, and then the `qiskit-pyext` crate should bind them
into its `fn _accelerate` when it's making the C extension.

Some examples of when it might be more appropriate to start a new crate:

* The feature you are developing will have a large amount of domain-specific Rust code and is a
  large self-contained module.  If it reasonably works in a single Rust file, you probably just want
  to put it here.

* The Rust code is for re-use within other Qiskit crates and maintainability of the code will be
  helped by using the crate system to provide API boundaries between the different sections.

* You want to start writing your own procedural macros.
