# `qiskit-accelerate`

This crate provides a bits-and-pieces Rust libary for small, self-contained functions
that are used by the main Python-space components to accelerate certain tasks.  If you're trying to
speed up one particular Python function by replacing its innards with a Rust one, this is the best
place to put the code.  This is _usually_ the right place to put new Rust/Python code.

The `qiskit-pyext` crate is what actually builds the C extension modules.  Modules in here should define
themselves has being submodules of `qiskit._accelerate`, and then the `qiskit-pyext` crate should bind them
into its `fn _accelerate` when it's making the C extension.

Specifically, adding a rust module `my_module` to Qiskit such that it can be called from within Python involves
the following steps:
1. Add `pub fn my_module(m: &Bound<PyModule>) -> PyResult<()>` to your `my_module.rs` file with `m.add_wrapped(wrap_pyfunction!(mymodulefunction))?;` for new functions and `m.add_class::<MyModuleClass>()?;` for new classes.
2. Add `pub mod my_module` to  `crates/accelerate/src/lib.rs`
3. To `crates/pyext/src/lib.rs` 
   * Add my_module::my_module to `use qiskit_accelerate::{`
   * Add `m.add_wrapped(wrap_pymodule!(my_module))?;`
5. To `qiskit/__init__.py` add `sys.modules["qiskit._accelerate.my_module”] = _accelerate.module`
6. Compile, and you should be done. Within Python you can now `import qiskit._accelerate.my_module`
