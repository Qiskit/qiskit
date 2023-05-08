# `qiskit._qasm2`

This crate is the bulk of the OpenQASM 2 parser.  Since OpenQASM 2 is a simple language, it doesn't
bother with an AST construction step, but produces a simple linear bytecode stream to pass to a
small Python interpreter (in `qiskit.qasm2`).  This started off life as a vendored version of [the
package `qiskit-qasm2`](https://pypi.org/project/qiskit-qasm2).
