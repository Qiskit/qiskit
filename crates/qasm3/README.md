# `qiskit._qasm3`

This crate is the Rust-level Qiskit interface to an OpenQASM 3 parser.  The parser itself does not know
about Qiskit, and this crate interfaces with it in a Qiskit-specific manner to produce `QuantumCircuit`s.
