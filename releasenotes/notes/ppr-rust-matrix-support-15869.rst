---
features:
  - |
    Added Rust-side matrix support for ``PauliProductRotationGate`` by reusing
    the existing Pauli matrix construction logic from the Rust
    ``SparsePauliOp`` implementation.