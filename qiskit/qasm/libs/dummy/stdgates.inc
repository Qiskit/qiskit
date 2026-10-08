// OpenQASM 3 standard gate library

// phase gate
gate p(lambda) a {}

// Pauli gate: bit-flip or NOT gate
gate x a {}
// Pauli gate: bit and phase flip
gate y a {}
// Pauli gate: phase flip
gate z a {}

// Clifford gate: Hadamard
gate h a {}
// Clifford gate: sqrt(Z) or S gate
gate s a {}
// Clifford gate: inverse of sqrt(Z)
gate sdg a {}

// sqrt(S) or T gate
gate t a {}
// inverse of sqrt(S)
gate tdg a {}

// sqrt(NOT) gate
gate sx a {}

// Rotation around X-axis
gate rx(theta) a {}
// rotation around Y-axis
gate ry(theta) a {}
// rotation around Z axis
gate rz(lambda) a {}

// controlled-NOT
gate cx c, t {}
// controlled-Y
gate cy a, b {}
// controlled-Z
gate cz a, b {}
// controlled-phase
gate cp(lambda) a, b {}
// controlled-rx
gate crx(theta) a, b {}
// controlled-ry
gate cry(theta) a, b {}
// controlled-rz
gate crz(theta) a, b {}
// controlled-H
gate ch a, b {}

// swap
gate swap a, b {}

// Toffoli
gate ccx a, b, c {}
// controlled-swap
gate cswap a, b, c {}

// four parameter controlled-U gate with relative phase
gate cu(theta, phi, lambda, gamma) c, t {}

// Gates for OpenQASM 2 backwards compatibility
// CNOT
gate CX c, t {}
// phase gate
gate phase(lambda) q {}
// controlled-phase
gate cphase(lambda) a, b {}
// identity or idle gate
gate id a {}
// IBM Quantum experience gates
gate u1(lambda) q {}
gate u2(phi, lambda) q {}
gate u3(theta, phi, lambda) q {}
