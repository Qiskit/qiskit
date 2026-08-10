// OpenQASM 3.0 standard gate library
//
// Vendored from the OpenQASM 3.0 project at commit 4ca1d793833b24a1 of
// https://github.com/openqasm/openqasm.
//
// Used under the terms of its Apache-2.0 license.
// Copyright 2017-2024 OpenQASM Contributors.


// phase gate
gate p(λ) a { ctrl @ gphase(λ) a; }

// Pauli gate: bit-flip or NOT gate
gate x a { U(π, 0, π) a; gphase(-π/2);}
// Pauli gate: bit and phase flip
gate y a { U(π, π/2, π/2) a; gphase(-π/2);}
 // Pauli gate: phase flip
gate z a { p(π) a; }

// Clifford gate: Hadamard
gate h a { U(π/2, 0, π) a; gphase(-π/4);}
// Clifford gate: sqrt(Z) or S gate
gate s a { pow(0.5) @ z a; }
// Clifford gate: inverse of sqrt(Z)
gate sdg a { inv @ pow(0.5) @ z a; }

// sqrt(S) or T gate
gate t a { pow(0.5) @ s a; }
// inverse of sqrt(S)
gate tdg a { inv @ pow(0.5) @ s a; }

// sqrt(NOT) gate
gate sx a { pow(0.5) @ x a; }

// Rotation around X-axis
gate rx(θ) a { U(θ, -π/2, π/2) a; gphase(-θ/2);}
// rotation around Y-axis
gate ry(θ) a { U(θ, 0, 0) a; gphase(-θ/2);}
// rotation around Z axis
gate rz(λ) a { gphase(-λ/2); U(0, 0, λ) a; }

// controlled-NOT
gate cx a, b { ctrl @ x a, b; }
// controlled-Y
gate cy a, b { ctrl @ y a, b; }
// controlled-Z
gate cz a, b { ctrl @ z a, b; }
// controlled-phase
gate cp(λ) a, b { ctrl @ p(λ) a, b; }
// controlled-rx
gate crx(θ) a, b { ctrl @ rx(θ) a, b; }
// controlled-ry
gate cry(θ) a, b { ctrl @ ry(θ) a, b; }
// controlled-rz
gate crz(θ) a, b { ctrl @ rz(θ) a, b; }
// controlled-H
gate ch a, b { ctrl @ h a, b; }

// swap
gate swap a, b { cx a, b; cx b, a; cx a, b; }

// Toffoli
gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }
// controlled-swap
gate cswap a, b, c { ctrl @ swap a, b, c; }

// four parameter controlled-U gate with relative phase γ
gate cu(θ, φ, λ, γ) a, b { p(γ-θ/2) a; ctrl @ U(θ, φ, λ) a, b; }

// Gates for OpenQASM 2 backwards compatibility
// CNOT
gate CX a, b { ctrl @ U(π, 0, π) a, b; }
// phase gate
gate phase(λ) q { U(0, 0, λ) q; }
// controlled-phase
gate cphase(λ) a, b { ctrl @ phase(λ) a, b; }
// identity or idle gate
gate id a { U(0, 0, 0) a; }
// IBM Quantum experience gates
gate u1(λ) q { U(0, 0, λ) q; }
gate u2(φ, λ) q { gphase(-(φ+λ+π/2)/2); U(π/2, φ, λ) q; }
gate u3(θ, φ, λ) q { gphase(-(φ+λ+θ)/2); U(θ, φ, λ) q; }
