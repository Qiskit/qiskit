// OpenQASM 3 standard gate library

// phase gate
gate p(lambda) a { ctrl @ gphase(lambda) a; }

// Pauli gate: bit-flip or NOT gate
gate x a { U(pi, 0, pi) a; }
// Pauli gate: bit and phase flip
gate y a { U(pi, pi/2, pi/2) a; }
// Pauli gate: phase flip
gate z a { p(pi) a; }

// Clifford gate: Hadamard
gate h a { U(pi/2, 0, pi) a; }
// Clifford gate: sqrt(Z) or S gate
gate s a { pow(1/2) @ z a; }
// Clifford gate: inverse of sqrt(Z)
gate sdg a { inv @ pow(1/2) @ z a; }

// sqrt(S) or T gate
gate t a { pow(1/2) @ s a; }
// inverse of sqrt(S)
gate tdg a { inv @ pow(1/2) @ s a; }

// sqrt(NOT) gate
gate sx a { pow(1/2) @ x a; }

// Rotation around X-axis
gate rx(theta) a { U(theta, -pi/2, pi/2) a; }
// rotation around Y-axis
gate ry(theta) a { U(theta, 0, 0) a; }
// rotation around Z axis
gate rz(lambda) a { gphase(-lambda/2); U(0, 0, lambda) a; }

// controlled-NOT
gate cx c, t { ctrl @ x c, t; }
// controlled-Y
gate cy a, b { ctrl @ y a, b; }
// controlled-Z
gate cz a, b { ctrl @ z a, b; }
// controlled-phase
gate cp(lambda) a, b { ctrl @ p(lambda) a, b; }
// controlled-rx
gate crx(theta) a, b { ctrl @ rx(theta) a, b; }
// controlled-ry
gate cry(theta) a, b { ctrl @ ry(theta) a, b; }
// controlled-rz
gate crz(theta) a, b { ctrl @ rz(theta) a, b; }
// controlled-H
gate ch a, b { ctrl @ h a, b; }

// swap
gate swap a, b { cx a, b; cx b, a; cx a, b; }

// Toffoli
gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }
// controlled-swap
gate cswap a, b, c { ctrl @ swap a, b, c; }

// four parameter controlled-U gate with relative phase
gate cu(theta, phi, lambda, gamma) c, t { p(gamma) c; ctrl @ U(theta, phi, lambda) c, t; }

// Gates for OpenQASM 2 backwards compatibility
// CNOT
gate CX c, t { ctrl @ U(pi, 0, pi) c, t; }
// phase gate
gate phase(lambda) q { U(0, 0, lambda) q; }
// controlled-phase
gate cphase(lambda) a, b { ctrl @ phase(lambda) a, b; }
// identity or idle gate
gate id a { U(0, 0, 0) a; }
// IBM Quantum experience gates
gate u1(lambda) q { U(0, 0, lambda) q; }
gate u2(phi, lambda) q { gphase(-(phi+lambda)/2); U(pi/2, phi, lambda) q; }
gate u3(theta, phi, lambda) q { gphase(-(phi+lambda)/2); U(theta, phi, lambda) q; }