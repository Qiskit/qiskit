// Quantum Experience (QE) Standard Header
// file: qelib1.inc

// --- QE Hardware primitives ---

// 3-parameter 2-pulse single qubit gate
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
// 2-parameter 1-pulse single qubit gate
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
// 1-parameter 0-pulse single qubit gate
gate u1(lambda) q { U(0,0,lambda) q; }
// controlled-NOT
gate cx c,t { CX c,t; }
// idle gate (identity)
gate id a { U(0,0,0) a; }
// idle gate (identity) with length gamma*sqglen
gate u0(gamma) q { U(0,0,0) q; }

// --- QE Standard Gates ---

// Pauli gate: bit-flip
gate x a { u3(pi,0,pi) a; }
// Pauli gate: bit and phase flip
gate y a { u3(pi,pi/2,pi/2) a; }
// Pauli gate: phase flip
gate z a { u1(pi) a; }
// Clifford gate: Hadamard
gate h a { u2(0,pi) a; }
// Clifford gate: sqrt(Z) phase gate
gate s a { u1(pi/2) a; }
// Clifford gate: conjugate of sqrt(Z)
gate sdg a { u1(-pi/2) a; }
// C3 gate: sqrt(S) phase gate
gate t a { u1(pi/4) a; }
// C3 gate: conjugate of sqrt(S)
gate tdg a { u1(-pi/4) a; }

// --- Standard rotations ---
// Rotation around X-axis
gate rx(theta) a { u3(theta, -pi/2,pi/2) a; }
// rotation around Y-axis
gate ry(theta) a { u3(theta,0,0) a; }
// rotation around Z axis
gate rz(phi) a { u1(phi) a; }

// --- QE Standard User-Defined Gates  ---

// controlled-Phase
gate cz a,b { h b; cx a,b; h b; }
// controlled-Y
gate cy a,b { sdg b; cx a,b; s b; }
// swap
gate swap a,b { cx a,b; cx b,a; cx a,b; }
// controlled-H
gate ch a,b {
h b; sdg b;
cx a,b;
h b; t b;
cx a,b;
t b; h b; s b; x b; s a;
}
// C3 gate: Toffoli
gate ccx a,b,c
{
  h c;
  cx b,c; tdg c;
  cx a,c; t c;
  cx b,c; tdg c;
  cx a,c; t b; t c; h c;
  cx a,b; t a; tdg b;
  cx a,b;
}
// cswap (Fredkin)
gate cswap a,b,c
{
  cx c,b;
  ccx a,b,c;
  cx c,b;
}
// controlled rz rotation
gate crz(lambda) a,b
{
  u1(lambda/2) b;
  cx a,b;
  u1(-lambda/2) b;
  cx a,b;
}
// controlled phase rotation
gate cu1(lambda) a,b
{
  u1(lambda/2) a;
  cx a,b;
  u1(-lambda/2) b;
  cx a,b;
  u1(lambda/2) b;
}
// controlled-U
gate cu3(theta,phi,lambda) c, t
{
  // implements controlled-U(theta,phi,lambda) with  target t and control c
  u1((lambda-phi)/2) t;
  cx c,t;
  u3(-theta/2,0,-(phi+lambda)/2) t;
  cx c,t;
  u3(theta/2,phi,0) t;
}
// two-qubit ZZ rotation
gate rzz(theta) a,b
{
  cx a,b;
  u1(theta) b;
  cx a,b;
}
