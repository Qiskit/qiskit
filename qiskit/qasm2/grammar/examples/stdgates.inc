// Standard gate library

// phase gate (Z-rotation by lambda)
gate phase(lambda) q { U(0, 0, lambda) q; }
// controlled-NOT
gate CX c, t { ctrl @ U(pi, 0, pi) c, t; }
gate cx c, t { CX c, t; }
// idle gate (identity)
gate id a { U(0, 0, 0) a; }
// Pauli gate: bit-flip
gate x a { U(pi, 0, pi) a; }
// Pauli gate: bit and phase flip
gate y a { U(pi, pi / 2, pi / 2) a; }
// Pauli gate: phase flip
gate z a { phase(pi) a; }
// Clifford gate: Hadamard
gate h a { U(pi / 2, 0, pi) a; }
// Clifford gate: sqrt(Z) phase gate
gate s a { phase(pi / 2) a; }
// Clifford gate: conjugate of sqrt(Z)
gate sdg a { phase(-pi / 2) a; }
// C3 gate: sqrt(S) phase gate
gate t a { phase(pi/4) a; }
// C3 gate: conjugate of sqrt(S)
gate tdg a { phase(-pi/4) a; }
// Rotation around X-axis
gate rx(theta) a { U(theta, -pi / 2, pi / 2) a; }
// rotation around Y-axis
gate ry(theta) a { U(theta, 0, 0) a; }
// rotation around Z axis
gate rz(phi) a { phase(phi) a; }
// controlled-Phase
gate cz a, b { h b; cx a, b; h b; }
// controlled-Y
gate cy a, b { sdg b; cx a, b; s b; }
// controlled-H
gate ch a, b {
  h b;
  sdg b;
  cx a, b;
  h b;
  t b;
  cx a, b;
  t b; s a;
  h b;
  s b;
  x b;
}
// Toffoli
gate ccx a, b, c
{
  h c;
  cx b, c;
  tdg c;
  cx a, c;
  t c;
  cx b, c;
  tdg c;
  cx a, c;
  t b; t c; h c;
  cx a, b;
  t a; tdg b;
  cx a, b;
}
// controlled-swap
gate cswap a, b, c
{
  cx c, b;
  ccx a, b, c;
  cx c, b;
}
// controlled-rz
gate crz(lambda) a, b
{
  phase(lambda / 2) b;
  cx a, b;
  phase(-lambda / 2) b;
  cx a, b;
}
// controlled-phase
gate cphase(lambda) a, b
{
  phase(lambda / 2) a;
  cx a, b;
  phase(-lambda / 2) b;
  cx a, b;
  phase(lambda / 2) b;
}
// controlled-U
gate cu(theta, phi, lambda) c, t
{
  // implements controlled-U(theta,phi,lambda) with  target t and control c
  phase((lambda - phi)/2) t;
  cx c,t;
  U(-theta / 2, 0, -(phi + lambda) / 2) t;
  cx c, t;
  U(theta / 2, phi, 0) t;
}
