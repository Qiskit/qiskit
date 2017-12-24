// quantum Fourier transform for the Quantum Experience
// on the first 3 qubits

OPENQASM 2.0;
include "qelib1.inc";

// Register declarations
qreg q[5];
creg c[5];

// Choose starting state
// ** Put your code here **

// Quantum Fourier transform
// For this small example, we have full connectivity.

h q[0];
// cu1(pi/2) q[0],q[1];
u1(pi/4) q[0];
cx q[0],q[1];
u1(-pi/4) q[1];
cx q[0],q[1];
u1(pi/4) q[1];
// end cu1
h q[1];
// cu1(pi/4) q[0],q[2];
u1(pi/8) q[0];
cx q[0],q[2];
u1(-pi/8) q[2];
cx q[0],q[2];
u1(pi/8) q[2];
// end cu1
// cu1(pi/2) q[1],q[2];
u1(pi/4) q[1];
cx q[1],q[2];
u1(-pi/4) q[2];
cx q[1],q[2];
u1(pi/4) q[2];
// end cu1
h q[2];

// Outputs are q[2], q[1], q[0]

// Choose measurement
// ** Put your code here **

measure q -> c;
