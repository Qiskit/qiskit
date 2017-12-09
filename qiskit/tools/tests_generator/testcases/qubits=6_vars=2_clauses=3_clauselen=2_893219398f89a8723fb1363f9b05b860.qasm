// Quantum code for the specified SAT problem.

include "qelib1.inc";

// Declare all needed (qu)bits
qreg v[3];
qreg c[3];
qreg a[1];
creg m[2];

// Prepare uniform superposition
h v[1];
h v[2];

// Marking with oracle evaluation
x c[0];
x c[1];
x c[2];
ccx v[1], v[2], c[0];
x v[1];
cx v[1], c[1];
x v[1];
x v[2];
cx v[2], c[2];
x v[2];
ccx c[0], c[1], a[0];
ccx c[2], a[0], v[0];
ccx c[0], c[1], a[0];
x v[2];
cx v[2], c[2];
x v[1];
x v[2];
cx v[1], c[1];
x v[1];
ccx v[1], v[2], c[0];

// Amplitude amplification
h v[1];
h v[2];
x v[0];
x v[1];
x v[2];
h v[0];
ccx v[1], v[2], v[0];
h v[0];
x v[0];
x v[1];
x v[2];
h v[0];
h v[1];
h v[2];

// Measurements
measure v[1] -> m[0];
measure v[2] -> m[1];