OPENQASM 2.0;
// https://github.com/QISKit/qiskit-sdk-py/issues/81
include "qelib1.inc";
qreg q[4];
creg c[4];
// -X-.-----
// -Y-+-S-.-
// -Z-.-T-+-
// ---+-H---
x q[0];
y q[1];
z q[2];
cx q[0], q[1];
cx q[2], q[3];
s q[1];
t q[2];
h q[3];
cx q[1], q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
