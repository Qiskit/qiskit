OPENQASM 2.0;
// https://github.com/QISKit/qiskit-sdk-py/issues/111
include "qelib1.inc";
qreg q[4];
creg c[4];

y q[0];
z q[2];
h q[2];
cx q[1], q[0];
y q[2];
t q[2];
z q[2];
cx q[1], q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
