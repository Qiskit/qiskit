include "qelib1.inc";
qreg q[2];
creg c[2];

x q[1];
cx q[0],q[1];
h q[0];
h q[1];
cx q[0],q[1];
h q[0];
h q[1];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
