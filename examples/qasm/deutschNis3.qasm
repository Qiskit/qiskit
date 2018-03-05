include "qelib1.inc";
qreg q[5];
creg c[5];

h q[0];
h q[1];
h q[2];
h q[2];
z q[0];
cx q[1],q[2];
h q[2];
h q[0];
h q[1];
h q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
