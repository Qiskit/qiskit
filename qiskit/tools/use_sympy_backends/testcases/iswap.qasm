// Name of Experiment: iswap v4

OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

x q[0];
s q[0];
s q[1];
h q[0];
cx q[0],q[1];
h q[0];
h q[1];
cx q[0],q[1];
h q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
