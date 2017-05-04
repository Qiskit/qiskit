// One randomized benchmarking sequence
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
barrier q;
cz q[0],q[1];
barrier q;
s q[0];
cz q[0],q[1];
barrier q;
s q[0];
z q[0];
h q[0];
barrier q;
measure q -> c;
