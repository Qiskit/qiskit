OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[2];
h q;
qreg v[1];



cx q[0],v[0];
// h v;
 //h q;

measure q[0] -> c[0];
measure v[0] -> c[1];
