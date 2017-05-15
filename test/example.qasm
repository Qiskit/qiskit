OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[3];
 h q[0];
qreg v[2];
h v[1];


cx q[0],v[0];
//cx q[0],v[1];
// h v;
 //h q;

reset v[1];

measure q[0] -> c[0];
measure v[0] -> c[1];
measure v[1] -> c[2];
