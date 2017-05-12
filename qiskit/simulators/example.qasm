OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[2];
h q;
qreg v[1];
h v;


cx q[0],v[0];
h v;
h q;
