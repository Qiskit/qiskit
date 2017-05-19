OPENQASM 2.0;
include "qelib1.inc";
gate pre q { }   // pre-rotation
gate post q { }  // post-rotation
qreg q[1];
creg c[1];
pre q[0];
barrier q;
h q[0];
barrier q;
post q[0];
measure q[0] -> c[0];
