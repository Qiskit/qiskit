// quantum teleportation example
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
// optional post-rotation for state tomography
gate post q { }
u3(0.3,0.2,0.1) q[0];
h q[1];
cx q[1],q[2];
barrier q;
cx q[0],q[1];
h q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
if(c==1) z q[2];
if(c==2) x q[2];
if(c==3) y q[2];
post q[2];
measure q[2] -> c[2];
