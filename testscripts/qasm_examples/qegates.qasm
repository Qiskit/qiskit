// Example with operations supported on a 5 qubit device
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
u3(3.2,10.6,-1) q[0];
barrier q[0],q[1];
CX q[0],q[1];
u2(.2,sin(.01)) q[1];
u1(.001) q[0];
measure q -> c;
