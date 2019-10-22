// plaquette check
OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

x q[1];
x q[4];
barrier q;

cx q[0], q[2];
cx q[1], q[2];
cx q[3], q[2];
cx q[4], q[2];
barrier q;
measure q[0]->c[0];
measure q[1]->c[1];
measure q[2]->c[2];
measure q[3]->c[3];
measure q[4]->c[4];
