// A simple 8 qubit example entangling two 4 qubit registers
OPENQASM 2.0;
include "qelib1.inc";

qreg a[4];
qreg b[4];
creg c[4];
creg d[4];
h a;
cx a, b;
barrier a;
barrier b;
measure a[0]->c[0];
measure a[1]->c[1];
measure a[2]->c[2];
measure a[3]->c[3];
measure b[0]->d[0];
measure b[1]->d[1];
measure b[2]->d[2];
measure b[3]->d[3];
