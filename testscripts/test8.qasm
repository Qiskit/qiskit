OPENQASM 2.0;
include "qelib1.inc";
qreg r[4];
cx r[1], r[0];
cx r[2], r[1];
cx r[3], r[2];
