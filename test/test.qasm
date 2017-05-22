// A simple 8 qubit example
OPENQASM 2.0;
include "qelib1.inc";

qreg a[4];
qreg b[4];
creg ans[5];
h a[3];
cx a[3],b[0];
