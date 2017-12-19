OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
u1(-pi) qr[0];
u1(-pi/2) qr[0];
measure qr[0] -> cr[0];