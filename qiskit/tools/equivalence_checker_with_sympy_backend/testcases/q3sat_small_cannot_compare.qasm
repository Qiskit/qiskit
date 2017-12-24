include "qelib1.inc";
qreg var[3];
qreg conj[3];
qreg anci[1];
creg m[2];

h var[2];

x conj[0];
x conj[1];
x conj[2];
x var[2];
x var[2];

x var[2];

measure var[2] -> m[1];
