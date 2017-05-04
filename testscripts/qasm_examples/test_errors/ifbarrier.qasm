// if barrier example - should be rejected by parser
OPENQASM 2.0;
qreg q[5];
creg c[5];
if(c==5) barrier q;
