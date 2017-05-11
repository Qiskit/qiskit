OPENQASM 2.0;
qreg q[5];
creg c[5];
gate foo q { U(0,0,0) q; }
if(c==0) foo q[0];
