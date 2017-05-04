// barriers with different arguments
OPENQASM 2.0;
qreg q[10];
qreg r[2];
qreg v[3];
barrier q,r,v;
barrier q[0],r[1],v[2];
barrier q[0],r,v[1];
barrier q,r[0],v;
