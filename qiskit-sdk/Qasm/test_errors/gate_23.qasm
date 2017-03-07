IBMQASM 2.0;
gate g1 a,b {
  CX a,b;
}
qreg q[3];
qreg r[2];
g1 q[1],q[1];
