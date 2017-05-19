IBMQASM 2.0;
gate g1(p1) a,b {
  U(p1,p1,p1) a;
  CX a,b;
}
qreg q[3];
qreg r[2];
g1 q[0],q[1];
