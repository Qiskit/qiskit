IBMQASM 2.0;
gate g1(p1,p2) a,b {
  U(p1,p2,0) a;
  U(0,p1,p2) b;
  CX a,b;
}
qreg q[3];
qreg r[2];
g1(0.0,0.0) q,r;
