IBMQASM 2.0;
gate g1(u,v) a,b {
}
qreg q[2];
gate g2 a,b,c {
  g1(0.0,0.0) q[0],q[1];
}
