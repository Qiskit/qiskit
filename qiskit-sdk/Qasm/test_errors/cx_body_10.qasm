IBMQASM 2.0;
qreg q[2];
gate g a,b {
  CX q,q[1];
}
