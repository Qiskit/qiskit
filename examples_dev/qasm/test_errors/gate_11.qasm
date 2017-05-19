IBMQASM 2.0;
gate g1(p1,p2) a,b {
CX a,b; // no indent
}
qreg q[2];
g1 q[0],q[1];
