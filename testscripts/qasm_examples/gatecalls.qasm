// Test a variety of gate call signatures
OPENQASM 2.0;
include "qelib1.inc";
gate test1 a, b, c
{
  h a;
  h b;
  h c;
}
gate test2() a, b, c
{
  x a;
  x b;
  x c;
}
gate test3(u) a, b, c
{
  u1(u) a;
  u1(u) b;
  u1(u) c;
}
gate test4(v1,v2,v3,v4) a, b, c, d
{
  u1(v1) a;
  u1(v2) b;
  u1(v3) c;
  u1(v4) d;
}
qreg q[5];
qreg r[5];
qreg v[5];
test1   q[0],q[1],q[2];
test1   q,r[0],v[1];
test1   q,r[0],v;
test1() q[0],r,v;
test1   q,r,v;
test2() q[0],q[1],q[2];
test2   q,r[0],v[1];
test2() q,r[0],v;
test2   q[0],r,v;
test2() q,r,v;
test3(0.1) q[0],q[1],q[2];
test3(0.2) q,r[0],v[1];
test3(0.3) q,r[0],v;
test3(0.4) q[0],r,v;
test3(0.5) q,r,v;
test4(0.11,0.12,0.13,0.14) q[0],q[1],r,v;
