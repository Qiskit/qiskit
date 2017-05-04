// Test barriers in gates
OPENQASM 2.0;
include "junk.incl";
gate new() a,b,c
{
  CX a,b;
  id c;
  barrier a,c;
  CX b,c;
  id a;
  barrier a,b;
  CX a,b;
  barrier a,b,c;
}
qreg q[2];
qreg r[2];
qreg s[1];
creg c[2];
U(0.3*0.2/1,0.1,0) q[0];
barrier q;
CX q[0],q[1];
U(0,0,pi/2) r;
CX r[0],s;
my(sin(0.3)/cos(0.3)) q[1],q[0];
barrier q;
U(.3,.3,-.2) q[0];
CX q[0],q[1];
U(.3,.3,-.2) q[1];
id q[0];
new() q[0],q[1],r[0];
barrier q;
measure q -> c;
measure r[0] -> c[0];
