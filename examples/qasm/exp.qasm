// parameter expressions and opaque gates
OPENQASM 2.0;
qreg q[1];
qreg r[5];
qreg s[5];
creg c[1];
creg d[3];
gate mundane q {
  U(.1,.2,.3) q;
}
opaque magic q;
opaque magic2(x) q;
U(0.3*0.2/1,cos(0.3*pi)*.1,0) q[0];
CX r,s;
measure q -> c;
if(c==1) U(cos(pi/2.0),sin(pi/2.0),0.0) s[3];
barrier q;
reset q;
reset s;
barrier q,s;
magic q[0];
mundane q[0];
barrier q[0],r[0],s;
magic2(0.5) q[0];
if(d==0) mundane r[0];
if(d==1) magic r[1];
if(d==2) magic2(0.2) r[2];
magic2(0.3) r[3];
mundane r[4];
