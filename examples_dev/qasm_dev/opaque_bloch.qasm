OPENQASM 2.0;
gate cx a,b { CX a,b; }
gate u1(a) b { U(0,0,a) b; }
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate h a { u2(0,pi) a; }
opaque bloch(u) v;
qreg r[10];
creg c[10];
h r[0];
cx r[0],r[1];
bloch(1) r[0];
cx r[0],r[2];
bloch(2) r[2];
cx r[2],r[3];
u1(pi/2) r[3];
bloch(2) r[3];
h r[3];
cx r[3],r[4];
measure r->c;
