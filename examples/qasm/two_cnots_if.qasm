OPENQASM 2.0;
qreg r0[1];
qreg r1[1];
creg c[2];
gate cx a,b { CX a,b; }
cx r0[0],r1[0];
if(c==1) cx r1[0],r0[0];
