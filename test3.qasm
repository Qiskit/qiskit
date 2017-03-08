OPENQASM 2.0;
qreg qqq[3];
qreg t[1];
creg tc[1];
gate cx a,b { CX a,b; }
cx qqq[0],qqq[1];
cx qqq[1],qqq[2];
cx qqq[2],qqq[0];
cx qqq[0],t[0];
measure t[0] -> tc[0];
