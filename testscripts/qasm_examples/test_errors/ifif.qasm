// nested ifs should be rejected by the parser
OPENQASM 2.0; 
qreg q[5]; 
creg c[5]; 
if(c==1) if(c==2) CX q[0],q[1];
