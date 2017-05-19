// quantum ripple-carry adder
// 8-bit adder made out of 2 4-bit adders from adder.qasm
// Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
gate majority a,b,c 
{ 
  cx c,b; 
  cx c,a; 
  ccx a,b,c; 
}
gate unmaj a,b,c 
{ 
  ccx a,b,c; 
  cx c,a; 
  cx a,b; 
}

// add a to b, storing result in b
gate add4 a0,a1,a2,a3,b0,b1,b2,b3,cin,cout 
{
  majority cin,b0,a0;
  majority a0,b1,a1;
  majority a1,b2,a2;
  majority a2,b3,a3;
  cx a3,cout;
  unmaj a2,b3,a3;
  unmaj a1,b2,a2;
  unmaj a0,b1,a1;
  unmaj cin,b0,a0;
}

// add two 8-bit numbers by calling the 4-bit ripple-carry adder
// carry bit on output lives in carry[0]
qreg carry[2];
qreg a[8];
qreg b[8];
creg ans[8];
creg carryout[1];
// set input states
x a[0]; // a = 00000001
x b;
x b[6]; // b = 10111111
// output should be 11000000 0

add4 a[0],a[1],a[2],a[3],b[0],b[1],b[2],b[3],carry[0],carry[1];
add4 a[4],a[5],a[6],a[7],b[4],b[5],b[6],b[7],carry[1],carry[0];

measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure b[4] -> ans[4];
measure b[5] -> ans[5];
measure b[6] -> ans[6];
measure b[7] -> ans[7];
measure carry[0] -> carryout[0];
