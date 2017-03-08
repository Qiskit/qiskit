OPENQASM 2.0;
gate v x,y,z
{
  U(0,0,0) x;
  barrier x,y;
  U(0,0,0) y;
  barrier x,z;
  U(0,0,0) z;
  barrier x,y,z;
  U(0,0,0) x;
}
qreg q[3];
v q[0],q[1],q[2];
