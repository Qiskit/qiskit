// Repetition code syndrome measurement
OPENQASM 3;
include "stdgates.inc";

qubit q[3];
qubit a[2];
bit c[3];
bit syn[2];

def syndrome qubit[3]:d, qubit[2]:a -> bit[2] {
  bit b[2];
  cx d[0], a[0];
  cx d[1], a[0];
  cx d[1], a[1];
  cx d[2], a[1];
  measure a -> b;
  return b;
}
reset q;
reset a;
x q[0]; // insert an error
barrier q;
syn = syndrome q, a;
if(int[2](syn)==1) x q[0];
if(int[2](syn)==2) x q[2];
if(int[2](syn)==3) x q[1];
c = measure q;
