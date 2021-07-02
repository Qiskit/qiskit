/*
 * Prepare a parameterized number of Bell pairs
 * and teleport a qubit using them.
 */
OPENQASM 3;
include "stdgates.inc";

const n_pairs = 10;  // number of teleportations to do

def bellprep qubit[2]:q {
  reset q;
  h q[0];
  cx q[0], q[1];
}

def xprepare qubit:q {
  reset q;
  h q;
}

qubit input;
bit output;
qubit q[2*n_pairs];

xprepare input;
rz(pi / 4) input;

let io = input;
for i in [0: n_pairs - 1] {
  let bp = q[2 * i, 2 * i + 1];
  bit pf[2];
  bellprep bp;
  cx io, bp[0];
  h io;
  pf[0] = measure io;
  pf[1] = bp[0];
  if (pf[0]==1) z bp[1];
  if (pf[1]==1) x bp[1];
  let io = bp[1];
}

h io;
output = measure io;  // should get zero
