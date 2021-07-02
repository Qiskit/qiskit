// QFT and measure, version 2
OPENQASM 3;
include "stdgates.inc";

qubit q[4];
bit c0;
bit c1;
bit c2;
bit c3;

reset q;
h q;
barrier q;
h q[0];
measure q[0] -> c0;
if(c0 == 1) { rz(pi / 2) q[1]; }
h q[1];
measure q[1] -> c1;
if(c0==1) { rz(pi / 4) q[2]; }
if(c1==1) { rz(pi / 2) q[2]; }
h q[2];
measure q[2] -> c2;
if(c0 == 1) { rz(pi / 8) q[3]; }
if(c1 == 1) { rz(pi / 4) q[3]; }
if(c2 == 1) { rz(pi / 2) q[3]; }
h q[3];
measure q[3] -> c3;
