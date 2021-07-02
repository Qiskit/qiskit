// QFT and measure, version 1
OPENQASM 3;
include "stdgates.inc";

qubit q[4];
bit c[4];
reset q;
h q;
barrier q;
h q[0];
c[0] = measure q[0];
if(int[4](c) == 1) { rz(pi / 2) q[1]; }
h q[1];
c[1] = measure q[1];
if(int[4](c) == 1){ rz(pi / 4) q[2]; }
if(int[4](c) == 2){ rz(pi / 2) q[2]; }
if(int[4](c) == 3){ rz(pi / 2 + pi / 4) q[2]; }
h q[2];
c[2] = measure q[2];
if(int[4](c) == 1) rz(pi / 8) q[3];
if(int[4](c) == 2) rz(pi / 4) q[3];
if(int[4](c) == 3) rz(pi/4+pi/8) q[3];
if(int[4](c) == 4) rz(pi / 2) q[3];
if(int[4](c) == 5) rz(pi / 2 + pi / 8) q[3];
if(int[4](c) == 6) rz(pi / 2+ pi / 4) q[3];
if(int[4](c) == 7) rz(pi / 2 + pi / 4 + pi / 8) q[3];
h q[3];
c[3] = measure q[3];
