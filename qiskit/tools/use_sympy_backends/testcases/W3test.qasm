// Name of Experiment: W3test v1
// example of W-state |001> + |010> + |100> 
// found by user lukasknips at http://ibm.biz/qiskit-W3-Example

OPENQASM 2.0;
include "qelib1.inc";


qreg q[5];
creg c[5];

u3(-1.23096,0,0) q[0];
u3(pi/4,0,0) q[1];
cx q[0],q[2];
z q[2];
h q[2];
cx q[1],q[2];
z q[2];
u3(pi/4,0,0) q[1];
h q[2];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
