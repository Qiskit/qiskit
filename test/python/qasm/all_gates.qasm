OPENQASM 2.0;
include "qelib1.inc";

gate rcccx_dg q0,q1,q2,q3 { u2(-2*pi,pi) q3; u1(pi/4) q3; cx q2,q3; u1(-pi/4) q3; u2(-2*pi,pi) q3; u1(pi/4) q3; cx q1,q3; u1(-pi/4) q3; cx q0,q3; u1(pi/4) q3; cx q1,q3; u1(-pi/4) q3; cx q0,q3; u2(-2*pi,pi) q3; u1(pi/4) q3; cx q2,q3; u1(-pi/4) q3; u2(-2*pi,pi) q3; }
gate rcccx q0,q1,q2,q3 { u2(0,pi) q3; u1(pi/4) q3; cx q2,q3; u1(-pi/4) q3; u2(0,pi) q3; cx q0,q3; u1(pi/4) q3; cx q1,q3; u1(-pi/4) q3; cx q0,q3; u1(pi/4) q3; cx q1,q3; u1(-pi/4) q3; u2(0,pi) q3; u1(pi/4) q3; cx q2,q3; u1(-pi/4) q3; u2(0,pi) q3; }
gate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }
gate mcx_5272411680 q0,q1,q2,q3,q4 { h q4; cu1(-pi/2) q3,q4; h q4; rcccx q0,q1,q2,q3; h q4; cu1(pi/2) q3,q4; h q4; rcccx_dg q0,q1,q2,q3; c3sx q0,q1,q2,q4; }

qreg q[5];
creg c[5];

// note that the order and where the gates are applied to is important!

// "abstract" gates (for legacy)
U(0.2,0.1,0.6) q[0];  // Creates u
CX q[0], q[1];        // Creates cx

// the hardware primitives
u3(0.2,0.1,0.6) q[0];
u2(0.1,0.6) q[0];
u1(0.6) q[0];
id q[0];
cx q[0], q[1];

// the standard single qubit gates
u(0.2, 0.1, 0.6) q[0];
p(0.6) q[0];
x q[0];
y q[0];
z q[0];
h q[0];
s q[0];
t q[0];
sdg q[0];
tdg q[0];
sx q[0];
sxdg q[0];

// the standard rotations
rx(0.1) q[0];
ry(0.1) q[0];
rz(0.1) q[0];

// the barrier
barrier q;

// the standard user-defined gates
swap q[0], q[1];
cswap q[0], q[1], q[2];

cy q[0], q[1];
cz q[0], q[1];
ch q[0], q[1];
csx q[0], q[1];
cu1(0.6) q[0], q[1];
cu3(0.2,0.1,0.6) q[0], q[1];
cp(0.6) q[0], q[1];
cu(0.2,0.1,0.6,0) q[0], q[1];
ccx q[0], q[1], q[2];

crx(0.6) q[0], q[1];
cry(0.6) q[0], q[1];
crz(0.6) q[0], q[1];

rxx(0.2) q[0], q[1];
rzz(0.2) q[0], q[1];

rccx q[0], q[1], q[2];
rcccx q[0], q[1], q[2], q[3];
mcx q[0], q[1], q[2], q[3];
c3sx q[0], q[1], q[2], q[3];
// mcx_c4 q[0], q[1], q[2], q[3], q[4];  // no option to test this gate

barrier q;

// measure
measure q->c;
