//Rewriting the code in pulse definitions backend=alt_almaden
// pulses defined for specific devices
// user could add custom pulses
pulse x90p_q0 [samples]
//fill this out OR just get from the library
pulse x90m_q0 [samples]


//include pulses
//included lib above

gate u3(theta, phi, lambda) 0 {
    fc(-lambda) d0;
    pulse x90p_q0 d0;
    fc(-theta) d0;
    pulse x90m_q0 d0;
    fc(-phi) d0;
}

gate u3(theta, phi, lambda) 1 {
    fc(-lambda) d1;
    pulse x90p_q1 d1;
    fc(-theta) d1;
    pulse x90m_q1 d1;
    fc(-phi) d1;
}


//usage: u3(pi, pi, pi)

gate u2(phi, lambda) 0 {
    fc(-lambda) d0;
    pulse Y90p_q0 d0;
    fc(-phi) d0;
}

gate u1(lambda) 0 {
    fc(-lambda) d0;
}

gate x 0 {
    pulse xp_d0 d0;
}

-----

gate_custom id 0 {
    pulse QId_d0 d0;
}

-----

gate measure 0 {
    pulse M_m0 m0;
    acquire m0 creg[0];
    acquire m1 creg[1];
    acquire m2 creg[2]; # continue on for all qubits ...
}

-----

gate cx 0, 1 {
    fc(np.pi/2) d0;
    pulse Ym_d0 d0;
    pulse x90p_d1 d1;
    barrier d0, d1, u0;
    pulse CR90p_d1 d1;
    barrier d0, d1, u0;
    pulse xp_d0 d0;
    barrier d0, d1, u0;
    pulse CR90m_d1 d1;
}


//quantum teleportation example after

OPENQASM 3.0;
include "qelib1.inc";
qreg q[3];
creg c0[1];
creg c1[1];
creg c2[1];

//u3(0.3,0.2,0.1) q[0];
fc(-0.1) d0;
pulse x90p_q0 d0;
fc(-0.3) d0;
pulse x90m_q0 d0;
fc(-0.2) d0;


h q[1];

cx q[1],q[2];

barrier q;

//cx q[0],q[1];
fc(np.pi/2) d0;
pulse Ym_d0 d0;
pulse x90p_d1 d1;
barrier d0, d1, u0;
pulse CR90p_d1 d1;
barrier d0, d1, u0;
pulse xp_d0 d0;
barrier d0, d1, u0;
pulse CR90m_d1 d1;

h q[0];

// measure q[0] -> c0[0];
pulse M_m0 m0;
acquire a0 c0[0];


// measure q[1] -> c1[0];
pulse M_m1 m1;
acquire a0 c1[0];

if(c0==1) z q[2];
//if(c1==1) x q[2];
if(c1==1) pulse xp_d2 d2;

//measure q[2] -> c2[0];
pulse M_m2 m2;
acquire a0 c2[0];
