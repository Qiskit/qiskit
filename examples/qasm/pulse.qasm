//Rewriting the code in pulse definitions backend=alt_almaden
// pulses defined for specific devices
//declare channels
dch d0;
dch d1;
uch u0;
uch u1;
mch m0;
mch m1;


// user could add custom pulses
pulse x90p_q0  [[0.004,0.009],
[0.029,0.05],[0.8,-0.355]]
[0.004,-0.009]
//fill this out OR just get from the library
pulse x90m_q0 [samples]

//include pulses
//included lib above

gate u3(theta, phi, lambda) a {
    framechange(-lambda) d0;
    pulse x90p_q0 d0;
    framechange(-theta) d0;
    pulse x90m_q0 d0;
    framechange(-phi) d0;
}

gate u3(theta, phi, lambda) a {
    framechange(-lambda) d1;
    pulse x90p_q1 d1;
    framechange(-theta) d1;
    pulse x90m_q1 d1;
    framechange(-phi) d1;
}


//usage: u3(pi, pi, pi)

gate u2(phi, lambda) a {
    framechange(-lambda) d0;
    pulse y90p_q0 d0;
    framechange(-phi) d0;
}

gate u1(lambda) a {
    framechange(-lambda) d0;
}

gate x a {
    pulse xp_d0 d0;
}


gate id a {
    pulse qId_d0 d0;
}

gate measure a {
    pulse m_m0 m0;
    acquire m0 creg[0];
    acquire m1 creg[1];
    acquire m2 creg[2];
    // continue on for all qubits ...
}

gate cx a, b {
    framechange(np.pi/2) d0;
    pulse ym_d0 d0;
    pulse x90p_d1 d1;
    ch_barrier d0, d1, u0;
    pulse cR90p_d1 d1;
    ch_barrier d0, d1, u0;
    pulse xp_d0 d0;
    ch_barrier d0, d1, u0;
    pulse cR90m_d1 d1;
}


//quantum teleportation example after

OPENQASM 3.0;
include "qelib1.inc";
qreg q[3];
creg c0[1];
creg c1[1];
creg c2[1];

//u3(0.3,0.2,0.1) q[0];
framechange(-0.1) d0;
pulse x90p_q0 d0;
framechange(-0.3) d0;
pulse x90m_q0 d0;
framechange(-0.2) d0;


h q[1];

cx q[1],q[2];

barrier q;

//cx q[0],q[1];
framechange(np.pi/2) d0;
pulse ym_d0 d0;
pulse x90p_d1 d1;
ch_barrier d0, d1, u0;
pulse cR90p_d1 d1;
ch_barrier d0, d1, u0;
pulse xp_d0 d0;
ch_barrier d0, d1, u0;
pulse cR90m_d1 d1;

h q[0];

// measure q[0] -> c0[0];
pulse m_m0 m0;
acquire a0 c0[0];


// measure q[1] -> c1[0];
pulse m_m1 m1;
acquire a0 c1[0];

if(c0==1) z q[2];
//if(c1==1) x q[2];
if(c1==1) pulse xp_d2 d2;

//measure q[2] -> c2[0];
pulse m_m2 m2;
acquire a0 c2[0];
