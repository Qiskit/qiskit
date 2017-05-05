"""Test the cost function for different basis functions."""
import sys
import numpy as np
sys.path.append("..")
from quantum_optimization import trial_funtion_optimization
from scripts.qhelpers.misc import program_to_text

entangler_map = {0: [2], 1: [2], 3: [2], 4: [2]}

m = 1
n = 6
theta = np.zeros(m*n)
trial_circuit = trial_funtion_optimization(n, m, theta, entangler_map)

program = [trial_circuit]
print(program_to_text(program))

"""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
u2(0.0,3.141592653589793) q[5];
u2(0.0,3.141592653589793) q[4];
u2(0.0,3.141592653589793) q[3];
u2(0.0,3.141592653589793) q[2];
u2(0.0,3.141592653589793) q[1];
u2(0.0,3.141592653589793) q[0];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
u2(0.0,3.141592653589793) q[2];
cx q[0],q[2];
u3(0.0,0.0,0.0) q[0];
u2(0.0,3.141592653589793) q[2];
u2(0.0,3.141592653589793) q[2];
cx q[1],q[2];
u2(0.0,3.141592653589793) q[2];
u2(0.0,3.141592653589793) q[2];
u3(0.0,0.0,0.0) q[1];
cx q[3],q[2];
u2(0.0,3.141592653589793) q[2];
u2(0.0,3.141592653589793) q[2];
u3(0.0,0.0,0.0) q[3];
cx q[4],q[2];
u3(0.0,0.0,0.0) q[4];
u2(0.0,3.141592653589793) q[2];
u3(0.0,0.0,0.0) q[2];
u3(0.0,0.0,0.0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[5] -> c[5];
measure q[4] -> c[4];
measure q[3] -> c[3];
measure q[2] -> c[2];
measure q[1] -> c[1];
measure q[0] -> c[0];
"""
