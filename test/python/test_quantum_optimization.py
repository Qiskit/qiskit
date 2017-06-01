"""Test the cost function for different basis functions."""
import sys
import numpy as np
import sys
sys.path.append("../..")
from tools.optimizationtools import trial_funtion_optimization

entangler_map = {0: [2], 1: [2], 3: [2], 4: [2]}

m = 1
n = 6
theta = np.zeros(m * n)
trial_circuit = trial_funtion_optimization(n, m, theta, entangler_map)

print(trial_circuit.qasm())
"""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
cz q[0],q[2];
cz q[1],q[2];
cz q[3],q[2];
cz q[4],q[2];
ry(0.000000000000000) q[0];
ry(0.000000000000000) q[1];
ry(0.000000000000000) q[2];
ry(0.000000000000000) q[3];
ry(0.000000000000000) q[4];
ry(0.000000000000000) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
"""
