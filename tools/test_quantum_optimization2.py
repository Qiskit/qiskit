"""Test the cost function for different basis functions."""
import sys
sys.path.append("..")
import numpy as np
from quantum_optimization import trial_funtion_optimization
from scripts.qhelpers.misc import program_to_text

entangler_map = {0: [2], 1: [2], 3: [2], 4: [2]}

m = 1
n = 6
theta = np.zeros(m*n)
trial_circuit = trial_funtion_optimization(n, m, theta, entangler_map)

program = [trial_circuit]
print(program_to_text(program))
