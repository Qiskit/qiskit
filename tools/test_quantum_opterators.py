"""Test the cost function for different basis functions."""
import numpy as np
from operators import gate_qij


gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
gate_qij(gate, 0, 1, 2)
print(gate_qij)
