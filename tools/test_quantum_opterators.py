"""Test the cost function for different basis functions."""
import numpy as np
from operators import operator_qi, operator_qij

# single qubit Test
gate = np.array([[1, 0], [0, -1]])

print(operator_qi(gate, 0, 3))
print(operator_qi(gate, 1, 3))
print(operator_qi(gate, 2, 3))

<<<<<<< Updated upstream
cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
print(" --------------------")
print(operator_qij(gate, 0, 1, 2))
=======
cnot = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
print(" --------------------")
print(operator_qij(cnot, 0, 1, 2))
print(operator_qij(cnot, 1, 0, 2))


print(operator_qij(cnot, 0, 1, 3))
print(operator_qij(cnot, 1, 2, 3))
>>>>>>> Stashed changes
