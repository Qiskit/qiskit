import numpy as np

from qiskit.opflow import MatrixOp

H = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

h = MatrixOp(H)
h_circ = h.to_circuit()

print(h_circ)