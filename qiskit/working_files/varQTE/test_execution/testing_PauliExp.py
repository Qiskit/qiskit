import numpy as np
from qiskit import QuantumCircuit, Aer

from qiskit.opflow import StateFn, Z, I, Y, X, PauliExpectation, Gradient, QFI, CircuitStateFn
from qiskit.circuit import Parameter

a = Parameter('a')
b = Parameter('b')

qc = QuantumCircuit(2)
qc.ry(a, 1)
qc.cx(1, 0)
qc.ry(a, 0)

H = Z^X

# op = StateFn(H, is_measurement=True) @ CircuitStateFn(qc)
op = CircuitStateFn(qc)

gradient = QFI('lin_comb_full').convert(op)

init_params = [np.pi/3]
params = [a]
print(gradient)
print(gradient.assign_parameters(dict(zip(params, init_params))).eval())

gradient = PauliExpectation().convert(gradient)
print(gradient)
print(gradient.assign_parameters(dict(zip(params, init_params))).eval())



