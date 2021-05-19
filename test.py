from functools import partial

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import StateFn, Z, X, ListOp, Zero
from qiskit.opflow.gradients import SAMGradient
grad_values = np.array([list([-1.0, -1.0, -1.0]), list([2.0, 2.0, 2.0])])
print(grad_values)

grad_values = grad_values.transpose()

eps = 0.05 * grad_values / np.linalg.norm(grad_values)

print(eps)


# params = [Parameter('a'), Parameter('b')]
# params_op = ListOp([param * (~Zero @ Zero) for param in params])
# p = params_op.bind_parameters({params[0]: 1.0, params[1]: 2.0}).eval()

# ham = 0.5 * X - 1 * Z
# a = Parameter('a')
# b = Parameter('b')
# params = [a, b]
# qc = QuantumCircuit(1)
# qc.h(0)
# qc.rz(a, 0)
# qc.ry(b, 0)
# op = ~StateFn(ham) @ StateFn(qc)
#
# sam_grad_op = SAMGradient().convert(op, params)
# sam_grad_op2 = SAMGradient(second_order=True).convert(op, params)
# result = sam_grad_op.bind_parameters({params[0]: 1.0, params[1]: 2.0}).eval()
# result2 = sam_grad_op2.bind_parameters({params[0]: 1.0, params[1]: 2.0}).eval()
#
# print(result)
# print(result2)
#
