from jax import grad, jit
import jax.numpy as jnp

import numpy as np
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.aqua.operators import StateFn, MatrixOp, CircuitOp, X, I, PauliOp, Z, Y
from qiskit.aqua.operators.gradients import NaturalGradient
from scipy.linalg import expm

np.random.seed = 2

H = (Y ^ X)
num_qubits = H.num_qubits
H = H.to_matrix()



time = 1
time_steps = 3

evolution_op = lambda t: expm(-1j * H * t)
init_state = np.ones(2 ** num_qubits) / np.sqrt(2 ** num_qubits)
target_state = lambda t: np.dot(evolution_op(t), init_state)
dt_target_state = lambda t: np.dot(-1j * H, target_state(t))

error = 0
for j in range(time_steps):
    # generate exact state

    target = target_state(j * time / time_steps)
    # dt_target = dt_target_state(j * time / time_steps)

    # et_ket = 1j*dt_target - np.dot(H, target)
    et_ket = np.ones(2 ** num_qubits) / np.sqrt(2 ** num_qubits) * j * 1e-2
    dt_target = -1j*(et_ket + np.dot(H, target))
    et = np.dot(np.conj(np.transpose(et_ket)), et_ket)
    print('et 1', et)
    et_other = np.dot(np.conj(np.transpose(dt_target)), dt_target) + \
               np.dot(np.conj(np.transpose(target)), np.dot(np.linalg.matrix_power(H, 2),
                                                                                   target)) - \
               2 * np.imag(np.dot(np.conj(np.transpose(dt_target)), np.dot(H, target)))
    print('et 2', np.round(et_other, 6))
    # check et
    error += time/time_steps * np.sqrt(et)

    print('e at time t', np.round(et, 3))
    print('error bound at time t', np.round(error, 3))

