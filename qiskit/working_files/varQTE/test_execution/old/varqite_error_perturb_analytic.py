from jax import grad, jit
import jax.numpy as jnp

import numpy as np
import scipy as sp
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

def inner_prod(x, y):
    return np.matmul(np.conj(np.transpose(x)), y)

evolution_op = lambda t: expm(-1 * H * t)
init_state = np.ones(2 ** num_qubits) / np.sqrt(2 ** num_qubits)
target_state = lambda t: np.dot(evolution_op(t), init_state)/\
                         np.sqrt(inner_prod(init_state, inner_prod(evolution_op(2*t), init_state)))

dt_target_state = lambda t: np.dot(-1 * H, target_state(t)) - \
                            np.dot(evolution_op(t), init_state) * 0.5 / \
                            (inner_prod(init_state,
                                        inner_prod(evolution_op(2*t), init_state))) ** (3/2) * \
                            inner_prod(init_state, inner_prod(-2*H, inner_prod(evolution_op(2*t),
                                                                          init_state)))

error = 0
for j in range(1, time_steps+1):
    # generate exact state

    target = target_state(j * time / time_steps)
    # dt_target = dt_target_state(j * time / time_steps)
    # print('dt target', dt_target)

    energy = np.dot(np.conj(np.transpose(target)), np.dot(H, target))

    # et_ket = dt_target + np.dot(H, target) - energy * target
    # print('et ket', et_ket)
    et_ket = np.ones(2 ** num_qubits) / np.sqrt(2 ** num_qubits) * j * 1e-2
    dt_analytic = - (np.dot(H, target) - energy * target)
    dt_target = et_ket + dt_analytic
    et = np.matmul(np.conj(np.transpose(et_ket)), et_ket)
    print('et 1', et)
    # alt_re = -2 * np.real(np.matmul(np.conj(np.transpose(dt_target)), target)) * energy + \
    #          2 * np.real(np.matmul(np.conj(np.transpose(dt_target)), np.matmul(H, target)))

    et_other = np.matmul(np.conj(np.transpose(dt_target)), dt_target) + \
               np.matmul(np.conj(np.transpose(target)),
                      np.matmul(np.matmul(H, H), target)) - energy ** 2 +\
               2 * np.real(np.matmul(np.transpose(np.conj(dt_target)), np.matmul(H, target))) - \
               2 * np.real(np.matmul(np.transpose(np.conj(dt_target)),
                                     np.matmul(energy*np.eye(2 ** num_qubits), target)))

    # TODO 2 * np.real(np.matmul(np.transpose(np.conj(dt_target)), np.matmul(H, target))) wrong!!!

    print('et 2', np.round(et_other, 6))
    # check et
    error += time/time_steps * np.sqrt(et)

    print('e at time t', np.round(et, 3))
    print('error bound at time t', np.round(error, 3))

