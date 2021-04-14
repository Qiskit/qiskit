import numpy as np
from qiskit import QuantumCircuit, Aer, execute, transpile, BasicAer
from qiskit.quantum_info.random import random_unitary

# %%

# np.random.seed(280680)

def deep_decompose(circ):
    new_circ = circ.decompose()

    while new_circ.count_ops() != circ.count_ops():
        circ = new_circ
        new_circ = circ.decompose()

    return circ

def get_state(circ):
    backend = BasicAer.get_backend('statevector_simulator')
    state_vector = execute(circ, backend).result().get_statevector()

    return state_vector


def h():
    return 1/np.sqrt(2) * np.array([[1,1],[1,-1]])

# %%
agate = np.array([[-1, 0],[0, -1]])

u1 = np.zeros((2,2), complex)
u1[:, 0] = np.random.rand(2) + 1j * np.random.rand(2)
u1[:, 0] = u1[:, 0] / np.linalg.norm(u1[:,0])
u1[:,1] = [np.conj(u1[1,0]), -np.conj(u1[0,0])]

u2 = np.zeros((2,2), complex)
u2[:, 0] = np.random.rand(2) + 1j * np.random.rand(2)
u2[:, 0] = u2[:, 0] / np.linalg.norm(u2[:,0])
u2[:,1] = [np.conj(u2[1,0]), -np.conj(u2[0,0])]

phase = np.angle(u1[0,0])
u1 = np.e**(-1j*phase) * u1

gates = [random_unitary(2).data for i in range(2 ** 4)]

# %%
print('original circuit --------')
size = int(np.log2(len(gates))) +1
c2 = QuantumCircuit(size)
# c2.h(0)
target = 0
controls = list(range(1, size))
c2.uc(gates, controls, target, False, True)
c2_state = get_state(c2)
print('c2_state: ', c2_state)
# print('phase: ', c2.global_phase)
# print('circuit: \n', c2.draw())

# %%
print('decomposed circuit --------')
dc2 = transpile(c2, basis_gates=['cx', 'u1', 'u2', 'u3', 'id' ], optimization_level=0)
# dc2 = dc2.reverse_bits()

# dc2 = deep_decompose(c2)
# dc2.global_phase -= 0.25 * np.pi
state = get_state(dc2)
# print(dc2.draw())

print('state: ', state)
# print('phase: ', dc2.global_phase)
# print('circuit: \n', dc2.draw())
# print('phase state: ', np.e**(-1j * dc2.global_phase)*state)
#
print('gates: ')
print(gates[0])
# dc3 = dc2.reverse_bits()
# dc3.global_phase = dc2.global_phase
# print(dc3.global_phase)


# print((np.angle(state[0]) - np.angle(c2_state[0])/2) / np.pi)
# print((np.angle(state[2]) - np.angle(c2_state[2])/2) / np.pi)

# %%
#print(dc2.global_phase)

# print(dc2.draw())
# a = QuantumCircuit(2)
# a.h([0,1])
# a.diagonal([-1,-1j,1,1], a.qubits)
# get_state(a.decompose())

# print(gates[0])
# %%
# a = QuantumCircuit(2)
# state = np.random.rand(4)
# state = state / np.linalg.norm(state)
# a.isometry(state, a.qubits, None)
#
# print('state: ', state)
# # a.global_phase = 0
# print(get_state(a))
# print((np.angle(c2_state[0])- np.angle(state[0])) / np.pi)
#
# n = 3
# v = np.random.rand(2**n) + np.random.rand(2**n)*1j
# v = v / np.linalg.norm(v)
# a = QuantumCircuit(n)
# a.isometry(v, a.qubits, None)
# print('--------------')
# print(v[0])
# vcalc = get_state(deep_decompose(a))
# print(vcalc[0])
