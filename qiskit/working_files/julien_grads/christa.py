"""The Pauli-Two-Design application."""

# pylint: disable=invalid-name

import numpy as np

from qiskit import Aer, IBMQ
from qiskit.circuit.library import PauliTwoDesign
from qiskit.opflow import StateFn, I, Z, CircuitSampler, X, SummedOp
from qiskit.utils import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel

# local imports
from cost_functions import (
    get_objective, get_overlap, get_vanilla_gradient, get_finite_difference_gradient,
    get_natural_gradient
)
from generalized_spsa import GSPSA
from gradient_descent import GradientDescent
from run_optimizers import run_optimizers
from plot_data import plot_data
from pauli_two_design import pauli_two_design

import matplotlib.pyplot as plt

from qiskit.circuit.library import EfficientSU2, RealAmplitudes

import networkx as nx
from qiskit_optimization.applications import Maxcut

# problem settings

np.random.seed = 131
# num_qubits = 9
# reps = 3
# seed = 42
# target_energy = -1

g = nx.Graph()
g.add_nodes_from(range(4))
g.add_edges_from([(0, 1), (1, 2), (0, 2), (1, 4), (4, 3), (2, 3)])
maxcut = Maxcut(g)
# nx.draw(g)
# plt.show()
# print('after draw')
qp = maxcut.to_quadratic_program()
H, offset = qp.to_ising()
observable = H
print('H ising ', observable.to_pauli_op())

circuit = RealAmplitudes(observable.num_qubits, entanglement='sca', reps=5)
# ansatz = EfficientSU2(observable.num_qubits, entanglement='sca', reps=d)

# Define a set of initial parameters
parameters = circuit.ordered_parameters
# init_param_values = np.zeros(len(ansatz.ordered_parameters))
# for i in range(ansatz.num_qubits):
#     init_param_values[-(i + 1)] = np.pi / 2


# observable and ansatz
# observable = SummedOp([(Z ^ Z) ^ (I ^ (num_qubits - 2)), -0.3 * (I ^ X) ^ (I ^ (num_qubits - 2))])
             # (X ^ I) ^ (I ^ (num_qubits - 2))])
# circuit, parameters = pauli_two_design(num_qubits, reps, seed)
ansatz = StateFn(circuit)

# execution parameters
initial_point = np.random.random(circuit.num_parameters)

# backend = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=8192)
backend = QuantumInstance(Aer.get_backend('statevector_simulator'))

# TOKEN = '41a450e0c5bf37ee6ffb442fb3cf07358fd5c2a291d8ae11c535af1ccb1e683d7e4cadf6c2ff24e0cd88e1cf' \
#         '286476a91ed0a78d1a336a5ac363783196e48448'
# IBMQ.save_account(TOKEN, overwrite=True)
# provider = IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
#
# backend_name = 'ibmq_montreal'
# backend_ibmq = provider.get_backend(backend_name)
# properties = backend_ibmq.properties()
# coupling_map = backend_ibmq.configuration().coupling_map
# noise_model = NoiseModel.from_backend(properties)
# shots = 1000
# qi_ibmq_noise_model = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
#                                        noise_model=noise_model, optimization_level=0, shots=shots,
#                                        seed_transpiler=2)
sampler = CircuitSampler(backend)


# define the optimizer settings
common = {'maxiter': 100,
          'learning_rate': 0.1,
          'trust_region': False,
          'tolerance': -1}  # do all steps

spsa = {'perturbation': 0.01,
        'blocking': False}


# get the objective and the overlap function for natural spsa
objective_fn = get_objective(observable, ansatz, parameters, sampler)
overlap_fn = get_overlap(ansatz, parameters, sampler)
vanilla_gradient = get_vanilla_gradient(observable, ansatz, parameters, sampler)
nat_grad = get_natural_gradient(observable, ansatz, parameters, sampler)
# finite_difference_gradient = get_finite_difference_gradient(
#     spsa['perturbation'], observable, ansatz, parameters, sampler
# )

# define the optimizers
optimizers = {
    'Gradient descent': (GradientDescent(**common), {'gradient_function': vanilla_gradient}),
    'Natural gradient descent': (GradientDescent(**common), {'gradient_function': nat_grad}),
}

runs = [1, 1]

# run the optimizers
saveas = 'grad_vs_nat_grad'
data = run_optimizers(optimizers, objective_fn,
                      initial_point, saveas, seed=None, runs=runs)

# plot the results
plot_data(data, saveas)
