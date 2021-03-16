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
    get_objective, get_overlap, get_vanilla_gradient, get_finite_difference_gradient
)
from generalized_spsa import GSPSA
from gradient_descent import GradientDescent
from run_optimizers import run_optimizers
from plot_data import plot_data
from pauli_two_design import pauli_two_design

# problem settings
num_qubits = 9
reps = 3
seed = 42
target_energy = -1

# observable and ansatz
observable = SummedOp([(Z ^ Z) ^ (I ^ (num_qubits - 2)), -0.3 * (I ^ X) ^ (I ^ (num_qubits - 2))])
             # (X ^ I) ^ (I ^ (num_qubits - 2))])
circuit, parameters = pauli_two_design(num_qubits, reps, seed)
ansatz = StateFn(circuit)

# execution parameters
initial_point = np.random.random(circuit.num_parameters)

# backend = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=8192)
backend = QuantumInstance(Aer.get_backend('statevector_simulator'))

TOKEN = '41a450e0c5bf37ee6ffb442fb3cf07358fd5c2a291d8ae11c535af1ccb1e683d7e4cadf6c2ff24e0cd88e1cf' \
        '286476a91ed0a78d1a336a5ac363783196e48448'
IBMQ.save_account(TOKEN, overwrite=True)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')

backend_name = 'ibmq_montreal'
backend_ibmq = provider.get_backend(backend_name)
properties = backend_ibmq.properties()
coupling_map = backend_ibmq.configuration().coupling_map
noise_model = NoiseModel.from_backend(properties)
shots = 1000
qi_ibmq_noise_model = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                       noise_model=noise_model, optimization_level=0, shots=shots,
                                       seed_transpiler=2)
sampler = CircuitSampler(qi_ibmq_noise_model)


# define the optimizer settings
common = {'maxiter': 200,
          'learning_rate': 0.1,
          'trust_region': False,
          'tolerance': -1}  # do all steps

spsa = {'perturbation': 0.01,
        'blocking': False}


# get the objective and the overlap function for natural spsa
objective_fn = get_objective(observable, ansatz, parameters, sampler)
overlap_fn = get_overlap(ansatz, parameters, sampler)
vanilla_gradient = get_vanilla_gradient(
    observable, ansatz, parameters, sampler)
finite_difference_gradient = get_finite_difference_gradient(
    spsa['perturbation'], observable, ansatz, parameters, sampler
)

# define the optimizers
optimizers = {
    'SPSA': GSPSA(**common, **spsa),
    'Gradient descent': (GradientDescent(**common), {'gradient_function': vanilla_gradient}),
    'Finite difference': (GradientDescent(**common), {'gradient_function':
                                                          finite_difference_gradient}),
}

runs = [1, 1, 1]

# run the optimizers
saveas = 'pauli_two_design'
data = run_optimizers(optimizers, objective_fn,
                      initial_point, saveas, seed=2, runs=runs)

# plot the results
plot_data(data, saveas)
