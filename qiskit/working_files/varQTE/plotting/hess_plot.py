import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

import numpy as np
import scipy as sp
import os

from qiskit import QuantumCircuit

from qiskit.circuit import ParameterExpression, ParameterVector, Parameter
from qiskit.circuit.library import EfficientSU2, RealAmplitudes

from qiskit.aqua.operators.evolutions.varqtes.varqite import VarQITE
from qiskit.aqua.operators.evolutions.varqtes.varqrte import VarQRTE
from qiskit.aqua.operators.evolutions.matrix_evolution import MatrixEvolution

from qiskit.aqua.operators import StateFn, CircuitStateFn, ListOp, ComposedOp, SummedOp
from qiskit.aqua.operators.operator_globals import  Z, I, Y, X
from qiskit.aqua.operators.gradients import NaturalGradient, CircuitQFI, CircuitGradient, \
    Gradient, QFI, Hessian
from qiskit.aqua.algorithms.minimum_eigen_solvers.qaoa.var_form import QAOAVarForm

# Evolution time
t = 1
# Instantiate the model ansatz
# entangler_map = [[i+1, i] for i in range(num_qubits - 1)]


# Define the model Hamiltonian
# TODO Pauli SummedOp
# H = SummedOp([0.3 * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, - 0.5 * I ^ Z ^ I ^ I])
# H = t * SummedOp([Y ^ X,  X ^ I])
H = t * (Y ^ Z)
num_qubits = H.num_qubits

num_time_steps = [5, 10, 20]
depths = [1, 2, 3]
parameters = [Parameter('a'), Parameter('b')]
ansatz = QuantumCircuit(2)
ansatz.ry(parameters[0], 0)
ansatz.rz(parameters[1], 1)

params = [np.pi/2, np.pi/2]


# init_params = np.zeros(len(ansatz.ordered_parameters))
# for i in range(ansatz.num_qubits):
#     init_params[-(i + 1)] = np.pi / 2
# ansatz = EfficientSU2(H.num_qubits, reps=, entanglement=e)
# init_params = np.zeros(len(ansatz.ordered_parameters))
# for i in range(ansatz.num_qubits):
#     init_params[-(ansatz.num_qubits + i + 1)] = np.pi / 2

# initial_point = np.random.random(len(ansatz.ordered_parameters))
# parameters = ansatz.ordered_parameters
# op = ~StateFn(observable) @ ansatz
op = ~StateFn(H) @ StateFn(ansatz)

hess = Hessian(hess_method='lin_comb').convert(op, params=parameters)

print(hess)