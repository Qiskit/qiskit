
import unittest
from test.python.opflow import QiskitOpflowTestCase
from itertools import product
import numpy as np
from ddt import ddt, data, idata, unpack

try:
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from qiskit import QuantumCircuit, QuantumRegister, BasicAer
from qiskit.test import slow_test
from qiskit.utils import QuantumInstance
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.utils import algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import CG
from qiskit.opflow import I, X, Y, Z, StateFn, CircuitStateFn, ListOp, CircuitSampler, TensoredOp
from qiskit.opflow.gradients import Gradient, NaturalGradient, Hessian
from qiskit.opflow.gradients.qfi import QFI
from qiskit.opflow.gradients.circuit_qfis import LinCombFull, OverlapBlockDiag, OverlapDiag
from qiskit.circuit import Parameter
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes


method = "lin_comb"
a = Parameter("a")
b = Parameter("b")
params = [a, b]

qc = QuantumCircuit(2)
qc.h(1)
qc.h(0)
qc.sdg(1)
qc.cz(0, 1)
qc.ry(params[0], 0)
qc.rz(params[1], 0)
qc.h(1)

qc_grad_a = QuantumCircuit(2)
qc_grad_a.h(1)
qc_grad_a.h(0)
qc_grad_a.sdg(1)
qc_grad_a.cz(0, 1)
qc_grad_a.y(0)
qc_grad_a.ry(params[0], 0)
qc_grad_a.rz(params[1], 0)
qc_grad_a.h(1)

qc_grbd_b = QuantumCircuit(2)
qc_grbd_b.h(1)
qc_grbd_b.h(0)
qc_grbd_b.sdg(1)
qc_grbd_b.cz(0, 1)
qc_grbd_b.ry(params[0], 0)
qc_grbd_b.z(0)
qc_grbd_b.rz(params[1], 0)
qc_grbd_b.h(1)

obs = (Z ^ X) - (Y ^ Y)
op = obs.to_matrix_op().primitive.data
# op = StateFn(obs, is_measurement=True) @ CircuitStateFn(primitive=qc)

state = StateFn(qc).assign_parameters(dict(zip(params, [0, np.pi / 2]))).eval().primitive.data
grad_a = StateFn(qc_grad_a).assign_parameters(dict(zip(params, [0, np.pi / 2]))).eval().primitive.data
grad_b = StateFn(qc_grbd_b).assign_parameters(dict(zip(params, [0, np.pi / 2]))).eval().primitive.data

shots = 10000

values = [[0, np.pi / 2], [np.pi / 4, np.pi / 4], [np.pi / 3, np.pi / 9]]
correct_values = [[-2.0, 0], [-1.0, -2.41421356], [-0.34202014, -3.5069806]]