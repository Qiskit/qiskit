import unittest
from test.python.opflow import QiskitOpflowTestCase
from itertools import product
import numpy as np
from ddt import ddt, data, idata, unpack
from sympy import Symbol, cos

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
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes

method = "lin_comb"
a = Parameter("a")
b = Parameter("b")
params = [a, b]
backends = ["statevector_simulator", "qasm_simulator"]

qc = QuantumCircuit(2)
qc.h(1)
qc.h(0)
qc.sdg(1)
qc.cz(0, 1)
qc.ry(params[0], 0)
qc.rz(params[1], 0)
qc.h(1)

obs = (Z ^ X) - (Y ^ Y)
op = ~StateFn(obs) @ CircuitStateFn(primitive=qc, coeff=1.0)

shots = 100000

values = [[0, np.pi / 2], [np.pi / 4, np.pi / 4], [np.pi / 3, np.pi / 9]]
for value in values:
    results = []
    for backend_type in backends:
        backend = BasicAer.get_backend(backend_type)

        q_instance = QuantumInstance(backend=backend, shots=shots)

        grad = NaturalGradient(grad_method=method).gradient_wrapper(
            operator=op, bind_params=params, backend=q_instance
        )

        result = grad(value)
        results.append(result)
    print(results)
    grad_eval = NaturalGradient(grad_method=method).gradient_wrapper(
        operator=op, bind_params=params, backend=None
    )
    print("eval grad", grad_eval(value))
