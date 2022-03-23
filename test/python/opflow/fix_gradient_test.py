import unittest
from test.python.opflow import QiskitOpflowTestCase
from itertools import product
import numpy as np
from ddt import ddt, data, idata, unpack

from qiskit import QuantumCircuit, QuantumRegister, BasicAer
from qiskit.test import slow_test
from qiskit.utils import QuantumInstance
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.utils import algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import CG
from qiskit.opflow import (
    I,
    X,
    Y,
    Z,
    StateFn,
    CircuitStateFn,
    ListOp,
    CircuitSampler,
    TensoredOp,
    SummedOp,
    ComposedOp,
)
from qiskit.opflow.gradients import Gradient, NaturalGradient, Hessian
from qiskit.opflow.gradients.circuit_gradients import LinComb
from qiskit.opflow.gradients.qfi import QFI
from qiskit.opflow.gradients.circuit_qfis import LinCombFull, OverlapBlockDiag, OverlapDiag
from qiskit.circuit import Parameter
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.utils import optionals

if optionals.HAS_JAX:
    import jax.numpy as jnp


def test_gradient_p_imag():
    """Test the state gradient for p
    |psi(a)> = 1/sqrt(2)[[1, exp(ia)]]
    <da psi(a)|X|psi(a)> = -iexp(-ia)/2 <1|H(|0>+exp(ia)|1>)
    Im(<psi(a)|X|da psi(a)>) = 0.5 cos(a)
    """
    ham = X
    a = Parameter("a")
    params = a
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.h(q)
    qc.p(a, q[0])
    op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

    state_grad = LinComb().convert(operator=op, params=params, aux_meas_op=(-1) * Y)
    values_dict = [{a: np.pi / 4}, {a: 0}, {a: np.pi / 2}]
    correct_values = [1 / np.sqrt(2), 1, 0]

    for i, value_dict in enumerate(values_dict):
        print(state_grad.assign_parameters(value_dict).eval())
        print(correct_values[i])
        np.testing.assert_array_almost_equal(
            state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
        )


def test_qfi_p_imag():
    """Test the state QFI for p
    |psi(a)> = 1/sqrt(2)[[1, exp(ia)]]
    <da psi(a)|X|psi(a)> = -iexp(-ia)/2 <1|H(|0>+exp(ia)|1>)
    Im(<psi(a)|X|da psi(a)>) = 0.5 cos(a)
    """
    x = Parameter("x")
    y = Parameter("y")
    circuit = QuantumCircuit(1)
    circuit.ry(y, 0)
    circuit.rx(x, 0)
    state = StateFn(circuit)

    dx = (
        lambda x, y: (-1)
        * 0.5j
        * np.array(
            [
                [
                    -1j * np.sin(x / 2) * np.cos(y / 2) + np.cos(x / 2) * np.sin(y / 2),
                    np.cos(x / 2) * np.cos(y / 2) - 1j * np.sin(x / 2) * np.sin(y / 2),
                ]
            ]
        )
    )

    dy = (
        lambda x, y: (-1)
        * 0.5j
        * np.array(
            [
                [
                    -1j * np.cos(x / 2) * np.sin(y / 2) + np.sin(x / 2) * np.cos(y / 2),
                    1j * np.cos(x / 2) * np.cos(y / 2) - 1 * np.sin(x / 2) * np.sin(y / 2),
                ]
            ]
        )
    )

    state_grad = LinCombFull().convert(
        operator=state, params=[x, y], aux_meas_op=-1 * Y, phase_fix=False
    )
    values_dict = [{x: 0, y: np.pi / 4}, {x: 0, y: np.pi / 2}, {x: np.pi / 2, y: 0}]

    for i, value_dict in enumerate(values_dict):
        print(state_grad.assign_parameters(value_dict).eval())
        x_ = list(value_dict.values())[0]
        y_ = list(value_dict.values())[1]
        correct_values = [
            [
                4 * np.imag(np.dot(dx(x_, y_), np.conj(np.transpose(dx(x_, y_))))[0][0]),
                4 * np.imag(np.dot(dy(x_, y_), np.conj(np.transpose(dx(x_, y_))))[0][0]),
            ],
            [
                4 * np.imag(np.dot(dy(x_, y_), np.conj(np.transpose(dx(x_, y_))))[0][0]),
                4 * np.imag(np.dot(dy(x_, y_), np.conj(np.transpose(dy(x_, y_))))[0][0]),
            ],
        ]

        np.testing.assert_array_almost_equal(
            state_grad.assign_parameters(value_dict).eval(), correct_values, decimal=3
        )


def test_unittest():
    """Test the probability Hessian using linear combination of unitaries method

    d^2p0/da^2 = - sin(a)sin(b) / 2
    d^2p1/da^2 =  sin(a)sin(b) / 2
    d^2p0/dadb = cos(a)cos(b) / 2
    d^2p1/dadb = - cos(a)cos(b) / 2
    """

    a = Parameter("a")
    b = Parameter("b")
    params = [(a, a), (a, b)]

    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.h(q)
    qc.rz(a, q[0])
    qc.rx(b, q[0])

    op = CircuitStateFn(primitive=qc, coeff=1.0)

    prob_hess = Hessian(hess_method="lin_comb").convert(operator=op, params=params)
    values_dict = [{a: np.pi / 4, b: 0}, {a: np.pi / 4, b: np.pi / 4}, {a: np.pi / 2, b: np.pi}]
    correct_values = [
        [[0, 0], [1 / (2 * np.sqrt(2)), -1 / (2 * np.sqrt(2))]],
        [[-1 / 4, 1 / 4], [1 / 4, -1 / 4]],
        [[0, 0], [0, 0]],
    ]
    for i, value_dict in enumerate(values_dict):
        for j, prob_hess_result in enumerate(prob_hess.assign_parameters(value_dict).eval()):
            np.testing.assert_array_almost_equal(prob_hess_result, correct_values[i][j], decimal=1)


# test_qfi_p_imag()
test_unittest()

state = (
    lambda a, b: 1
    / np.sqrt(2)
    * np.array(
        [
            [
                np.exp(-0.5j * a) * np.cos(b / 2) - 1j * np.exp(0.5j * a) * np.sin(b / 2),
                -1j * np.exp(-0.5j * a) * np.sin(b / 2) + np.exp(0.5j * a) * np.cos(b / 2),
            ]
        ]
    )
)


da_state = (
    lambda a, b: -0.5j
    / np.sqrt(2)
    * np.array(
        [
            [
                np.exp(-0.5j * a) * np.cos(b / 2) + 1j * np.exp(0.5j * a) * np.sin(b / 2),
                -1j * np.exp(-0.5j * a) * np.sin(b / 2) - np.exp(0.5j * a) * np.cos(b / 2),
            ]
        ]
    )
)

db_state = (
    lambda a, b: -0.5j
    / np.sqrt(2)
    * np.array(
        [
            [
                -1j * np.exp(-0.5j * a) * np.sin(b / 2) + np.exp(0.5j * a) * np.cos(b / 2),
                np.exp(-0.5j * a) * np.cos(b / 2) - 1j * np.exp(0.5j * a) * np.sin(b / 2),
            ]
        ]
    )
)


# a: [np.pi / 4], b: [0]}

da0 = lambda a, b: np.dot(
    np.conj(state(a, b)), np.dot([[1, 0], [0, 0]], np.transpose(da_state(a, b)))
)
da1 = lambda a, b: np.dot(
    np.conj(state(a, b)), np.dot([[0, 0], [0, 1]], np.transpose(da_state(a, b)))
)

db0 = lambda a, b: np.dot(
    np.conj(state(a, b)), np.dot([[1, 0], [0, 0]], np.transpose(db_state(a, b)))
)
db1 = lambda a, b: np.dot(
    np.conj(state(a, b)), np.dot([[0, 0], [0, 1]], np.transpose(db_state(a, b)))
)

a_ = np.pi / 4
b_ = 0

print(state(a_, b_))

print("Imag da0 ", 2 * np.imag(da0(a_, b_)))
print("Imag da1 ", 2 * np.imag(da1(a_, b_)))
print("Imag db0 ", 2 * np.imag(db0(a_, b_)))
print("Imag db1 ", 2 * np.imag(db1(a_, b_)))
#
print("da0 ", 2 * da0(a_, b_))
print("da1 ", 2 * da1(a_, b_))
print("db0 ", 2 * db0(a_, b_))
print("db1 ", 2 * db1(a_, b_))
#
# Re(<state|O|dstate>) =0.5(<state|O|dstate> + <dstate|O|state>)
# O = |0><0|

# print('Real da0 ', 2*np.real(da0(a_, b_)))
# print('REal da1 ', 2*np.real(da1(a_, b_)))
# print('REal db0 ', 2*np.real(db0(a_, b_)))
# print('Real db1 ', 2*np.real(db1(a_, b_)))

# print(np.dot([[1, 0], [0, 0]], np.transpose(da_state(a_,b_))))
