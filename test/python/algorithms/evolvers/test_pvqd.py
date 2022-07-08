# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for PVQD."""

from functools import partial
from ddt import ddt, data, unpack
import numpy as np

from qiskit.test import QiskitTestCase

from qiskit import BasicAer
from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.algorithms.evolvers import EvolutionProblem
from qiskit.algorithms.evolvers.pvqd import PVQD
from qiskit.algorithms.optimizers import L_BFGS_B, GradientDescent, SPSA, OptimizerResult
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Z, I, MatrixExpectation


# pylint: disable=unused-argument, invalid-name
def gradient_supplied(fun, x0, jac, info):
    """A mock optimizer that checks whether the gradient is supported or not."""
    result = OptimizerResult()
    result.x = x0
    result.fun = 0
    info["has_gradient"] = jac is not None

    return result


class WhatAmI(Gate):
    """An custom opaque gate that can be inverted but not decomposed."""

    def __init__(self, angle):
        super().__init__(name="whatami", num_qubits=2, params=[angle])

    def inverse(self):
        return WhatAmI(-self.params[0])


@ddt
class TestPVQD(QiskitTestCase):
    """Tests for the pVQD algorithm."""

    def setUp(self):
        super().setUp()
        self.sv_backend = BasicAer.get_backend("statevector_simulator")
        self.qasm_backend = BasicAer.get_backend("qasm_simulator")
        self.expectation = MatrixExpectation()
        self.hamiltonian = 0.1 * (Z ^ Z) + (I ^ X) + (X ^ I)
        self.observable = Z ^ Z
        self.ansatz = EfficientSU2(2, reps=1)
        self.initial_parameters = np.zeros(self.ansatz.num_parameters)

    @data((True, "sv"), (False, "qasm"))
    @unpack
    def test_pvqd(self, gradient, backend_type):
        """Test a simple evolution."""
        time = 0.02

        if gradient:
            optimizer = GradientDescent(maxiter=1)
        else:
            optimizer = L_BFGS_B(maxiter=1)

        backend = self.sv_backend if backend_type == "sv" else self.qasm_backend

        # run pVQD keeping track of the energy and the magnetization
        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            timestep=0.01,
            optimizer=optimizer,
            quantum_instance=backend,
            expectation=self.expectation,
        )
        problem = EvolutionProblem(
            self.hamiltonian, time, aux_operators=[self.hamiltonian, self.observable]
        )
        result = pvqd.evolve(problem)

        self.assertTrue(len(result.fidelities) == 3)
        self.assertTrue(np.all(result.times == [0.0, 0.01, 0.02]))
        self.assertTrue(np.asarray(result.observables).shape == (3, 2))
        num_parameters = self.ansatz.num_parameters
        self.assertTrue(
            len(result.parameters) == 3
            and np.all([len(params) == num_parameters for params in result.parameters])
        )

    def test_invalid_timestep(self):
        """Test raises if the timestep is larger than the evolution time."""
        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            timestep=1,
            optimizer=L_BFGS_B(),
            quantum_instance=self.sv_backend,
            expectation=self.expectation,
        )
        problem = EvolutionProblem(
            self.hamiltonian, time=0.01, aux_operators=[self.hamiltonian, self.observable]
        )

        with self.assertRaises(ValueError):
            _ = pvqd.evolve(problem)

    def test_initial_guess_and_observables(self):
        """Test doing no optimizations stays at initial guess."""
        initial_guess = np.zeros(self.ansatz.num_parameters)

        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            timestep=0.01,
            optimizer=SPSA(maxiter=0, learning_rate=0.1, perturbation=0.01),
            initial_guess=initial_guess,
            quantum_instance=self.sv_backend,
            expectation=self.expectation,
        )
        problem = EvolutionProblem(
            self.hamiltonian, time=0.1, aux_operators=[self.hamiltonian, self.observable]
        )

        result = pvqd.evolve(problem)

        observables = result.aux_ops_evaluated
        self.assertEqual(observables[0], 0.1)  # expected energy
        self.assertEqual(observables[1], 1)  # expected magnetization

    def test_gradient_supported(self):
        """Test the gradient support is correctly determined."""
        # gradient supported here
        empty = QuantumCircuit(2)
        wrapped = EfficientSU2(2)  # a circuit wrapped into a big instruction
        plain = wrapped.decompose()  # a plain circuit with already supported instructions

        # gradients not supported on the following circuits
        x = Parameter("x")
        duplicated = QuantumCircuit(2)
        duplicated.rx(x, 0)
        duplicated.rx(x, 1)

        needs_chainrule = QuantumCircuit(2)
        needs_chainrule.rx(2 * x, 0)

        custom_gate = WhatAmI(x)
        unsupported = QuantumCircuit(2)
        unsupported.append(custom_gate, [0, 1])

        tests = [
            (empty, True),  # tuple: (circuit, gradient support)
            (wrapped, True),
            (plain, True),
            (duplicated, False),
            (needs_chainrule, False),
            (unsupported, False),
        ]

        # used to store the info if a gradient callable is passed into the
        # optimizer of not
        info = {"has_gradient": None}
        optimizer = partial(gradient_supplied, info=info)

        pvqd = PVQD(
            ansatz=empty,
            initial_parameters=np.array([]),
            timestep=0.01,
            optimizer=optimizer,
            quantum_instance=self.sv_backend,
            expectation=self.expectation,
        )
        problem = EvolutionProblem(self.hamiltonian, time=0.01)
        for circuit, expected_support in tests:
            with self.subTest(circuit=circuit, expected_support=expected_support):
                pvqd.ansatz = circuit
                pvqd.initial_parameters = np.zeros(circuit.num_parameters)
                _ = pvqd.evolve(problem)
                self.assertEqual(info["has_gradient"], expected_support)

    def test_invalid_setup(self):
        """Test appropriate error is raised if attributes are missing or incompatible."""
        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            timestep=0.01,
            optimizer=L_BFGS_B(maxiter=1),
            quantum_instance=self.qasm_backend,
            expectation=self.expectation,
        )
        problem = EvolutionProblem(self.hamiltonian, time=0.01)

        args_to_test = [
            ("ansatz", self.ansatz),
            ("initial_parameters", self.initial_parameters),
            ("timestep", 0.01),
            ("optimizer", L_BFGS_B(maxiter=1)),
            ("quantum_instance", self.qasm_backend),
            ("expectation", self.expectation),
        ]

        for attr, value in args_to_test:
            with self.subTest(msg=f"missing: {attr}"):
                # set attribute to None to invalidate the setup
                setattr(pvqd, attr, None)

                with self.assertRaises(ValueError):
                    _ = pvqd.evolve(problem)

                # set the correct value again
                setattr(pvqd, attr, value)

        # check PVQD is running now that all arguments are set
        result = pvqd.evolve(problem)
        self.assertIsNotNone(result.evolved_state)
