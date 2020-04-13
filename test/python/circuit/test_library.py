# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of quantum circuits."""

from collections import defaultdict
from ddt import ddt, data, unpack
import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit import BasicAer, execute, transpile
from qiskit.circuit import (QuantumCircuit, QuantumRegister, Parameter, ParameterExpression,
                            ParameterVector)
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.random.utils import random_circuit
from qiskit.extensions.standard import XGate, RXGate, CRXGate

from qiskit.circuit.library import Permutation, XOR, InnerProduct
from qiskit.circuit.library.n_local import NLocal
from qiskit.circuit.library.arithmetic import (LinearPauliRotations, PolynomialPauliRotations,
                                               IntegerComparator, PiecewiseLinearPauliRotations,
                                               WeightedAdder)


class TestBooleanLogicLibrary(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def test_permutation(self):
        """Test permutation circuit."""
        circuit = Permutation(num_qubits=4, pattern=[1, 0, 3, 2])
        expected = QuantumCircuit(4)
        expected.swap(0, 1)
        expected.swap(2, 3)
        self.assertEqual(circuit, expected)

    def test_permutation_bad(self):
        """Test that [0,..,n-1] permutation is required (no -1 for last element)"""
        self.assertRaises(CircuitError, Permutation, 4, [1, 0, -1, 2])

    def test_xor(self):
        """Test xor circuit."""
        circuit = XOR(num_qubits=3, amount=4)
        expected = QuantumCircuit(3)
        expected.x(2)
        self.assertEqual(circuit, expected)

    def test_inner_product(self):
        """Test inner product circuit."""
        circuit = InnerProduct(num_qubits=3)
        expected = QuantumCircuit(*circuit.qregs)
        expected.cz(0, 3)
        expected.cz(1, 4)
        expected.cz(2, 5)
        self.assertEqual(circuit, expected)


@ddt
class TestFunctionalPauliRotations(QiskitTestCase):
    """Test the functional Pauli rotations."""

    def assertFunctionIsCorrect(self, function_circuit, reference):
        """Assert that ``function_circuit`` implements the reference function ``reference``."""
        num_state_qubits = function_circuit.num_state_qubits
        num_ancilla_qubits = function_circuit.num_ancilla_qubits
        circuit = QuantumCircuit(num_state_qubits + 1 + num_ancilla_qubits)
        circuit.h(list(range(num_state_qubits)))
        circuit.append(function_circuit.to_instruction(), list(range(circuit.num_qubits)))

        backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(circuit, backend).result().get_statevector()

        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[num_ancilla_qubits:]
            probabilities[i] += np.real(np.abs(statevector_amplitude) ** 2)

        unrolled_probabilities = []
        unrolled_expectations = []
        for i, probability in probabilities.items():
            x, last_qubit = int(i[1:], 2), i[0]
            if last_qubit == '0':
                expected_amplitude = np.cos(reference(x)) / np.sqrt(2**num_state_qubits)
            else:
                expected_amplitude = np.sin(reference(x)) / np.sqrt(2**num_state_qubits)

            unrolled_probabilities += [probability]
            unrolled_expectations += [np.real(np.abs(expected_amplitude) ** 2)]

        np.testing.assert_almost_equal(unrolled_probabilities, unrolled_expectations)

    @data(
        ([1, 0.1], 3),
        ([0, 0.4, 2], 2),
    )
    @unpack
    def test_polynomial_function(self, coeffs, num_state_qubits):
        """Test the polynomial rotation."""
        def poly(x):
            res = sum(coeff * x**i for i, coeff in enumerate(coeffs))
            return res

        polynome = PolynomialPauliRotations(num_state_qubits, [2 * coeff for coeff in coeffs])
        self.assertFunctionIsCorrect(polynome, poly)

    def test_polynomial_rotations_mutability(self):
        """Test the mutability of the linear rotations circuit."""

        polynomial_rotations = PolynomialPauliRotations()

        with self.subTest(msg='missing number of state qubits'):
            with self.assertRaises(AttributeError):  # no state qubits set
                print(polynomial_rotations.draw())

        with self.subTest(msg='default setup, just setting number of state qubits'):
            polynomial_rotations.num_state_qubits = 2
            self.assertFunctionIsCorrect(polynomial_rotations, lambda x: x / 2)

        with self.subTest(msg='setting non-default values'):
            polynomial_rotations.coeffs = [0, 1.2 * 2, 0.4 * 2]
            self.assertFunctionIsCorrect(polynomial_rotations, lambda x: 1.2 * x + 0.4 * x ** 2)

        with self.subTest(msg='changing of all values'):
            polynomial_rotations.num_state_qubits = 4
            polynomial_rotations.coeffs = [1 * 2, 0, 0, -0.5 * 2]
            self.assertFunctionIsCorrect(polynomial_rotations, lambda x: 1 - 0.5 * x**3)

    @data(
        (2, 0.1, 0),
        (4, -2, 2),
        (1, 0, 0)
    )
    @unpack
    def test_linear_function(self, num_state_qubits, slope, offset):
        """Test the linear rotation arithmetic circuit."""
        def linear(x):
            return offset + slope * x

        linear_rotation = LinearPauliRotations(num_state_qubits, slope * 2, offset * 2)
        self.assertFunctionIsCorrect(linear_rotation, linear)

    def test_linear_rotations_mutability(self):
        """Test the mutability of the linear rotations circuit."""

        linear_rotation = LinearPauliRotations()

        with self.subTest(msg='missing number of state qubits'):
            with self.assertRaises(AttributeError):  # no state qubits set
                print(linear_rotation.draw())

        with self.subTest(msg='default setup, just setting number of state qubits'):
            linear_rotation.num_state_qubits = 2
            self.assertFunctionIsCorrect(linear_rotation, lambda x: x / 2)

        with self.subTest(msg='setting non-default values'):
            linear_rotation.slope = -2.3 * 2
            linear_rotation.offset = 1 * 2
            self.assertFunctionIsCorrect(linear_rotation, lambda x: 1 - 2.3 * x)

        with self.subTest(msg='changing all values'):
            linear_rotation.num_state_qubits = 4
            linear_rotation.slope = 0.2 * 2
            linear_rotation.offset = 0.1 * 2
            self.assertFunctionIsCorrect(linear_rotation, lambda x: 0.1 + 0.2 * x)

    @data(
        (1, [0], [1], [0]),
        (2, [0, 2], [-0.5, 1], [2, 1]),
        (3, [0, 2, 5], [1, 0, -1], [0, 2, 2]),
        (2, [1, 2], [1, -1], [2, 1]),
        (3, [0, 1], [1, 0], [0, 1])
    )
    @unpack
    def test_piecewise_linear_function(self, num_state_qubits, breakpoints, slopes, offsets):
        """Test the piecewise linear rotations."""
        def pw_linear(x):
            for i, point in enumerate(reversed(breakpoints)):
                if x >= point:
                    return offsets[-(i + 1)] + slopes[-(i + 1)] * (x - point)
            return 0

        pw_linear_rotations = PiecewiseLinearPauliRotations(num_state_qubits, breakpoints,
                                                            [2 * slope for slope in slopes],
                                                            [2 * offset for offset in offsets])

        self.assertFunctionIsCorrect(pw_linear_rotations, pw_linear)

    def test_piecewise_linear_rotations_mutability(self):
        """Test the mutability of the linear rotations circuit."""

        pw_linear_rotations = PiecewiseLinearPauliRotations()

        with self.subTest(msg='missing number of state qubits'):
            with self.assertRaises(AttributeError):  # no state qubits set
                print(pw_linear_rotations.draw())

        with self.subTest(msg='default setup, just setting number of state qubits'):
            pw_linear_rotations.num_state_qubits = 2
            self.assertFunctionIsCorrect(pw_linear_rotations, lambda x: x / 2)

        with self.subTest(msg='setting non-default values'):
            pw_linear_rotations.breakpoints = [0, 2]
            pw_linear_rotations.slopes = [-1 * 2, 1 * 2]
            pw_linear_rotations.offsets = [0, -1.2 * 2]
            self.assertFunctionIsCorrect(pw_linear_rotations,
                                         lambda x: -1.2 + (x - 2) if x >= 2 else -x)

        with self.subTest(msg='changing all values'):
            pw_linear_rotations.num_state_qubits = 4
            pw_linear_rotations.breakpoints = [1, 3, 6]
            pw_linear_rotations.slopes = [-1 * 2, 1 * 2, -0.2 * 2]
            pw_linear_rotations.offsets = [0, -1.2 * 2, 2 * 2]

            def pw_linear(x):
                if x >= 6:
                    return 2 - 0.2 * (x - 6)
                if x >= 3:
                    return -1.2 + (x - 3)
                if x >= 1:
                    return -(x - 1)
                return 0

            self.assertFunctionIsCorrect(pw_linear_rotations, pw_linear)


@ddt
class TestIntegerComparator(QiskitTestCase):
    """Text Fixed Value Comparator"""

    def assertComparisonIsCorrect(self, comp, num_state_qubits, value, geq):
        """Assert that the comparator output is correct."""
        qc = QuantumCircuit(comp.num_qubits)  # initialize circuit
        qc.h(list(range(num_state_qubits)))  # set equal superposition state
        qc.append(comp, list(range(comp.num_qubits)))  # add comparator

        # run simulation
        backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(qc, backend).result().get_statevector()
        for i, amplitude in enumerate(statevector):
            prob = np.abs(amplitude)**2
            if prob > 1e-6:
                # equal superposition
                self.assertEqual(True, np.isclose(1.0, prob * 2.0**num_state_qubits))
                b_value = '{0:b}'.format(i).rjust(qc.width(), '0')
                x = int(b_value[(-num_state_qubits):], 2)
                comp_result = int(b_value[-num_state_qubits-1], 2)
                if geq:
                    self.assertEqual(x >= value, comp_result == 1)
                else:
                    self.assertEqual(x < value, comp_result == 1)

    @data(
        # n, value, geq
        [1, 0, True],
        [1, 1, True],
        [2, -1, True],
        [3, 5, True],
        [3, 2, True],
        [3, 2, False],
        [4, 6, False]
    )
    @unpack
    def test_fixed_value_comparator(self, num_state_qubits, value, geq):
        """Test the fixed value comparator circuit."""
        # build the circuit with the comparator
        comp = IntegerComparator(num_state_qubits, value, geq=geq)
        self.assertComparisonIsCorrect(comp, num_state_qubits, value, geq)

    def test_mutability(self):
        """Test changing the arguments of the comparator."""

        comp = IntegerComparator()

        with self.subTest(msg='missing num state qubits and value'):
            with self.assertRaises(AttributeError):
                print(comp.draw())

        comp.num_state_qubits = 2

        with self.subTest(msg='missing value'):
            with self.assertRaises(AttributeError):
                print(comp.draw())

        comp.value = 0
        comp.geq = True

        with self.subTest(msg='updating num state qubits'):
            comp.num_state_qubits = 1
            self.assertComparisonIsCorrect(comp, 1, 0, True)

        with self.subTest(msg='updating the value'):
            comp.num_state_qubits = 3
            comp.value = 2
            self.assertComparisonIsCorrect(comp, 3, 2, True)

        with self.subTest(msg='updating geq'):
            comp.geq = False
            self.assertComparisonIsCorrect(comp, 3, 2, False)


class TestAquaApplications(QiskitTestCase):
    """Test applications of the arithmetic library in Aqua use-cases."""

    def test_asian_barrier_spread(self):
        """Test the asian barrier spread model."""
        try:
            from qiskit.aqua.circuits import WeightedSumOperator, FixedValueComparator as Comparator
            from qiskit.aqua.components.uncertainty_problems import (
                UnivariatePiecewiseLinearObjective as PwlObjective,
                MultivariateProblem
            )
            from qiskit.aqua.components.uncertainty_models import MultivariateLogNormalDistribution
        except ImportError:
            import warnings
            warnings.warn('Qiskit Aqua is not installed, skipping the application test.')
            return

        # number of qubits per dimension to represent the uncertainty
        num_uncertainty_qubits = 2

        # parameters for considered random distribution
        spot_price = 2.0  # initial spot price
        volatility = 0.4  # volatility of 40%
        interest_rate = 0.05  # annual interest rate of 5%
        time_to_maturity = 40 / 365  # 40 days to maturity

        # resulting parameters for log-normal distribution
        # pylint: disable=invalid-name
        mu = ((interest_rate - 0.5 * volatility**2) * time_to_maturity + np.log(spot_price))
        sigma = volatility * np.sqrt(time_to_maturity)
        mean = np.exp(mu + sigma**2/2)
        variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        stddev = np.sqrt(variance)

        # lowest and highest value considered for the spot price; in between,
        # an equidistant discretization is considered.
        low = np.maximum(0, mean - 3*stddev)
        high = mean + 3*stddev

        # map to higher dimensional distribution
        # for simplicity assuming dimensions are independent and identically distributed)
        dimension = 2
        num_qubits = [num_uncertainty_qubits]*dimension
        low = low * np.ones(dimension)
        high = high * np.ones(dimension)
        mu = mu * np.ones(dimension)
        cov = sigma ** 2 * np.eye(dimension)

        # construct circuit factory
        distribution = MultivariateLogNormalDistribution(num_qubits=num_qubits,
                                                         low=low,
                                                         high=high,
                                                         mu=mu,
                                                         cov=cov)

        # determine number of qubits required to represent total loss
        weights = []
        for n in num_qubits:
            for i in range(n):
                weights += [2**i]

        num_sum_qubits = WeightedSumOperator.get_required_sum_qubits(weights)

        # create circuit factoy
        agg = WeightedSumOperator(sum(num_qubits), weights)

        # set the strike price (should be within the low and the high value of the uncertainty)
        strike_price_1 = 3
        strike_price_2 = 4

        # set the barrier threshold
        barrier = 2.5

        # map strike prices and barrier threshold from [low, high] to {0, ..., 2^n-1}
        max_value = 2**num_sum_qubits - 1
        low_ = low[0]
        high_ = high[0]

        mapped_strike_price_1 = (strike_price_1 - dimension*low_) / \
            (high_ - low_) * (2**num_uncertainty_qubits - 1)
        mapped_strike_price_2 = (strike_price_2 - dimension*low_) / \
            (high_ - low_) * (2**num_uncertainty_qubits - 1)
        mapped_barrier = (barrier - low) / (high - low) * (2**num_uncertainty_qubits - 1)

        conditions = []
        for i in range(dimension):
            # target dimension of random distribution and corresponding condition
            conditions += [(i, Comparator(num_qubits[i], mapped_barrier[i] + 1, geq=False))]

        # set the approximation scaling for the payoff function
        c_approx = 0.25

        # setup piecewise linear objective fcuntion
        breakpoints = [0, mapped_strike_price_1, mapped_strike_price_2]
        slopes = [0, 1, 0]
        offsets = [0, 0, mapped_strike_price_2 - mapped_strike_price_1]
        f_min = 0
        f_max = mapped_strike_price_2 - mapped_strike_price_1
        bull_spread_objective = PwlObjective(
            num_sum_qubits, 0, max_value, breakpoints, slopes, offsets, f_min, f_max, c_approx)

        # define overall multivariate problem
        asian_barrier_spread = MultivariateProblem(
            distribution, agg, bull_spread_objective, conditions=conditions)

        num_req_qubits = asian_barrier_spread.num_target_qubits
        num_req_ancillas = asian_barrier_spread.required_ancillas()

        qr = QuantumRegister(num_req_qubits, name='q')
        qr_ancilla = QuantumRegister(num_req_ancillas, name='q_a')
        qc = QuantumCircuit(qr, qr_ancilla)

        asian_barrier_spread.build(qc, qr, qr_ancilla)
        job = execute(qc, backend=BasicAer.get_backend('statevector_simulator'))

        # evaluate resulting statevector
        value = 0
        for i, amplitude in enumerate(job.result().get_statevector()):
            b = ('{0:0%sb}' % asian_barrier_spread.num_target_qubits).format(
                i)[-asian_barrier_spread.num_target_qubits:]
            prob = np.abs(amplitude)**2
            if prob > 1e-4 and b[0] == '1':
                value += prob
                # all other states should have zero probability due to ancilla qubits
                if i > 2**num_req_qubits:
                    break

        # map value to original range
        mapped_value = asian_barrier_spread.value_to_estimation(
            value) / (2**num_uncertainty_qubits - 1) * (high_ - low_)
        expected = 0.83188
        self.assertAlmostEqual(mapped_value, expected, places=5)


@ddt
class TestWeightedAdder(QiskitTestCase):
    """Test the weighted adder circuit."""

    def assertSummationIsCorrect(self, adder):
        """Assert that ``adder`` correctly implements the summation w.r.t. its set weights."""

        circuit = QuantumCircuit(adder.num_qubits)
        circuit.h(list(range(adder.num_state_qubits)))
        circuit.append(adder.to_instruction(), list(range(adder.num_qubits)))

        backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(circuit, backend).result().get_statevector()

        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[adder.num_ancilla_qubits:]
            probabilities[i] += np.real(np.abs(statevector_amplitude) ** 2)

        expectations = defaultdict(float)
        for x in range(2**adder.num_state_qubits):
            bits = np.array(list(bin(x)[2:].zfill(adder.num_state_qubits)), dtype=int)
            summation = bits.dot(adder.weights[::-1])

            entry = bin(summation)[2:].zfill(adder.num_sum_qubits) \
                + bin(x)[2:].zfill(adder.num_state_qubits)
            expectations[entry] = 1 / 2 ** adder.num_state_qubits

        for state, probability in probabilities.items():
            self.assertAlmostEqual(probability, expectations[state])

    @data(
        [0],
        [1, 2, 1],
        [4],
    )
    def test_summation(self, weights):
        """Test the weighted adder on some examples."""
        adder = WeightedAdder(len(weights), weights)
        self.assertSummationIsCorrect(adder)

    def test_mutability(self):
        """Test the mutability of the weighted adder."""
        adder = WeightedAdder()

        with self.subTest(msg='missing number of state qubits'):
            with self.assertRaises(AttributeError):
                print(adder.draw())

        with self.subTest(msg='default weights'):
            adder.num_state_qubits = 3
            default_weights = 3 * [1]
            self.assertListEqual(adder.weights, default_weights)

        with self.subTest(msg='specify weights'):
            adder.weights = [3, 2, 1]
            self.assertSummationIsCorrect(adder)

        with self.subTest(msg='mismatching number of state qubits and weights'):
            with self.assertRaises(ValueError):
                adder.weights = [0, 1, 2, 3]
                print(adder.draw())

        with self.subTest(msg='change all attributes'):
            adder.num_state_qubits = 4
            adder.weights = [2, 0, 1, 1]
            self.assertSummationIsCorrect(adder)


@ddt
class TestNLocal(QiskitTestCase):
    """Test the n-local circuit class."""

    def assertCircuitEqual(self, qc1, qc2, visual=False, verbosity=0, transpiled=True):
        """An equality test specialized to circuits."""
        basis_gates = ['id', 'u1', 'u3', 'cx']
        qc1_transpiled = transpile(qc1, basis_gates=basis_gates)
        qc2_transpiled = transpile(qc2, basis_gates=basis_gates)

        if verbosity > 0:
            print('-- circuit 1:')
            print(qc1)
            print('-- circuit 2:')
            print(qc2)
            print('-- transpiled circuit 1:')
            print(qc1_transpiled)
            print('-- transpiled circuit 2:')
            print(qc2_transpiled)

        if verbosity > 1:
            print('-- dict:')
            for key in qc1.__dict__.keys():
                if key == '_data':
                    print(key)
                    print(qc1.__dict__[key])
                    print(qc2.__dict__[key])
                else:
                    print(key, qc1.__dict__[key], qc2.__dict__[key])

        if transpiled:
            qc1, qc2 = qc1_transpiled, qc2_transpiled

        if visual:
            self.assertEqual(qc1.draw(), qc2.draw())
        else:
            self.assertEqual(qc1, qc2)

    def test_empty_nlocal(self):
        """Test the creation of an empty NLocal."""
        nlocal = NLocal()
        self.assertEqual(nlocal.num_qubits, 0)
        self.assertEqual(nlocal.num_parameters, 0)
        self.assertEqual(nlocal.reps, 1)

        self.assertEqual(nlocal.to_circuit(), QuantumCircuit())

        for attribute in [nlocal.blocks, nlocal.entangler_maps, nlocal._reps_as_list()]:
            self.assertEqual(len(attribute), 0)

    @data(
        [(XGate(), [0])],
        [(XGate(), [0]), (XGate(), [2])],
        [(RXGate(0.2), [2]), (CRXGate(-0.2), [1, 3])],
    )
    def test_append_gates_to_empty_nlocal(self, gate_data):
        """Test appending gates to an empty nlocal."""
        nlocal = NLocal()

        max_num_qubits = 0
        for (_, indices) in gate_data:
            max_num_qubits = max(max_num_qubits, max(indices))

        reference = QuantumCircuit(max_num_qubits + 1)
        for (gate, indices) in gate_data:
            nlocal.append(gate, indices)
            reference.append(gate, indices)

        self.assertCircuitEqual(nlocal.to_circuit(), reference, verbosity=1)

    @data(
        [5, 3], [1, 5], [1, 1], [1, 2, 3, 10],
    )
    def test_append_circuit(self, num_qubits):
        """Test appending circuits to an nlocal."""
        # fixed depth of 3 gates per circuit
        depth = 3

        # keep track of a reference circuit
        reference = QuantumCircuit(max(num_qubits))

        # construct the NLocal from the first circuit
        first_circuit = random_circuit(num_qubits[0], depth)
        # TODO Terra bug: if this is to_gate it fails, since the QC adds an instruction not gate
        nlocal = NLocal(blocks=first_circuit.to_instruction())
        reference.append(first_circuit, list(range(num_qubits[0])))

        # append the rest
        for num in num_qubits[1:]:
            circuit = random_circuit(num, depth)
            nlocal.append(circuit)
            reference.append(circuit, list(range(num)))

        self.assertCircuitEqual(nlocal.to_circuit(), reference)

    @data(
        [5, 3], [1, 5], [1, 1], [1, 2, 3, 10],
    )
    def test_append_nlocal(self, num_qubits):
        """Test appending an nlocal to an nlocal."""
        # fixed depth of 3 gates per circuit
        depth = 3

        # keep track of a reference circuit
        reference = QuantumCircuit(max(num_qubits))

        # construct the NLocal from the first circuit
        first_circuit = random_circuit(num_qubits[0], depth)
        # TODO Terra bug: if this is to_gate it fails, since the QC adds an instruction not gate
        nlocal = NLocal(blocks=first_circuit.to_instruction())
        reference.append(first_circuit, list(range(num_qubits[0])))

        # print(nlocal.to_circuit())
        # print(reference)

        # append the rest
        for num in num_qubits[1:]:
            circuit = random_circuit(num, depth)
            nlocal.append(NLocal(blocks=circuit))
            reference.append(circuit, list(range(num)))
            # print(nlocal.to_circuit())
            # print(reference)

        self.assertCircuitEqual(nlocal.to_circuit(), reference)

    def test_iadd_overload(self):
        """Test the overloaded + operator."""
        num_qubits, depth = 2, 2

        # construct two circuits for adding
        first_circuit = random_circuit(num_qubits, depth)
        circuit = random_circuit(num_qubits, depth)

        # get a reference
        reference = first_circuit + circuit

        # convert the object to be appended to different types
        others = [circuit, circuit.to_instruction(), circuit.to_gate(), NLocal(circuit)]

        # try adding each type
        for other in others:
            nlocal = NLocal(blocks=first_circuit)
            nlocal += other
            with self.subTest(msg='type: {}'.format(type(other))):
                self.assertCircuitEqual(nlocal.to_circuit(), reference, verbosity=0)

    def test_parameter_getter_from_automatic_repetition(self):
        """Test getting and setting of the nlocal parameters."""
        circuit = QuantumCircuit(2)
        circuit.ry(Parameter('a'), 0)
        circuit.crx(Parameter('b'), 0, 1)

        # repeat circuit and check that parameters are duplicated
        reps = 3
        nlocal = NLocal(blocks=circuit, reps=reps)
        self.assertTrue(nlocal.num_parameters, 6)
        self.assertTrue(len(nlocal.parameters), 6)

    @data(list(range(6)), ParameterVector('θ', length=6))
    def test_parameter_setter_from_automatic_repetition(self, params):
        """Test getting and setting of the nlocal parameters.

        TODO Test the input ``[0, 1, Parameter('theta'), 3, 4, 5]`` once that's supported.
        """
        circuit = QuantumCircuit(2)
        circuit.ry(Parameter('a'), 0)
        circuit.crx(Parameter('b'), 0, 1)

        # repeat circuit and check that parameters are duplicated
        reps = 3
        nlocal = NLocal(blocks=circuit, reps=reps)
        nlocal.parameters = params

        param_set = set(p for p in params if isinstance(p, ParameterExpression))
        with self.subTest(msg='Test the parameters of the non-transpiled circuit'):
            # check the parameters of the final circuit
            self.assertEqual(nlocal.to_circuit().parameters, param_set)

        with self.subTest(msg='Test the parameters of the transpiled circuit'):
            basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']
            transpiled_circuit = transpile(nlocal.to_circuit(), basis_gates=basis_gates)
            self.assertEqual(transpiled_circuit.parameters, param_set)

    @data(list(range(6)), ParameterVector('θ', length=6), [0, 1, Parameter('theta'), 3, 4, 5])
    def test_parameters_setter(self, params):
        """Test setting the parameters via list."""
        # construct circuit with some parameters
        initial_params = ParameterVector('p', length=6)
        circuit = QuantumCircuit(1)
        for i, initial_param in enumerate(initial_params):
            circuit.ry(i * initial_param, 0)

        # create an NLocal from the circuit and set the new parameters
        nlocal = NLocal(blocks=circuit)
        nlocal.parameters = params

        param_set = set(p for p in params if isinstance(p, ParameterExpression))
        with self.subTest(msg='Test the parameters of the non-transpiled circuit'):
            # check the parameters of the final circuit
            self.assertEqual(nlocal.to_circuit().parameters, param_set)

        with self.subTest(msg='Test the parameters of the transpiled circuit'):
            basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']
            transpiled_circuit = transpile(nlocal.to_circuit(), basis_gates=basis_gates)
            self.assertEqual(transpiled_circuit.parameters, param_set)

    def test_repetetive_parameter_setting(self):
        """Test alternate setting of parameters and circuit construction."""
        x = Parameter('x')
        circuit = QuantumCircuit(1)
        circuit.rx(x, 0)

        nlocal = NLocal(blocks=circuit, reps=[0, 0, 0], insert_barriers=True)
        with self.subTest(msg='immediately after initialization'):
            self.assertEqual(len(nlocal.parameters), 3)

        with self.subTest(msg='after circuit construction'):
            as_circuit = nlocal.to_circuit()
            self.assertEqual(len(nlocal.parameters), 3)

        nlocal.parameters = [0, -1, 0]
        with self.subTest(msg='setting parameter to numbers'):
            as_circuit = nlocal.to_circuit()
            self.assertEqual(nlocal.parameters, [0, -1, 0])
            self.assertEqual(as_circuit.parameters, set())

        q = Parameter('q')
        nlocal.parameters = [x, q, q]
        with self.subTest(msg='setting parameter to Parameter objects'):
            as_circuit = nlocal.to_circuit()
            self.assertEqual(nlocal.parameters, [x, q, q])
            self.assertEqual(as_circuit.parameters, set({x, q}))
