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
from qiskit import BasicAer, execute
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import Permutation, XOR, InnerProduct
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
