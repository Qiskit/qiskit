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

"""Faster Amplitude Estimation."""

from typing import Optional, Union, List, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance
from qiskit.algorithms.exceptions import AlgorithmError

from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .estimation_problem import EstimationProblem


class FasterAmplitudeEstimation(AmplitudeEstimator):
    """The Faster Amplitude Estimation algorithm.

    The Faster Amplitude Estimation (FAE) [1] algorithm is a variant of Quantum Amplitude
    Estimation (QAE), where the Quantum Phase Estimation (QPE) by an iterative Grover search,
    similar to [2].

    Due to the iterative version of the QPE, this algorithm does not require any additional
    qubits, as the originally proposed QAE [3] and thus the resulting circuits are less complex.

    References:

        [1]: K. Nakaji. Faster Amplitude Estimation, 2020;
            `arXiv:2002.02417 <https://arxiv.org/pdf/2003.02417.pdf>`_
        [2]: D. Grinko et al. Iterative Amplitude Estimation, 2019;
            `arXiv:1912.05559 <http://arxiv.org/abs/1912.05559>`_
        [3]: G. Brassard et al. Quantum Amplitude Amplification and Estimation, 2000;
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_

    """

    def __init__(self,
                 delta: float,
                 maxiter: Optional[int] = None,
                 rescale: bool = True,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        r"""
        Args:
            delta: The probability that the true value is outside of the final confidence interval.
            maxiter: The number of iterations, the maximal power of Q is `2 ** (maxiter - 1)`.
            rescale: Whether to rescale the problem passed to `estimate`.
            quantum_instance: The quantum instance or backend to run the circuits.
        """
        super().__init__(quantum_instance)
        self._shots = (int(1944 * np.log(2 / delta)), int(972 * np.log(2 / delta)))
        self._rescale = rescale
        self._delta = delta
        self._maxiter = maxiter
        self._num_oracle_calls = 0

    def _cos_estimate(self, estimation_problem, k, shots):
        if self._quantum_instance is None:
            raise AlgorithmError('Quantum instance must be set.')

        if self._quantum_instance.is_statevector:
            circuit = self.construct_circuit(estimation_problem, k, measurement=False)
            statevector = self._quantum_instance.execute(circuit).get_statevector()

            # sum over all amplitudes where the objective qubits are 1
            prob = 0
            for i, amplitude in enumerate(statevector):
                # get bitstring of objective qubits
                full_state = bin(i)[2:].zfill(circuit.num_qubits)[::-1]
                state = ''.join([full_state[i] for i in estimation_problem.objective_qubits])

                # check if it is a good state
                if estimation_problem.is_good_state(state):
                    prob = prob + np.abs(amplitude) ** 2

            cos_estimate = 1 - 2 * prob
        else:
            circuit = self.construct_circuit(estimation_problem, k, measurement=True)

            self._quantum_instance.run_config.shots = shots
            counts = self._quantum_instance.execute(circuit).get_counts()
            self._num_oracle_calls += (2 * k + 1) * shots

            # TODO add good state handling
            good_counts = 0
            for state, count in counts.items():
                if estimation_problem.is_good_state(state):
                    good_counts += count

            cos_estimate = 1 - 2 * good_counts / shots

        return cos_estimate

    def _chernoff(self, cos, shots):
        width = np.sqrt(np.log(2 / self._delta) * 12 / shots)
        confint = [np.maximum(-1, cos - width), np.minimum(1, cos + width)]
        return confint

    def construct_circuit(self, estimation_problem: EstimationProblem, k: int,
                          measurement: bool = False
                          ) -> Union[QuantumCircuit, Tuple[QuantumCircuit, List[int]]]:
        r"""Construct the circuit :math:`Q^k X |0\rangle>`.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            estimation_problem: The estimation problem for which to construct the circuit.
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit :math:`Q^k X |0\rangle`.
        """
        num_qubits = max(estimation_problem.state_preparation.num_qubits,
                         estimation_problem.grover_operator.num_qubits)
        circuit = QuantumCircuit(num_qubits, name='circuit')

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            circuit.add_register(c)

        # add A operator
        circuit.compose(estimation_problem.state_preparation, inplace=True)

        # add Q^k
        if k != 0:
            circuit.compose(estimation_problem.grover_operator.power(k), inplace=True)

            # add optional measurement
        if measurement:
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, c[:])

        return circuit

    def estimate(self, estimation_problem: EstimationProblem) -> 'FasterAmplitudeEstimationResult':
        self._num_oracle_calls = 0

        if self._rescale:
            a_op = estimation_problem.state_preparation
            q_op = estimation_problem.grover_operator

            if not all(hasattr(q_op, attr) for attr in ['oracle', 'reflection_qubits']):
                raise ValueError('Can only automatically rescale the Grover operator if it '
                                 'provides the ``oracle`` and ``reflection_qubits`` attributes, '
                                 'like the qiskit.circuit.library.GroverOperator object. '
                                 'Ensure the amplitude is in [0, 0.25] and run the algorithm with '
                                 '``rescale=False`` if you want to use a custom grover operator.')

            # rescale the estimation problem
            # rescale the amplitude by a factor of 1/4 by adding an auxiliary qubit
            a_op = rescale_amplitudes(a_op, 0.25)

            # additionally control the oracle on the scaling qubit
            oracle = QuantumCircuit(*a_op.qregs)
            oracle.compose(q_op.oracle.control(ctrl_state='0'),
                           [oracle.qubits[-1]] + oracle.qubits[:-1],
                           inplace=True)

            # add the scaling qubit to the reflection qubits
            reflection_qubits = q_op.reflection_qubits + [a_op.num_qubits - 1]
            q_op = GroverOperator(oracle, a_op, reflection_qubits=reflection_qubits,
                                  insert_barriers=True)

            # add the scaling qubit to the good state qualifier
            def is_good_state(bitstr):
                return estimation_problem.is_good_state(bitstr[:-1]) and bitstr[-1] == '1'

            print(q_op.draw())
            print(q_op.decompose().draw())

            # create the rescaled estimation problem
            problem = EstimationProblem(
                a_op,
                q_op,
                estimation_problem.objective_qubits + [a_op.num_qubits - 1],
                estimation_problem.post_processing,  # post processing remains the same
                is_good_state
            )
        else:
            problem = estimation_problem

        if self._quantum_instance.is_statevector:
            cos = self._cos_estimate(problem, k=0, shots=1)
            theta = np.arccos(cos) / 2
            theta_ci = [theta, theta]
            theta_cis = [theta_ci]
            num_steps = num_first_stage_steps = 1

        else:
            theta_ci = [0, np.arcsin(0.25)]
            first_stage = True
            j_0 = self._maxiter

            theta_cis = [theta_ci]
            num_first_stage_steps = 0
            num_steps = 0

            def cos_estimate(power, shots):
                return self._cos_estimate(problem, power, shots)

            for j in range(1, self._maxiter + 1):
                num_steps += 1
                if first_stage:
                    num_first_stage_steps += 1
                    c = cos_estimate(2**(j - 1), self._shots[0])
                    chernoff_ci = self._chernoff(c, self._shots[0])
                    theta_ci = [np.arccos(x) / (2 ** (j + 1) + 2) for x in chernoff_ci[::-1]]

                    if 2 ** (j + 1) * theta_ci[1] >= 3 * np.pi / 8 and j < self._maxiter:
                        j_0 = j
                        v = 2 ** j * np.sum(theta_ci)
                        first_stage = False
                else:
                    cos = cos_estimate(2**(j - 1), self._shots[1])
                    cos_2 = cos_estimate(2 ** (j - 1) + 2 ** (j_0 - 1), self._shots[1])
                    sin = (cos * np.cos(v) - cos_2) / np.sin(v)
                    rho = np.arctan2(sin, cos)
                    n = int(((2 ** (j + 1) + 2) * theta_ci[1] - rho + np.pi / 3) / (2 * np.pi))

                    theta_ci = [(2 * np.pi * n + rho + sign * np.pi / 3) / (2 ** (j + 1) + 2)
                                for sign in [-1, 1]]
                theta_cis.append(theta_ci)

        theta = np.mean(theta_ci)
        rescaling = 4 if self._rescale else 1
        value = (rescaling * np.sin(theta)) ** 2
        value_ci = [(rescaling * np.sin(x)) ** 2 for x in theta_ci]

        result = FasterAmplitudeEstimationResult()
        result.num_oracle_queries = self._num_oracle_calls
        result.num_steps = num_steps
        result.num_first_state_steps = num_first_stage_steps
        if self._quantum_instance.is_statevector:
            result.success_probability = 1
        else:
            result.success_probability = 1 - (2 * self._maxiter - j_0) * self._delta

        result.estimation = value
        result.estimation_processed = problem.post_processing(value)
        result.confidence_interval = value_ci
        result.confidence_interval_processed = tuple(problem.post_processing(x)
                                                     for x in value_ci)
        result.theta_intervals = theta_cis

        return result


def rescale_amplitudes(circuit: QuantumCircuit, scaling_factor: float) -> QuantumCircuit:
    r"""Uses an auxiliary qubit to scale the amplitude of :math:`|1\rangle` by ``scaling_factor``.

    Explained in Section 2.1. of [1].

    For example, for a scaling factor of 0.25 this turns this circuit

    .. code-block::

                      ┌───┐
        state_0: ─────┤ H ├─────────■────
                  ┌───┴───┴───┐ ┌───┴───┐
          obj_0: ─┤ RY(0.125) ├─┤ RY(1) ├
                  └───────────┘ └───────┘

    into

    .. code-block::

                      ┌───┐
        state_0: ─────┤ H ├─────────■────
                  ┌───┴───┴───┐ ┌───┴───┐
          obj_0: ─┤ RY(0.125) ├─┤ RY(1) ├
                 ┌┴───────────┴┐└───────┘
      scaling_0: ┤ RY(0.50536) ├─────────
                 └─────────────┘

    References:

        [1]: K. Nakaji. Faster Amplitude Estimation, 2020;
            `arXiv:2002.02417 <https://arxiv.org/pdf/2003.02417.pdf>`_

    Args:
        circuit: The circuit whose amplitudes to rescale.
        scaling_factor: The rescaling factor.

    Returns:
        A copy of the circuit with an additional qubit and RY gate for the rescaling.
    """
    qr = QuantumRegister(1, 'scaling')
    rescaled = QuantumCircuit(*circuit.qregs, qr)
    rescaled.compose(circuit, circuit.qubits, inplace=True)
    rescaled.ry(2 * np.arcsin(scaling_factor), qr)
    return rescaled


class FasterAmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The result object for the Faster Amplitude Estimation algorithm."""

    @property
    def success_probability(self) -> int:
        """Return the success probability of the algorithm."""
        return self.get('success_probability')

    @success_probability.setter
    def success_probability(self, probability: int) -> None:
        """Set the success probability of the algorithm."""
        self.data['success_probability'] = probability

    @property
    def num_steps(self) -> int:
        """Return the total number of steps taken in the algorithm."""
        return self.get('num_steps')

    @num_steps.setter
    def num_steps(self, num_steps: int) -> None:
        """Set the total number of steps taken in the algorithm."""
        self.data['num_steps'] = num_steps

    @property
    def num_first_state_steps(self) -> int:
        """Return the number of steps taken in the first step of algorithm."""
        return self.get('num_first_state_steps')

    @num_first_state_steps.setter
    def num_first_state_steps(self, num_steps: int) -> None:
        """Set the number of steps taken in the first step of algorithm."""
        self.data['num_first_state_steps'] = num_steps

    @property
    def theta_intervals(self) -> List[List[float]]:
        """Return the confidence intervals for the angles in each iteration."""
        return self.get('theta_intervals')

    @theta_intervals.setter
    def theta_intervals(self, value: List[List[float]]) -> None:
        """Set the confidence intervals for the angles in each iteration."""
        self.data['theta_intervals'] = value
