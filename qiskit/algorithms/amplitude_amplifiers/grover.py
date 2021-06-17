# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Grover's search algorithm."""

import itertools
import operator
from typing import Iterator, List, Optional, Union

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.providers import Backend, BaseBackend
from qiskit.quantum_info import partial_trace
from qiskit.utils import QuantumInstance
from .amplification_problem import AmplificationProblem
from .amplitude_amplifier import AmplitudeAmplifier, AmplitudeAmplifierResult


class Grover(AmplitudeAmplifier):
    r"""Grover's Search algorithm.

    Grover's Search [1, 2] is a well known quantum algorithm for that can be used for
    searching through unstructured collections of records for particular targets
    with quadratic speedup compared to classical algorithms.

    Given a set :math:`X` of :math:`N` elements :math:`X=\{x_1,x_2,\ldots,x_N\}`
    and a boolean function :math:`f : X \rightarrow \{0,1\}`, the goal of an
    unstructured-search problem is to find an element :math:`x^* \in X` such
    that :math:`f(x^*)=1`.

    The search is called *unstructured* because there are no guarantees as to how
    the database is ordered.  On a sorted database, for instance, one could perform
    binary search to find an element in :math:`\mathbb{O}(\log N)` worst-case time.
    Instead, in an unstructured-search problem, there is no prior knowledge about
    the contents of the database. With classical circuits, there is no alternative
    but to perform a linear number of queries to find the target element.
    Conversely, Grover's Search algorithm allows to solve the unstructured-search
    problem on a quantum computer in :math:`\mathcal{O}(\sqrt{N})` queries.

    To carry out this search a so-called oracle is required, that flags a good element/state.
    The action of the oracle :math:`\mathcal{S}_f` is

    .. math::

        \mathcal{S}_f |x\rangle = (-1)^{f(x)} |x\rangle,

    i.e. it flips the phase of the state :math:`|x\rangle` if :math:`x` is a hit.
    The details of how :math:`S_f` works are unimportant to the algorithm; Grover's
    search algorithm treats the oracle as a black box.

    This class supports oracles in form of :class:`~qiskit.QuantumCircuit`

    With oracle at hand, Grover's Search constructs the Grover operator to amplify the amplitudes
    of the good states:

    .. math::

        \mathcal{Q} = H^{\otimes n} \mathcal{S}_0 H^{\otimes n} \mathcal{S}_f
                    = D \mathcal{S}_f,

    where :math:`\mathcal{S}_0` flips the phase of the all-zero state and acts as identity
    on all other states. Sometimes the first three operands are summarized as diffusion operator,
    which implements a reflection over the equal superposition state.

    If the number of solutions is known, we can calculate how often :math:`\mathcal{Q}` should be
    applied to find a solution with very high probability, see the method
    `optimal_num_iterations`. If the number of solutions is unknown, the algorithm tries different
    powers of Grover's operator, see the `iterations` argument, and after each iteration checks
    if a good state has been measured using `good_state`.

    The generalization of Grover's Search, Quantum Amplitude Amplification [3] uses a modified
    version of :math:`\mathcal{Q}` where the diffusion operator does not reflect about the
    equal superposition state, but another state specified via an operator :math:`\mathcal{A}`:

    .. math::

        \mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A}^\dagger \mathcal{S}_f.

    For more information, see the :class:`~qiskit.circuit.library.GroverOperator` in the
    circuit library.

    References:
        [1]: L. K. Grover (1996), A fast quantum mechanical algorithm for database search,
            `arXiv:quant-ph/9605043 <https://arxiv.org/abs/quant-ph/9605043>`_.
        [2]: I. Chuang & M. Nielsen, Quantum Computation and Quantum Information,
            Cambridge: Cambridge University Press, 2000. Chapter 6.1.2.
        [3]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
            Quantum Amplitude Amplification and Estimation.
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.

    """

    def __init__(
        self,
        iterations: Optional[Union[List[int], Iterator[int], int]] = None,
        growth_rate: Optional[float] = None,
        sample_from_iterations: bool = False,
        quantum_instance: Optional[Union[QuantumInstance, Backend, BaseBackend]] = None,
    ) -> None:
        r"""
        Args:
            iterations: Specify the number of iterations/power of Grover's operator to be checked.
                * If an int, only one circuit is run with that power of the Grover operator.
                If the number of solutions is known, this is option should be used with the optimal
                power. The optimal power can be computed with ``Grover.optimal_num_iterations``.
                * If a list, all the powers in the list are run in the specified order.
                * If an iterator, the powers yielded by the iterator are checked, until a maximum
                number of iterations or maximum power is reached.
            growth_rate: If specified, the iterator is set to increasing powers of ``growth_rate``,
                i.e. to ``int(growth_rate ** 1), int(growth_rate ** 2), ...`` until a maximum
                number of iterations is reached.
            sample_from_iterations: If True, instead of taking the values in ``iterations`` as
                powers of the Grover operator, a random integer sample between 0 and smaller value
                than the iteration is used as a power, see [1], Section 4.
            quantum_instance: A Quantum Instance or Backend to run the circuits.

        Raises:
            ValueError: If ``growth_rate`` is a float but not larger than 1.
            ValueError: If both ``iterations`` and ``growth_rate`` is set.

        References:
            [1]: Boyer et al., Tight bounds on quantum searching
                 `<https://arxiv.org/abs/quant-ph/9605034>`_
        """
        # set default value
        if growth_rate is None and iterations is None:
            growth_rate = 1.2

        if growth_rate is not None and iterations is not None:
            raise ValueError("Pass either a value for iterations or growth_rate, not both.")

        if growth_rate is not None:
            # yield iterations ** 1, iterations ** 2, etc. and casts to int
            self._iterations = map(lambda x: int(growth_rate ** x), itertools.count(1))
        elif isinstance(iterations, int):
            self._iterations = [iterations]
        else:
            self._iterations = iterations

        self._quantum_instance = None
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

        self._sample_from_iterations = sample_from_iterations

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Get the quantum instance.
        Returns:
            The quantum instance used to run this algorithm.
        """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
    ) -> None:
        """Set quantum instance.
        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    def amplify(self, amplification_problem: AmplificationProblem) -> "GroverResult":
        """Run the Grover algorithm.

        Args:
            amplification_problem: The amplification problem.

        Returns:
            The result as a ``GroverResult``, where e.g. the most likely state can be queried
            as ``result.top_measurement``.
        """
        if isinstance(self._iterations, list):
            max_iterations = len(self._iterations)
            max_power = np.inf  # no cap on the power
            iterator = iter(self._iterations)
        else:
            max_iterations = max(10, 2 ** amplification_problem.oracle.num_qubits)
            max_power = np.ceil(
                2 ** (len(amplification_problem.grover_operator.reflection_qubits) / 2)
            )
            iterator = self._iterations

        result = GroverResult()

        iterations = []
        top_measurement = "0" * len(amplification_problem.objective_qubits)
        oracle_evaluation = False
        all_circuit_results = []
        max_probability = 0
        shots = 0

        for _ in range(max_iterations):  # iterate at most to the max number of iterations
            # get next power and check if allowed
            power = next(iterator)

            if power > max_power:
                break

            iterations.append(power)  # store power

            # sample from [0, power) if specified
            if self._sample_from_iterations:
                power = np.random.randint(power)
            # Run a grover experiment for a given power of the Grover operator.
            if self._quantum_instance.is_statevector:
                qc = self.construct_circuit(amplification_problem, power, measurement=False)
                circuit_results = self._quantum_instance.execute(qc).get_statevector()
                num_bits = len(amplification_problem.objective_qubits)

                # trace out work qubits
                if qc.width() != num_bits:
                    indices = [
                        i
                        for i in range(qc.num_qubits)
                        if i not in amplification_problem.objective_qubits
                    ]
                    rho = partial_trace(circuit_results, indices)
                    circuit_results = np.diag(rho.data)

                max_amplitude = max(circuit_results.max(), circuit_results.min(), key=abs)
                max_amplitude_idx = np.where(circuit_results == max_amplitude)[0][0]
                top_measurement = np.binary_repr(max_amplitude_idx, num_bits)
                max_probability = np.abs(max_amplitude) ** 2
                shots = 1
            else:
                qc = self.construct_circuit(amplification_problem, power, measurement=True)
                circuit_results = self._quantum_instance.execute(qc).get_counts(qc)
                top_measurement = max(circuit_results.items(), key=operator.itemgetter(1))[0]
                shots = sum(circuit_results.values())
                max_probability = (
                    max(circuit_results.items(), key=operator.itemgetter(1))[1] / shots
                )

            all_circuit_results.append(circuit_results)
            oracle_evaluation = amplification_problem.is_good_state(top_measurement)
            if oracle_evaluation is True:
                break  # we found a solution

        result.iterations = iterations
        result.top_measurement = top_measurement
        result.assignment = amplification_problem.post_processing(top_measurement)
        result.oracle_evaluation = oracle_evaluation
        result.circuit_results = all_circuit_results
        result.max_probability = max_probability

        return result

    @staticmethod
    def optimal_num_iterations(num_solutions: int, num_qubits: int) -> int:
        """Return the optimal number of iterations, if the number of solutions is known.

        Args:
            num_solutions: The number of solutions.
            num_qubits: The number of qubits used to encode the states.

        Returns:
            The optimal number of iterations for Grover's algorithm to succeed.
        """
        amplitude = np.sqrt(num_solutions / 2 ** num_qubits)
        return round(np.arccos(amplitude) / (2 * np.arcsin(amplitude)))

    def construct_circuit(
        self, problem: AmplificationProblem, power: Optional[int] = None, measurement: bool = False
    ) -> QuantumCircuit:
        """Construct the circuit for Grover's algorithm with ``power`` Grover operators.

        Args:
            problem: The amplification problem for the algorithm.
            power: The number of times the Grover operator is repeated. If None, this argument
                is set to the first item in ``iterations``.
            measurement: Boolean flag to indicate if measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit

        Raises:
            ValueError: If no power is passed and the iterations are not an integer.
        """
        if power is None:
            if len(self._iterations) > 1:
                raise ValueError("Please pass ``power`` if the iterations are not an integer.")
            power = self._iterations[0]

        qc = QuantumCircuit(problem.oracle.num_qubits, name="Grover circuit")
        qc.compose(problem.state_preparation, inplace=True)
        if power > 0:
            qc.compose(problem.grover_operator.power(power), inplace=True)

        if measurement:
            measurement_cr = ClassicalRegister(len(problem.objective_qubits))
            qc.add_register(measurement_cr)
            qc.measure(problem.objective_qubits, measurement_cr)

        return qc


class GroverResult(AmplitudeAmplifierResult):
    """Grover Result."""

    def __init__(self) -> None:
        super().__init__()
        self._iterations = None
        self._circuit_results = None
        self._shots = None

    @property
    def iterations(self) -> List[int]:
        """All the powers of the Grover operator that have been tried.

        Returns:
            The powers of the Grover operator tested.
        """
        return self._iterations

    @iterations.setter
    def iterations(self, value: List[int]) -> None:
        """Set the powers of the Grover operator that have been tried.

        Args:
            value: A new value for the powers.
        """
        self._iterations = value
