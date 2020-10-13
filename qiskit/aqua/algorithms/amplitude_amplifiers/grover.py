# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Grover's search algorithm."""

from typing import Optional, Union, Dict, List, Any, Callable
import logging
import warnings
import operator
import math
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.providers import Backend
from qiskit.providers import BaseBackend
from qiskit.quantum_info import Statevector

from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils import get_subsystem_density_matrix, name_args
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.algorithms import QuantumAlgorithm, AlgorithmResult
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.components.oracles import Oracle


logger = logging.getLogger(__name__)


class Grover(QuantumAlgorithm):
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

    This class supports oracles in form of :class:`~qiskit.QuantumCircuit` or
    :class:`~qiskit.aqua.components.oracles.Oracle`. For example the
    :class:`~qiskit.aqua.components.oracles.LogicalExpressionOracle`
    can take as input a SAT problem in
    `DIMACS CNF format <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__
    and be used with Grover algorithm to find a satisfiable assignment.

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

    @name_args([
        ('oracle', ),
        ('good_state', {InitialState: 'init_state'}),
        ('state_preparation', {bool: 'incremental'}),
        ('iterations', ),
        ('post_processing', {float: 'lam'}),
        ('grover_operator', {list: 'rotation_counts'}),
        ('quantum_instance', {str: 'mct_mode'}),
        ('incremental', {(Backend, BaseBackend, QuantumInstance): 'quantum_instance'})
    ], skip=1)  # skip the argument 'self'
    def __init__(self,
                 oracle: Union[Oracle, QuantumCircuit, Statevector],
                 good_state: Optional[Union[Callable[[str], bool],
                                            List[str], List[int], Statevector]] = None,
                 state_preparation: Optional[QuantumCircuit] = None,
                 iterations: Union[int, List[int]] = 1,
                 sample_from_iterations: bool = False,
                 post_processing: Callable[[List[int]], List[int]] = None,
                 grover_operator: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[Union[QuantumInstance, Backend, BaseBackend]] = None,
                 init_state: Optional[InitialState] = None,
                 incremental: bool = False,
                 num_iterations: Optional[int] = None,
                 lam: Optional[float] = None,
                 rotation_counts: Optional[List[int]] = None,
                 mct_mode: Optional[str] = None,
                 ) -> None:
        # pylint: disable=line-too-long
        r"""
        Args:
            oracle: The oracle to flip the phase of good states, :math:`\mathcal{S}_f`.
            good_state: A callable to check if a given measurement corresponds to a good state.
                For convenience, a list of bitstrings, a list of integer or statevector can be
                passed instead of a function. If the input is a list of bitstrings, each bitstrings
                in the list represents a good state. If the input is a list of integer,
                each integer represent the index of the good state to be :math:`|1\rangle`.
                If it is a :class:`~qiskit.quantum_info.Statevector`, it represents a superposition
                of all good states.
            state_preparation: The state preparation :math:`\mathcal{A}`. If None then Grover's
                 Search by default uses uniform superposition.
            iterations: Specify the number of iterations/power of Grover's operator to be checked.
                It the number of solutions is known, this should be an integer specifying the
                optimal number of iterations (see ``optimal_num_iterations``). Alternatively,
                this can be a list of powers to check.
            sample_from_iterations: If True, instead of taking the values in ``iterations`` as
                powers of the Grover operator, a random integer sample between 0 and smaller value
                than the iteration is used as a power, see [1], Section 4.
            post_processing: An optional post processing applied to the top measurement. Can be used
                e.g. to convert from the bit-representation of the measurement `[1, 0, 1]` to a
                DIMACS CNF format `[1, -2, 3]`.
            grover_operator: A circuit implementing the Grover operator :math:`\mathcal{Q}`.
                If None, the operator is constructed automatically using the
                :class:`~qiskit.circuit.library.GroverOperator` from the circuit library.
            quantum_instance: A Quantum Instance or Backend to run the circuits.
            init_state: DEPRECATED, use ``state_preparation`` instead.
                 An optional initial quantum state. If None (default) then Grover's Search
                 by default uses uniform superposition to initialize its quantum state. However,
                 an initial state may be supplied, if useful, for example, if the user has some
                 prior knowledge regarding where the search target(s) might be located.
            incremental: DEPRECATED, use ``iterations`` instead.
                Whether to use incremental search mode (True) or not (False).
                Supplied *num_iterations* is ignored when True and instead the search task will
                be carried out in successive rounds, using circuits built with incrementally
                higher number of iterations for the repetition of the amplitude amplification
                until a target is found or the maximal number :math:`\log N` (:math:`N` being the
                total number of elements in the set from the oracle used) of iterations is
                reached. The implementation follows Section 4 of [2].
            num_iterations: DEPRECATED, use ``iterations`` instead.
                How many times the marking and reflection phase sub-circuit is
                repeated to amplify the amplitude(s) of the target(s). Has a minimum value of 1.
            lam: DEPRECATED, use ``iterations`` instead.
                For incremental search mode, the maximum number of repetition of amplitude
                amplification increases by factor lam in every round,
                :math:`R_{i+1} = lam \times R_{i}`. If this parameter is not set, the default
                value lam = 1.34 is used, which is proved to be optimal [1].
            rotation_counts: DEPRECATED, use ``iterations`` instead.
                For incremental mode, if rotation_counts is defined, parameter *lam*
                is ignored. rotation_counts is the list of integers that defines the number of
                repetition of amplitude amplification for each round.
            mct_mode: DEPRECATED, pass a custom ``grover_operator`` instead.
                Multi-Control Toffoli mode ('basic' | 'basic-dirty-ancilla' |
                'advanced' | 'noancilla')

        Raises:
            TypeError: If ``init_state`` is of unsupported type or is of type ``InitialState` but
                the oracle is not of type ``Oracle``.
            AquaError: evaluate_classically() missing from the input oracle
            TypeError: If ``oracle`` is of unsupported type.


        References:
            [1]: Boyer et al., Tight bounds on quantum searching
                 `<https://arxiv.org/abs/quant-ph/9605034>`_
        """
        super().__init__(quantum_instance)
        _check_deprecated_args(init_state, mct_mode, rotation_counts, lam, num_iterations)
        if init_state is not None:
            state_preparation = init_state

        if mct_mode is None:
            mct_mode = 'noancilla'

        self._oracle = oracle

        # if oracle is an Oracle class, extract the `good_state` callable
        if isinstance(oracle, Oracle):
            def is_good_state(bitstr):
                return oracle.evaluate_classically(bitstr)[0]

            good_state = is_good_state

        # Construct GroverOperator circuit
        if grover_operator is not None:
            self._grover_operator = grover_operator
        else:
            # wrap in method to hide the logic of handling deprecated arguments, can be simplified
            # once the deprecated arguments are removed
            self._grover_operator = _construct_grover_operator(oracle, state_preparation,
                                                               mct_mode)

        max_iterations = np.ceil(2 ** (len(self._grover_operator.reflection_qubits) / 2))
        if incremental:  # TODO remove 3 months after 0.8.0
            if rotation_counts is not None:
                iterations = rotation_counts
                self._sample_from_iterations = False
            else:
                if lam is None:
                    lam = 1.34

                iterations = []
                self._sample_from_iterations = True
                power = 1.0
                while power < max_iterations:
                    iterations.append(int(power))
                    power = lam * power

        elif num_iterations is not None:  # TODO remove 3 months after 0.8.0
            iterations = [num_iterations]
        elif not isinstance(iterations, list):
            iterations = [iterations]
        # else: already a list

        # cutoff if max_iterations is exceeded (legacy code, should considered for removal?)
        self._iterations = []
        for iteration in iterations:
            self._iterations += [iteration]
            if iteration > max_iterations:
                break

        # check the type of good_state
        _check_is_good_state(good_state)

        self._is_good_state = good_state
        self._sample_from_iterations = sample_from_iterations
        self._post_processing = post_processing
        self._incremental = incremental
        self._lam = lam
        self._rotation_counts = rotation_counts

        if incremental or (isinstance(iterations, list) and len(iterations) > 1):
            logger.debug('Incremental mode specified, \
                ignoring "num_iterations" and "num_solutions".')

        self._ret = {}  # type: Dict[str, Any]

    @staticmethod
    def optimal_num_iterations(num_solutions: int, num_qubits: int) -> int:
        """Return the optimal number of iterations, if the number of solutions is known.

        Args:
            num_solutions: The number of solutions.
            num_qubits: The number of qubits used to encode the states.

        Returns:
            The optimal number of iterations for Grover's algorithm to succeed.
        """
        return math.floor(np.pi * np.sqrt(2 ** num_qubits / num_solutions) / 4)

    def _run_experiment(self, power):
        """Run a grover experiment for a given power of the Grover operator."""
        if self._quantum_instance.is_statevector:
            qc = self.construct_circuit(power, measurement=False)
            result = self._quantum_instance.execute(qc)
            statevector = result.get_statevector(qc)
            num_bits = len(self._grover_operator.reflection_qubits)
            # trace out work qubits
            if qc.width() != num_bits:
                rho = get_subsystem_density_matrix(
                    statevector,
                    range(num_bits, qc.width())
                )
                statevector = np.diag(rho)
            max_amplitude = max(statevector.max(), statevector.min(), key=abs)
            max_amplitude_idx = np.where(statevector == max_amplitude)[0][0]
            top_measurement = np.binary_repr(max_amplitude_idx, num_bits)

        else:
            qc = self.construct_circuit(power, measurement=True)
            measurement = self._quantum_instance.execute(qc).get_counts(qc)
            self._ret['measurement'] = measurement
            top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]

        self._ret['top_measurement'] = top_measurement

        # as_list = [int(bit) for bit in top_measurement]
        # return self.post_processing(as_list), self.is_good_state(top_measurement)
        return self.post_processing(top_measurement), self.is_good_state(top_measurement)

    def is_good_state(self, bitstr: str) -> bool:
        """Check whether a provided bitstring is a good state or not.

        Args:
            bitstr: The measurement as bitstring.

        Returns:
            True if the measurement is a good state, False otherwise.
        """
        if callable(self._is_good_state):
            return self._is_good_state(bitstr)
        elif isinstance(self._is_good_state, list):
            if all(isinstance(good_bitstr, str) for good_bitstr in self._is_good_state):
                return bitstr in self._is_good_state
            else:
                return all(bitstr[good_index] == '1'  # type:ignore
                           for good_index in self._is_good_state)
        # else isinstance(self._is_good_state, Statevector) must be True
        return bitstr in self._is_good_state.probabilities_dict()

    def post_processing(self, measurement: List[int]) -> List[int]:
        """Do the post-processing to the measurement result

        Args:
            measurement: The measurement as list of int.

        Returns:
            Do the post-processing based on the post_processing argument.
            If the post_processing argument is None and the Oracle class is used as its oracle,
            oracle.evaluate_classically is used as the post_processing.
            Otherwise, just return the input bitstr
        """
        if self._post_processing is not None:
            return self._post_processing(measurement)

        if isinstance(self._oracle, Oracle):
            bitstr = measurement
            # bitstr = ''.join([str(bit) for bit in measurement])
            return self._oracle.evaluate_classically(bitstr)[1]

        return measurement

    def construct_circuit(self, power: Optional[int] = None,
                          measurement: bool = False) -> QuantumCircuit:
        """Construct the circuit for Grover's algorithm with ``power`` Grover operators.

        Args:
            power: The number of times the Grover operator is repeated. If None, this argument
                is set to the first item in ``iterations``.
            measurement: Boolean flag to indicate if measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """
        if power is None:
            power = self._iterations[0]

        qc = QuantumCircuit(self._grover_operator.num_qubits, name='Grover circuit')
        qc.compose(self._grover_operator.state_preparation, inplace=True)
        if power > 0:
            qc.compose(self._grover_operator.power(power), inplace=True)

        if measurement:
            measurement_cr = ClassicalRegister(len(self._grover_operator.reflection_qubits))
            qc.add_register(measurement_cr)
            qc.measure(self._grover_operator.reflection_qubits, measurement_cr)

        self._ret['circuit'] = qc
        return qc

    def _run(self) -> 'GroverResult':
        # If ``rotation_counts`` is specified, run Grover's circuit for the powers specified
        # in ``rotation_counts``. Once a good state is found (oracle_evaluation is True), stop.
        for power in self._iterations:
            if self._sample_from_iterations:
                power = self.random.integers(power)
            assignment, oracle_evaluation = self._run_experiment(power)
            if oracle_evaluation:
                break

        # TODO remove all former dictionary logic
        self._ret['result'] = assignment
        self._ret['oracle_evaluation'] = oracle_evaluation

        result = GroverResult()
        if 'measurement' in self._ret:
            result.measurement = dict(self._ret['measurement'])
        if 'top_measurement' in self._ret:
            result.top_measurement = self._ret['top_measurement']
        if 'circuit' in self._ret:
            result.circuit = self._ret['circuit']
        result.assignment = self._ret['result']
        result.oracle_evaluation = self._ret['oracle_evaluation']
        return result

    @property
    def grover_operator(self) -> QuantumCircuit:
        """Returns grover_operator."""
        return self._grover_operator


def _oracle_component_to_circuit(oracle: Oracle):
    """Convert an Oracle to a QuantumCircuit."""
    circuit = QuantumCircuit(oracle.circuit.num_qubits)

    _output_register = [i for i, qubit in enumerate(oracle.circuit.qubits)
                        if qubit in oracle.output_register[:]]

    circuit.x(_output_register)
    circuit.h(_output_register)
    circuit.compose(oracle.circuit, list(range(oracle.circuit.num_qubits)),
                    inplace=True)
    circuit.h(_output_register)
    circuit.x(_output_register)

    reflection_qubits = [i for i, qubit in enumerate(oracle.circuit.qubits)
                         if qubit in oracle.variable_register[:]]

    return circuit, reflection_qubits


def _construct_grover_operator(oracle, state_preparation, mct_mode):
    # check the type of state_preparation
    if isinstance(state_preparation, InitialState):
        warnings.warn('Passing an InitialState component is deprecated as of 0.8.0, and '
                      'will be removed no earlier than 3 months after the release date. '
                      'You should pass a QuantumCircuit instead.',
                      DeprecationWarning, stacklevel=3)
        if isinstance(oracle, Oracle):
            state_preparation = state_preparation.construct_circuit(
                mode='circuit', register=oracle.variable_register
                )
        else:
            raise TypeError('If init_state is of type InitialState, oracle must be of type '
                            'Oracle')
    elif not (isinstance(state_preparation, QuantumCircuit) or state_preparation is None):
        raise TypeError('Unsupported type "{}" of state_preparation'.format(
            type(state_preparation)))

    # check to oracle type and if necessary convert the deprecated Oracle component to
    # a circuit
    reflection_qubits = None
    if isinstance(oracle, Oracle):
        if not callable(getattr(oracle, "evaluate_classically", None)):
            raise AquaError(
                'Missing the evaluate_classically() method \
                    from the provided oracle instance.'
            )

        oracle, reflection_qubits = _oracle_component_to_circuit(oracle)
    elif not isinstance(oracle, (QuantumCircuit, Statevector)):
        raise TypeError('Unsupported type "{}" of oracle'.format(type(oracle)))

    grover_operator = GroverOperator(oracle=oracle,
                                     state_preparation=state_preparation,
                                     reflection_qubits=reflection_qubits,
                                     mcx_mode=mct_mode)
    return grover_operator


def _check_deprecated_args(init_state, mct_mode, rotation_counts, lam, num_iterations):
    """Check the deprecated args, can be removed 3 months after 0.8.0."""

    # init_state has been renamed to state_preparation
    if init_state is not None:
        warnings.warn('The init_state argument is deprecated as of 0.8.0, and will be removed '
                      'no earlier than 3 months after the release date. You should use the '
                      'state_preparation argument instead and pass a QuantumCircuit or '
                      'Statevector instead of an InitialState.',
                      DeprecationWarning, stacklevel=3)

    if mct_mode is not None:
        validate_in_set('mct_mode', mct_mode,
                        {'basic', 'basic-dirty-ancilla', 'advanced', 'noancilla'})
        warnings.warn('The mct_mode argument is deprecated as of 0.8.0, and will be removed no '
                      'earlier than 3 months after the release date. If you want to use a '
                      'special MCX mode you should use the GroverOperator in '
                      'qiskit.circuit.library directly and pass it to the grover_operator '
                      'keyword argument.', DeprecationWarning, stacklevel=3)

    if rotation_counts is not None:
        warnings.warn('The rotation_counts argument is deprecated as of 0.8.0, and will be '
                      'removed no earlier than 3 months after the release date. '
                      'If you want to use the incremental mode with the rotation_counts '
                      'argument or you should use the iterations argument instead and pass '
                      'a list of integers',
                      DeprecationWarning, stacklevel=3)

    if lam is not None:
        warnings.warn('The lam argument is deprecated as of 0.8.0, and will be '
                      'removed no earlier than 3 months after the release date. '
                      'If you want to use the incremental mode with the lam argument, '
                      'you should use the iterations argument instead and pass '
                      'a list of integers calculated with the lam argument.',
                      DeprecationWarning, stacklevel=3)

    if num_iterations is not None:
        validate_min('num_iterations', num_iterations, 1)
        warnings.warn('The num_iterations argument is deprecated as of 0.8.0, and will be '
                      'removed no earlier than 3 months after the release date. '
                      'If you want to use the num_iterations argument '
                      'you should use the iterations argument instead and pass an integer '
                      'for the number of iterations.',
                      DeprecationWarning, stacklevel=3)


def _check_is_good_state(is_good_state):
    """Check whether a provided is_good_state is one of the supported types or not"""
    is_compatible = False
    if callable(is_good_state):
        is_compatible = True
    if isinstance(is_good_state, list):
        if all(isinstance(good_bitstr, str) for good_bitstr in is_good_state) or \
           all(isinstance(good_index, int) for good_index in is_good_state):
            is_compatible = True
    if isinstance(is_good_state, Statevector):
        is_compatible = True

    if not is_compatible:
        raise TypeError('Unsupported type "{}" of is_good_state'.format(type(is_good_state)))


class GroverResult(AlgorithmResult):
    """Grover Result."""

    @property
    def measurement(self) -> Optional[Dict[str, int]]:
        """ returns measurement """
        return self.get('measurement')

    @measurement.setter
    def measurement(self, value: Dict[str, int]) -> None:
        """ set measurement """
        self.data['measurement'] = value

    @property
    def top_measurement(self) -> Optional[str]:
        """ return top measurement """
        return self.get('top_measurement')

    @top_measurement.setter
    def top_measurement(self, value: str) -> None:
        """ set top measurement """
        self.data['top_measurement'] = value

    @property
    def circuit(self) -> Optional[QuantumCircuit]:
        """ return circuit """
        return self.get('circuit')

    @circuit.setter
    def circuit(self, value: QuantumCircuit) -> None:
        """ set circuit """
        self.data['circuit'] = value

    @property
    def assignment(self) -> List[int]:
        """ return assignment """
        return self.get('assignment')

    @assignment.setter
    def assignment(self, value: List[int]) -> None:
        """ set assignment """
        self.data['assignment'] = value

    @property
    def oracle_evaluation(self) -> bool:
        """ return oracle evaluation """
        return self.get('oracle_evaluation')

    @oracle_evaluation.setter
    def oracle_evaluation(self, value: bool) -> None:
        """ set oracle evaluation """
        self.data['oracle_evaluation'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'GroverResult':
        """ create new object from a dictionary """
        return GroverResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'result':
            warnings.warn('result deprecated, use assignment property.', DeprecationWarning)
            return super().__getitem__('assignment')

        return super().__getitem__(key)
