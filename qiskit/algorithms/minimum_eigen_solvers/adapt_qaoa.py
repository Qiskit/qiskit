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

""" The Adaptive Derivative Assembled Problem Tailored - Quantum Approximate Optimization Algorithm. """

from functools import reduce
from itertools import combinations_with_replacement, permutations, product
from random import randint
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.opflow import ComposedOp, I, OperatorBase, X, Y, Z
from qiskit.opflow.expectations.expectation_factory import ExpectationFactory
from qiskit.opflow.primitive_ops import MatrixOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.quantum_info import Operator
from qiskit.utils import algorithm_globals

from .qaoa import QAOA


class AdaptQAOA(QAOA):
    """
    The Adaptive Derivative Assembled Problem Tailored - Quantum Approximate Optimization Algorithm.

    `ADAPT-QAOA <https://arxiv.org/abs/2005.10258>` __ is a variation of the well-known algorithm
    for finding solutions to combinatorial-optimization problems.

    The ADAPT-QAOA implementation directly extends :class:`QAOA` and inherits QAOA's optimization
    structure.
    However, unlike QAOA, which has a fixed form of the ansatz, ADAPT-QAOA takes an iterative approach
    to finding a more optimal ansatz for the given problem.

    An optional array of :math:`2p` parameter values, as the *initial_point*, may be provided as the
    starting **beta** and **gamma** parameters (as identically named in the
    original `QAOA paper <https://arxiv.org/abs/1411.4028>`__) for the ADAPT-QAOA ansatz.

    A list of operators or parameterized quantum circuits may optionally also be provided as a custom
    `mixer_pool`. The build options for the mixer pool contains the standard single-qubit X rotations
    and single-qubit Y mixers as well as the option of also including multi-qubit entangling gates.
    """

    def __init__(
        self,
        mixer_pool: Optional[Union[OperatorBase, QuantumCircuit]] = None,
        mixer_pool_type: Optional[str] = None,
        threshold: Optional[
            Callable[[int, float], None]
        ] = None,  # todo: add default value for threshold
        max_reps=1,
        **kwargs,
    ) -> None:
        """
        Args:
            optimizer: A classical optimizer.
            max_reps: An optional maximum number of repetitions of the ADAPT-QAOA circuit
                (defaults to 5).
            initial_point: An optional initial point (i.e. initial parameter values) for the optimizer.
            initial_state: An optional initial state to prepend the ADAPT-QAOA circuit with.
            gamma_init: An optional initial value for the parameter gamma to use as a starting
                value for the optimizer.
            beta_init: An optional initial value for the parameter beta to use as a starting value
                for the optimizer.
            mixer_pool: An optional custom list of Operators or QuantumCircuits that make up a pool
                from which mixers are chosen from.
                Cannot be used in conjunction with `mixer_pool_type`.
            mixer_pool_type: An optional string representing different mixer pool types `single`
            creates the same mixer pool as the
                standard QAOA. `singular` creates a mixer pool including mixers in `single` as well
                as additional single qubit
                mixers. `multi` creates a mixer pool including mixers from `single`, `singular` as
                well as multi-qubit entangling mixers.
                Cannot be used in conjuction with `mixer_pool`.
            threshold: A positive, real value in which the algorithm stops once the norm of the gradient
                is below this threshold.
            gradient: An optional gradient operator respectively a gradient function used for
                      optimization.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When None (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to True (defaults to False).
            include_custom: When `expectation` parameter here is None setting this to True will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Ignored if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
            quantum_instance: Quantum Instance or Backend

        Raises:
            AttributeError: If both a mixer pool and mixer pool type has been defined.
        """
        self._exp_seed = randint(1, 100)
        super().__init__(**kwargs)
        self.threshold = threshold

        if mixer_pool is not None and mixer_pool_type is not None:
            raise AttributeError(
                "A custom mixer pool can be passed in or a mixer pool type can be passed in but not both"
            )

        self.mixer_pool = mixer_pool
        self.mixer_pool_type = mixer_pool_type
        self.max_reps = max_reps

        self.optimal_mixer_list = (
            []
        )  # will be appending optimal mixers to this, first mixer is H see above
        self.reps = 1  # Overload QAOA's reps parameter & set it to 1.
        self.name = "AdaptQAOA"
        self.ansatz = None

        if self.mixer_pool_type is None:
            # TODO: check if this list of operators/circuits needs more preprocessing before use
            self.mixer_pool = mixer_pool
        if self.threshold is None:
            self.threshold = 0.01  # TODO: work out a way to better set a default threshold

        self.best_gamma, self.best_beta = [self.initial_point[-1]], []

    def _check_operator_ansatz(self, operator: OperatorBase) -> OperatorBase:
        # Recreates a circuit based on operator parameter.
        if operator != self._cost_operator:
            self._cost_operator = operator
            self.ansatz = QAOAAnsatz(
                operator,
                initial_state=self._initial_state,
                mixer_operator=self.optimal_mixer_list,
                name=self.name,
            ).decompose()  # TODO remove decompose once #6674 is fixed

    def compute_energy_gradient(self, mixer, cost_operator, ansatz=None) -> ComposedOp:
        """Computes the energy gradient of the cost operator wrt the mixer pool at an
            ansatz layer specified by the input 'state' and initial point.

        Returns:
            The mixer operator with the largest energy gradient along with the
            associated energy gradient.
        """

        from qiskit.opflow import commutator

        if not isinstance(cost_operator, MatrixOp):
            cost_operator = MatrixOp(Operator(cost_operator.to_matrix()))
        self.cost_operator = cost_operator
        wave_function = ansatz.assign_parameters(self.hyperparameter_dict)
        # construct expectation operator
        exp_hc = (0.01 * cost_operator).exp_i()  # TODO: Check if this is legit?
        # exp_hc = (self.best_gamma[-1] * cost_operator).exp_i()
        exp_hc_ad = exp_hc.adjoint().to_matrix()
        exp_hc = exp_hc.to_matrix()
        energy_grad_op = exp_hc_ad @ (commutator(cost_operator, mixer).to_matrix()) @ exp_hc
        energy_grad_op = PrimitiveOp(energy_grad_op)

        expectation = ExpectationFactory.build(
            operator=energy_grad_op,
            backend=self.quantum_instance,
            include_custom=self._include_custom,
        )
        observable_meas = expectation.convert(StateFn(energy_grad_op, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()
        return 1j*expect_op

    def _test_mixer_pool(self, operator: OperatorBase):
        self._check_problem_configuration(operator=operator)
        energy_gradients = []
        for mixer in self.mixer_pool:
            expect_op = self.compute_energy_gradient(mixer, operator, ansatz=self.ansatz)
            # run expectation circuit
            sampled_expect_op = self._circuit_sampler.convert(
                expect_op, params=self.hyperparameter_dict
            )
            meas = np.abs(sampled_expect_op.eval())
            energy_gradients.append(meas)
        max_energy_idx = np.argmax(energy_gradients)
        return self.mixer_pool[max_energy_idx], energy_gradients[max_energy_idx]

    def run_adapt(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ):
        """Runs ADAPT-QAOA for each iteration"""
        self.cost_operator = operator
        p, self.ansatz = 0, self.initial_state  # initialise layer loop counter and ansatz
        print("--------------------------------------------------------")
        print("Cost operator {}".format(operator))
        while p < self.max_reps:  # loop over number of maximum reps
            best_mixer, energy_norm = self._test_mixer_pool(operator=operator)
            print(f"REPETITION: {p}")
            print(f"Current energy norm | Threshold  =====> | {energy_norm} | {self.threshold} |")
            if energy_norm < self.threshold:  # Threshold stoppage condition
                break
            self.optimal_mixer_list.append(
                best_mixer
            )  # Append mixer associated with largest energy gradient to list
            self.ansatz = QAOAAnsatz(
                cost_operator=operator,
                initial_state=self._initial_state,
                mixer_operator=self.optimal_mixer_list,
                name=self.name,
            )
            result = self.compute_minimum_eigenvalue(operator=operator, aux_operators=aux_operators)
            opt_params = result.optimal_point
            self.best_beta, self.best_gamma = np.split(opt_params, 2)
            self.best_beta = list(self.best_beta)
            self.best_gamma = list(self.best_gamma)
            print("Optimal value", result.optimal_value)
            # print(self.ansatz.decompose().draw())
            # print(f"Optimal parameters: {result.optimal_parameters}")
            # print(f"Initial point: {self.initial_point}")
            # print(f"New mixer {self.optimal_mixer_list}")
            print("--------------------------------------------------------")
            p += 1
        print("Optimal mixers:", self.optimal_mixer_list)
        return result

    def _check_problem_configuration(self, operator: OperatorBase):
        # Generates the pool of mixers with respect to the cost operator size
        self.num_qubits = (
            operator.num_qubits
        )  # Compute the number of qubits from the problem operator
        # if not self.mixer_pool: # Set a generate a mixer pool if none was provided
        #     self.mixer_pool =
        #       adapt_mixer_pool(num_qubits=self.num_qubits, pool_type=self.mixer_pool_type)
        if isinstance(self.mixer_pool, list):
            mixer_n_qubits = [mixer.num_qubits for mixer in self.mixer_pool]
        else:
            mixer_n_qubits = self.mixer_pool.num_qubits
            self.mixer_pool = [self.mixer_pool]
        check_mixer_qubits = list(np.argwhere(mixer_n_qubits != self.num_qubits)[0])
        if check_mixer_qubits:
            err_str = ", ".join(map(str(x) for x in check_mixer_qubits))
            raise ValueError(
                f"One or more mixing operators specified at list indices {err_str}"
                " have an unequal number of respective qubits"
                "{mixer_n_qubits[check_mixer_qubits]} to the initialised"
                "cost operator {self.num_qubits}."
            )

    @property
    def mixer_pool(self) -> List:
        """Creates the mixer pool if not already defined

        Returns:
            List of mixers that make up the mixer pool.

        Raises:
            AttributeError: If operator and thus num_qubits has not yet been defined.
        """
        if not self._mixer_pool:
            if not hasattr(self, "num_qubits"):
                raise AttributeError(
                    "Unable to automatically generate operators in mixer pool "
                    "without reference to a problem operator."
                )
            self._mixer_pool = adapt_mixer_pool(
                num_qubits=self.num_qubits, pool_type=self.mixer_pool_type
            )
        return self._mixer_pool

    @mixer_pool.setter
    def mixer_pool(self, mixer_pool: List) -> None:
        self._mixer_pool = mixer_pool

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Returns an optional initial state as a circuit"""
        if self._initial_state is not None:
            return self._initial_state

        # if no initial state is passed and we know the number of qubits, then initialize it.
        if self.num_qubits > 0:
            initial_state = QuantumCircuit(self.num_qubits)
            initial_state.h(range(self.num_qubits))
            return initial_state
        # otherwise we cannot provide a default
        return None

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """Sets initial state."""
        self._initial_state = initial_state
        self._invalidate()

    @property
    def hyperparameter_dict(self) -> Dict:
        """Creates dictionary of hyperparameters including ansatz parameters

        Returns:
            Dictionary of hyperparameters
        """
        self._hyperparameter_dict = {}
        if self._ansatz_params:
            # reordered_params = self._reorder_parameters()
            self._hyperparameter_dict = dict(zip(self._ansatz_params, self.best_beta+self.best_gamma))
        return self._hyperparameter_dict

    @hyperparameter_dict.setter
    def hyperparameter_dict(self, hyperparameter_dict) -> Dict:
        self._hyperparameter_dict = hyperparameter_dict

    @property
    def cost_operator(self):
        """Returns an operator representing the cost of the optimization problem.

        Returns:
            OperatorBase: cost operator.
        """
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        """Sets cost operator & number of qubits for optimization problem.

        Args:
            cost_operator (OperatorBase, optional): cost operator to set.
        """
        self.num_qubits = cost_operator.num_qubits
        self._cost_operator = cost_operator

    @property
    def initial_point(self) -> np.ndarray:
        """Updates initial points

        Returns:
            Numpy array of intial points

        """
        if self._ansatz_params:
            # self.ansatz.parameter_bounds = [(-2 * np.pi, 2 * np.pi)] * self.ansatz.num_parameters
            if len(self._initial_point) != self.ansatz.num_parameters:
                self._update_initial_point()
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point) -> Optional[np.ndarray]:
        if initial_point is None:
            initial_point = self._generate_initial_point()
        self._initial_point = initial_point

    def _generate_initial_point(self, reps=1): # set initial value for gamma according to https://arxiv.org/abs/2005.10258
        return np.array([algorithm_globals.random.uniform(-2 * np.pi, 2 * np.pi),0.01])

    def _update_initial_point(self):
        # optimised_initial_point = self._reorder_parameters()
        # assert len(optimised_initial_point)==len(self._initial_point)
        self._initial_point = np.concatenate((self._initial_point, self._generate_initial_point()))

    def _reorder_parameters(self):
        reordered_params = np.zeros(len(self._initial_point))
        reordered_params[0:][::2] = np.array(self.best_gamma)
        reordered_params[1:][::2] = np.array(self.best_beta)
        return list(reordered_params)


def adapt_mixer_pool(
    num_qubits: int,
    add_single: bool = True,
    add_multi: bool = True,
    pool_type: str = None,
) -> List:
    """
    Gets all combinations of mixers in desired set (standard qaoa mixer, single qubit
        mixers, multi qubit mixers)
    Args:
        num_qubits: number of qubits
        add_single: whether to add single qubit to mixer pool (not standard qaoa x mixers)
        add_multi: whether to add multi qubit to mixer pool
        pool_type: Optional input overrides add_single and add_multi by respecifying
            these conditions based on the preset mixer pool classes: 'multi',
            'singular' and 'single'.

    Returns:
        List of all possible combinations of mixers.

    Raises:
        ValueError: If an unrecognisible mixer type has been provided.
    """
    if pool_type:
        if pool_type == "multi":
            add_multi, add_single = True, True
        elif pool_type == "singular":
            add_multi, add_single = False, True
        elif pool_type == "single":
            add_multi, add_single = False, False
        else:
            raise ValueError(
                "Unrecognised mixer pool type {}, modify this input to the available presets"
                " 'single', 'singular' or 'multi'."
            )

    # always include the all x's:
    mixer_pool = ["X" * num_qubits]
    if add_single:
        # y's
        mixer_pool.append("Y" * num_qubits)
        mixer_pool += [i * "I" + "X" + (num_qubits - i - 1) * "I" for i in range(num_qubits)]
        mixer_pool += [i * "I" + "Y" + (num_qubits - i - 1) * "I" for i in range(num_qubits)]
    if add_multi:
        indicies = list(permutations(range(num_qubits), 2))
        indicies = list(set(tuple(sorted(x)) for x in indicies))
        combos = list(combinations_with_replacement(["X", "Y", "Z"], 2))
        full_multi = list(product(indicies, combos))
        for item in full_multi:
            iden_str = list("I" * num_qubits)
            iden_str[item[0][0]] = item[1][0]
            iden_str[item[0][1]] = item[1][1]
            mixer_pool.append("".join(iden_str))

    op_dict = {"I": I, "X": X, "Y": Y, "Z": Z}

    def is_all_same(items):
        return all(x == items[0] for x in items)

    def string_to_op(qstring):
        if is_all_same(qstring):
            # case where its all X's or Y's
            gate = qstring[0]
            list_string = [
                i * "I" + gate + (len(qstring) - i - 1) * "I" for i in range(len(qstring))
            ]
            return sum(
                [
                    reduce(lambda a, b: a ^ b, [op_dict[char.upper()] for char in x])
                    for x in list_string
                ]
            )

        return reduce(lambda a, b: a ^ b, [op_dict[char.upper()] for char in qstring])

    mixer_pool = [string_to_op(mixer) for mixer in mixer_pool]
    return mixer_pool
