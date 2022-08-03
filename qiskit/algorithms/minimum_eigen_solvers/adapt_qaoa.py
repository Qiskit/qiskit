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

import logging
from typing import Callable, List, Optional, Union, Tuple
import warnings
import numpy as np
from scipy.stats import loguniform

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.n_local import RealAmplitudes
from qiskit.circuit.parameter import Parameter
from qiskit.quantum_info.operators.operator import Operator
from qiskit.opflow import (
    MatrixOp,
    PrimitiveOp,
    ComposedOp,
    OperatorBase,
    CircuitStateFn,
    StateFn,
    ExpectationBase,
    ExpectationFactory,
    GradientBase,
)
from qiskit.algorithms.optimizers import Optimizer, COBYLA, Minimizer
from qiskit.providers import Backend
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit.library.n_local.adaptqaoa_ansatz import AdaptQAOAAnsatz, commutator
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOA
from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import MinimumEigensolverResult
from qiskit.algorithms.exceptions import AlgorithmError
from qiskit.algorithms.list_or_dict import ListOrDict

logger = logging.getLogger(__name__)


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
        mixer_pool: Optional[List[Union[OperatorBase, QuantumCircuit]]] = None,
        mixer_pool_type: Optional[str] = None,
        max_reps=1,
        optimizer: Optional[Union[Optimizer, Minimizer]] = None,
        initial_state: Optional[QuantumCircuit] = None,
        initial_point: Optional[np.ndarray] = None,
        gradient: Optional[Union[GradientBase, Callable[[Union[np.ndarray, List]], List]]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
    ) -> None:
        r"""
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
            mixer_pool: An optional custom mixer or list of mixers that define the 'pool' of operators
                that the AdaptQAOA algorithm chooses optimal mixers from. The set of optimal mixers
                are selected from the pool are denoted as :math:`U(B, \beta)` in the original paper.
                Elements in mixer_pool may be operators or an optionally parameterized quantum circuit.
                Can only be used in conjunction with a NoneType `mixer_pool_type`.
            mixer_pool_type: An optional string representing the varied mixer pool types, with `single`
                creating the same mixer pool as the standard QAOA. `singular` creates a mixer pool
                including mixers in `single` as well as additional single qubit mixers. `multi` creates
                a mixer pool including mixers from `single`, `singular` as well as multi-qubit entangling
                mixers. Cannot be used in conjuction with a non-empty `mixer_pool` list.
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
        self.__ansatz = None
        super().__init__(
            optimizer=optimizer,
            initial_state=initial_state,
            gradient=gradient,
            expectation=expectation,
            include_custom=include_custom,
            max_evals_grouped=max_evals_grouped,
            callback=callback,
            quantum_instance=quantum_instance,
        )
        self._max_reps = max_reps
        self.mixer_pool_type = mixer_pool_type
        self.mixer_pool = mixer_pool
        self.initial_point = initial_point
        self._ground_state_energy = None
        self._solution = None
        self.__energy_idx = None
        self._gamma_ip = None
        self._beta_ip = None
        self._num_beta = None
        self._num_gamma = None

    def _check_operator_ansatz(
        self,
        operator: OperatorBase,
    ) -> OperatorBase:
        if operator != self._cost_operator:
            self._cost_operator = operator
            self._solution = False  # Flag problem solution
            self._ansatz = AdaptQAOAAnsatz(  # build the ansatz
                operator,
                initial_state=self.initial_state,
                mixer_pool=self._mixer_pool,
                mixer_pool_type=self._mixer_pool_type,
            )
            self._ansatz._check_configuration()  # check the ansatz config without building it

    def energy_gradient_expectation(
        self,
        mixer: OperatorBase,
        operator: OperatorBase,
        ansatz: QuantumCircuit,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        gamma_k: float,
        return_expectation: bool = False,
    ) -> ComposedOp:
        """
        For a specified ansatz, generate the expectation value measurement of the
        operator energy gradient with respect to a specified mixer
        (see: <https://arxiv.org/abs/2005.10258>) and return the runnable composition.
        Args:
            mixer: Mixer operator that energy gradient is computed with respect to.
            operator: Qubit operator of the Observable.
            ansatz: Circuit representing the ansatz.
            parameter: Parameters for the ansatz circuit.
            gamma_k: Floating point representing the kth-γ variational parameter used in the operator
                evolution of the ansatz.
            return_expectation: If True, return the ``ExpectationBase`` expectation converter used
                in the construction of the expectation value. Useful e.g. to compute the standard
                deviation of the expectation value.
        Returns:
            The Operator equalling the measurement of the ansatz :class:`StateFn` by the
            Observable's expectation :class:`StateFn`, and, optionally, the expectation converter.
        """
        if mixer.parameters:  # check if the mixer has any parameters
            num_beta = mixer.num_parameters
            if self._num_beta[-1] != num_beta:  # if so, modify the number of
                self._num_beta[-1] = num_beta  # beta parameters at the current depth
                self._update_points()  # update the initial points
            mixer_param_values = list(np.zeros(num_beta))  # Use zeros to compute commutator
            param_dict = dict(zip(mixer.parameters, mixer_param_values))
            mixer = mixer.assign_parameters(param_dict)  # assign
            mixer = MatrixOp(Operator(mixer)).to_matrix()  # convert to matrix for commutator

        # Construct the operator representing the energy gradient with respect to the mixer pool
        energy_grad_op = energy_grad_operator(
            mixer=mixer,
            operator=operator,
            gamma_k=gamma_k,
        )
        ret_expectation = self._get_ansatz_expectation(
            ansatz=ansatz,
            parameter=parameter,
            operator=energy_grad_op,
            return_expectation=return_expectation,
        )
        return ret_expectation

    def _get_ansatz_expectation(
        self,
        ansatz: QuantumCircuit,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Union[OperatorBase, Tuple[OperatorBase, ExpectationBase]]:

        expectation = ExpectationFactory.build(
            operator=operator,
            backend=self.quantum_instance,
            include_custom=self._include_custom,
        )
        wave_function = ansatz.assign_parameters(parameter) if parameter else ansatz
        ansatz_circuit_op = CircuitStateFn(wave_function)
        observable_meas = expectation.convert(StateFn(operator, is_measurement=True))
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()
        if return_expectation:
            return expect_op, expectation
        return expect_op

    def compute_mixer_pool_energy_grads(
        self,
        operator: OperatorBase,
        ansatz: QuantumCircuit,
        mixer_pool: Optional[Union[OperatorBase, QuantumCircuit]],
        parameter: Union[List[float], List[Parameter], np.ndarray],
        gamma_k: float,
    ) -> List[float]:
        """
        For a specified ansatz, generate the operator energy gradients with respect to
        a list of mixers specified by mixer_pool.

        Args:
            operator: Qubit operator of the Observable.
            ansatz: Circuit representing the ansatz.
            mixer_pool: List of mixers that the operator energy gradient is computed with respect to.
            parameter: Parameters for the ansatz circuit.
            gamma_k: Floating point representing the kth-γ variational parameter used in the operator
                evolution of the ansatz.
        Returns:
            A an ordered list of values denoting the mixer pool energy gradients.
        """
        mixer_pool = mixer_pool if isinstance(mixer_pool, list) else [mixer_pool]
        energy_gradients = []
        params = dict(zip(ansatz.parameters, parameter))
        for mixer in mixer_pool:
            expect_op = self.energy_gradient_expectation(
                mixer=mixer, operator=operator, ansatz=ansatz, parameter=parameter, gamma_k=gamma_k
            )
            # run expectation circuit
            sampled_expect_op = self._circuit_sampler.convert(
                expect_op,
                params=params,
            )
            meas = 1j * sampled_expect_op.eval()
            energy_gradients.append(meas)
        energy_gradients = np.abs(np.real(energy_gradients))  # get the norm
        return energy_gradients  # return list of energy gradients

    def compute_minimum_eigenvalue(
        self,
        operator: OperatorBase,
        custom_mixers: Optional[List[Union[OperatorBase, QuantumCircuit]]] = None,
        threshold: Optional[float] = None,
        solution_tolerance: Optional[float] = None,
        aux_operators: Optional[Optional[ListOrDict[OperatorBase]]] = None,
    ) -> MinimumEigensolverResult:
        """The Adaptive analogue to the compute_minimum_eigenvalue function of
        QAOA as described in <https://arxiv.org/abs/2005.10258>. For each layer
        to the maximum ansatz depth (via max_reps class attribute), custom mixer
        operators may be specified by custom_mixers. Overrides attribute
        compute_minimum_eigenvalue in VQE.
        Args:
            operator: Qubit operator of the Observable.
            custom_mixers: An ordered list of custom mixer operators to be used at each
                ansatz rep up to max_reps.
            threshold: A positive, real value that discontinues the algorithm stops if the norm
                of the energy gradient with respect to the mixer pool falls below it.
            solution_tolerance: A positive, real value that discontinues the algorithm if
                it is greater than the difference in ansatz and operator ground state energy.
                Defaults to zero. Defaults to zero.
            aux_operators: Optional list of auxiliary operators to be evaluated with
                the eigenstate of the minimum eigenvalue main result and their expectation
                values returned.
        Returns:
            Optimal solution in the form of a VQEResult class.
        Raises:
            AlgorithmError: If quantum instance is unspecified.
            AttributeError: If the length of custom_mixers exceeds max_reps.
            ValueError: If the threshold or solution_tolerance is less than zero.
        """

        if self.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend must be supplied to run the quantum algorithm."
            )

        custom_mixers = [] if custom_mixers is None else custom_mixers
        if not isinstance(custom_mixers, list):
            custom_mixers = [custom_mixers]
        if self.max_reps < len(custom_mixers):
            raise AttributeError(
                "Length of custom ansatz mixer list {} must be less than maximum "
                "ansatz depth {}.".format(len(custom_mixers), self.max_reps)
            )
        threshold = 0.0 if threshold is None else threshold
        if threshold < 0.0:
            raise ValueError("Specified threshold {} must be positive.".format(threshold))
        solution_tolerance = 0.0 if solution_tolerance is None else solution_tolerance
        if solution_tolerance < 0.0:
            raise ValueError(
                "Specified solution_tolerance {} must be positive.".format(solution_tolerance)
            )
        self._check_operator_ansatz(operator)
        self._update_ansatz_params()  # set algorithm initial parameters
        while self._reps <= self.max_reps:  # loop over number of maximum reps
            if self._reps <= len(custom_mixers):
                mixers = custom_mixers[self._reps - 1]
                self._ansatz.mixer_operators = mixers
            else:
                mixers = self._ansatz._mixer_op_pool
            energy_gradients = self.compute_mixer_pool_energy_grads(
                operator=operator,
                ansatz=self.__ansatz,
                mixer_pool=mixers,
                parameter=list(self._ret.optimal_point)
                if hasattr(self._ret, "optimal_point")
                else {},
                gamma_k=float(self.initial_point[-1]),
            )
            self.__energy_idx = np.argmax(
                energy_gradients
            )  # private variable for the index of max gradient change
            energy_norm = energy_gradients[
                self.__energy_idx
            ]  # find maximal value for norm energy gradient
            if energy_norm < threshold:  # Threshold stoppage condition
                self._solution = True
                stop_msg = str(
                    "energy gradient with respect to the mixer pool ({}) less "
                    "than threshold ({})".format(np.round(energy_norm, 3), np.round(threshold, 3))
                )
                break
            self.ansatz = self.__ansatz
            super().compute_minimum_eigenvalue(operator=operator, aux_operators=aux_operators)
            delta_energy = self._ret.optimal_value - self.ground_state_energy
            logger.info("Circuit depth: %s", self._reps)
            logger.info("Current energy norm: %s", energy_norm)
            logger.info("Best mixer: %s", self._ansatz.mixer_operators[-1])
            logger.info("Initial point: %s", self.initial_point)
            logger.info("Optimal parameters: %s", self._ret.optimal_parameters)
            logger.info("Relative Energy: %s", delta_energy)
            if (
                np.abs(delta_energy) <= solution_tolerance
            ):  # Stop the algorithm if the ansatz groundstate is less than system groundstate
                self._solution = True
                stop_msg = str(
                    "cost function energy ({}) less than solution tolerance ({})".format(
                        np.round(self._ret.optimal_value, 3), np.round(solution_tolerance, 3)
                    )
                )
                break
            self._reps += 1
            self._update_points()
        if self._solution:
            logger.info("Early stoppage condition satisfied; %s", stop_msg)
        self._solution = True
        return self._ret

    def construct_circuit(
        self,
        operator: OperatorBase,
        mixer_operators: Optional[List[Union[OperatorBase, QuantumCircuit]]] = None,
        parameter: Union[List[float], List[Parameter], np.ndarray] = None,
    ) -> AdaptQAOAAnsatz:
        """Return a QAOA ansatz circuit used to compute the expectation value.
        Custom mixing operators at each layer in the ansatz may be specified by an
        an ordered list of operators, with a list length equating to the ansatz depth.
        Overrides method construct_circuit in VQE.
        Args:
            operator: Qubit operator of the Observable
            mixer_operators: Operator or an ordered list of operators defining the
                mixer to be inserted at each layer in the ansatz.
            parameter: Parameters for the ansatz circuit.
        Returns:
            A QAOA-like ansatz circuit used to compute the expectation value.
        """

        self._check_operator_ansatz(operator)

        self._ansatz.mixer_operators = mixer_operators
        self._ansatz._build()
        self._ansatz.assign_parameters(parameter, inplace=True)
        return self._ansatz

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the ansatz for the current circuit depth"""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[QuantumCircuit]):
        """Sets and updates the ansatz
        Args:
            ansatz: The parameterized circuit used as an ansatz.
            If None is passed, RealAmplitudes is used by default.
        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        if self.__ansatz == ansatz:  # update the ansatz with mixer operators
            self._update_ansatz_params()
            self.__ansatz = self._ansatz
        else:
            self._ansatz = ansatz
        self._ansatz_params = list(self._ansatz.parameters)

    @property
    def ground_state_energy(self) -> float:
        """Returns the ground state energy of the cost operator
        Returns:
            Float: The problem Hamiltonian ground state energy.

        """
        if self._ground_state_energy is None:
            if self._cost_operator is not None:
                try:
                    self._ground_state_energy = min(
                        np.real(np.linalg.eig(self._cost_operator.to_matrix())[0])
                    )
                except:
                    self._ground_state_energy = 0
        return self._ground_state_energy

    @property
    def initial_point(self) -> np.array:
        """Returns the initial points in the circuit up to the current circuit depth.
        If no initial points are specified, the cost and mixer parameters will be set
        as 0.01 and 0.25 * pi respectively; see https://arxiv.org/abs/2005.10258.

        Returns:
            Numpy array of intial points for iterative layers.
        """
        if not self._solution:
            num_params = self._num_gamma + sum(self._num_beta[: self._reps])
            num_pts = len(self._gamma_ip) + len(self._beta_ip)
            if num_pts < num_params:
                self._update_points()
            return self._init_point_rep
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point) -> Optional[np.ndarray]:
        """
        Specifies initial point.
        Raises:
            AttributeError: If number of initial points dont match the current circuit depth.
        """
        if initial_point is None:
            self.__user_ip = False
        else:
            self.__user_ip = (
                initial_point.tolist() if not isinstance(initial_point, list) else initial_point
            )
            if len(initial_point) != 2 * self.max_reps:
                raise AttributeError(
                    "The number of user specified initial points ({}) must "
                    "be at least twice the maximum ansatz depth ({})".format(
                        len(initial_point), 2 * self.max_reps
                    )
                )
        self._init_point_rep = []
        self._initial_point = initial_point

    @property
    def optimizer(self) -> Optimizer:
        """Returns optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer]):
        """Sets the optimizer as COBYLA if otherwise unspecified.
        Args:
            optimizer: The optimizer to be used. If None is passed, COBYLA is used by default.
        """
        if optimizer is None:
            optimizer = COBYLA()
        optimizer.set_max_evals_grouped(self.max_evals_grouped)
        self._optimizer = optimizer

    @property
    def mixer_pool_type(self) -> str:
        """Returns the class of mixer pool type.
        Returns:
            str: mixer pool type.
        """
        return self._mixer_pool_type

    @mixer_pool_type.setter
    def mixer_pool_type(self, mixer_pool_type: str):
        """Sets the mixer pool type.
        Args:
            mixer_pool_type: A string that represents the preset mixer pool classes.
        Raises:
            KeyError: If the specified mixer pool type is not one of 'single', 'singular' or 'multi'.
        """
        if mixer_pool_type not in ["single", "singular", "multi", None]:
            raise KeyError(
                "Unrecognised mixer pool type '{}', modify this input to the available presets"
                " 'single', 'singular' or 'multi'.".format(mixer_pool_type)
            )
        self._mixer_pool_type = mixer_pool_type

    @property
    def mixer_pool(self) -> None:
        """Returns a list of mixing operators that define the mixer pool
        Returns:
            List[Union[OperatorBase, QuantumCircuit]]: mixers pool.
        """
        return self._mixer_pool

    @mixer_pool.setter
    def mixer_pool(self, mixer_pool: List[Union[OperatorBase, QuantumCircuit]]) -> None:
        """
        Args:
            mixer_pool: A list of operators that define the pool of operators that
            the eigensolver may drawn an optimal mixer solutions from.
        Raises:
            AttributeError: If mixer_pool and mixer_pool_type are not None, mixer_pool
            cannot be set.
        """
        if mixer_pool is not None:
            if not isinstance(mixer_pool, list):
                mixer_pool = [mixer_pool]
            if self._mixer_pool_type is not None:
                raise AttributeError(
                    "Unable to set mixer pool as the provided mixer "
                    "pool type {} is not None".format(self._mixer_pool_type)
                )
        self._mixer_pool = mixer_pool
        self._ansatz.mixer_pool = mixer_pool  # reset the ansatz mixer pool

    @property
    def max_reps(self):
        """Returns the maximum number of algorithm repetitions
        Returns:
            int: The maximum allowed ansatz depth
        """
        return self._max_reps

    @max_reps.setter
    def max_reps(self, max_reps: int) -> None:
        """Sets the maximum ansatz depth.
        Raises:
            ValueError: If the maximum number of reps is < 1.
        """
        if max_reps < 1:
            raise ValueError(
                "Specified maximum number of algorithm repetitions {} "
                "cannot be less than 1.".format(max_reps)
            )
        self._max_reps = max_reps

    def _update_points(self, num_gamma: int = None, num_beta: list = None):
        """Updates the initial points of the ansatz for the current depth"""
        if num_gamma is None:
            num_gamma = self._num_gamma
            if 0 < self._num_gamma < self._reps:
                self._num_gamma = self._reps  # move to the next initial point

        if num_beta is None:
            num_beta = self._num_beta
            if len(num_beta) < self._reps:
                self._num_beta.append(0)    # Move to next IP by increasing list length

        if self.__user_ip:  # if user defined initial points
            self._beta_ip = self.__user_ip[: sum(num_beta)]
            self._gamma_ip = self.__user_ip[-1 * num_gamma :]
            if np.any([bool(abs(_) < 1e-3) for _ in self._gamma_ip]):
                warnings.warn(
                    "To avoid critical points in the cost function evaluation, "
                    "the use of initial points |γ| ≲ 1e-3 is discouraged."
                )
        else:  # no initial points are specified

            def rand_beta(num_beta):
                new_beta = []
                for _ in range(num_beta):  # randomly sample β from {0, +pi/2, -pi/2}
                    beta_i = np.random.uniform(-np.pi / 2, np.pi / 2)
                    if (abs(beta_i) < 1e-6) or (abs(beta_i + np.pi / 4) <= 1e-6):
                        beta_i = rand_beta(1)[0]  # if sample is near a critical point, resample
                    new_beta.append(beta_i)
                return new_beta

            gamma_diff = np.abs(len(self._gamma_ip) - num_gamma)
            if gamma_diff != 0:  # if there aren't enough gamma pts,
                # generate a log-uniform sample 1e-4 ≲ γ ≲ 1e-1
                self._gamma_ip.extend([loguniform.rvs(1e-2, 1e-1) for _ in range(gamma_diff)])
            if len(self._beta_ip) < sum(num_beta):  # if there aren't enough beta pts
                self._beta_ip.extend(rand_beta(num_beta[-1]))  # use zeros instead
        self._init_point_rep = self._beta_ip + self._gamma_ip

    def _update_ansatz_params(self):
        if (
            self._ansatz._mixer_operators is None
        ):  # set the initial parameters for iterative algorithm
            self._reps = 1
            self._beta_ip, self._gamma_ip = [], []
            self._ret, self._ground_state_energy = None, None
            mixer_operators = []  # initialise an empty list to append mixer operators to
            self.__ansatz = self._ansatz.initial_state  # an iterative circuit
        else:  # update ansatz parameters
            mixer_operators = self._ansatz._mixer_operators
            if len(mixer_operators) < self._reps:
                mixer_operators += [self._ansatz._mixer_pool[self.__energy_idx]]
        self.construct_circuit(operator=self._cost_operator, mixer_operators=mixer_operators)
        self._num_gamma = self._ansatz._num_cost  # cost operator parameters
        self._num_beta = self._ansatz._num_mixer  # mixer parameters


def energy_grad_operator(
    mixer: OperatorBase, operator: OperatorBase, gamma_k: float
) -> OperatorBase:
    """Computes the operator-valued energy gradient with respect to
    the mixer pool.
    Args:
        mixer: Mixer operator that energy gradient is computed with respect to.
        operator: Qubit operator of the Observable.
        gamma_k: Floating point representing the kth-γ variational parameter used in the operator
            evolution of the ansatz.
    Returns:
        Energy of the hamiltonian of each parameter, and, optionally, the expectation
        converter.
    """
    # compute evolution of operator with kth γ variational parameter
    exp_hc = (gamma_k * operator).exp_i().to_matrix()
    exp_hc_ad = np.transpose((np.conjugate(exp_hc)))  # and its adjoint
    # convert operator to matrix_op if its not already of type MatrixOp
    mat_op = operator.to_matrix_op() if not isinstance(operator, MatrixOp) else operator
    # conjugate the operator, mixer commutator with the operator evolution wrt γ
    energy_grad_op = exp_hc_ad @ (commutator(mat_op, mixer)) @ exp_hc
    return PrimitiveOp(energy_grad_op)
