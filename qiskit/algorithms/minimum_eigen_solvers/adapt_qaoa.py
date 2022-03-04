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
from math import gamma
from pickle import FALSE, NONE
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from .qaoa import QAOA
from qiskit.circuit.library.n_local.adaptqaoa_ansatz import AdaptQAOAAnsatz
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import ComposedOp, OperatorBase, ExpectationBase
from qiskit.opflow.expectations.expectation_factory import ExpectationFactory
from qiskit.opflow.primitive_ops import MatrixOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.quantum_info import Operator
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow.gradients import GradientBase
from qiskit.providers import Backend, BaseBackend
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA

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
        optimizer: Optimizer = None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        gamma_init: Optional[float] = 0.01,
        beta_init: Optional[float] = np.pi / 4,
        mixer_pool: Optional[Union[OperatorBase, QuantumCircuit]] = None,
        mixer_pool_type: Optional[str] = None,
        threshold: Optional[Callable[[int, float], None]] = None,
        max_reps=1,
        optimizer: Optimizer = None,
        initial_state: Optional[QuantumCircuit] = None,
        initial_point: Optional[np.ndarray] = None,
        solution_tolerance: float = 1e-16,
        gradient: Optional[Union[GradientBase, Callable[[Union[np.ndarray, List]], List]]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
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
        self.max_reps = max_reps
        self._reps = 1
        self._initial_state = initial_state
        self._cost_operator = None
        self.__ansatz = False
        self._mixer_pool = mixer_pool
        self._mixer_pool_type = mixer_pool_type
        super().__init__(
            optimizer=optimizer,
            initial_point=initial_point,
            gradient=gradient,
            expectation=expectation,
            include_custom=include_custom,
            max_evals_grouped=max_evals_grouped,
            callback=callback,
            quantum_instance=quantum_instance,
        )

        self.name = "AdaptQAOA"
        self._optimal_parameters = {}
        self.threshold = threshold
        self.solution_tolerance = solution_tolerance

    def _check_operator_ansatz(self, operator: OperatorBase) -> OperatorBase:
        # Initialises the algorithms necessary operators 
        print("158, _check_operator_ansatz!!")
        if operator != self._cost_operator:
            self._cost_operator = operator
            adaptansatz = AdaptQAOAAnsatz(
                operator, self._reps, initial_state=self.initial_state, mixer_operators=self._mixer_pool,
                mixer_pool_type=self._mixer_pool_type          
            )   # AdaptQAOAAnsatz will construct the mixer_pool if one was not already provided.
            adaptansatz._check_configuration()
            self.__mixer_pool = adaptansatz._mixer_operators    # mixer pool composed entirely of OperatorBase types
            if self.mixer_pool is None:     # set the mixer_pool from the ansatz if not already specified.
                self.mixer_pool = adaptansatz.mixer_operators
            self.__ansatz = adaptansatz.initial_state            # store initial state as private variable of _ansatz_step

    def compute_energy_gradient(
        self, 
        mixer: OperatorBase,
        operator: OperatorBase, 
        parameters: dict = None, 
        ansatz: QuantumCircuit = None
        ) -> ComposedOp:
        """Computes the energy gradient of the cost operator wrt the mixer pool at an
            ansatz layer specified by the input 'state' and initial point.

        Returns:
            The mixer operator with the largest energy gradient along with the
            associated energy gradient.
        """

        from qiskit.opflow import commutator
        #TODO: What if mixer/ operator are parameterised circuits??
        if not isinstance(operator, MatrixOp):  
            operator = MatrixOp(Operator(operator.to_matrix()))

        if parameters is None:
            parameters = self._optimal_parameters

        wave_function = ansatz.assign_parameters(self._optimal_parameters)
        # construct expectation operator
        if self.__user_specified_ip:    # if user specified initial pts choose
            gamma_k = self.__user_specified_ip[self._reps]  # next gamma in sequence
        else:
            gamma_k = 0.01  # otherwise use preset value
        exp_hc = (gamma_k * operator).exp_i()  # Use first gamma hyperparameter
        exp_hc_ad = exp_hc.adjoint().to_matrix()
        exp_hc = exp_hc.to_matrix()
        energy_grad_op = (              # compute the energy gradient with respect to the current mixer
            exp_hc_ad
            @ (commutator(operator, mixer).to_matrix())
            @ exp_hc
        )
        energy_grad_op = PrimitiveOp(energy_grad_op)
        expectation = ExpectationFactory.build(
            operator=energy_grad_op,
            backend=self.quantum_instance,
            include_custom=self._include_custom,
        )
        observable_meas = expectation.convert(StateFn(energy_grad_op, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()
        return expect_op

    def _test_mixer_pool(self, operator: OperatorBase):
        energy_gradients = []
        for mixer in self.__mixer_pool:
            expect_op = self.compute_energy_gradient(mixer, operator, ansatz=self.__ansatz)
            # run expectation circuit
            sampled_expect_op = self._circuit_sampler.convert(
                expect_op, params=self._optimal_parameters
            )
            meas = -1j * sampled_expect_op.eval()
            energy_gradients.append(meas)
        # Compute mixer index associated with the largest gradient change
        max_energy_idx = np.argmax(np.real(energy_gradients))
        self.mixer_operators.append(self.mixer_pool[max_energy_idx]) # Append mixer to mixer_operators list
        return np.abs(energy_gradients[max_energy_idx]) # return the norm of the energy gradient

    def compute_minimum_eigenvalue(
        self,
        operator: OperatorBase,
        aux_operators: Optional[List[Optional[OperatorBase]]] = None,
    ):
        """Runs ADAPT-QAOA for each iteration"""
        self._check_operator_ansatz(operator)
        while self._reps <= self.max_reps:  # loop over number of maximum reps
            energy_norm = self._test_mixer_pool(operator=operator)
            logger.info(f"Circuit depth: {self._reps}")
            logger.info(f"Current energy norm: {energy_norm}")
            logger.info(f"Best mixer: {self.mixer_operators[-1]}")
            self.ansatz = self.__ansatz
            if energy_norm < self.threshold:  # Threshold stoppage condition
                break
            result = super().compute_minimum_eigenvalue(
                operator=operator, aux_operators=aux_operators
            )
            self._optimal_parameters = result.optimal_parameters
            if np.abs(result.optimal_value - self.ground_state_energy) <= self.solution_tolerance:
                break     # Stop the algorithm if the ansatz groundstate sufficiently approximates the true groundstate
            
            self._reps += 1
            # logger.info(f"Initial point: {self.initial_point}")
            # logger.info(f"Optimal parameters: {result.optimal_parameters}")
            # logger.info(f"Relative Energy: {result.optimal_value - self.ground_state_energy}")

            # print('-----------------------------------------------------------------')
            # print('Depth: {}, Energy: {}'.format(self._reps-1,np.abs(result.optimal_value - self.ground_state_energy)))
            # # print('Depth: {}, Energy: {}'.format(self._reps-1,np.abs(result.optimal_value - self.ground_state_energy)))
            # print('Depth: {}, Energy: {}'.format(self._reps-1,result.optimal_value))

            # print('Initial points: ', self.initial_point)
            # print('Optimal parameters: ', self._optimal_parameters)
            # print('Best mixer: ', self.mixer_operators[-1])  
            # print()
            # print(self.get_optimal_circuit().decompose().draw())
            # print()
            # print('-----------------------------------------------------------------')
        return result

#TODO: Check that you need ansatz property/ setter
    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the ansatz for the current circuit depth"""
        return self._ansatz

        
    @ansatz.setter
    def ansatz(self, ansatz: Optional[QuantumCircuit]):
        """Sets the ansatz.

        Args:
            ansatz: The parameterized circuit used as an ansatz.
            If None is passed, RealAmplitudes is used by default.

        """
        if ansatz is None:
            ansatz = RealAmplitudes()  

        if (self.__ansatz == ansatz):
            ansatz = AdaptQAOAAnsatz(   # update the ansatz to include the optimal mixers
                cost_operator=self._cost_operator,
                initial_state=self._initial_state,
                mixer_operators=self.mixer_operators,
                mixer_pool_type=self.mixer_pool_type,          
                name=self.name
            )
            ansatz._build()
            if hasattr(ansatz,'_num_mixer'):
                num_beta, num_gamma =  ansatz._num_mixer, ansatz._num_cost
                points = self._update_initial_points(
                                                    num_gamma = num_gamma, 
                                                    num_beta = num_beta
                                                    )
                ansatz.initial_point = points
                self.__ansatz = ansatz

#TODO: Make a callback to initial_point here; the user should be able to specify the ordered list of initial points with operator/ circuit mixer variations
        self._ansatz = ansatz
        self._ansatz_params = list(ansatz.parameters)


    @property
    def mixer_pool(self) -> List:
        """Creates the mixer pool if not already defined

        Returns:
            List of mixers that make up the mixer pool.

        Raises:
            AttributeError: If operator and thus num_qubits has not yet been defined.
        """
        return self._mixer_pool

    @mixer_pool.setter
    def mixer_pool(self, mixer_pool: List) -> None:
        """
        Args:
            mixer_pool: A list of operators that define the pool of operators that
            the eigensolver may drawn an optimal mixer solutions from.        
        """

        self._mixer_pool = mixer_pool

    @property
    def mixer_operators(self) -> List:
        if not hasattr(self,'_mixer_operators'):
            self._mixer_operators = []
        return self._mixer_operators

    @mixer_operators.setter
    def mixer_operators(self, mixer_operators) -> List:
        self._mixer_operators = mixer_operators

    @property
    def ground_state_energy(self) -> float:
        """Returns the ground state energy of the cost operator
            Returns:
                Float: The problem Hamiltonian ground state energy.
        
        """
        if self._cost_operator is not None:
            return min(np.real(np.linalg.eig(self._cost_operator.to_matrix())[0]))
        else:
            return None

    @property
    def threshold(self) -> float:
        """Returns the gradient threshold if specified, otherwise defaults to zero.
            Returns:
                Float: Specifies the minimum value, or 'threshold' of the energy gradient
                with respect to the mixer pool that will stop the eigensolver. Defaults to
                zero if None is provided.
            """
        if self._threshold is None:
            self._threshold = 0
        return self._threshold      

    @threshold.setter
    def threshold(self, threshold) -> Optional[Callable[[int, float], None]]:
        """Sets the threshold
        Args:
            threshold: The threshold of the energy gradient with respect to the mixer pool."""
        self._threshold = threshold

    @property
    def initial_point(self) -> np.array:
        """Returns the user specified initial points 

        Returns:
            Numpy array of intial points for iterative layers.

        """
        if self.mixer_operators:
            return self.__ansatz.initial_point
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point) -> Optional[np.ndarray]:
        """
        Specifies initial point.
        Raises:
            AttributeError: If number of initial points dont match the current circuit depth.
        """
        if initial_point is None:
            self.__user_specified_ip = False
        else:
            if not isinstance(initial_point, list):
                initial_point = list(initial_point)
            self.__user_specified_ip = initial_point
            if len(initial_point) != 2 * self.max_reps:
                raise AttributeError(
                    "The number of user specified initial points ({}) must "
                    "be equal to twice the maximum ansatz depth ({})".format(       #TODO: Update this
                        len(initial_point), 2 * self.max_reps
                    )
                )
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
            optimizer = COBYLA(maxiter=10e+9, tol=1e-30)

        optimizer.set_max_evals_grouped(self.max_evals_grouped)
        self._optimizer = optimizer

    @property 
    def mixer_pool_type(self) -> str:
        """Returns the class of mixer pool type.
        Returns:
            str: mixer pool type"""
        return self._mixer_pool_type

    @mixer_pool_type.setter
    def mixer_pool_type(self, mixer_pool_type: str):
        """Sets the mixer pool type.
        Args:
            mixer_pool_type: A string that represents the preset mixer pool classes.
        Raises:
            KeyError: If mixer pool type is not in the set of presets.
        """
        self._mixer_pool_type = mixer_pool_type

    def _update_initial_points(self, num_gamma: int, num_beta: list):
        if self.__user_specified_ip:
           gamma_ip = self.__user_specified_ip[:self._reps]
           beta_ip = self.__user_specified_ip[self.max_reps: self.max_reps + self._reps]

        else:   # set initial value for gamma according to https://arxiv.org/abs/2005.10258
            gamma_ip, beta_ip  = [0.01] * num_gamma, [-0.25 * np.pi] * len(num_beta)

        updated_points = []
        for rep, nbeta in enumerate(num_beta):
            updated_points.extend(beta_ip[rep : rep + 1] * nbeta)
        updated_points += gamma_ip
        return updated_points
        