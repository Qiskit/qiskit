# -*- coding: utf-8 -*-

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

"""The Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1304.3061
"""

from typing import Optional, List, Callable, Union, Dict, Any
import logging
import warnings
from time import time
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.operators import (OperatorBase, ExpectationBase, ExpectationFactory, StateFn,
                                   CircuitStateFn, LegacyBaseOperator, ListOp, I, CircuitSampler)
from qiskit.aqua.components.optimizers import Optimizer, SLSQP
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.utils.validation import validate_min
from ..vq_algorithm import VQAlgorithm, VQResult
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult

logger = logging.getLogger(__name__)

# disable check for var_forms, optimizer setter because of pylint bug
# pylint: disable=no-member


class VQE(VQAlgorithm, MinimumEigensolver):
    r"""The Variational Quantum Eigensolver algorithm.

    `VQE <https://arxiv.org/abs/1304.3061>`__ is a hybrid algorithm that uses a
    variational technique and interleaves quantum and classical computations in order to find
    the minimum eigenvalue of the Hamiltonian :math:`H` of a given system.

    An instance of VQE requires defining two algorithmic sub-components:
    a trial state (ansatz) from Aqua's :mod:`~qiskit.aqua.components.variational_forms`, and one
    of the classical :mod:`~qiskit.aqua.components.optimizers`. The ansatz is varied, via its set
    of parameters, by the optimizer, such that it works towards a state, as determined by the
    parameters applied to the variational form, that will result in the minimum expectation value
    being measured of the input operator (Hamiltonian).

    An optional array of parameter values, via the *initial_point*, may be provided as the
    starting point for the search of the minimum eigenvalue. This feature is particularly useful
    such as when there are reasons to believe that the solution point is close to a particular
    point.  As an example, when building the dissociation profile of a molecule,
    it is likely that using the previous computed optimal solution as the starting
    initial point for the next interatomic distance is going to reduce the number of iterations
    necessary for the variational algorithm to converge.  Aqua provides an
    `initial point tutorial <https://github.com/Qiskit/qiskit-tutorials-community/blob/master
    /chemistry/h2_vqe_initial_point.ipynb>`__ detailing this use case.

    The length of the *initial_point* list value must match the number of the parameters
    expected by the variational form being used. If the *initial_point* is left at the default
    of ``None``, then VQE will look to the variational form for a preferred value, based on its
    given initial state. If the variational form returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the variational form provides ``None`` as the lower bound, then VQE
    will default it to :math:`-2\pi`; similarly, if the variational form returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.

    .. note::

        The VQE stores the parameters of ``var_form`` sorted by name to map the values
        provided by the optimizer to the circuit. This is done to ensure reproducible results,
        for example such that running the optimization twice with same random seeds yields the
        same result. Also, the ``optimal_point`` of the result object can be used as initial
        point of another VQE run by passing it as ``initial_point`` to the initializer.

    """

    def __init__(self,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 var_form: Optional[Union[QuantumCircuit, VariationalForm]] = None,
                 optimizer: Optional[Optimizer] = None,
                 initial_point: Optional[np.ndarray] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False,
                 max_evals_grouped: int = 1,
                 aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                             LegacyBaseOperator]]]] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        """

        Args:
            operator: Qubit operator of the Observable
            var_form: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            expectation: The Expectation converter for taking the average value of the
                Observable over the var_form state function. When ``None`` (the default) an
                :class:`~qiskit.aqua.operators.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time.
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                eigenstate of the minimum eigenvalue main result and their expectation values
                returned. For instance in chemistry these can be dipole operators, total particle
                count operators so we can get values for these at the ground state.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                variational form, the evaluated mean and the evaluated standard deviation.`
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('max_evals_grouped', max_evals_grouped, 1)
        if var_form is None:
            var_form = RealAmplitudes()

        if optimizer is None:
            optimizer = SLSQP()

        # set the initial point to the preferred parameters of the variational form
        if initial_point is None and hasattr(var_form, 'preferred_init_points'):
            initial_point = var_form.preferred_init_points

        self._max_evals_grouped = max_evals_grouped
        self._circuit_sampler = None  # type: Optional[CircuitSampler]
        self._expectation = expectation
        self._user_valid_expectation = self._expectation is not None
        self._include_custom = include_custom
        self._expect_op = None
        self._operator = None

        super().__init__(var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=self._energy_evaluation,
                         initial_point=initial_point,
                         quantum_instance=quantum_instance)
        self._ret = None  # type: Dict[str, Any]
        self._eval_time = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback

        if operator is not None:
            self.operator = operator
        self.aux_operators = aux_operators

        self._eval_count = 0
        logger.info(self.print_settings())

    @property
    def operator(self) -> Optional[OperatorBase]:
        """ Returns operator """
        return self._operator

    @operator.setter
    def operator(self, operator: Union[OperatorBase, LegacyBaseOperator]) -> None:
        """ set operator """
        if isinstance(operator, LegacyBaseOperator):
            operator = operator.to_opflow()
        self._operator = operator
        self._expect_op = None
        self._check_operator_varform()
        # Expectation was not passed by user, try to create one
        if not self._user_valid_expectation:
            self._try_set_expectation_value_from_factory()

    def _try_set_expectation_value_from_factory(self) -> None:
        if self.operator is not None and self.quantum_instance is not None:
            self._set_expectation(ExpectationFactory.build(operator=self.operator,
                                                           backend=self.quantum_instance,
                                                           include_custom=self._include_custom))

    def _set_expectation(self, exp: ExpectationBase) -> None:
        self._expectation = exp
        self._user_valid_expectation = False
        self._expect_op = None

    @QuantumAlgorithm.quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance, BaseBackend]) -> None:
        """ set quantum_instance """
        super(VQE, self.__class__).quantum_instance.__set__(self, quantum_instance)

        self._circuit_sampler = CircuitSampler(self._quantum_instance)
        # Expectation was not passed by user, try to create one
        if not self._user_valid_expectation:
            self._try_set_expectation_value_from_factory()

    @property
    def expectation(self) -> ExpectationBase:
        """ The expectation value algorithm used to construct the expectation measurement from
        the observable. """
        return self._expectation

    @expectation.setter
    def expectation(self, exp: ExpectationBase) -> None:
        self._set_expectation(exp)
        self._user_valid_expectation = self._expectation is not None

    @property
    def aux_operators(self) -> Optional[List[Optional[OperatorBase]]]:
        """ Returns aux operators """
        return self._aux_operators

    @aux_operators.setter
    def aux_operators(self,
                      aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                                  LegacyBaseOperator]]]]) -> None:
        """ Set aux operators """
        # We need to handle the array entries being Optional i.e. having value None
        self._aux_op_nones = None
        if isinstance(aux_operators, list):
            self._aux_op_nones = [op is None for op in aux_operators]
            zero_op = I.tensorpower(self.operator.num_qubits) * 0.0
            converted = [op.to_opflow() if op else zero_op for op in aux_operators]
            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            converted = [zero_op if op == 0 else op for op in converted]
            aux_operators = ListOp(converted)
        elif isinstance(aux_operators, LegacyBaseOperator):
            aux_operators = [aux_operators.to_opflow()]
        self._aux_operators = aux_operators

    def _check_operator_varform(self):
        """Check that the number of qubits of operator and variational form match."""
        if self.operator is not None and self.var_form is not None:
            if self.operator.num_qubits != self.var_form.num_qubits:
                # try to set the number of qubits on the variational form, if possible
                try:
                    self.var_form.num_qubits = self.operator.num_qubits
                    self._var_form_params = sorted(self.var_form.parameters, key=lambda p: p.name)
                except AttributeError:
                    raise AquaError("The number of qubits of the variational form does not match "
                                    "the operator, and the variational form does not allow setting "
                                    "the number of qubits using `num_qubits`.")

    @VQAlgorithm.optimizer.setter  # type: ignore
    def optimizer(self, optimizer: Optimizer):
        """ Sets optimizer """
        super(VQE, self.__class__).optimizer.__set__(self, optimizer)  # type: ignore
        if optimizer is not None:
            optimizer.set_max_evals_grouped(self._max_evals_grouped)

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    def print_settings(self):
        """
        Preparing the setting of VQE into a string.

        Returns:
            str: the formatted setting of VQE
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.__class__.__name__)
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        if hasattr(self._var_form, 'setting'):
            ret += "{}".format(self._var_form.setting)
        elif hasattr(self._var_form, 'print_settings'):
            ret += "{}".format(self._var_form.print_settings())
        elif isinstance(self._var_form, QuantumCircuit):
            ret += "var_form is a custom circuit"
        else:
            ret += "var_form has not been set"
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

    def construct_expectation(self,
                              parameter: Union[List[float], List[Parameter], np.ndarray]
                              ) -> OperatorBase:
        r"""
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Args:
            parameter: Parameters for the ansatz circuit.

        Returns:
            The Operator equalling the measurement of the ansatz :class:`StateFn` by the
            Observable's expectation :class:`StateFn`.

        Raises:
            AquaError: If no operator has been provided.
        """
        if self.operator is None:
            raise AquaError("The operator was never provided.")

        # ensure operator and varform are compatible
        self._check_operator_varform()

        if isinstance(self.var_form, QuantumCircuit):
            param_dict = dict(zip(self._var_form_params, parameter))  # type: Dict
            wave_function = self.var_form.assign_parameters(param_dict)
        else:
            wave_function = self.var_form.construct_circuit(parameter)

        # Expectation was never created, try to create one
        if self._expectation is None:
            self._try_set_expectation_value_from_factory()

        # If setting the expectation failed, raise an Error:
        if self._expectation is None:
            raise AquaError('No expectation set and could not automatically set one, please '
                            'try explicitly setting an expectation or specify a backend so it '
                            'can be chosen automatically.')

        observable_meas = self.expectation.convert(StateFn(self.operator, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        return observable_meas.compose(ansatz_circuit_op).reduce()

    def construct_circuit(self,
                          parameter: Union[List[float], List[Parameter], np.ndarray]
                          ) -> List[QuantumCircuit]:
        """Return the circuits used to compute the expectation value.

        Args:
            parameter: Parameters for the ansatz circuit.

        Returns:
            A list of the circuits used to compute the expectation value.
        """
        expect_op = self.construct_expectation(parameter).to_circuit_op()

        circuits = []

        # recursively extract circuits
        def extract_circuits(op):
            if isinstance(op, CircuitStateFn):
                circuits.append(op.primitive)
            elif isinstance(op, ListOp):
                for op_i in op.oplist:
                    extract_circuits(op_i)

        extract_circuits(expect_op)

        return circuits

    def supports_aux_operators(self) -> bool:
        return True

    def _run(self) -> 'VQEResult':
        """Run the algorithm to compute the minimum eigenvalue.

        Returns:
            The result of the VQE algorithm as ``VQEResult``.

        Raises:
            AquaError: Wrong setting of operator and backend.
        """
        if self.operator is None:
            raise AquaError("The operator was never provided.")

        self._check_operator_varform()

        self._quantum_instance.circuit_summary = True

        self._eval_count = 0
        vqresult = self.find_minimum(initial_point=self.initial_point,
                                     var_form=self.var_form,
                                     cost_fn=self._energy_evaluation,
                                     optimizer=self.optimizer)

        # TODO remove all former dictionary logic
        self._ret = {}
        self._ret['num_optimizer_evals'] = vqresult.optimizer_evals
        self._ret['min_val'] = vqresult.optimal_value
        self._ret['opt_params'] = vqresult.optimal_point
        self._ret['eval_time'] = vqresult.optimizer_time
        self._ret['opt_params_dict'] = vqresult.optimal_parameters

        if self._ret['num_optimizer_evals'] is not None and \
                self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        result = VQEResult()
        result.combine(vqresult)
        result.eigenvalue = vqresult.optimal_value + 0j
        result.eigenstate = self.get_optimal_vector()

        self._ret['energy'] = self.get_optimal_cost()
        self._ret['eigvals'] = np.asarray([self._ret['energy']])
        self._ret['eigvecs'] = np.asarray([result.eigenstate])

        if self.aux_operators:
            self._eval_aux_ops()
            # TODO remove when ._ret is deprecated
            result.aux_operator_eigenvalues = self._ret['aux_ops'][0]

        result.cost_function_evals = self._eval_count

        return result

    def _eval_aux_ops(self, threshold=1e-12):
        # Create new CircuitSampler to avoid breaking existing one's caches.
        sampler = CircuitSampler(self.quantum_instance)

        aux_op_meas = self.expectation.convert(StateFn(self.aux_operators, is_measurement=True))
        aux_op_expect = aux_op_meas.compose(CircuitStateFn(self.get_optimal_circuit()))
        values = np.real(sampler.convert(aux_op_expect).eval())

        # Discard values below threshold
        aux_op_results = (values * (np.abs(values) > threshold))
        # Deal with the aux_op behavior where there can be Nones or Zero qubit Paulis in the list
        self._ret['aux_ops'] = [None if is_none else [result]
                                for (is_none, result) in zip(self._aux_op_nones, aux_op_results)]
        self._ret['aux_ops'] = np.array([self._ret['aux_ops']])

    def compute_minimum_eigenvalue(
            self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                        LegacyBaseOperator]]]] = None
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        return self._run()

    def _energy_evaluation(self, parameters: Union[List[float], np.ndarray]
                           ) -> Union[float, List[float]]:
        """Evaluate energy at given parameters for the variational form.

        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            parameters: The parameters for the variational form.

        Returns:
            Energy of the hamiltonian of each parameter.


        Raises:
            RuntimeError: If the variational form has no parameters.
        """
        if not self._expect_op:
            self._expect_op = self.construct_expectation(self._var_form_params)

        num_parameters = self.var_form.num_parameters
        if self._var_form.num_parameters == 0:
            raise RuntimeError('The var_form cannot have 0 parameters.')

        parameter_sets = np.reshape(parameters, (-1, num_parameters))
        # Create dict associating each parameter with the lists of parameterization values for it
        param_bindings = dict(zip(self._var_form_params,
                                  parameter_sets.transpose().tolist()))  # type: Dict

        start_time = time()
        sampled_expect_op = self._circuit_sampler.convert(self._expect_op, params=param_bindings)
        means = np.real(sampled_expect_op.eval())

        if self._callback is not None:
            variance = np.real(self._expectation.compute_variance(sampled_expect_op))
            estimator_error = np.sqrt(variance / self.quantum_instance.run_config.shots)
            for i, param_set in enumerate(parameter_sets):
                self._eval_count += 1
                self._callback(self._eval_count, param_set, means[i], estimator_error[i])
        else:
            self._eval_count += len(means)

        end_time = time()
        logger.info('Energy evaluation returned %s - %.5f (ms), eval count: %s',
                    means, (end_time - start_time) * 1000, self._eval_count)

        return means if len(means) > 1 else means[0]

    def get_optimal_cost(self) -> float:
        """Get the minimal cost or energy found by the VQE."""
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the "
                            "algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self) -> QuantumCircuit:
        """Get the circuit with the optimal parameters."""
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the "
                            "algorithm to find optimal params.")
        if isinstance(self.var_form, VariationalForm):
            return self._var_form.construct_circuit(self._ret['opt_params'])
        return self.var_form.assign_parameters(self._ret['opt_params_dict'])

    def get_optimal_vector(self) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the optimal circuit. """
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the "
                            "algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector']

    @property
    def optimal_params(self) -> List[float]:
        """The optimal parameters for the variational form."""
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']


class VQEResult(VQResult, MinimumEigensolverResult):
    """ VQE Result."""

    @property
    def cost_function_evals(self) -> int:
        """ Returns number of cost optimizer evaluations """
        return self.get('cost_function_evals')

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """ Sets number of cost function evaluations """
        self.data['cost_function_evals'] = value

    def __getitem__(self, key: object) -> object:
        if key == 'eval_count':
            warnings.warn('eval_count deprecated, use cost_function_evals property.',
                          DeprecationWarning)
            return super().__getitem__('cost_function_evals')

        try:
            return VQResult.__getitem__(self, key)
        except KeyError:
            return MinimumEigensolverResult.__getitem__(self, key)
