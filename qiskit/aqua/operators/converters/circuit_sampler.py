# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" CircuitSampler Class """

from typing import Optional, Dict, List, Union, cast, Any
import logging
from functools import partial

from qiskit.providers import BaseBackend
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit import QiskitError
from qiskit.aqua import QuantumInstance
from qiskit.aqua.utils.backend_utils import is_aer_provider, is_statevector_backend
from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.operator_globals import Zero
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.state_fns.state_fn import StateFn
from qiskit.aqua.operators.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.aqua.operators.converters.converter_base import ConverterBase

logger = logging.getLogger(__name__)


class CircuitSampler(ConverterBase):
    """
    The CircuitSampler traverses an Operator and converts any CircuitStateFns into
    approximations of the state function by a DictStateFn or VectorStateFn using a quantum
    backend. Note that in order to approximate the value of the CircuitStateFn, it must 1) send
    state function through a depolarizing channel, which will destroy all phase information and
    2) replace the sampled frequencies with **square roots** of the frequency, rather than the raw
    probability of sampling (which would be the equivalent of sampling the **square** of the
    state function, per the Born rule.

    The CircuitSampler aggressively caches transpiled circuits to handle re-parameterization of
    the same circuit efficiently. If you are converting multiple different Operators,
    you are better off using a different CircuitSampler for each Operator to avoid cache thrashing.
    """

    def __init__(self,
                 backend: Union[BaseBackend, QuantumInstance] = None,
                 statevector: Optional[bool] = None,
                 param_qobj: bool = False,
                 attach_results: bool = False) -> None:
        """
        Args:
            backend: The quantum backend or QuantumInstance to use to sample the circuits.
            statevector: If backend is a statevector backend, whether to replace the
                CircuitStateFns with DictStateFns (from the counts) or VectorStateFns (from the
                statevector). ``None`` will set this argument automatically based on the backend.
            param_qobj: (TODO, not yet available) Whether to use Aer's parameterized Qobj
                capability to avoid re-assembling the circuits.
            attach_results: Whether to attach the data from the backend ``Results`` object for
                a given ``CircuitStateFn``` to an ``execution_results`` field added the converted
                ``DictStateFn`` or ``VectorStateFn``.

        Raises:
            ValueError: Set statevector or param_qobj True when not supported by backend.
        """
        self._quantum_instance = backend if isinstance(backend, QuantumInstance) else\
            QuantumInstance(backend=backend)
        self._statevector = statevector if statevector is not None \
            else self.quantum_instance.is_statevector
        self._param_qobj = param_qobj
        self._attach_results = attach_results

        self._check_quantum_instance_and_modes_consistent()

        # Object state variables
        self._last_op = None
        self._reduced_op_cache = None
        self._circuit_ops_cache = {}  # type: Dict[int, CircuitStateFn]
        self._transpiled_circ_cache = None  # type: Optional[List[Any]]
        self._transpile_before_bind = True
        self._binding_mappings = None

    def _check_quantum_instance_and_modes_consistent(self) -> None:
        """ Checks whether the statevector and param_qobj settings are compatible with the
        backend

        Raises:
            ValueError: statevector or param_qobj are True when not supported by backend.
        """
        if self._statevector and not is_statevector_backend(self.quantum_instance.backend):
            raise ValueError('Statevector mode for circuit sampling requires statevector '
                             'backend, not {}.'.format(self.quantum_instance.backend))

        if self._param_qobj and not is_aer_provider(self.quantum_instance.backend):
            raise ValueError('Parameterized Qobj mode requires Aer '
                             'backend, not {}.'.format(self.quantum_instance.backend))

    @property
    def backend(self) -> BaseBackend:
        """ Returns the backend.

        Returns:
             The backend used by the CircuitSampler
        """
        return self.quantum_instance.backend

    @backend.setter
    def backend(self, backend: BaseBackend):
        """ Sets backend without additional configuration. """
        self.set_backend(backend)

    def set_backend(self, backend: BaseBackend, **kwargs) -> None:
        """ Sets backend with configuration.

        Raises:
            ValueError: statevector or param_qobj are True when not supported by backend.
        """
        self.quantum_instance = QuantumInstance(backend)
        self.quantum_instance.set_config(**kwargs)

    @property
    def quantum_instance(self) -> QuantumInstance:
        """ Returns the quantum instance.

        Returns:
             The QuantumInstance used by the CircuitSampler
        """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance, BaseBackend]) -> None:
        """ Sets the QuantumInstance.

        Raises:
            ValueError: statevector or param_qobj are True when not supported by backend.
        """
        if isinstance(quantum_instance, BaseBackend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance
        self._check_quantum_instance_and_modes_consistent()

    # pylint: disable=arguments-differ
    def convert(self,
                operator: OperatorBase,
                params: Optional[Dict[Union[ParameterExpression, ParameterVector],
                                      Union[float, List[float], List[List[float]]]]] = None
                ) -> OperatorBase:
        r"""
        Converts the Operator to one in which the CircuitStateFns are replaced by
        DictStateFns or VectorStateFns. Extracts the CircuitStateFns out of the Operator,
        caches them, calls ``sample_circuits`` below to get their converted replacements,
        and replaces the CircuitStateFns in operator with the replacement StateFns.

        Args:
            operator: The Operator to convert
            params: A dictionary mapping parameters to either single binding values or lists of
                binding values. The dictionary can also contain pairs of ParameterVectors with
                lists of parameters or lists of lists of parameters to bind to them.

        Returns:
            The converted Operator with CircuitStateFns replaced by DictStateFns or VectorStateFns.
        """
        if self._last_op is None or id(operator) != id(self._last_op):
            # Clear caches
            self._last_op = operator
            self._reduced_op_cache = None
            self._circuit_ops_cache = None
            self._transpiled_circ_cache = None
            self._transpile_before_bind = True

        if not self._reduced_op_cache:
            operator_dicts_replaced = operator.to_circuit_op()
            self._reduced_op_cache = operator_dicts_replaced.reduce()

        if not self._circuit_ops_cache:
            self._circuit_ops_cache = {}
            self._extract_circuitstatefns(self._reduced_op_cache)

        if params:
            p_0 = list(params.values())[0]  # type: ignore
            num_parameterizations = len(cast(List, p_0))
            param_bindings = [{param: value_list[i]  # type: ignore
                               for (param, value_list) in params.items()}
                              for i in range(num_parameterizations)]
        else:
            param_bindings = None
            num_parameterizations = 1

        # Don't pass circuits if we have in the cache, the sampling function knows to use the cache
        circs = list(self._circuit_ops_cache.values()) if not self._transpiled_circ_cache else None
        p_b = cast(List[Dict[ParameterExpression, List[float]]], param_bindings)
        sampled_statefn_dicts = self.sample_circuits(circuit_sfns=circs,
                                                     param_bindings=p_b)

        def replace_circuits_with_dicts(operator, param_index=0):
            if isinstance(operator, CircuitStateFn):
                return sampled_statefn_dicts[id(operator)][param_index]
            elif isinstance(operator, ListOp):
                return operator.traverse(partial(replace_circuits_with_dicts,
                                                 param_index=param_index))
            else:
                return operator

        if params:
            return ListOp([replace_circuits_with_dicts(self._reduced_op_cache, param_index=i)
                           for i in range(num_parameterizations)])
        else:
            return replace_circuits_with_dicts(self._reduced_op_cache, param_index=0)

    def _extract_circuitstatefns(self, operator: OperatorBase) -> None:
        r"""
        Recursively extract the ``CircuitStateFns`` contained in operator into the
        ``_circuit_ops_cache`` field.
        """
        if isinstance(operator, CircuitStateFn):
            self._circuit_ops_cache[id(operator)] = operator
        elif isinstance(operator, ListOp):
            for op in operator.oplist:
                self._extract_circuitstatefns(op)

    def sample_circuits(self,
                        circuit_sfns: Optional[List[CircuitStateFn]] = None,
                        param_bindings: Optional[List[Dict[ParameterExpression,
                                                           List[float]]]] = None
                        ) -> Dict[int, Union[StateFn, List[StateFn]]]:
        r"""
        Samples the CircuitStateFns and returns a dict associating their ``id()`` values to their
        replacement DictStateFn or VectorStateFn. If param_bindings is provided,
        the CircuitStateFns are broken into their parameterizations, and a list of StateFns is
        returned in the dict for each circuit ``id()``. Note that param_bindings is provided here
        in a different format than in ``convert``, and lists of parameters within the dict is not
        supported, and only binding dicts which are valid to be passed into Terra can be included
        in this list.

        Args:
            circuit_sfns: The list of CircuitStateFns to sample.
            param_bindings: The parameterizations to bind to each CircuitStateFn.

        Returns:
            The dictionary mapping ids of the CircuitStateFns to their replacement StateFns.
        """
        if circuit_sfns or not self._transpiled_circ_cache:
            if self._statevector:
                circuits = [op_c.to_circuit(meas=False) for op_c in circuit_sfns]
            else:
                circuits = [op_c.to_circuit(meas=True) for op_c in circuit_sfns]

            try:
                self._transpiled_circ_cache = self.quantum_instance.transpile(circuits)
            except QiskitError:
                logger.debug(r'CircuitSampler failed to transpile circuits with unbound '
                             r'parameters. Attempting to transpile only when circuits are bound '
                             r'now, but this can hurt performance due to repeated transpilation.')
                self._transpile_before_bind = False
                self._transpiled_circ_cache = circuits
        else:
            circuit_sfns = list(self._circuit_ops_cache.values())

        if param_bindings is not None:
            if self._param_qobj:
                ready_circs = self._transpiled_circ_cache
                self._prepare_parameterized_run_config(param_bindings)
            else:
                ready_circs = [circ.assign_parameters(binding)
                               for circ in self._transpiled_circ_cache
                               for binding in param_bindings]
        else:
            ready_circs = self._transpiled_circ_cache

        results = self.quantum_instance.execute(ready_circs,
                                                had_transpiled=self._transpile_before_bind)

        # Wipe parameterizations, if any
        # self.quantum_instance._run_config.parameterizations = None

        sampled_statefn_dicts = {}
        for i, op_c in enumerate(circuit_sfns):
            # Taking square root because we're replacing a statevector
            # representation of probabilities.
            reps = len(param_bindings) if param_bindings is not None else 1
            c_statefns = []
            for j in range(reps):
                circ_index = (i * reps) + j
                circ_results = results.data(circ_index)

                if 'expval_measurement' in circ_results.get('snapshots', {}).get(
                        'expectation_value', {}):
                    snapshot_data = results.data(circ_index)['snapshots']
                    avg = snapshot_data['expectation_value']['expval_measurement'][0]['value']
                    if isinstance(avg, (list, tuple)):
                        # Aer versions before 0.4 use a list snapshot format
                        # which must be converted to a complex value.
                        avg = avg[0] + 1j * avg[1]
                    # Will be replaced with just avg when eval is called later
                    num_qubits = circuit_sfns[0].num_qubits
                    result_sfn = (Zero ^ num_qubits).adjoint() * avg
                elif self._statevector:
                    result_sfn = StateFn(op_c.coeff * results.get_statevector(circ_index))
                else:
                    shots = self.quantum_instance._run_config.shots
                    result_sfn = StateFn({b: (v / shots) ** 0.5 * op_c.coeff
                                          for (b, v) in results.get_counts(circ_index).items()})
                if self._attach_results:
                    result_sfn.execution_results = circ_results
                c_statefns.append(result_sfn)
            sampled_statefn_dicts[id(op_c)] = c_statefns
        return sampled_statefn_dicts

    # TODO build Aer re-parameterized Qobj.
    def _prepare_parameterized_run_config(self,
                                          param_bindings: List[Dict[Any, List[float]]]) -> None:
        raise NotImplementedError
        # Wipe parameterizations, if any
        # self.quantum_instance._run_config.parameterizations = None

        # if not self._binding_mappings:
        #     phony_binding = {k: str(k) for k in param_bindings[0].keys()}
        #     phony_bound_circuits = [circ.bind_parameters(phony_binding)
        #                             for circ in self._transpiled_circ_cache]
        #     qobj = self.quantum_instance.assemble(phony_bound_circuits)
        #     # for circ in qobj:
        #     #     mapping = None
        #     #     for
        #
        # # self.quantum_instance._run_config.parameterizations = [params_circ]
