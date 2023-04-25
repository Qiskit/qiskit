# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""CircuitSampler Class"""


import logging
from functools import partial
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from qiskit import QiskitError
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.opflow.converters.converter_base import ConverterBase
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.dict_state_fn import DictStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.providers import Backend
from qiskit.utils.backend_utils import is_aer_provider, is_statevector_backend
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.utils.deprecation import deprecate_func

logger = logging.getLogger(__name__)


class CircuitSampler(ConverterBase):
    """
    Deprecated: The CircuitSampler traverses an Operator and converts any CircuitStateFns into
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

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        backend: Union[Backend, QuantumInstance],
        statevector: Optional[bool] = None,
        param_qobj: bool = False,
        attach_results: bool = False,
        caching: str = "last",
    ) -> None:
        """
        Args:
            backend: The quantum backend or QuantumInstance to use to sample the circuits.
            statevector: If backend is a statevector backend, whether to replace the
                CircuitStateFns with DictStateFns (from the counts) or VectorStateFns (from the
                statevector). ``None`` will set this argument automatically based on the backend.
            attach_results: Whether to attach the data from the backend ``Results`` object for
                a given ``CircuitStateFn``` to an ``execution_results`` field added the converted
                ``DictStateFn`` or ``VectorStateFn``.
            param_qobj: Whether to use Aer's parameterized Qobj capability to avoid re-assembling
                the circuits.
            caching: The caching strategy. Can be `'last'` (default) to store the last operator
                that was converted, set to `'all'` to cache all processed operators.

        Raises:
            ValueError: Set statevector or param_qobj True when not supported by backend.
        """
        super().__init__()

        self._quantum_instance = (
            backend if isinstance(backend, QuantumInstance) else QuantumInstance(backend=backend)
        )
        self._statevector = (
            statevector if statevector is not None else self.quantum_instance.is_statevector
        )
        self._param_qobj = param_qobj
        self._attach_results = attach_results

        self._check_quantum_instance_and_modes_consistent()

        # Object state variables
        self._caching = caching
        self._cached_ops: Dict[int, OperatorCache] = {}

        self._last_op: Optional[OperatorBase] = None
        self._reduced_op_cache = None
        self._circuit_ops_cache: Dict[int, CircuitStateFn] = {}
        self._transpiled_circ_cache: Optional[List[Any]] = None
        self._transpiled_circ_templates: Optional[List[Any]] = None
        self._transpile_before_bind = True

    def _check_quantum_instance_and_modes_consistent(self) -> None:
        """Checks whether the statevector and param_qobj settings are compatible with the
        backend

        Raises:
            ValueError: statevector or param_qobj are True when not supported by backend.
        """
        if self._statevector and not is_statevector_backend(self.quantum_instance.backend):
            raise ValueError(
                "Statevector mode for circuit sampling requires statevector "
                "backend, not {}.".format(self.quantum_instance.backend)
            )

        if self._param_qobj and not is_aer_provider(self.quantum_instance.backend):
            raise ValueError(
                "Parameterized Qobj mode requires Aer "
                "backend, not {}.".format(self.quantum_instance.backend)
            )

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance.

        Returns:
             The QuantumInstance used by the CircuitSampler
        """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance, Backend]) -> None:
        """Sets the QuantumInstance.

        Raises:
            ValueError: statevector or param_qobj are True when not supported by backend.
        """
        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance
        self._check_quantum_instance_and_modes_consistent()

    def convert(
        self,
        operator: OperatorBase,
        params: Optional[Dict[Parameter, Union[float, List[float], List[List[float]]]]] = None,
    ) -> OperatorBase:
        r"""
        Converts the Operator to one in which the CircuitStateFns are replaced by
        DictStateFns or VectorStateFns. Extracts the CircuitStateFns out of the Operator,
        caches them, calls ``sample_circuits`` below to get their converted replacements,
        and replaces the CircuitStateFns in operator with the replacement StateFns.

        Args:
            operator: The Operator to convert
            params: A dictionary mapping parameters to either single binding values or lists of
                binding values.

        Returns:
            The converted Operator with CircuitStateFns replaced by DictStateFns or VectorStateFns.
        Raises:
            OpflowError: if extracted circuits are empty.
        """
        # check if the operator should be cached
        op_id = operator.instance_id
        # op_id = id(operator)
        if op_id not in self._cached_ops.keys():
            # delete cache if we only want to cache one operator
            if self._caching == "last":
                self.clear_cache()

            # convert to circuit and reduce
            operator_dicts_replaced = operator.to_circuit_op()
            self._reduced_op_cache = operator_dicts_replaced.reduce()

            # extract circuits
            self._circuit_ops_cache = {}
            self._extract_circuitstatefns(self._reduced_op_cache)
            if not self._circuit_ops_cache:
                raise OpflowError(
                    "Circuits are empty. "
                    "Check that the operator is an instance of CircuitStateFn or its ListOp."
                )
            self._transpiled_circ_cache = None
            self._transpile_before_bind = True
        else:
            # load the cached circuits
            self._reduced_op_cache = self._cached_ops[op_id].reduced_op_cache
            self._circuit_ops_cache = self._cached_ops[op_id].circuit_ops_cache
            self._transpiled_circ_cache = self._cached_ops[op_id].transpiled_circ_cache
            self._transpile_before_bind = self._cached_ops[op_id].transpile_before_bind
            self._transpiled_circ_templates = self._cached_ops[op_id].transpiled_circ_templates

        return_as_list = False
        if params is not None and len(params.keys()) > 0:
            p_0 = list(params.values())[0]
            if isinstance(p_0, (list, np.ndarray)):
                num_parameterizations = len(p_0)
                param_bindings = [
                    {param: value_list[i] for param, value_list in params.items()}  # type: ignore
                    for i in range(num_parameterizations)
                ]
                return_as_list = True
            else:
                num_parameterizations = 1
                param_bindings = [params]

        else:
            param_bindings = None
            num_parameterizations = 1

        # Don't pass circuits if we have in the cache, the sampling function knows to use the cache
        circs = list(self._circuit_ops_cache.values()) if not self._transpiled_circ_cache else None
        p_b = cast(List[Dict[Parameter, float]], param_bindings)
        sampled_statefn_dicts = self.sample_circuits(circuit_sfns=circs, param_bindings=p_b)

        def replace_circuits_with_dicts(operator, param_index=0):
            if isinstance(operator, CircuitStateFn):
                return sampled_statefn_dicts[id(operator)][param_index]
            elif isinstance(operator, ListOp):
                return operator.traverse(
                    partial(replace_circuits_with_dicts, param_index=param_index)
                )
            else:
                return operator

        # store the operator we constructed, if it isn't stored already
        if op_id not in self._cached_ops.keys():
            op_cache = OperatorCache()
            op_cache.reduced_op_cache = self._reduced_op_cache
            op_cache.circuit_ops_cache = self._circuit_ops_cache
            op_cache.transpiled_circ_cache = self._transpiled_circ_cache
            op_cache.transpile_before_bind = self._transpile_before_bind
            op_cache.transpiled_circ_templates = self._transpiled_circ_templates
            self._cached_ops[op_id] = op_cache

        if return_as_list:
            return ListOp(
                [
                    replace_circuits_with_dicts(self._reduced_op_cache, param_index=i)
                    for i in range(num_parameterizations)
                ]
            )
        else:
            return replace_circuits_with_dicts(self._reduced_op_cache, param_index=0)

    def clear_cache(self) -> None:
        """Clear the cache of sampled operator expressions."""
        self._cached_ops = {}

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

    def sample_circuits(
        self,
        circuit_sfns: Optional[List[CircuitStateFn]] = None,
        param_bindings: Optional[List[Dict[Parameter, float]]] = None,
    ) -> Dict[int, List[StateFn]]:
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
        Raises:
            OpflowError: if extracted circuits are empty.
        """
        if not circuit_sfns and not self._transpiled_circ_cache:
            raise OpflowError("CircuitStateFn is empty and there is no cache.")

        if circuit_sfns:
            self._transpiled_circ_templates = None
            if self._statevector or circuit_sfns[0].from_operator:
                circuits = [op_c.to_circuit(meas=False) for op_c in circuit_sfns]
            else:
                circuits = [op_c.to_circuit(meas=True) for op_c in circuit_sfns]

            try:
                self._transpiled_circ_cache = self.quantum_instance.transpile(
                    circuits, pass_manager=self.quantum_instance.unbound_pass_manager
                )
            except QiskitError:
                logger.debug(
                    r"CircuitSampler failed to transpile circuits with unbound "
                    r"parameters. Attempting to transpile only when circuits are bound "
                    r"now, but this can hurt performance due to repeated transpilation."
                )
                self._transpile_before_bind = False
                self._transpiled_circ_cache = circuits
        else:
            circuit_sfns = list(self._circuit_ops_cache.values())

        if param_bindings is not None:
            if self._param_qobj:
                start_time = time()
                ready_circs = self._prepare_parameterized_run_config(param_bindings)
                end_time = time()
                logger.debug("Parameter conversion %.5f (ms)", (end_time - start_time) * 1000)
            else:
                start_time = time()
                ready_circs = [
                    circ.assign_parameters(_filter_params(circ, binding))
                    for circ in self._transpiled_circ_cache
                    for binding in param_bindings
                ]
                end_time = time()
                logger.debug("Parameter binding %.5f (ms)", (end_time - start_time) * 1000)
        else:
            ready_circs = self._transpiled_circ_cache

        # run transpiler passes on bound circuits
        if self._transpile_before_bind and self.quantum_instance.bound_pass_manager is not None:
            ready_circs = self.quantum_instance.transpile(
                ready_circs, pass_manager=self.quantum_instance.bound_pass_manager
            )

        results = self.quantum_instance.execute(
            ready_circs, had_transpiled=self._transpile_before_bind
        )

        if param_bindings is not None and self._param_qobj:
            self._clean_parameterized_run_config()

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

                if "expval_measurement" in circ_results:
                    avg = circ_results["expval_measurement"]
                    # Will be replaced with just avg when eval is called later
                    num_qubits = circuit_sfns[0].num_qubits
                    result_sfn = DictStateFn(
                        "0" * num_qubits,
                        coeff=avg * op_c.coeff,
                        is_measurement=op_c.is_measurement,
                        from_operator=op_c.from_operator,
                    )
                elif self._statevector:
                    result_sfn = StateFn(
                        op_c.coeff * results.get_statevector(circ_index),
                        is_measurement=op_c.is_measurement,
                    )
                else:
                    shots = self.quantum_instance._run_config.shots
                    result_sfn = DictStateFn(
                        {
                            b: (v / shots) ** 0.5 * op_c.coeff
                            for (b, v) in results.get_counts(circ_index).items()
                        },
                        is_measurement=op_c.is_measurement,
                        from_operator=op_c.from_operator,
                    )
                if self._attach_results:
                    result_sfn.execution_results = circ_results
                c_statefns.append(result_sfn)
            sampled_statefn_dicts[id(op_c)] = c_statefns
        return sampled_statefn_dicts

    def _build_aer_params(
        self,
        circuit: QuantumCircuit,
        building_param_tables: Dict[Tuple[int, int], List[float]],
        input_params: Dict[Parameter, float],
    ) -> None:
        def resolve_param(inst_param):
            if not isinstance(inst_param, ParameterExpression):
                return None
            param_mappings = {}
            for param in inst_param._parameter_symbols.keys():
                if param not in input_params:
                    raise ValueError(f"unexpected parameter: {param}")
                param_mappings[param] = input_params[param]
            return float(inst_param.bind(param_mappings))

        gate_index = 0
        for instruction in circuit.data:
            param_index = 0
            for inst_param in instruction.operation.params:
                val = resolve_param(inst_param)
                if val is not None:
                    param_key = (gate_index, param_index)
                    if param_key in building_param_tables:
                        building_param_tables[param_key].append(val)
                    else:
                        building_param_tables[param_key] = [val]
                param_index += 1
            gate_index += 1

    def _prepare_parameterized_run_config(
        self, param_bindings: List[Dict[Parameter, float]]
    ) -> List[Any]:

        self.quantum_instance._run_config.parameterizations = []

        if self._transpiled_circ_templates is None or len(self._transpiled_circ_templates) != len(
            self._transpiled_circ_cache
        ):

            # temporally resolve parameters of self._transpiled_circ_cache
            # They will be overridden in Aer from the next iterations
            self._transpiled_circ_templates = [
                circ.assign_parameters(_filter_params(circ, param_bindings[0]))
                for circ in self._transpiled_circ_cache
            ]

        for circ in self._transpiled_circ_cache:
            building_param_tables: Dict[Tuple[int, int], List[float]] = {}
            for param_binding in param_bindings:
                self._build_aer_params(circ, building_param_tables, param_binding)
            param_tables = []
            for gate_and_param_indices in building_param_tables:
                gate_index = gate_and_param_indices[0]
                param_index = gate_and_param_indices[1]
                param_tables.append(
                    [[gate_index, param_index], building_param_tables[(gate_index, param_index)]]
                )
            self.quantum_instance._run_config.parameterizations.append(param_tables)

        return self._transpiled_circ_templates

    def _clean_parameterized_run_config(self) -> None:
        self.quantum_instance._run_config.parameterizations = []


def _filter_params(circuit, param_dict):
    """Remove all parameters from ``param_dict`` that are not in ``circuit``."""
    return {param: value for param, value in param_dict.items() if param in circuit.parameters}


class OperatorCache:
    """A struct to cache an operator along with the circuits in contains."""

    reduced_op_cache = None  # the reduced operator
    circuit_ops_cache: Optional[Dict[int, CircuitStateFn]] = None  # the extracted circuits
    transpiled_circ_cache = None  # the transpiled circuits
    transpile_before_bind = True  # whether to transpile before binding parameters in the operator
    transpiled_circ_templates: Optional[List[Any]] = None  # transpiled circuit templates for Aer
