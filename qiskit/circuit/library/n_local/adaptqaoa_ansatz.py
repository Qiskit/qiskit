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

"""A generalized QAOA quantum circuit with a support of custom initial states and mixers."""

# pylint: disable=cyclic-import

import itertools
from typing import List, Optional, Tuple, Union
import numpy as np
from itertools import combinations_with_replacement, permutations, product

from qiskit.circuit.library.n_local import QAOAAnsatz
from qiskit.circuit.library.evolved_operator_ansatz import _is_pauli_identity
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp, PauliOp, OperatorBase
from qiskit.pulse.builder import num_qubits
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp


def _reordered_indices(num_cost: int, num_mixer: list, opt_inputs: bool = False):
    # Create a permutation of indices to take us from (cost_1, mixer_1, cost_2, mixer_2, ...)
    # to (cost_1, cost_2, ..., mixer_1, mixer_2, ...), or if the mixer is a circuit
    # with more than 1 parameters, from (cost_1, mixer_1a, mixer_1b, cost_2, ...)
    # to (cost_1, cost_2, ..., mixer_1a, mixer_1b, mixer_2a, mixer_2b, ...)
    gamma_indices, beta_indices = [], []
    if opt_inputs:
        cost_p, mix_p, p = 0, 0, 0
        while p < num_cost + len(num_mixer):
            if mix_p < len(num_mixer):
                p_beta = num_mixer[mix_p] + p
                beta_indices.extend(list(range(p, p_beta)))
                p += num_mixer[cost_p]
            gamma_indices.extend(list(range(p, p + 1)))
            mix_p += 1
            cost_p += 1
            p += 1
    else:
        rep = 0
        for rep_cost in range(num_cost):
            if num_cost>0:
                gamma_indices.extend(list(range(rep,rep+1)))
            rep+=1
            if rep_cost < len(num_mixer):
                rep_mix = num_mixer[rep_cost]
                beta_indices.extend(list(range(rep,rep_mix+rep)))
            rep+=rep_mix

    return gamma_indices, beta_indices


def adapt_mixer_pool(
    num_qubits: int, add_single: bool = True, add_multi: bool = True, pool_type: str = None
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
                "Unrecognised mixer pool type '{}', modify this input to the available presets"
                " 'single', 'singular' or 'multi'.".format(pool_type)
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

    mixer_circ_list = []
    for mix_str in mixer_pool:
        if mix_str == len(mix_str) * mix_str[0]:
            gate = mix_str[0]
            list_string = [
                i * "I" + gate + (len(mix_str) - i - 1) * "I" for i in range(len(mix_str))
            ]
            op_list = [(op, 1) for op in list_string]
            op = PauliSumOp(SparsePauliOp.from_list(op_list))
        else:
            op = PauliOp(Pauli(mix_str))
        mixer_circ_list.append(op)

    return mixer_circ_list

class AdaptQAOAAnsatz(QAOAAnsatz):
    """A generalized QAOA quantum circuit with a support of custom initial states and mixers.

    References:

        [1]: Farhi et al., A Quantum Approximate Optimization Algorithm.
            `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """

    def __init__(
        self,
        cost_operator=None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        mixer_operators: Optional[Union[OperatorBase, QuantumCircuit]] = None,
        mixer_pool_type: str = None,
        name: str = "AdaptQAOA",
    ):
        r"""
        Args:
            cost_operator (OperatorBase, optional): The operator representing the cost of
                the optimization problem, denoted as :math:`U(C, \gamma)` in the original paper.
                Must be set either in the constructor or via property setter.
            reps (int): The integer parameter p, which determines the depth of the circuit,
                as specified in the original paper, default is 1.
            initial_state (QuantumCircuit, optional): An optional initial state to use.
                If `None` is passed then a set of Hadamard gates is applied as an initial state
                to all qubits.
            mixer_operator (OperatorBase or QuantumCircuit, List[QuantumCircuit], optional): An optional
                custom mixer or list of mixer to be used instead of the global X-rotations, denoted
                as :math:`U(B, \beta)` in the original paper. Can be an operator or an optionally
                parameterized quantum circuit.
            name (str): A name of the circuit, default 'qaoa'
        """
        super().__init__(reps=reps, name=name, cost_operator=cost_operator)
        self._reps = reps
        self._initial_state = initial_state
        self._mixer_pool_type = mixer_pool_type
        self._mixer_pool = mixer_operators
        # set this circuit as a not-built circuit
        self._bounds = None

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True
        if not isinstance(self.mixer_operators, list):
            self.mixer_operator = self.mixer_operators
            return super()._check_configuration()

        if not super(QAOAAnsatz, self)._check_configuration(raise_on_failure):
            return False

        if self.cost_operator is None:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    "The operator representing the cost of the optimization problem is not set"
                )
        self._num_gamma = 0 if _is_pauli_identity(self.cost_operator) else 1

        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    f"The number of qubits of the initial state {self.initial_state.num_qubits}"
                    f"does not match the number of qubits of the cost operator {self.num_qubits}"
                )
        if self.mixer_operators is not None and not hasattr(self,'_config_check'):
            # Check that the dimensionality of the mixer operator pool is equal to the cost operator
            nmix_qubits, nmix_params, mixer_operators = [], [], []
            for mixer in self.mixer_operators: 
                nmix_qubits.append(mixer.num_qubits) #
                if isinstance(mixer, QuantumCircuit): # For the purposes of efficient energy gradient computation 
                    nmix_params.append(mixer.num_parameters)
                    if not nmix_params[-1]:
                        mixer = PrimitiveOp(Operator(mixer)) # (i.e. the commutator) we must convert mixer circuits
                else:
                    nmix_params.append(0 if _is_pauli_identity(mixer) else 1)
                mixer_operators.append(mixer)           # to opreators here
            if len(set(nmix_params))>1:
                valid = False
                if raise_on_failure:
                    raise ValueError(
                        f"Inconsistent number of optimisable parameters in the mixer pool."
                    )
            self._num_beta = nmix_params[-1:]
            _config_check = np.where(np.array(nmix_qubits) != self.num_qubits)[0]
            if bool(sum(_config_check)):
                valid = False
                err_str = str(tuple(set(_config_check)))
                if raise_on_failure:
                    raise AttributeError(
                        f"Operators at index location(s) {err_str} in the specified mixer pool"
                        f" have an unequal number of qubits {[nmix_qubits[i] for i in _config_check]}"
                        f" to the cost operator {self.num_qubits}."
                    )
            self._mixer_operators = mixer_operators # set self._mixer_operators as the operator representation
            self._config_check = True   # set class attribute to avoid doing this check twice
        return valid

    @property
    def parameter_bounds(self) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """
        
        if hasattr(self,'_num_cost'):
            # self._bounds = [(-0.5 * np.pi, 0.5 * np.pi)] * sum(self._num_mixer)
            # self._bounds += [(-2 * np.pi, 2 * np.pi)] * self._num_cost
            self._bounds = [(-0.5 * np.pi, 0.5 * np.pi)] * sum(self._num_mixer)
            self._bounds += [(-2 * np.pi, 2 * np.pi)] * self._num_cost

        return self._bounds

    @parameter_bounds.setter
    def parameter_bounds(
        self, bounds: Optional[List[Tuple[Optional[float], Optional[float]]]]
    ) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    @property
    def operators(self):
        """The operators that are evolved in this circuit.

        Returns:
             List[Union[OperatorBase, QuantumCircuit]]: The operators to be evolved (and circuits)
                in this ansatz.
        """
        if self._operators is None:
            if isinstance(self.mixer_operators, list) and hasattr(self,'_config_check'):
                varied_operators = list(
                    itertools.chain.from_iterable(
                        [[self.cost_operator, mixer] for mixer in self.mixer_operators]
                    )
                )
                self._operators = varied_operators
            else:
                return [self.cost_operator, self.mixer_operator]
        return self._operators
    
    @operators.setter
    def operators(self, operators) -> List:
        "Sets list of ansatz operators"
        self._operators = operators

    @property
    def mixer_operators(self):
        """Creates the mixer pool if not already defined
        Returns:
            List of mixers that make up the mixer pool.
        Raises:
            AttributeError: If operator and thus num_qubits has not yet been defined.
        """
        if self._mixer_pool is not None:
            return self._mixer_pool
        # if no mixer is passed and we know the number of qubits, then initialize it.
        if self.cost_operator is not None:
            mixer_pool = adapt_mixer_pool(
                num_qubits=self.num_qubits, pool_type=self._mixer_pool_type
            )
            self._mixer_pool = mixer_pool
            return mixer_pool
        return None

    @mixer_operators.setter
    def mixer_operators(self, mixer_operators) -> None:
        """Sets mixer pool.
        Args:
            mixer_operator: a list of operators or circuits to set.
        """
        self._mixer_pool = mixer_operators
        self._invalidate()

    @property 
    def mixer_pool_type(self) -> str:
        """Returns the class of mixer pool if specified; otherwise will return a multi-qubit
            mixer pool as a default.
        Returns:
            str: mixer pool type"""
        if self._mixer_pool_type is None:  
            self._mixer_pool_type = 'multi'
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
    
    def _build(self):
        if self._is_built:
            return
        super(QAOAAnsatz, self)._build()

        num_mixer = []
        for mix in self.mixer_operators:
            if isinstance(mix, QuantumCircuit):
                num_mixer.append(mix.num_parameters)
            else:
                num_mixer.append(0 if _is_pauli_identity(mix) else 1)
        self._num_mixer = num_mixer
        self._num_cost = 0 if _is_pauli_identity(self.cost_operator) else len(num_mixer)

        betas = ParameterVector("β", sum(self._num_mixer))
        gammas = ParameterVector("γ", self._num_cost)
        gamma_index, beta_index = _reordered_indices(num_mixer=self._num_mixer, #TODO: Order is reversed here!!
                                                    num_cost=self._num_cost)
        reordered = np.array([None]*(sum(self._num_mixer)+self._num_cost))
        reordered[gamma_index] = [_ for _ in gammas]
        reordered[beta_index] = [_ for _ in betas]
        self.assign_parameters(dict(zip(self.ordered_parameters, reordered)), inplace=True)
