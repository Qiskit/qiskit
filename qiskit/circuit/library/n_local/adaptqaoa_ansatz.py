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

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.converters import isinstanceint
from qiskit.opflow import PauliSumOp, PauliOp, OperatorBase, PrimitiveOp
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit import ParameterVector
from qiskit.circuit.parametertable import ParameterTable
from qiskit.circuit.library.evolved_operator_ansatz import _is_pauli_identity
from qiskit.circuit.library.n_local import QAOAAnsatz

def commutator(a,b):
    "Qiskit commutator produces incorrect results, using this instead."
    a = a if isinstance(a, np.ndarray) else a.to_matrix()
    b = b if isinstance(b, np.ndarray) else b.to_matrix()    
    return np.matmul(a, b) - np.matmul(b, a)

def _reordered_indices(num_cost: int, num_mixer: list):
    # Create a permutation of indices to take us from (cost_1, mixer_1, cost_2, mixer_2, ...)
    # to (cost_1, cost_2, ..., mixer_1, mixer_2, ...), or if the mixer is a circuit
    # with more than 1 parameters, from (cost_1, mixer_1a, mixer_1b, cost_2, ...)
    # to (cost_1, cost_2, ..., mixer_1a, mixer_1b, mixer_2a, mixer_2b, ...)
    gamma_indices, beta_indices = [], []
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

# TODO: Implement compute_energy_gradient in here rather than adapt_qaoa.
"""
I think it makes more logical sense for compute_energy_gradient/ _test_mixer_pool functions 
to be placed in adaptansatz. The reason is because we would like to build the ansatz based
on the user-defined cost function.

The parameter reps is somewhat useless here since the length of mixer_operators should define
the ansatz depth. This is because for len(mixer_operators)==1 and reps>1, parameter optimisation
for a set of initial values must completed (which is the function of adaptqaoa class.)



"""

def adapt_mixer_pool(
    num_qubits: int, add_single: bool = True, add_multi: bool = True, pool_type: str = None,
    circuit_rep: bool = False, circ_params: Optional[Union[np.ndarray, List]] = None, num_params: int = 1,
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
        circ_rep: Boolean value when set to True will return circuit representation of mixer pool. 
            Defaults to False.
        circ_params: List of indices that specify qubits to be parameterised in the circuit representation.
        num_params: The number of parameters to be optimised per single qubit operation. 

    Returns:
        List of all possible combinations of mixers.

    Raises:
        KeyError: If an unrecognisible mixer type has been provided.
    """
    from qiskit.circuit import Parameter
    from qiskit.opflow import I, X, Y, Z
    if pool_type:
        if pool_type == "multi":
            add_multi, add_single = True, True
        elif pool_type == "singular":
            add_multi, add_single = False, True
        elif pool_type == "single":
            add_multi, add_single = False, False
        else:
            raise KeyError(
                "Unrecognised mixer pool type '{}', modify this input to the available presets"
                " 'single', 'singular' or 'multi'.".format(pool_type)
            )
    op_dict = {"I": I, "X": X, "Y": Y, "Z": Z}
    if circuit_rep:
        from qiskit.circuit.library import RXGate, RYGate, RZGate
        if circ_params is not None:
            circ_params = [circ_params] if isinstanceint(circ_params) else circ_params
            circ_params = circ_params.tolist() if not isinstance(circ_params, list) else circ_params
            check_params = [i for i in circ_params if i > num_qubits]
            if check_params:
                raise ValueError("Qubit parameter indices {} out of range for {} specified "
                    "number of qubits.".format(check_params, num_qubits))
            theta = Parameter("")
            param_op_dict = {"I": I, "X": RXGate(theta), "Y": RYGate(theta), "Z": RZGate(theta)}
        else:
            circ_params = []    
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
        if circuit_rep:
            mixer = QuantumCircuit(num_qubits)
            for p_i in range(num_params):
                param = [Parameter("θ"+str(p_i))]
                for q_i, mix in enumerate(mix_str):
                    if q_i in circ_params:
                        qbit_op = param_op_dict[mix]
                        qbit_op.params = param
                        mixer.append(qbit_op,[q_i])
                    else:
                        qbit_op = op_dict[mix]
                        mixer.append(qbit_op,[q_i])
        else:
            if mix_str == len(mix_str) * mix_str[0]:
                gate = mix_str[0]
                list_string = [
                    i * "I" + gate + (len(mix_str) - i - 1) * "I" for i in range(len(mix_str))
                ]
                op_list = [(op, 1) for op in list_string]
                mixer = PauliSumOp(SparsePauliOp.from_list(op_list))
            else:
                mixer = PauliOp(Pauli(mix_str))
        mixer_circ_list.append(mixer)
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
        initial_state: Optional[QuantumCircuit] = None,
        mixer_pool: Optional[Union[OperatorBase, QuantumCircuit]] = None,
        mixer_pool_type: str = None,
        name: str = "AdaptQAOA",
    ):
        r"""
        Args:
            cost_operator (OperatorBase, optional): The operator representing the cost of
                the optimization problem, denoted as :math:`U(C, \gamma)` in the original paper.
                Must be set either in the constructor or via property setter.
            initial_state (QuantumCircuit, optional): An optional initial state to use.
                If `None` is passed then a set of Hadamard gates is applied as an initial state
                to all qubits.
            mixer_pool (OperatorBase or QuantumCircuit, List[QuantumCircuit], optional): An optional
                custom mixer or list of mixers that define the 'pool' of operators that the AdaptQAOA
                algorithm chooses optimal mixers from. The set of optimal mixers selected from the pool
                are denoted as :math:`U(B, \beta)` in the original paper. Elements in mixer_pool may be
                operators or an optionally parameterized quantum circuit.
            mixer_pool_type: An optional string representing the varied mixer pool types, with `single`
                creating the same mixer pool as the standard QAOA. `singular` creates a mixer pool 
                including mixers in `single` as well as additional single qubit mixers. `multi` creates 
                a mixer pool including mixers from `single`, `singular` as well as multi-qubit entangling 
                mixers.
                Cannot be used in conjuction with a non-empty `mixer_pool` list.
            name (str): A name of the circuit, default 'qaoa'
        """
        super().__init__(name=name)
        self._initial_state = initial_state
        self._mixer_pool = mixer_pool
        self._mixer_operators = None
        self._mixer_pool_type = mixer_pool_type
        # set this circuit as a not-built circuit
        self._bounds = None

        self.cost_operator = cost_operator

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True
        if self.mixer_operators and not isinstance(self.mixer_operators, list):
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

        if not self._config_check:
            # Check that the dimensionality of the mixer operator pool is equal to the cost operator
            if self.mixer_operators:
                _, valid = self._check_mixers(self.mixer_operators)
            if valid:
                self._mixer_pool, valid  = self._check_mixers(self.mixer_pool) # set self._mixer_pool as the operator representation
            self._config_check = True   # set class attribute to avoid doing this check twice
        return valid

    def _check_mixers(self, mixer_operators: Optional[Union[OperatorBase, QuantumCircuit]], raise_on_failure: bool = True):
        valid = True
        nmix_qubits, nmix_params, mixer_list = [], [], []
        for mixer in mixer_operators:
            nmix_qubits.append(mixer.num_qubits) #
            if isinstance(mixer, QuantumCircuit): # For the purposes of efficient energy gradient computation 
                nmix_params.append(mixer.num_parameters)
                if not nmix_params[-1]:
                    mixer = PrimitiveOp(Operator(mixer)) # (i.e. the commutator) we must convert mixer circuits
            else:
                nmix_params.append(0 if _is_pauli_identity(mixer) else 1)
            mixer_list.append(mixer)           # to opreators here
        if len(set(nmix_params))>1:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    f"Inconsistent number of optimisable parameters in the mixer pool."
                )
        self._num_mixer = nmix_params[-1:]
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
        return mixer_list, valid

    @property
    def parameter_bounds(self) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """
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
            if isinstance(self.mixer_operators, list) and self._config_check:
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
    def cost_operator(self):
        """Returns an operator representing the cost of the optimization problem.
        Returns:
            OperatorBase: cost operator.
        """
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        """Sets cost operator.
        Args:
            cost_operator (OperatorBase, optional): cost operator to set.
        """
        self._cost_operator = cost_operator
        self.qregs = [QuantumRegister(self.num_qubits, name="q")]
        self._num_cost = 0 if _is_pauli_identity(self.cost_operator) else 1
        self._num_mixer = [0]
        self._config_check = False
        self._invalidate()
        
    @property
    def mixer_operators(self):
        """An ordered list of mixer operators to be used in the construction of the ansatz
        with length equivalent to the circuit depth.
        Returns:
            List of mixers to be used on ansatz layers.
        Raises:
            AttributeError: If operator and thus num_qubits has not yet been defined.
        """
        return self._mixer_operators

    @mixer_operators.setter
    def mixer_operators(self, mixer_operators) -> None:
        """Sets the list of mixer operators to be used in the ansatz construction
        Args:
            mixer_operator: a list of operators or circuits to set.
        """
        self._operators = None
        self._is_built = False
        self._mixer_operators = mixer_operators

    @property
    def mixer_pool(self):
        if self._mixer_pool is not None:
            return self._mixer_pool
        # if no mixer pool is passed and we know the number of qubits, then initialize it.
        if self.cost_operator is not None:  # Generate the complete set of mixers for   
            all_mixers = adapt_mixer_pool(  # problem dimension
                num_qubits=self.num_qubits, pool_type=self._mixer_pool_type
            )
            mixer_pool = []
            for mix in all_mixers:  # remove mixers that commute with cost operator
                comm = commutator(self.cost_operator,mix)
                if np.count_nonzero(comm)>0: 
                    mixer_pool.append(mix)
            self._mixer_pool = mixer_pool
        return self._mixer_pool
        
    @mixer_pool.setter
    def mixer_pool(self, mixer_pool) -> None:
        self._mixer_pool = mixer_pool
        self._config_check = False
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

    def construct_ansatz(self, mixer_operators: List):
        """ Constructs an ansatz with an ordered list of mixer operators, 
        where the circuit depth is equal to the mixer_operator list length.

        Args:
            mixer_operators: A list of mixer operators.
        """
        if not isinstance(mixer_operators, list):
            mixer_operators = [mixer_operators]

        _config_check = self._config_check  # save
        if self._mixer_pool is None:
            self._check_mixers(mixer_operators = mixer_operators)
        elif np.any([_ not in self._mixer_pool for _ in mixer_operators]):
            self._check_mixers(mixer_operators)
            self._config_check = True
        self.mixer_operators = mixer_operators
        self._build()
        self._config_check = _config_check
    
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
        gamma_index, beta_index = _reordered_indices(num_mixer=self._num_mixer,
                                                    num_cost=self._num_cost)
        reordered = np.array([None] * (sum(self._num_mixer) + self._num_cost))
        reordered[gamma_index] = [_ for _ in gammas]
        reordered[beta_index] = [_ for _ in betas]
        self.assign_parameters(dict(zip(self.ordered_parameters, reordered)), inplace=True)