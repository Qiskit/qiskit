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
from functools import reduce
from itertools import combinations_with_replacement, permutations, product

from sympy.logic.boolalg import Not
from qiskit.circuit.library.evolved_operator_ansatz import EvolvedOperatorAnsatz, _is_pauli_identity
from qiskit.circuit.library.n_local import QAOAAnsatz
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.opflow import PauliSumOp, PauliOp, OperatorBase
from qiskit.quantum_info import Pauli, SparsePauliOp


def _reorder_parameters(num_mixer: Union[int, List], num_cost: int, reps: int):
    if not isinstance(num_mixer, list):
        num_mixer = [num_mixer] * reps
    num_beta_params = sum(num_mixer)
    betas = ParameterVector("β", num_beta_params)
    gammas = ParameterVector("γ", reps * num_cost)
    # Create a permutation to take us from (cost_1, mixer_1, cost_2, mixer_2, ...)
    # to (cost_1, cost_2, ..., mixer_1, mixer_2, ...), or if the mixer is a circuit
    # with more than 1 parameters, from (cost_1, mixer_1a, mixer_1b, cost_2, ...)
    # to (cost_1, cost_2, ..., mixer_1a, mixer_1b, mixer_2a, mixer_2b, ...)
    reordered = []
    for rep in range(reps):
        reordered.extend(gammas[rep * num_cost : (rep + 1) * num_cost])
        reordered.extend(betas[rep * num_mixer[rep] : (rep + 1) * num_mixer[rep]])
    return reordered


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
        # if mixer_operators is not None:
        #     self.mixer_operators = mixer_operators
        # if mixer_pool_type is not None:
        #     self.mixer_pool_type = mixer_pool_type
        # set this circuit as a not-built circuit
        self._bounds = None

        # store cost operator and set the registers if the operator is not None
        # self.cost_operator = super().cost_operator

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

        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    f"The number of qubits of the initial state {self.initial_state.num_qubits}"
                    f"does not match the number of qubits of the cost operator {self.num_qubits}"
                )
        if self.mixer_operators is not None:
            # Check that the dimensionality of the mixer operator pool is equal to the cost operator
            n_mixer_qubits = np.array([mixer.num_qubits for mixer in self.mixer_operators])
            check_mixer_qubits = np.argwhere(n_mixer_qubits != self.num_qubits)
            if len(check_mixer_qubits):
                valid = False
                err_str = str(tuple(set(check_mixer_qubits)))
                if raise_on_failure:
                    raise AttributeError(
                        f"Operators at index location(s) {err_str} in the specified mixer pool"
                        f" have an unequal number of qubits {n_mixer_qubits[check_mixer_qubits]}"
                        f" to the cost operator {self.num_qubits}."
                    )
        return valid

    @property
    def parameter_bounds(self) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """
        if self._bounds is not None:
            return self._bounds

        # if the mixer is a circuit, we set no bounds
        if isinstance(self.mixer_operators, QuantumCircuit):     #TODO: FIX THIS!
            return None
        # default bounds: None for gamma (cost operator), [-pi/2, pi/2] for gamma (mixer operator)
        beta_bounds = (-0.5 * np.pi, 0.5 * np.pi)
        gamma_bounds = (None, None)
        bounds = []

        if not _is_pauli_identity(self.mixer_operators): #TODO: Fix this!
            bounds += self.reps * [beta_bounds]

        if not _is_pauli_identity(self.cost_operator):
            bounds += self.reps * [gamma_bounds]

        return bounds

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
            if isinstance(self.mixer_operators, list):
                varied_operators = list(
                    itertools.chain.from_iterable(
                        [[self.cost_operator, mixer] for mixer in self.mixer_operators]
                    )
                )
                self._operators = varied_operators
        #TODO: see if a condition can be placed here to return nothing when doing check_operator_ansatz: e.g.
        if self._mixer_pool==self.mixer_operators:
            pass      # this will prevent an ansatz of the full mixer pool from being constructed
        return self._operators
    
    @operators.setter
    def operators(self, operators) -> List:
        "Sets list of ansatz operators"
        self._operators = operators

    @property
    def mixer_operators(self):
        """Returns an optional mixer operator expressed as an operator or a quantum circuit.

        Returns:
            OperatorBase or QuantumCircuit, optional: mixer operator or circuit.
        """
        if self._mixer_pool is not None:
            return self._mixer_pool
        else:
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
            mixer_operator (OperatorBase or QuantumCircuit, optional): mixer operator or circuit
                to set.
        """
        self._mixer_pool = mixer_operators
        self._invalidate()

    @property 
    def mixer_pool_type(self) -> str:
        if self._mixer_pool_type is None:
            self._mixer_pool_type = 'multi'
        return self._mixer_pool_type

    @mixer_pool_type.setter
    def mixer_pool_type(self, mixer_pool_type: str):
        self._mixer_pool_type = mixer_pool_type
    

    def _build(self):
        if self._data is not None:
            return

        super(QAOAAnsatz, self)._build()

    # keep old parameter order: first cost operator, then mixer operators
        num_cost = 0 if _is_pauli_identity(self.cost_operator) else 1
        if isinstance(self.mixer_operators, list):
            num_mixer = []
            for reps, mix in enumerate(self.operators[1:][::2], 1):
                if isinstance(mix, QuantumCircuit):
                    num_mixer.append(mix.num_parameters)
                else:
                    num_mixer.append(0 if _is_pauli_identity(mix) else 1)
        else:
            reps = self.reps
            if isinstance(self.mixer_pool, QuantumCircuit):
                num_mixer = self.mixer_pool.num_parameters
            else:
                num_mixer = 0 if _is_pauli_identity(self.mixer_pool) else 1

        reordered = _reorder_parameters(num_mixer=num_mixer, num_cost=num_cost, reps=reps)
        self.assign_parameters(dict(zip(self.ordered_parameters, reordered)), inplace=True)