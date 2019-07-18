# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from copy import deepcopy
import itertools
from functools import reduce
import logging
import json
from operator import add as op_add, sub as op_sub
import sys
import warnings

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli
from qiskit.qasm import pi
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua.utils import find_regs_by_name
from qiskit.aqua.utils.backend_utils import is_statevector_backend
from qiskit.aqua.operators.base_operator import BaseOperator
from qiskit.aqua.operators.common import measure_pauli_z, covariance, kernel_F2, \
                                         suzuki_expansion_slice_pauli_list, check_commutativity, evolution_instruction


logger = logging.getLogger(__name__)


class WeightedPauliOperator(BaseOperator):

    def __init__(self, paulis, basis=None, atol=1e-12, name=None):
        """
        Args:
            paulis ([[complex, Pauli]]): the list of weighted Paulis, where a weighted pauli is composed of
                                         a length-2 list and the first item is the weight and
                                         the second item is the Pauli object.
            basis (list[tuple(object, [int])], optional): the grouping basis, each element is a tuple composed of the basis
                                                          and the indices to paulis which are belonged to that group.
                                                          e.g., if tpb basis is used, the object will be a pauli.
                                                          by default, the group is equal to non-grouping, each pauli is its own basis.
            atol (float, optional): the threshold used in truncating paulis
            name (str, optional): the name of operator.
        """
        # plain store the paulis, the group information is store in the basis
        self._paulis_table = None
        self._paulis = paulis
        self._basis = [(pauli[1], [i]) for i, pauli in enumerate(paulis)] if basis is None else basis
        # combine the paulis and remove those with zero weight
        self.simplify()
        self._aer_paulis = None
        self._atol = atol
        self._name = name if name is not None else ''

    @classmethod
    def from_list(cls, paulis, weights=None, name=None):
        """
        Create a WeightedPauliOperator via a pair of list.

        Args:
            paulis ([Pauli]): the list of Paulis
            weights ([complex], optional): the list of weights, if it is None, all weights are 1.
            name (str, optional): name of the operator.

        Returns:
            WeightedPauliOperator
        """
        if weights is not None and len(weights) != len(paulis):
            raise ValueError("The length of weights and paulis must be the same.")
        if weights is None:
            weights = [1.0] * len(paulis)
        return cls(paulis=[[w, p] for w, p in zip(weights, paulis)], name=name)

    @property
    def paulis(self):
        return self._paulis

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, new_value):
        self._atol = new_value

    @property
    def basis(self):
        return self._basis

    @property
    def num_qubits(self):
        """
        number of qubits required for the operator.

        Returns:
            int: number of qubits

        """
        if not self.is_empty():
            return self._paulis[0][1].numberofqubits
        else:
            logger.warning("Operator is empty, Return 0.")
            return 0

    @property
    def aer_paulis(self):
        """
        Returns: the weighted paulis formatted for the aer simulator.
        """
        if getattr(self, '_aer_paulis', None) is None:
            aer_paulis = []
            for weight, pauli in self._paulis:
                new_weight = [weight.real, weight.imag]
                new_pauli = pauli.to_label()
                aer_paulis.append([new_weight, new_pauli])
            self._aer_paulis = aer_paulis
        return self._aer_paulis

    def __eq__(self, other):
        """Overload == operation"""
        # need to clean up the zeros
        self.simplify()
        other.simplify()
        if len(self._paulis) != len(other.paulis):
            return False
        for weight, pauli in self._paulis:
            found_pauli = False
            other_weight = 0.0
            for weight2, pauli2 in other.paulis:
                if pauli == pauli2:
                    found_pauli = True
                    other_weight = weight2
                    break
            if not found_pauli and other_weight != 0.0:  # since we might have 0 weights of paulis.
                return False
            if weight != other_weight:
                return False
        return True

    def _add_or_sub(self, other, operation, copy=True):
        """
        Add two operators either extend (in-place) or combine (copy) them.
        The addition performs optimized combiniation of two operators.
        If `other` has identical basis, the coefficient are combined rather than
        appended.

        Args:
            other (WeightedPauliOperator): to-be-combined operator
            operation (callable or str): add or sub callable from operator
            copy (bool): working on a copy or self

        Returns:
            WeightedPauliOperator

        Raises:
            AquaError: two operators have different number of qubits.
        """

        if not self.is_empty() and not other.is_empty():
            if self.num_qubits != other.num_qubits:
                raise AquaError("Can not add/sub two operators with different number of qubits.")

        ret_op = self.copy() if copy else self

        for pauli in other.paulis:
            pauli_label = pauli[1].to_label()
            idx = ret_op._paulis_table.get(pauli_label, None)
            if idx is not None:
                ret_op._paulis[idx][0] = operation(ret_op._paulis[idx][0], pauli[0])
            else:
                ret_op._paulis_table[pauli_label] = len(ret_op._paulis)
                ret_op._basis.append((pauli[1], [len(ret_op._paulis)]))
                pauli[0] = operation(0.0, pauli[0])
                ret_op._paulis.append(pauli)
        return ret_op

    def add(self, other, copy=False):
        """Perform self + other.

        Args:
            other (WeightedPauliOperator): to-be-combined operator
            copy (bool): working on a copy or self, if True, the results are written back to self.

        Returns:
            WeightedPauliOperator
        """

        return self._add_or_sub(other, op_add, copy=copy)

    def sub(self, other, copy=False):
        """Perform self - other.

        Args:
            other (WeightedPauliOperator): to-be-combined operator
            copy (bool): working on a copy or self, if True, the results are written back to self.

        Returns:
            WeightedPauliOperator
        """

        return self._add_or_sub(other, op_sub, copy=copy)

    def __add__(self, other):
        """Overload + operator."""
        return self.add(other, copy=True)

    def __iadd__(self, other):
        """Overload += operator."""
        return self.add(other, copy=False)

    def __sub__(self, other):
        """Overload - operator."""
        return self.sub(other, copy=True)

    def __isub__(self, other):
        """Overload -= operator."""
        return self.sub(other, copy=False)

    def _scaling_weight(self, scaling_factor, copy=False):
        """
        Constantly scaling all weights of paulis.

        Args:
            scaling_factor (complex): the scaling factor
            copy (bool): return a copy or modify in-place

        Returns:
            WeightedPauliOperator: a copy of the scaled one.

        Raises:
            ValueError: the scaling factor is not a valid type.
        """
        if not isinstance(scaling_factor, (int, float, complex, np.int, np.float, np.complex)):
            raise ValueError("Type of scaling factor is a valid type. {} if given.".format(scaling_factor.__class__))
        ret = self.copy() if copy else self
        for idx in range(len(ret._paulis)):
            ret._paulis[idx] = [ret._paulis[idx][0] * scaling_factor, ret._paulis[idx][1]]
        return ret

    def multiply(self, other):
        """
        Perform self * other, and the phases are tracked.

        Args:
            other (WeightedPauliOperator): an operator

        Returns:
            WeightedPauliOperator: the multiplied operator
        """
        ret_op = WeightedPauliOperator(paulis=[])
        for existed_weight, existed_pauli in self.paulis:
            for weight, pauli in other.paulis:
                new_pauli, sign = Pauli.sgn_prod(existed_pauli, pauli)
                new_weight = existed_weight * weight * sign
                pauli_term = [new_weight, new_pauli]
                ret_op += WeightedPauliOperator(paulis=[pauli_term])
        return ret_op

    def __rmul__(self, other):
        """Overload other * self."""
        if isinstance(other, (int, float, complex, np.int, np.float, np.complex)):
            return self._scaling_weight(other, copy=True)
        else:
            return other.multiply(self)

    def __mul__(self, other):
        """Overload self * other."""
        if isinstance(other, (int, float, complex, np.int, np.float, np.complex)):
            return self._scaling_weight(other, copy=True)
        else:
            return self.multiply(other)

    def __neg__(self):
        """Overload unary -."""
        return self._scaling_weight(-1.0, copy=True)

    def __str__(self):
        """Overload str()."""
        curr_repr = 'paulis'
        length = len(self._paulis)
        name = "" if self._name == '' else "{}: ".format(self._name)
        ret = "{}Representation: {}, qubits: {}, size: {}".format(name, curr_repr, self.num_qubits, length)
        return ret

    def print_details(self):
        """
        Print out the operator in details.

        Returns:
            str: a formatted string describes the operator.
        """
        if self.is_empty():
            return "Operator is empty."
        ret = ""
        for weight, pauli in self._paulis:
            ret = ''.join([ret, "{}\t{}\n".format(pauli.to_label(), weight)])

        return ret

    def copy(self):
        """Get a copy of self."""
        return deepcopy(self)

    def simplify(self, copy=False):
        """
        #TODO: note change the behavior
        Merge the paulis whose bases are identical and the pauli with zero coefficient
        would be removed.

        Args:
            copy (bool): simplify on a copy or self

        Returns:
            WeightedPauliOperator: the simplified operator
        """

        op = self.copy() if copy else self

        new_paulis = []
        new_paulis_table = {}
        for curr_weight, curr_pauli in op.paulis:
            pauli_label = curr_pauli.to_label()
            new_idx = new_paulis_table.get(pauli_label, None)
            if new_idx is not None:
                new_paulis[new_idx][0] += curr_weight
            else:
                new_paulis_table[pauli_label] = len(new_paulis)
                new_paulis.append([curr_weight, curr_pauli])

        op._paulis = new_paulis
        op._paulis_table = {weighted_pauli[1].to_label(): i for i, weighted_pauli in enumerate(new_paulis)}
        op.chop(0.0)
        return op

    def chop(self, threshold=None, copy=False):
        """
        Eliminate the real and imagine part of weight in each pauli by `threshold`.
        If pauli's weight is less then `threshold` in both real and imagine parts, the pauli is removed.

        Note:
            If weight is real-only, the imag part is skipped.

        Args:
            threshold (float): the threshold is used to remove the paulis
            copy (bool): chop on a copy or self

        Returns:
            WeightedPauliOperator: if copy is True, the original operator is unchanged; otherwise,
                                   the operator is mutated.
        """
        threshold = self._atol if threshold is None else threshold

        def chop_real_imag(weight):
            temp_real = weight.real if np.absolute(weight.real) >= threshold else 0.0
            temp_imag = weight.imag if np.absolute(weight.imag) >= threshold else 0.0
            if temp_real == 0.0 and temp_imag == 0.0:
                return 0.0
            else:
                new_weight = temp_real + 1j * temp_imag
                return new_weight

        op = self.copy() if copy else self

        if op.is_empty():
            return op

        paulis = []
        old_to_new_indices = {}
        curr_idx = 0
        for idx, weighted_pauli in enumerate(op.paulis):
            weight, pauli = weighted_pauli
            new_weight = chop_real_imag(weight)
            if new_weight != 0.0:
                old_to_new_indices[idx] = curr_idx
                curr_idx += 1
                paulis.append([new_weight, pauli])

        op._paulis = paulis
        op._paulis_table = {weighted_pauli[1].to_label(): i for i, weighted_pauli in enumerate(paulis)}
        # update the grouping info, since this method only remove pauli, we can handle it here for both
        # pauli and tpb grouped pauli
        new_basis = []
        for basis, indices in op.basis:
            new_indices = []
            for idx in indices:
                new_idx = old_to_new_indices.get(idx, None)
                if new_idx is not None:
                    new_indices.append(new_idx)
            if len(new_indices) > 0:
                new_basis.append((basis, new_indices))
        op._basis = new_basis
        return op

    def commute_with(self, other):
        return check_commutativity(self, other)

    def anticommute_with(self, other):
        return check_commutativity(self, other, anti=True)

    def is_empty(self):
        """
        Check Operator is empty or not.

        Returns:
            bool: is empty?
        """
        if self._paulis is None:
            return True
        elif len(self._paulis) == 0:
            return True
        elif len(self._paulis[0]) == 0:
            return True
        else:
            return False

    def to_grouped_weighted_pauli_operator(self, grouping_func=None, **kwargs):
        """

        Args:
            grouping_func (Callable): a grouping callback to group paulis, and this callback will be fed with the paulis
                                      and kwargs arguments
            kwargs: other arguments needed for grouping func.

        Returns:
            object: the type depending on the `grouping_func`.
        """
        return grouping_func(self._paulis, **kwargs)

    def to_matrix_operator(self):
        """
        Convert to matrix operator.

        Returns:
            MatrixOperator:
        """
        from qiskit.aqua.operators.matrix_operator import MatrixOperator
        if self.is_empty():
            return MatrixOperator(None)

        hamiltonian = 0
        for weight, pauli in self._paulis:
            hamiltonian += weight * pauli.to_spmatrix()
        return MatrixOperator(matrix=hamiltonian)

    @classmethod
    def from_file(cls, file_name, before_04=False):
        """
        Load paulis in a file to construct an Operator.

        Args:
            file_name (str): path to the file, which contains a list of Paulis and coefficients.
            before_04 (bool): support the format before Aqua 0.4.

        Returns:
            Operator class: the loaded operator.
        """
        with open(file_name, 'r') as file:
            return cls.from_dict(json.load(file), before_04=before_04)

    def to_file(self, file_name):
        """
        Save operator to a file in pauli representation.

        Args:
            file_name (str): path to the file

        """
        with open(file_name, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, dictionary, before_04=False):
        """
        Load paulis in a dict to construct an Operator, \
        the dict must be represented as follows: label and coeff (real and imag). \
        E.g.: \
           {'paulis': \
               [ \
                   {'label': 'IIII', \
                    'coeff': {'real': -0.33562957575267038, 'imag': 0.0}}, \
                   {'label': 'ZIII', \
                    'coeff': {'real': 0.28220597164664896, 'imag': 0.0}}, \
                    ... \
                ] \
            } \

        Args:
            dictionary (dict): dictionary, which contains a list of Paulis and coefficients.
            before_04 (bool): support the format before Aqua 0.4.

        Returns:
            Operator: the loaded operator.
        """
        if 'paulis' not in dictionary:
            raise AquaError('Dictionary missing "paulis" key')

        paulis = []
        for op in dictionary['paulis']:
            if 'label' not in op:
                raise AquaError('Dictionary missing "label" key')

            pauli_label = op['label']
            if 'coeff' not in op:
                raise AquaError('Dictionary missing "coeff" key')

            pauli_coeff = op['coeff']
            if 'real' not in pauli_coeff:
                raise AquaError('Dictionary missing "real" key')

            coeff = pauli_coeff['real']
            if 'imag' in pauli_coeff:
                coeff = complex(pauli_coeff['real'], pauli_coeff['imag'])

            pauli_label = pauli_label[::-1] if before_04 else pauli_label
            paulis.append([coeff, Pauli.from_label(pauli_label)])

        return cls(paulis=paulis)

    def to_dict(self):
        """
        Save operator to a dict in pauli representation.

        Returns:
            dict: a dictionary contains an operator with pauli representation.
        """
        ret_dict = {"paulis": []}
        for coeff, pauli in self._paulis:
            op = {"label": pauli.to_label()}
            if isinstance(coeff, complex):
                op["coeff"] = {"real": np.real(coeff),
                               "imag": np.imag(coeff)
                               }
            else:
                op["coeff"] = {"real": coeff}

            ret_dict["paulis"].append(op)

        return ret_dict

    def evaluate_with_statevector(self, quantum_state):
        """

        Args:
            quantum_state (numpy.ndarray): a quantum state.

        Returns:
            float: the mean value
            float: the standard deviation
        """
        # convert to matrix first?
        mat_op = self.to_matrix_operator()
        avg = np.vdot(quantum_state, mat_op.matrix.dot(quantum_state))
        return avg, 0.0

    def construct_evaluation_circuit(self, operator_mode=None, input_circuit=None, backend=None, qr=None, cr=None,
                                     use_simulator_operator_mode=False, wave_function=None, is_statevector=None,
                                     circuit_name_prefix=''):
        """
        Construct the circuits for evaluation, which calculating the expectation <psi|H|psi>.

        Args:
            operator_mode (str): representation of operator, including paulis, grouped_paulis and matrix
            input_circuit (QuantumCircuit): the quantum circuit.
            wave_function (QuantumCircuit): the quantum circuit.
            backend (BaseBackend, optional): backend selection for quantum machine.
            is_statevector (bool, optional): indicate which type of simulator are going to use.
            qr (QuantumRegister, optional): the quantum register associated with the input_circuit
            cr (ClassicalRegister, optional): the classical register associated with the input_circuit
            use_simulator_operator_mode (bool, optional): if aer_provider is used, we can do faster
                                                evaluation for pauli mode on statevector simualtion
            circuit_name_prefix (str, optional): a prefix of circuit name

        Returns:
            list[QuantumCircuit]: a list of quantum circuits and each circuit with a unique name:
                                  circuit_name_prefix + Pauli string

        Raises:
            AquaError: Can not find quantum register with `q` as the name and do not provide
                       quantum register explicitly
            AquaError: The provided qr is not in the input_circuit
            AquaError: Neither backend nor is_statevector is provided
        """
        if operator_mode is not None:
            warnings.warn("operator_mode option is deprecated and it will be removed after 0.6, "
                          "Every operator knows which mode is using, not need to indicate the mode.", DeprecationWarning)

        if input_circuit is not None:
            warnings.warn("input_circuit option is deprecated and it will be removed after 0.6, "
                          "Use `wave_function` instead.", DeprecationWarning)
            wave_function = input_circuit
        else:
            if wave_function is None:
                raise AquaError("wave_function must not be None.")

        if qr is None:
            qr = find_regs_by_name(wave_function, 'q')
            if qr is None:
                raise AquaError("Either providing the quantum register (qr) explicitly"
                                "or used `q` as the name in the input circuit.")
        else:
            if not wave_function.has_register(qr):
                raise AquaError("The provided QuantumRegister (qr) is not in the circuit.")

        if backend is not None:
            warnings.warn("backend option is deprecated and it will be removed after 0.6, "
                          "Use `is_statevector` instead", DeprecationWarning)
            is_statevector = is_statevector_backend(backend)
        else:
            if is_statevector is None:
                raise AquaError("Either backend or is_statevector need to be provided.")

        if is_statevector:
            if use_simulator_operator_mode:
                circuits = [wave_function.copy(name=circuit_name_prefix + 'aer_mode')]
            else:
                n_qubits = self.num_qubits
                circuits = [wave_function.copy(name=circuit_name_prefix + 'psi')]
                for _, pauli in self._paulis:
                    circuit = wave_function.copy(name=circuit_name_prefix + pauli.to_label())
                    if np.all(np.logical_not(pauli.z)) and np.all(np.logical_not(pauli.x)):  # all I
                        continue
                    for qubit_idx in range(n_qubits):
                        if not pauli.z[qubit_idx] and pauli.x[qubit_idx]:
                            circuit.u3(pi, 0.0, pi, qr[qubit_idx])  # x
                        elif pauli.z[qubit_idx] and not pauli.x[qubit_idx]:
                            circuit.u1(pi, qr[qubit_idx])  # z
                        elif pauli.z[qubit_idx] and pauli.x[qubit_idx]:
                            circuit.u3(pi, pi/2, pi/2, qr[qubit_idx])  # y
                    circuits.append(circuit)
        else:
            n_qubits = self.num_qubits
            circuits = []
            base_circuit = QuantumCircuit() + wave_function
            if cr is not None:
                if not base_circuit.has_register(cr):
                    base_circuit.add_register(cr)
            else:
                cr = find_regs_by_name(base_circuit, 'c', qreg=False)
                if cr is None:
                    cr = ClassicalRegister(n_qubits, name='c')
                    base_circuit.add_register(cr)

            for basis, indices in self._basis:
                circuit = base_circuit.copy(name=circuit_name_prefix + basis.to_label())
                for qubit_idx in range(n_qubits):
                    if basis.x[qubit_idx]:
                        if basis.z[qubit_idx]:
                            # Measure Y
                            circuit.u1(-np.pi/2, qr[qubit_idx])  # sdg
                            circuit.u2(0.0, pi, qr[qubit_idx])  # h
                        else:
                            # Measure X
                            circuit.u2(0.0, pi, qr[qubit_idx])  # h
                circuit.barrier(qr)
                circuit.measure(qr, cr)
                circuits.append(circuit)

        return circuits

    # def evaluation_instruction(self, is_statevector, use_simulator_operator_mode=False):
    #     """
    #
    #     Args:
    #         is_statevector (bool): will it be run on statevector simulator or not
    #         use_simulator_operator_mode: will it use qiskit aer simulator operator mode
    #
    #     Returns:
    #         OrderedDict: Pauli-instruction pair.
    #     """
    #     # TODO:
    #     pass
    #     instructions = {}
    #     if is_statevector:
    #         if use_simulator_operator_mode:
    #             instructions['aer_mode'] = Instruction('aer_mode', self.num_qubits)
    #         else:
    #             instructions['psi'] = Instruction('psi', self.num_qubits)
    #             for _, pauli in self._paulis:
    #                 inst = Instruction(pauli.to_label(), self.num_qubits)
    #                 if np.all(np.logical_not(pauli.z)) and np.all(np.logical_not(pauli.x)):  # all I
    #                     continue
    #                 for qubit_idx in range(self.num_qubits):
    #                     if not pauli.z[qubit_idx] and pauli.x[qubit_idx]:
    #                         inst.u3(pi, 0.0, pi, qr[qubit_idx])  # x gate
    #                     elif pauli.z[qubit_idx] and not pauli.x[qubit_idx]:
    #                         inst.u1(pi, qubit_idx)  # z gate
    #                     elif pauli.z[qubit_idx] and pauli.x[qubit_idx]:
    #                         inst.u3(pi, pi / 2, pi / 2, qubit_idx)  # y gate
    #                 instructions[pauli.to_label()] = inst
    #     else:
    #         for basis, indices in self._basis:
    #             inst = Instruction(basis.to_label(), self.num_qubits)
    #             for qubit_idx in range(self.num_qubits):
    #                 if basis.x[qubit_idx]:
    #                     if basis.z[qubit_idx]:  # pauli Y
    #                         inst.u1(pi / 2, qubit_idx).inverse()  # s
    #                         inst.u2(0.0, pi, qubit_idx)  # h
    #                     else:  # pauli X
    #                         inst.u2(0.0, pi, qubit_idx)  # h
    #             instructions[basis.to_label()] = inst
    #
    #     return instructions

    def evaluate_with_result(self, operator_mode=None, circuits=None, backend=None, result=None,
                             use_simulator_operator_mode=False, is_statevector=None,
                             circuit_name_prefix=''):
        """
        This method can be only used with the circuits generated by the `construct_evaluation_circuit` method
        with the same `circuit_name_prefix` since the circuit names are tied to some meanings.

        Calculate the evaluated value with the measurement results.

        Args:
            operator_mode (str): representation of operator, including paulis, grouped_paulis and matrix
            circuits (list of qiskit.QuantumCircuit): the quantum circuits.
            result (qiskit.Result): the result from the backend.
            backend (BaseBackend, optional): backend for quantum machine.
            is_statevector (bool, optional): indicate which type of simulator are used.
            use_simulator_operator_mode (bool): if aer_provider is used, we can do faster
                           evaluation for pauli mode on statevector simualtion
            circuit_name_prefix (str): a prefix of circuit name

        Returns:
            float: the mean value
            float: the standard deviation
        """
        if operator_mode is not None:
            warnings.warn("operator_mode option is deprecated and it will be removed after 0.6, "
                          "Every operator knows which mode is using, not need to indicate the mode.", DeprecationWarning)
        if circuits is not None:
            warnings.warn("circuits option is deprecated and it will be removed after 0.6, "
                          "we will retrieve the circuit via its unique name directly.", DeprecationWarning)

        avg, std_dev, variance = 0.0, 0.0, 0.0
        if backend is not None:
            warnings.warn("backend option is deprecated and it will be removed after 0.6, "
                          "Use `is_statevector` instead", DeprecationWarning)
            is_statevector = is_statevector_backend(backend)
        else:
            if is_statevector is None:
                raise AquaError("Either backend or is_statevector need to be provided.")

        if is_statevector:
            if use_simulator_operator_mode:
                temp = result.data(circuit_name_prefix + 'aer_mode')['snapshots']['expectation_value']['test'][0]['value']
                avg = temp[0] + 1j * temp[1]
            else:
                quantum_state = np.asarray(result.get_statevector(circuit_name_prefix + 'psi'))
                for weight, pauli in self._paulis:
                    if np.all(np.logical_not(pauli.z)) and np.all(np.logical_not(pauli.x)):  # all I
                        avg += weight
                    else:
                        quantum_state_i = np.asarray(result.get_statevector(circuit_name_prefix + pauli.to_label()))
                        avg += weight * (np.vdot(quantum_state, quantum_state_i))
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the expectation from measurement results:")
                TextProgressBar(sys.stderr)
            # pick the first result to get the total number of shots
            num_shots = sum(list(result.get_counts(0).values()))
            results = parallel_map(WeightedPauliOperator._routine_compute_mean_and_var,
                                   [([self._paulis[idx] for idx in indices],
                                     result.get_counts(circuit_name_prefix + basis.to_label()))
                                    for basis, indices in self._basis],
                                   num_processes=aqua_globals.num_processes)
            for result in results:
                avg += result[0]
                variance += result[1]
            std_dev = np.sqrt(variance / num_shots)
        return avg, std_dev

    @staticmethod
    def _routine_compute_mean_and_var(args):
        paulis, measured_results = args
        avg_paulis = []
        avg = 0.0
        variance = 0.0
        for weight, pauli in paulis:
            observable = measure_pauli_z(measured_results, pauli)
            avg += weight * observable
            avg_paulis.append(observable)

        for idx_1, weighted_pauli_1 in enumerate(paulis):
            weight_1, pauli_1 = weighted_pauli_1
            for idx_2, weighted_pauli_2 in enumerate(paulis):
                weight_2, pauli_2 = weighted_pauli_2
                variance += weight_1 * weight_2 * covariance(measured_results, pauli_1, pauli_2,
                                                             avg_paulis[idx_1], avg_paulis[idx_2])

        return avg, variance

    def reorder_paulis(self):
        """
        Reorder the paulis based on the basis and return the reordered paulis.

        Returns:
            [[complex, paulis]]: the ordered paulis based on the basis.
        """

        # if each pauli belongs to its group, no reordering it needed.
        if len(self._basis) == len(self._paulis):
            return self._paulis

        paulis = []
        new_basis = []
        curr_count = 0
        for basis, indices in self._basis:
            sub_paulis = []
            for idx in indices:
                sub_paulis.append(self._paulis[idx])
            new_basis.append((basis, range(curr_count, curr_count + len(sub_paulis))))
            paulis.extend(sub_paulis)
            curr_count += len(sub_paulis)

        self._paulis = paulis
        self._basis = new_basis

        return self._paulis

    def evolve(self, state_in=None, evo_time=0, num_time_slices=1, quantum_registers=None,
               expansion_mode='trotter', expansion_order=1):
        """
        Carry out the eoh evolution for the operator under supplied specifications.

        Args:
            state_in (QuantumCircuit): a circuit describes the input state
            evo_time (int): The evolution time
            num_time_slices (int): The number of time slices for the expansion
            quantum_registers (QuantumRegister): The QuantumRegister to build the QuantumCircuit off of
            expansion_mode (str): The mode under which the expansion is to be done.
                Currently support 'trotter', which follows the expansion as discussed in
                http://science.sciencemag.org/content/273/5278/1073,
                and 'suzuki', which corresponds to the discussion in
                https://arxiv.org/pdf/quant-ph/0508139.pdf
            expansion_order (int): The order for suzuki expansion

        Returns:
            The constructed QuantumCircuit.

        """
        # pylint: disable=no-member
        if num_time_slices <= 0 or not isinstance(num_time_slices, int):
            raise ValueError('Number of time slices should be a non-negative integer.')
        if expansion_mode not in ['trotter', 'suzuki']:
            raise NotImplementedError('Expansion mode {} not supported.'.format(expansion_mode))

        if quantum_registers is None:
            quantum_registers = QuantumRegister(self.num_qubits)
        # TODO: sanity check between register and qc

        pauli_list = self.reorder_paulis()

        if len(pauli_list) == 1:
            slice_pauli_list = pauli_list
        else:
            if expansion_mode == 'trotter':
                slice_pauli_list = pauli_list
            # suzuki expansion
            else:
                slice_pauli_list = suzuki_expansion_slice_pauli_list(
                    pauli_list,
                    1,
                    expansion_order
                )
        instruction = evolution_instruction(slice_pauli_list, evo_time, num_time_slices)
        qc = QuantumCircuit(quantum_registers)
        qc.append(instruction, quantum_registers)
        return qc

    def find_Z2_symmetries(self):
        """
        Finds Z2 Pauli-type symmetries of an Operator.

        Returns:
            [Pauli]: the list of Pauli objects representing the Z_2 symmetries
            [Pauli]: the list of single - qubit Pauli objects to construct the Cliffors operators
            [WeightedPauliOperator]: the list of Clifford unitaries to block diagonalize Operator
            [int]: the list of support of the single-qubit Pauli objects used to build the clifford operators
        """

        pauli_symmetries = []
        sq_paulis = []
        cliffords = []
        sq_list = []

        stacked_paulis = []

        if self.is_empty():
            logger.info("Operator is empty.")
            # TODO: return None or empty list?
            return [], [], [], []

        for pauli in self._paulis:
            stacked_paulis.append(np.concatenate((pauli[1].x, pauli[1].z), axis=0).astype(np.int))

        stacked_matrix = np.array(np.stack(stacked_paulis))
        symmetries = kernel_F2(stacked_matrix)

        if len(symmetries) == 0:
            logger.info("No symmetry is found.")
            return [], [], [], []

        stacked_symmetries = np.stack(symmetries)
        symm_shape = stacked_symmetries.shape

        for row in range(symm_shape[0]):

            pauli_symmetries.append(Pauli(stacked_symmetries[row, : symm_shape[1] // 2],
                                          stacked_symmetries[row, symm_shape[1] // 2:]))

            stacked_symm_del = np.delete(stacked_symmetries, (row), axis=0)
            for col in range(symm_shape[1] // 2):
                # case symmetries other than one at (row) have Z or I on col qubit
                Z_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (stacked_symm_del[symm_idx, col] == 0
                            and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] in (0, 1)):
                        Z_or_I = False
                if Z_or_I:
                    if ((stacked_symmetries[row, col] == 1 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 0) or
                            (stacked_symmetries[row, col] == 1 and
                             stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2),
                                               np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = False
                        sq_paulis[row].x[col] = True
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row) have X or I on col qubit
                X_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (stacked_symm_del[symm_idx, col] in (0, 1) and
                            stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0):
                        X_or_I = False
                if X_or_I:
                    if ((stacked_symmetries[row, col] == 0 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 1) or
                            (stacked_symmetries[row, col] == 1 and
                             stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = True
                        sq_paulis[row].x[col] = False
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row)  have Y or I on col qubit
                Y_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not ((stacked_symm_del[symm_idx, col] == 1 and
                             stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 1)
                            or (stacked_symm_del[symm_idx, col] == 0 and
                                stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0)):
                        Y_or_I = False
                if Y_or_I:
                    if ((stacked_symmetries[row, col] == 0 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 1) or
                            (stacked_symmetries[row, col] == 1 and
                             stacked_symmetries[row, col + symm_shape[1] // 2] == 0)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = True
                        sq_paulis[row].x[col] = True
                        sq_list.append(col)
                        break

        for sq_pauli, pauli_symm in zip(sq_paulis, pauli_symmetries):
            clifford = WeightedPauliOperator(paulis=[[1 / np.sqrt(2), pauli_symm], [1 / np.sqrt(2), sq_pauli]])
            cliffords.append(clifford)

        return pauli_symmetries, sq_paulis, cliffords, sq_list

    @classmethod
    def load_from_file(cls, file_name, before_04=False):
        warnings.warn("load_from_file is deprecated and it will be removed after 0.6, "
                      "Use `from_file` instead", DeprecationWarning)
        return cls.from_file(file_name, before_04)

    def save_to_file(self, file_name):
        warnings.warn("save_to_file is deprecated and it will be removed after 0.6, "
                      "Use `to_file` instead", DeprecationWarning)
        self.to_file(file_name)

    @classmethod
    def load_from_dict(cls, dictionary, before_04=False):
        warnings.warn("load_from_dict is deprecated and it will be removed after 0.6, "
                      "Use `from_dict` instead", DeprecationWarning)
        return cls.from_dict(dictionary, before_04)

    def save_to_dict(self):
        warnings.warn("save_to_dict is deprecated and it will be removed after 0.6, "
                      "Use `to_dict` instead", DeprecationWarning)
        return self.to_dict()

    def print_operators(self):
        warnings.warn("print_operators() is deprecated and it will be removed after 0.6, "
                      "Use `print_details()` instead", DeprecationWarning)

        return self.print_details()

    def _simplify_paulis(self):
        warnings.warn("_simplify_paulis() is deprecated and it will be removed after 0.6, "
                      "Use `simplify()` instead", DeprecationWarning)
        self.simplify()

    def _eval_directly(self, quantum_state):
        warnings.warn("_eval_directly() is deprecated and it will be removed after 0.6, "
                      "Use `evaluate_with_statevector()` instead, and it returns tuple (mean, std) now.",
                      DeprecationWarning)
        return self.evaluate_with_statevector(quantum_state)

    def get_flat_pauli_list(self):
        warnings.warn("get_flat_pauli_list() is deprecated and it will be removed after 0.6. "
                      "Use `reorder_paulis()` instead", DeprecationWarning)
        return self.reorder_paulis()

    def scaling_coeff(self, scaling_factor):
        warnings.warn("scaling_coeff method is deprecated and it will be removed after 0.6. "
                      "Use `* operator` with the scalar directly.", DeprecationWarning)
        self._scaling_weight(scaling_factor)

    def zeros_coeff_elimination(self):
        warnings.warn("zeros_coeff_elimination method is deprecated and it will be removed after 0.6. "
                      "Use chop(0.0) to remove terms with 0 weight.", DeprecationWarning)
        self.chop(0.0)

    def to_grouped_paulis(self):
        warnings.warn("to_grouped_paulis method is deprecated and it will be removed after 0.6. "
                      "Use `to_grouped_weighted_pauli_operator` and providing your own grouping func.",
                      DeprecationWarning)

    def to_matrix(self):
        warnings.warn("to_matrix method is deprecated and it will be removed after 0.6. "
                      "Use `to_matrix_operator` instead.",
                      DeprecationWarning)
        return self.to_matrix_operator()
