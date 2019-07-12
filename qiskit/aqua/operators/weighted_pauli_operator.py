# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy
import itertools
from functools import reduce
import logging
import json
from operator import iadd as op_iadd, isub as op_isub
import sys
from collections import OrderedDict
import warnings

import numpy as np
from scipy import sparse as scisparse
from scipy import linalg as scila
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction
from qiskit.quantum_info import Pauli
from qiskit.qasm import pi
from qiskit.assembler.run_config import RunConfig
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua.utils import PauliGraph, compile_and_run_circuits, find_regs_by_name
from qiskit.aqua.utils.backend_utils import is_statevector_backend

from qiskit.aqua.operators.base_operator import BaseOperator
from qiskit.aqua.operators.common import measure_pauli_z, covariance, kernel_F2, suzuki_expansion_slice_pauli_list, check_commutativity


logger = logging.getLogger(__name__)


class WeightedPauliOperator(BaseOperator):

    def __init__(self, paulis, basis=None, atol=1e-12):
        """
        Args:
            paulis ([[complex, Pauli]]): the list of weighted Paulis, where a weighted pauli is composed of
                                         a length-2 list and the first item is the weight and
                                         the second item is the Pauli object.
            basis (list[tuple(object, [int])]): the grouping basis, each element is a tuple composed of the basis
                                                and the indices to paulis which are belonged to that group.
                                                e.g., if tpb basis is used, the object will be a pauli.
                                                by default, the group is equal to non-grouping, each pauli is its own basis.
        """
        # plain store the paulis, the group information is store in the basis
        self._paulis_table = None
        self._paulis = paulis
        # combine the paulis and remove those with zero weight
        self.simplify()
        self._basis = [(pauli[1], [i]) for i, pauli in enumerate(paulis)] if basis is None else basis
        self._aer_paulis = None
        self._atol = atol

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
        other = other.simplify()
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

    def _extend_or_combine(self, other, mode, operation=op_iadd):
        """
        Add two operators either extend (in-place) or combine (copy) them.
        The addition performs optimized combiniation of two operators.
        If `other` has identical basis, the coefficient are combined rather than
        appended.

        Args:
            other (Operator): to-be-combined operator
            mode (str): in-place or not.

        Returns:
            Operator: the operator.

        Raises:
            ValueError: the mode are not in ['inplace', 'non-inplace']
        """

        if mode not in ['inplace', 'non-inplace']:
            ValueError("'mode' should be either 'inplace' or 'inplace' but {} is specified.".format(mode))

        lhs = self if mode == 'inplace' else  self.copy()

        for pauli in other.paulis:
            pauli_label = pauli[1].to_label()
            idx = lhs._paulis_table.get(pauli_label, None)
            if idx is not None:
                lhs._paulis[idx][0] = operation(lhs._paulis[idx][0], pauli[0])
            else:
                lhs._paulis_table[pauli_label] = len(lhs._paulis)
                pauli[0] = operation(0.0, pauli[0])
                lhs._paulis.append(pauli)

        return lhs

    def __add__(self, other):
        """Overload + operator."""
        return self._extend_or_combine(other, 'non-inplace', op_iadd)

    def __iadd__(self, other):
        """Overload += operator."""
        return self._extend_or_combine(other, 'inplace', op_iadd)

    def __sub__(self, other):
        """Overload - operator."""
        return self._extend_or_combine(other, 'non-inplace', op_isub)

    def __isub__(self, other):
        """Overload -= operator."""
        return self._extend_or_combine(other, 'inplace', op_isub)

    def __mul__(self, other):
        """Overload * operator."""
        ret_pauli = WeightedPauliOperator(paulis=[])
        for existed_weight, existed_pauli in self._paulis:
            for weight, pauli in other._paulis:
                new_pauli, sign = Pauli.sgn_prod(existed_pauli, pauli)
                new_weight = existed_weight * weight * sign
                if abs(new_weight) > self._atol:
                    pauli_term = [new_weight, new_pauli]
                    ret_pauli += WeightedPauliOperator(paulis=[pauli_term])
        return ret_pauli

    def __neg__(self):
        """Overload unary -."""
        return self.copy().scaling(-1.0)

    def __str__(self):
        """Overload str()."""
        curr_repr = 'paulis'
        length = len(self._paulis)
        ret = "Representation: {}, qubits: {}, size: {}".format(curr_repr, self.num_qubits, length)
        return ret

    # TODO: figure out a good name?
    def print_operators(self):
        """
        Print out the operator in details.

        Returns:
            str: a formated operator.

        Raises:
            ValueError: if `print_format` is not supported.
        """

        if self.is_empty():
            return "Pauli list is empty."
        ret = ""
        for weight, pauli in self._paulis:
            ret = ''.join([ret, "{}\t{}\n".format(pauli.to_label(), weight)])
        return ret

    def copy(self):
        """Get a copy of self."""
        return copy.deepcopy(self)

    def simplify(self):
        """
        #TODO: note change the behavior
        Merge the paulis whose bases are identical and the pauli with zero coefficient
        would be removed.

        Usually used in construction.
        """
        new_paulis = []
        new_paulis_table = {}
        for curr_weight, curr_pauli in self._paulis:
            pauli_label = curr_pauli.to_label()
            new_idx = new_paulis_table.get(pauli_label, None)
            if new_idx is not None:
                new_paulis[new_idx][0] += curr_weight
            else:
                new_paulis_table[pauli_label] = len(new_paulis)
                new_paulis.append([curr_weight, curr_pauli])

        self._paulis = new_paulis
        self.remove_zero_weights()
        # self._paulis_table = new_paulis_table
        return self

    # def zeros_coeff_elimination(self):
    def remove_zero_weights(self):
        """
        Elinminate paulis whose weights are zeros.

        The difference from `_simplify_paulis` method is that, this method will not remove duplicated
        paulis.
        """
        new_paulis = [[weight, pauli] for weight, pauli in self._paulis if weight != 0]
        self._paulis = new_paulis
        self._paulis_table = {pauli[1].to_label(): i for i, pauli in enumerate(self._paulis)}

        return self

    def scaling(self, scaling_factor):
        """
        Constantly scaling all weights of paulis.

        Args:
            scaling_factor (complex): the scaling factor

        Returns:
            WeightedPauliOperator: self, the scaled one.
        """
        for idx in range(len(self._paulis)):
            self._paulis[idx] = [self._paulis[idx][0] * scaling_factor, self._paulis[idx][1]]
        return self

    def is_commute(self, other):
        return check_commutativity(self, other)

    def is_anticommute(self, other):
        return check_commutativity(self, other, anti=True)

    # TODO: need this shortcut method?
    def evaluate_with_statevector(self, quantum_state):
        # convert to matrix first?
        matrix = self.to_operator()
        avg = np.vdot(quantum_state, matrix.dot(quantum_state))
        return avg

    def to_matrix(self):
        """

        Returns:
            MatrixOperator:

        Raises:
            AquaError: the operator is empty.

        """
        if self.is_empty():
            raise AquaError("Can not convert an empty WeightedPauliOperator to MatrixOperator.")

        hamiltonian = 0
        for weight, pauli in self._paulis:
            hamiltonian += weight * pauli.to_spmatrix()
        return MatrixOperator(matrix=hamiltonian)

    def chop(self, threshold=None):
        """
        Eliminate the real and imagine part of weight in each pauli by `threshold`.
        If pauli's weight is less then `threshold` in both real and imagine parts, the pauli is removed.

        Note:
            If weight is real-only, the imag part is skipped.

        Args:
            threshold (float): the threshold is used to remove the paulis

        Returns:
            WeightedPauliOperator

        Raises:
            AquaError: if operator is empty
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

        if self.is_empty():
            raise AquaError("Operator is empty.")

        for i in range(len(self._paulis)):
            self._paulis[i][0] = chop_real_imag(self._paulis[i][0])
        paulis = [[weight, pauli] for weight, pauli in self._paulis if weight != 0.0]
        self._paulis = paulis
        self._paulis_table = {pauli[1].to_label(): i for i, pauli in enumerate(self._paulis)}
        return self

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

    @classmethod
    def load_from_file(cls, file_name, before_04=False):
        """
        Load paulis in a file to construct an Operator.

        Args:
            file_name (str): path to the file, which contains a list of Paulis and coefficients.
            before_04 (bool): support the format before Aqua 0.4.

        Returns:
            Operator class: the loaded operator.
        """
        with open(file_name, 'r') as file:
            return cls.load_from_dict(json.load(file), before_04=before_04)

    def save_to_file(self, file_name):
        """
        Save operator to a file in pauli representation.

        Args:
            file_name (str): path to the file

        """
        with open(file_name, 'w') as f:
            json.dump(self.save_to_dict(), f)

    @classmethod
    def load_from_dict(cls, dictionary, before_04=False):
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

    def save_to_dict(self):
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

    def construct_evaluation_circuit(self, wave_function, backend=None, is_statevector=None, qr=None, cr=None,
                                     use_simulator_operator_mode=False, circuit_name_prefix=''):
        """
        Construct the circuits for evaluation, which calculating the expectation <psi|H|psi>.

        Args:
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
                    circuit = QuantumCircuit(name=circuit_name_prefix + pauli.to_label()) + wave_function
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
                circuit = QuantumCircuit(name=circuit_name_prefix + basis.to_label()) + base_circuit
                for qubit_idx in range(n_qubits):
                    if basis.x[qubit_idx]:
                        if basis.z[qubit_idx]:
                            # Measure Y
                            circuit.u1(pi/2, qr[qubit_idx]).inverse()  # s
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

    def evaluate_with_result(self, result, backend=None, is_statevector=None, use_simulator_operator_mode=False,
                             circuit_name_prefix=''):
        """
        This method can be only used with the circuits generated by the `construct_evaluation_circuit` method
        with the same `circuit_name_prefix` since the circuit names are tied to some meanings.

        Calculate the evaluated value with the measurement results.

        Args:
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
        avg, std_dev, variance = 0.0, 0.0, 0.0
        is_statevector = is_statevector_backend(backend)

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
                                   [([self._paulis[idx] for idx in indices], result.get_counts(basis.to_label()))
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

        for idx_1, weighted_pauli_1 in paulis:
            weight_1, pauli_1 = weighted_pauli_1
            for idx_2, weighted_pauli_2 in paulis:
                weight_2, pauli_2 = weighted_pauli_2
                variance += weight_1 * weight_2 * covariance(measured_results, pauli_1, pauli_2,
                                                             avg_paulis[idx_1], avg_paulis[idx_2])

        return avg, variance

    def to_grouped_paulis(self, grouping_func=None, **kwargs):
        """

        Args:
            grouping_func (Callable): a grouping callback to group paulis, and this callback will be fed with the paulis
                                      and kwargs arguments
            kwargs: other arguments needed for grouping func.

        Returns:
            object: the type depending on the `grouping_func`.
        """
        return grouping_func(self._paulis, **kwargs)

    @staticmethod
    def construct_evolution_circuit(slice_pauli_list, evo_time, num_time_slices, state_registers,
                                    ancillary_registers=None, ctl_idx=0, unitary_power=None, use_basis_gates=True,
                                    shallow_slicing=False):
        """
        Construct the evolution circuit according to the supplied specification.

        Args:
            slice_pauli_list (list): The list of pauli terms corresponding to a single time slice to be evolved
            evo_time (int): The evolution time
            num_time_slices (int): The number of time slices for the expansion
            state_registers (QuantumRegister): The Qiskit QuantumRegister corresponding to the qubits of the system
            ancillary_registers (QuantumRegister): The optional Qiskit QuantumRegister corresponding to the control
                qubits for the state_registers of the system
            ctl_idx (int): The index of the qubit of the control ancillary_registers to use
            unitary_power (int): The power to which the unitary operator is to be raised
            use_basis_gates (bool): boolean flag for indicating only using basis gates when building circuit.
            shallow_slicing (bool): boolean flag for indicating using shallow qc.data reference repetition for slicing

        Returns:
            QuantumCircuit: The Qiskit QuantumCircuit corresponding to specified evolution.

        """
        qc = QuantumCircuit(state_registers)
        instruction = WeightedPauliOperator.evolution_instruction(slice_pauli_list, evo_time, num_time_slices,
                                                                  ancillary_registers, ctl_idx, unitary_power,
                                                                  use_basis_gates, shallow_slicing)
        if ancillary_registers is None:
            qc.append(instruction, state_registers)
        else:
            qc.append(instruction, [state_registers, ancillary_registers])
        return qc

    @staticmethod
    def evolution_instruction(slice_pauli_list, evo_time, num_time_slices, ancillary_registers=None,
                              ctl_idx=0, unitary_power=None, use_basis_gates=True, shallow_slicing=False):
        """
        Construct the evolution circuit according to the supplied specification.

        Args:
            slice_pauli_list (list): The list of pauli terms corresponding to a single time slice to be evolved
            evo_time (int): The evolution time
            num_time_slices (int): The number of time slices for the expansion
            ancillary_registers (QuantumRegister): The optional Qiskit QuantumRegister corresponding to the control
                qubits for the state_registers of the system
            ctl_idx (int): The index of the qubit of the control ancillary_registers to use
            unitary_power (int): The power to which the unitary operator is to be raised
            use_basis_gates (bool): boolean flag for indicating only using basis gates when building circuit.
            shallow_slicing (bool): boolean flag for indicating using shallow qc.data reference repetition for slicing

        Returns:
            QuantumCircuit: The Qiskit QuantumCircuit corresponding to specified evolution.
        """
        state_registers = QuantumRegister(slice_pauli_list[0][1].numberofqubits)
        qc_slice = QuantumCircuit(state_registers, name='Evolution')
        if ancillary_registers is not None:
            qc_slice.add_register(ancillary_registers)

        # for each pauli [IXYZ]+, record the list of qubit pairs needing CX's
        cnot_qubit_pairs = [None] * len(slice_pauli_list)
        # for each pauli [IXYZ]+, record the highest index of the nontrivial pauli gate (X,Y, or Z)
        top_XYZ_pauli_indices = [-1] * len(slice_pauli_list)

        for pauli_idx, pauli in enumerate(reversed(slice_pauli_list)):
            n_qubits = pauli[1].numberofqubits
            # changes bases if necessary
            nontrivial_pauli_indices = []
            for qubit_idx in range(n_qubits):
                # pauli I
                if not pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
                    continue

                if cnot_qubit_pairs[pauli_idx] is None:
                    nontrivial_pauli_indices.append(qubit_idx)

                if pauli[1].x[qubit_idx]:
                    # pauli X
                    if not pauli[1].z[qubit_idx]:
                        if use_basis_gates:
                            qc_slice.u2(0.0, pi, state_registers[qubit_idx])
                        else:
                            qc_slice.h(state_registers[qubit_idx])
                    # pauli Y
                    elif pauli[1].z[qubit_idx]:
                        if use_basis_gates:
                            qc_slice.u3(pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                        else:
                            qc_slice.rx(pi / 2, state_registers[qubit_idx])
                # pauli Z
                elif pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
                    pass
                else:
                    raise ValueError('Unrecognized pauli: {}'.format(pauli[1]))

            if len(nontrivial_pauli_indices) > 0:
                top_XYZ_pauli_indices[pauli_idx] = nontrivial_pauli_indices[-1]

            # insert lhs cnot gates
            if cnot_qubit_pairs[pauli_idx] is None:
                cnot_qubit_pairs[pauli_idx] = list(zip(
                    sorted(nontrivial_pauli_indices)[:-1],
                    sorted(nontrivial_pauli_indices)[1:]
                ))

            for pair in cnot_qubit_pairs[pauli_idx]:
                qc_slice.cx(state_registers[pair[0]], state_registers[pair[1]])

            # insert Rz gate
            if top_XYZ_pauli_indices[pauli_idx] >= 0:
                if ancillary_registers is None:
                    lam = (2.0 * pauli[0] * evo_time / num_time_slices).real
                    if use_basis_gates:
                        qc_slice.u1(lam, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    else:
                        qc_slice.rz(lam, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                else:
                    unitary_power = (2 ** ctl_idx) if unitary_power is None else unitary_power
                    lam = (2.0 * pauli[0] * evo_time / num_time_slices * unitary_power).real

                    if use_basis_gates:
                        qc_slice.u1(lam / 2, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc_slice.cx(ancillary_registers[ctl_idx], state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc_slice.u1(-lam / 2, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc_slice.cx(ancillary_registers[ctl_idx], state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    else:
                        qc_slice.crz(lam, ancillary_registers[ctl_idx],
                                     state_registers[top_XYZ_pauli_indices[pauli_idx]])

            # insert rhs cnot gates
            for pair in reversed(cnot_qubit_pairs[pauli_idx]):
                qc_slice.cx(state_registers[pair[0]], state_registers[pair[1]])

            # revert bases if necessary
            for qubit_idx in range(n_qubits):
                if pauli[1].x[qubit_idx]:
                    # pauli X
                    if not pauli[1].z[qubit_idx]:
                        if use_basis_gates:
                            qc_slice.u2(0.0, pi, state_registers[qubit_idx])
                        else:
                            qc_slice.h(state_registers[qubit_idx])
                    # pauli Y
                    elif pauli[1].z[qubit_idx]:
                        if use_basis_gates:
                            qc_slice.u3(-pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                        else:
                            qc_slice.rx(-pi / 2, state_registers[qubit_idx])

        # repeat the slice
        if shallow_slicing:
            logger.info('Under shallow slicing mode, the qc.data reference is repeated shallowly. '
                        'Thus, changing gates of one slice of the output circuit might affect other slices.')
            qc_slice.data *= num_time_slices
            qc = qc_slice
        else:
            qc = QuantumCircuit()
            for _ in range(num_time_slices):
                qc += qc_slice
        return qc.to_instruction()

    def evolve(self, evo_time=0, num_time_slices=1, expansion_mode='trotter', expansion_order=1, qr=None):
        """
        Carry out the eoh evolution for the operator under supplied specifications.

        Args:
            evo_time (int): The evolution time
            num_time_slices (int): The number of time slices for the expansion
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

        if qr is None:
            qr = QuantumRegister(self.num_qubits)
        pauli_list = self._paulis

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
        circuit = self.construct_evolution_circuit(slice_pauli_list, evo_time, num_time_slices, qr)
        return circuit

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
            #TODO: return None or empty list?
            return [], [], [], []

        for pauli in self._paulis:
            stacked_paulis.append(np.concatenate((pauli[1].x, pauli[1].z), axis=0).astype(np.int))

        stacked_matrix = np.array(np.stack(stacked_paulis))
        symmetries = kernel_F2(stacked_matrix)

        if len(symmetries) == 0:
            logger.info("No symmetry is found.")
            # TODO: return None or empty list?
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
