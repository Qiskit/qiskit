# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Weighted Pauli Operator """

from typing import List, Optional, Tuple, Union
from copy import deepcopy
import itertools
import logging
import json
from operator import add as op_add, sub as op_sub
import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import AquaError, aqua_globals
from .base_operator import LegacyBaseOperator
from .common import (measure_pauli_z, covariance, pauli_measurement,
                     kernel_F2, suzuki_expansion_slice_pauli_list,
                     check_commutativity, evolution_instruction)

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name

class WeightedPauliOperator(LegacyBaseOperator):
    """ Weighted Pauli Operator """

    def __init__(self,
                 paulis: List[List[Union[complex, Pauli]]],
                 basis: Optional[List[Tuple[object, List[int]]]] = None,
                 z2_symmetries: 'Z2Symmetries' = None,
                 atol: float = 1e-12,
                 name: Optional[str] = None) -> None:
        """
        Args:
            paulis: the list of weighted Paulis, where a weighted pauli is
                    composed of a length-2 list and the first item is the
                    weight and the second item is the Pauli object.
            basis: the grouping basis, each element is a tuple composed of the basis
                    and the indices to paulis which belong to that group.
                    e.g., if tpb basis is used, the object will be a pauli.
                    By default, the group is equal to non-grouping, each pauli is its own basis.
            z2_symmetries: recording the z2 symmetries info
            atol: the threshold used in truncating paulis
            name: the name of operator.
        """
        super().__init__(basis, z2_symmetries, name)
        # plain store the paulis, the group information is store in the basis
        self._paulis_table = None
        self._paulis = paulis
        self._basis = \
            [(pauli[1], [i]) for i, pauli in enumerate(paulis)] if basis is None else basis
        # combine the paulis and remove those with zero weight
        self.simplify()
        self._atol = atol

    @classmethod
    def from_list(cls, paulis, weights=None, name=None):
        """
        Create a WeightedPauliOperator via a pair of list.

        Args:
            paulis (list[Pauli]): the list of Paulis
            weights (list[complex], optional): the list of weights,
                                               if it is None, all weights are 1.
            name (str, optional): name of the operator.

        Returns:
            WeightedPauliOperator: operator

        Raises:
            ValueError: The length of weights and paulis must be the same
        """
        if weights is not None and len(weights) != len(paulis):
            raise ValueError("The length of weights and paulis must be the same.")
        if weights is None:
            weights = [1.0] * len(paulis)
        return cls(paulis=[[w, p] for w, p in zip(weights, paulis)], name=name)

    # pylint: disable=arguments-differ
    def to_opflow(self, reverse_endianness=False):
        """ to op flow """
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.operators import PrimitiveOp

        pauli_ops = []
        for [w, p] in self.paulis:
            pauli = Pauli.from_label(str(p)[::-1]) if reverse_endianness else p
            # Adding the imaginary is necessary to handle the imaginary coefficients in UCCSD.
            # TODO fix these or add support for them in Terra.
            pauli_ops += [PrimitiveOp(pauli, coeff=np.real(w) + np.imag(w))]
        return sum(pauli_ops)

    @property
    def paulis(self):
        """ get paulis """
        return self._paulis

    @property
    def atol(self):
        """ get atol """
        return self._atol

    @atol.setter
    def atol(self, new_value):
        """ set atol """
        self._atol = new_value

    @property
    def num_qubits(self):
        """
        Number of qubits required for the operator.

        Returns:
            int: number of qubits
        """
        if not self.is_empty():
            return self._paulis[0][1].num_qubits
        else:
            logger.warning("Operator is empty, Return 0.")
            return 0

    def __eq__(self, other):
        """ Overload == operation """
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
        The addition performs optimized combination of two operators.
        If `other` has identical basis, the coefficient are combined rather than
        appended.

        Args:
            other (WeightedPauliOperator): to-be-combined operator
            operation (callable or str): add or sub callable from operator
            copy (bool): working on a copy or self

        Returns:
            WeightedPauliOperator: operator

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
                new_pauli = deepcopy(pauli)
                ret_op._paulis_table[pauli_label] = len(ret_op._paulis)
                ret_op._basis.append((new_pauli[1], [len(ret_op._paulis)]))
                new_pauli[0] = operation(0.0, pauli[0])
                ret_op._paulis.append(new_pauli)
        return ret_op

    def add(self, other, copy=False):
        """
        Perform self + other.

        Args:
            other (WeightedPauliOperator): to-be-combined operator
            copy (bool): working on a copy or self, if False, the results are written back to self.

        Returns:
            WeightedPauliOperator: operator
        """

        return self._add_or_sub(other, op_add, copy=copy)

    def sub(self, other, copy=False):
        """
        Perform self - other.

        Args:
            other (WeightedPauliOperator): to-be-combined operator
            copy (bool): working on a copy or self, if False, the results are written back to self.

        Returns:
            WeightedPauliOperator: operator
        """

        return self._add_or_sub(other, op_sub, copy=copy)

    def __add__(self, other):
        """ Overload + operator """
        return self.add(other, copy=True)

    def __iadd__(self, other):
        """ Overload += operator """
        return self.add(other, copy=False)

    def __sub__(self, other):
        """ Overload - operator """
        return self.sub(other, copy=True)

    def __isub__(self, other):
        """ Overload -= operator """
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
            raise ValueError(
                "Type of scaling factor is a valid type. {} if given.".format(
                    scaling_factor.__class__))
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
        """ Overload other * self """
        if isinstance(other, (int, float, complex, np.int, np.float, np.complex)):
            return self._scaling_weight(other, copy=True)
        else:
            return other.multiply(self)

    def __mul__(self, other):
        """ Overload self * other """
        if isinstance(other, (int, float, complex, np.int, np.float, np.complex)):
            return self._scaling_weight(other, copy=True)
        else:
            return self.multiply(other)

    def __neg__(self):
        """Overload unary - """
        return self._scaling_weight(-1.0, copy=True)

    def __str__(self):
        """ Overload str() """
        curr_repr = 'paulis'
        length = len(self._paulis)
        name = "" if self._name == '' else "{}: ".format(self._name)
        ret = "{}Representation: {}, qubits: {}, size: {}".format(name,
                                                                  curr_repr,
                                                                  self.num_qubits, length)
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
        """ Get a copy of self """
        return deepcopy(self)

    def simplify(self, copy=False):
        """
        Merge the paulis whose bases are identical and the pauli with zero coefficient
        would be removed.

        Note:
            This behavior of this method is slightly changed,
            it will remove the paulis whose weights are zero.

        Args:
            copy (bool): simplify on a copy or self

        Returns:
            WeightedPauliOperator: the simplified operator
        """

        op = self.copy() if copy else self

        new_paulis = []
        new_paulis_table = {}
        old_to_new_indices = {}
        curr_idx = 0
        for curr_weight, curr_pauli in op.paulis:
            pauli_label = curr_pauli.to_label()
            new_idx = new_paulis_table.get(pauli_label, None)
            if new_idx is not None:
                new_paulis[new_idx][0] += curr_weight
                old_to_new_indices[curr_idx] = new_idx
            else:
                new_paulis_table[pauli_label] = len(new_paulis)
                old_to_new_indices[curr_idx] = len(new_paulis)
                new_paulis.append([curr_weight, curr_pauli])
            curr_idx += 1

        op._paulis = new_paulis
        op._paulis_table = new_paulis_table

        # update the grouping info, since this method only reduce the number
        # of paulis, we can handle it here for both
        # pauli and tpb grouped pauli
        # should have a better way to rebuild the basis here.
        new_basis = []
        for basis, indices in op.basis:
            new_indices = []
            found = False
            if new_basis:
                for b, ind in new_basis:
                    if b == basis:
                        new_indices = ind
                        found = True
                        break
            for idx in indices:
                new_idx = old_to_new_indices[idx]
                if new_idx is not None and new_idx not in new_indices:
                    new_indices.append(new_idx)
            if new_indices and not found:
                new_basis.append((basis, new_indices))
        op._basis = new_basis
        op.chop(0.0)
        return op

    def rounding(self, decimals, copy=False):
        """
        Rounding the weight.

        Args:
            decimals (int): rounding the weight to the decimals.
            copy (bool): chop on a copy or self

        Returns:
            WeightedPauliOperator: operator
        """
        op = self.copy() if copy else self

        op._paulis = [[np.around(weight, decimals=decimals), pauli] for weight, pauli in op.paulis]

        return op

    def chop(self, threshold=None, copy=False):
        """
        Eliminate the real and imagine part of weight in each pauli by `threshold`.
        If pauli's weight is less then `threshold` in both real and imaginary parts,
        the pauli is removed.

        Note:
            If weight is real-only, the imaginary part is skipped.

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
        op._paulis_table = \
            {weighted_pauli[1].to_label(): i for i, weighted_pauli in enumerate(paulis)}
        # update the grouping info, since this method only remove pauli,
        # we can handle it here for both
        # pauli and tpb grouped pauli
        new_basis = []
        for basis, indices in op.basis:
            new_indices = []
            for idx in indices:
                new_idx = old_to_new_indices.get(idx, None)
                if new_idx is not None:
                    new_indices.append(new_idx)
            if new_indices:
                new_basis.append((basis, new_indices))
        op._basis = new_basis
        return op

    def commute_with(self, other):
        """ Commutes with """
        return check_commutativity(self, other)

    def anticommute_with(self, other):
        """ Anti commutes with """
        return check_commutativity(self, other, anti=True)

    def is_empty(self):
        """
        Check Operator is empty or not.

        Returns:
            bool: True if empty, False otherwise
        """
        if not self._paulis:
            return True
        elif not self._paulis[0]:
            return True
        else:
            return False

    @classmethod
    def from_file(cls, file_name, before_04=False):
        """
        Load paulis in a file to construct an Operator.

        Args:
            file_name (str): path to the file, which contains a list of Paulis and coefficients.
            before_04 (bool): support the format before Aqua 0.4.

        Returns:
            WeightedPauliOperator: the loaded operator.
        """
        with open(file_name, 'r') as file:
            return cls.from_dict(json.load(file), before_04=before_04)

    def to_file(self, file_name):
        """
        Save operator to a file in pauli representation.

        Args:
            file_name (str): path to the file

        """
        with open(file_name, 'w') as file:
            json.dump(self.to_dict(), file)

    @classmethod
    def from_dict(cls, dictionary, before_04=False):
        """
        Load paulis from a dictionary to construct an Operator. The dictionary must
        comprise the key 'paulis' having a value which is an array of pauli dicts.
        Each dict in this array must be represented by label and coeff (real and imag)
        such as in the following example:

        .. code-block:: python

           {'paulis':
               [
                   {'label': 'IIII',
                    'coeff': {'real': -0.33562957575267038, 'imag': 0.0}},
                   {'label': 'ZIII',
                    'coeff': {'real': 0.28220597164664896, 'imag': 0.0}},
                    ...
               ]
            }

        Args:
            dictionary (dict): dictionary, which contains a list of Paulis and coefficients.
            before_04 (bool): support the format before Aqua 0.4.

        Returns:
            WeightedPauliOperator: the operator created from the input dictionary.

        Raises:
            AquaError: Invalid dictionary
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
        Raises:
            AquaError: if Operator is empty
        """
        if self.is_empty():
            raise AquaError("Operator is empty, check the operator.")
        # convert to matrix first?
        # pylint: disable=import-outside-toplevel
        from .op_converter import to_matrix_operator
        mat_op = to_matrix_operator(self)
        avg = np.vdot(quantum_state, mat_op._matrix.dot(quantum_state))
        return avg, 0.0

    # pylint: disable=arguments-differ
    def construct_evaluation_circuit(self, wave_function, statevector_mode,
                                     qr=None, cr=None, use_simulator_snapshot_mode=False,
                                     circuit_name_prefix=''):
        r"""
        Construct the circuits for evaluation, which calculating the expectation <psi\|H\|psi>.

        At statevector mode: to simplify the computation, we do not build the whole
        circuit for <psi|H|psi>, instead of
        that we construct an individual circuit <psi\|, and a bundle circuit for H\|psi>

        Args:
            wave_function (QuantumCircuit): the quantum circuit.
            statevector_mode (bool): indicate which type of simulator are going to use.
            qr (QuantumRegister, optional): the quantum register associated with the input_circuit
            cr (ClassicalRegister, optional): the classical register associated
                                              with the input_circuit
            use_simulator_snapshot_mode (bool, optional): if aer_provider is used, we can do faster
                                                          evaluation for pauli mode on
                                                          statevector simulation
            circuit_name_prefix (str, optional): a prefix of circuit name

        Returns:
            list[QuantumCircuit]: a list of quantum circuits and each circuit with a unique name:
                                  circuit_name_prefix + Pauli string

        Raises:
            AquaError: if Operator is empty
            AquaError: if quantum register is not provided explicitly and
                       cannot find quantum register with `q` as the name
            AquaError: The provided qr is not in the wave_function
        """
        if self.is_empty():
            raise AquaError("Operator is empty, check the operator.")
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if qr is None:
            qr = find_regs_by_name(wave_function, 'q')
            if qr is None:
                raise AquaError("Either provide the quantum register (qr) explicitly or use"
                                " `q` as the name of the quantum register in the input circuit.")
        else:
            if not wave_function.has_register(qr):
                raise AquaError("The provided QuantumRegister (qr) is not in the circuit.")

        n_qubits = self.num_qubits
        instructions = self.evaluation_instruction(statevector_mode, use_simulator_snapshot_mode)
        circuits = []
        if use_simulator_snapshot_mode:
            circuit = wave_function.copy(name=circuit_name_prefix + 'snapshot_mode')
            # Add expectation value snapshot instruction
            instr = instructions.get('expval_snapshot', None)
            if instr is not None:
                circuit.append(instr, qr)
            circuits.append(circuit)
        elif statevector_mode:
            circuits.append(wave_function.copy(name=circuit_name_prefix + 'psi'))
            for _, pauli in self._paulis:
                inst = instructions.get(pauli.to_label(), None)
                if inst is not None:
                    circuit = wave_function.copy(name=circuit_name_prefix + pauli.to_label())
                    circuit.append(inst, qr)
                    circuits.append(circuit)
        else:
            base_circuit = wave_function.copy()
            if cr is not None:
                if not base_circuit.has_register(cr):
                    base_circuit.add_register(cr)
            else:
                cr = find_regs_by_name(base_circuit, 'c', qreg=False)
                if cr is None:
                    cr = ClassicalRegister(n_qubits, name='c')
                    base_circuit.add_register(cr)

            for basis, _ in self._basis:
                circuit = base_circuit.copy(name=circuit_name_prefix + basis.to_label())
                circuit.append(instructions[basis.to_label()], qargs=qr, cargs=cr)
                circuits.append(circuit)

        return circuits

    def evaluation_instruction(self, statevector_mode, use_simulator_snapshot_mode=False):
        """
        Args:
            statevector_mode (bool): will it be run on statevector simulator or not
            use_simulator_snapshot_mode (bool): will it use qiskit aer simulator operator mode

        Returns:
            dict: Pauli-instruction pair.

        Raises:
            AquaError: if Operator is empty
        """
        if self.is_empty():
            raise AquaError("Operator is empty, check the operator.")
        instructions = {}
        qr = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(qr)
        if use_simulator_snapshot_mode and self.paulis:
            # pylint: disable=import-outside-toplevel
            from qiskit.providers.aer.extensions import SnapshotExpectationValue
            snapshot = SnapshotExpectationValue('expval', self.paulis, variance=True)
            instructions = {'expval_snapshot': snapshot}
        elif statevector_mode:
            for _, pauli in self._paulis:
                tmp_qc = qc.copy(name="Pauli " + pauli.to_label())
                if np.all(np.logical_not(pauli.z)) and np.all(np.logical_not(pauli.x)):  # all I
                    continue
                # This explicit barrier is needed for statevector simulator since Qiskit-terra
                # will remove global phase at default compilation level but the results here
                # rely on global phase.
                tmp_qc.barrier(list(range(self.num_qubits)))
                tmp_qc.append(pauli.to_instruction(), list(range(self.num_qubits)))
                instructions[pauli.to_label()] = tmp_qc.to_instruction()
        else:
            cr = ClassicalRegister(self.num_qubits)
            qc.add_register(cr)
            for basis, _ in self._basis:
                tmp_qc = qc.copy(name="Pauli " + basis.to_label())
                tmp_qc = pauli_measurement(tmp_qc, basis, qr, cr, barrier=True)
                instructions[basis.to_label()] = tmp_qc.to_instruction()
        return instructions

    # pylint: disable=arguments-differ
    def evaluate_with_result(self, result, statevector_mode, use_simulator_snapshot_mode=False,
                             circuit_name_prefix=''):
        """
        This method can be only used with the circuits generated by the
        :meth:`construct_evaluation_circuit` method with the same `circuit_name_prefix`
        name since the circuit names are tied to some meanings.

        Calculate the evaluated value with the measurement results.

        Args:
            result (qiskit.Result): the result from the backend.
            statevector_mode (bool): indicate which type of simulator are used.
            use_simulator_snapshot_mode (bool): if aer_provider is used, we can do faster
                                                evaluation for pauli mode on
                                                statevector simulation
            circuit_name_prefix (str): a prefix of circuit name

        Returns:
            float: the mean value
            float: the standard deviation

        Raises:
            AquaError: if Operator is empty
        """
        if self.is_empty():
            raise AquaError("Operator is empty, check the operator.")

        avg, std_dev, variance = 0.0, 0.0, 0.0
        if use_simulator_snapshot_mode:
            snapshot_data = result.data(
                circuit_name_prefix + 'snapshot_mode')['snapshots']
            avg = snapshot_data['expectation_value']['expval'][0]['value']
            if isinstance(avg, (list, tuple)):
                # Aer versions before 0.4 use a list snapshot format
                # which must be converted to a complex value.
                avg = avg[0] + 1j * avg[1]
        elif statevector_mode:
            quantum_state = np.asarray(result.get_statevector(circuit_name_prefix + 'psi'))
            for weight, pauli in self._paulis:
                # all I
                if np.all(np.logical_not(pauli.z)) and np.all(np.logical_not(pauli.x)):
                    avg += weight
                else:
                    quantum_state_i = \
                        result.get_statevector(circuit_name_prefix + pauli.to_label())
                    avg += (weight * (np.vdot(quantum_state, quantum_state_i)))
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
            for res in results:
                avg += res[0]
                variance += res[1]
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

    def reorder_paulis(self) -> List[List[Union[complex, Pauli]]]:
        """
        Reorder the paulis based on the basis and return the reordered paulis.

        Returns:
            the ordered paulis based on the basis.
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

    # pylint: disable=arguments-differ
    def evolve(self, state_in=None, evo_time=0, num_time_slices=1, quantum_registers=None,
               expansion_mode='trotter', expansion_order=1):
        """
        Carry out the eoh evolution for the operator under supplied specifications.

        Args:
            state_in (QuantumCircuit): a circuit describes the input state
            evo_time (Union(complex, float, Parameter, ParameterExpression)): The evolution time
            num_time_slices (int): The number of time slices for the expansion
            quantum_registers (QuantumRegister): The QuantumRegister to build
                                                 the QuantumCircuit off of
            expansion_mode (str): The mode under which the expansion is to be done.
                Currently support 'trotter', which follows the expansion as discussed in
                http://science.sciencemag.org/content/273/5278/1073,
                and 'suzuki', which corresponds to the discussion in
                https://arxiv.org/pdf/quant-ph/0508139.pdf
            expansion_order (int): The order for suzuki expansion

        Returns:
            QuantumCircuit: The constructed circuit.

        Raises:
            AquaError: quantum_registers must be in the provided state_in circuit
            AquaError: if operator is empty
        """
        if self.is_empty():
            raise AquaError("Operator is empty, can not evolve.")

        if state_in is not None and quantum_registers is not None:
            if not state_in.has_register(quantum_registers):
                raise AquaError("quantum_registers must be in the provided state_in circuit.")
        elif state_in is None and quantum_registers is None:
            quantum_registers = QuantumRegister(self.num_qubits)
            qc = QuantumCircuit(quantum_registers)
        elif state_in is not None and quantum_registers is None:
            # assuming the first register is for evolve
            quantum_registers = state_in.qregs[0]
            qc = QuantumCircuit() + state_in
        else:
            qc = QuantumCircuit(quantum_registers)

        instruction = self.evolve_instruction(evo_time, num_time_slices,
                                              expansion_mode, expansion_order)
        qc.append(instruction, quantum_registers)
        return qc

    def evolve_instruction(self, evo_time=0, num_time_slices=1,
                           expansion_mode='trotter', expansion_order=1):
        """
        Carry out the eoh evolution for the operator under supplied specifications.

        Args:
            evo_time (Union(complex, float, Parameter, ParameterExpression)): The evolution time
            num_time_slices (int): The number of time slices for the expansion
            expansion_mode (str): The mode under which the expansion is to be done.
                Currently support 'trotter', which follows the expansion as discussed in
                http://science.sciencemag.org/content/273/5278/1073,
                and 'suzuki', which corresponds to the discussion in
                https://arxiv.org/pdf/quant-ph/0508139.pdf
            expansion_order (int): The order for suzuki expansion

        Returns:
            QuantumCircuit: The constructed QuantumCircuit.

        Raises:
            ValueError: Number of time slices should be a non-negative integer
            NotImplementedError: expansion mode not supported
            AquaError: if operator is empty
        """
        if self.is_empty():
            raise AquaError("Operator is empty, can not build evolve instruction.")
        # pylint: disable=no-member
        if num_time_slices <= 0 or not isinstance(num_time_slices, int):
            raise ValueError('Number of time slices should be a non-negative integer.')
        if expansion_mode not in ['trotter', 'suzuki']:
            raise NotImplementedError('Expansion mode {} not supported.'.format(expansion_mode))

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
        return instruction


class Z2Symmetries:
    """ Z2 Symmetries """

    def __init__(self, symmetries, sq_paulis, sq_list, tapering_values=None):
        """
        Args:
            symmetries (list[Pauli]): the list of Pauli objects representing the Z_2 symmetries
            sq_paulis (list[Pauli]): the list of single - qubit Pauli objects to construct the
                                     Clifford operators
            sq_list (list[int]): the list of support of the single-qubit Pauli objects used to build
                                 the Clifford operators
            tapering_values (list[int], optional): values determines the sector.

        Raises:
            AquaError: Invalid paulis
        """
        if len(symmetries) != len(sq_paulis):
            raise AquaError("Number of Z2 symmetries has to be the same as number "
                            "of single-qubit pauli x.")

        if len(sq_paulis) != len(sq_list):
            raise AquaError("Number of single-qubit pauli x has to be the same "
                            "as length of single-qubit list.")

        if tapering_values is not None:
            if len(sq_list) != len(tapering_values):
                raise AquaError("The length of single-qubit list has "
                                "to be the same as length of tapering values.")

        self._symmetries = symmetries
        self._sq_paulis = sq_paulis
        self._sq_list = sq_list
        self._tapering_values = tapering_values

    @property
    def symmetries(self):
        """ return symmetries """
        return self._symmetries

    @property
    def sq_paulis(self):
        """ returns sq paulis """
        return self._sq_paulis

    @property
    def cliffords(self):
        """
        Get clifford operators, build based on symmetries and single-qubit X.

        Returns:
            list[WeightedPauliOperator]: a list of unitaries used to diagonalize the Hamiltonian.
        """
        cliffords = [WeightedPauliOperator(paulis=[[1 / np.sqrt(2), pauli_symm],
                                                   [1 / np.sqrt(2), sq_pauli]])
                     for pauli_symm, sq_pauli in zip(self._symmetries, self._sq_paulis)]
        return cliffords

    @property
    def sq_list(self):
        """ returns sq list """
        return self._sq_list

    @property
    def tapering_values(self):
        """ returns tapering values """
        return self._tapering_values

    @tapering_values.setter
    def tapering_values(self, new_value):
        """ set tapering values """
        self._tapering_values = new_value

    def __str__(self):
        ret = ["Z2 symmetries:"]
        ret.append("Symmetries:")
        for symmetry in self._symmetries:
            ret.append(symmetry.to_label())
        ret.append("Single-Qubit Pauli X:")
        for x in self._sq_paulis:
            ret.append(x.to_label())
        ret.append("Cliffords:")
        for c in self.cliffords:
            ret.append(c.print_details())
        ret.append("Qubit index:")
        ret.append(str(self._sq_list))
        ret.append("Tapering values:")
        if self._tapering_values is None:
            possible_values = \
                [str(list(coeff)) for coeff in itertools.product([1, -1],
                                                                 repeat=len(self._sq_list))]
            possible_values = ', '.join(x for x in possible_values)
            ret.append("  - Possible values: " + possible_values)
        else:
            ret.append(str(self._tapering_values))

        ret = "\n".join(ret)
        return ret

    def copy(self) -> 'Z2Symmetries':
        """
        Get a copy of self.

        Returns:
            copy
        """
        return deepcopy(self)

    def is_empty(self):
        """
        Check the z2_symmetries is empty or not.

        Returns:
            bool: empty
        """
        if self._symmetries != [] and self._sq_paulis != [] and self._sq_list != []:
            return False
        else:
            return True

    @classmethod
    def find_Z2_symmetries(cls, operator) -> 'Z2Symmetries':  # pylint: disable=invalid-name
        """
        Finds Z2 Pauli-type symmetries of an Operator.

        Returns:
            a z2_symmetries object contains symmetries,
            single-qubit X, single-qubit list.
        """
        # pylint: disable=invalid-name
        pauli_symmetries = []
        sq_paulis = []
        sq_list = []

        stacked_paulis = []

        if operator.is_empty():
            logger.info("Operator is empty.")
            return cls([], [], [], None)

        for pauli in operator.paulis:
            stacked_paulis.append(np.concatenate((pauli[1].x, pauli[1].z), axis=0).astype(np.int))

        stacked_matrix = np.array(np.stack(stacked_paulis))
        symmetries = kernel_F2(stacked_matrix)

        if not symmetries:
            logger.info("No symmetry is found.")
            return cls([], [], [], None)

        stacked_symmetries = np.stack(symmetries)
        symm_shape = stacked_symmetries.shape

        for row in range(symm_shape[0]):

            pauli_symmetries.append(Pauli(stacked_symmetries[row, : symm_shape[1] // 2],
                                          stacked_symmetries[row, symm_shape[1] // 2:]))

            stacked_symm_del = np.delete(stacked_symmetries, row, axis=0)
            for col in range(symm_shape[1] // 2):
                # case symmetries other than one at (row) have Z or I on col qubit
                Z_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (stacked_symm_del[symm_idx, col] == 0
                            and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] in (0, 1)):
                        Z_or_I = False
                if Z_or_I:
                    if ((stacked_symmetries[row, col] == 1
                         and stacked_symmetries[row, col + symm_shape[1] // 2] == 0)
                            or (stacked_symmetries[row, col] == 1
                                and stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2),
                                               np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = False
                        sq_paulis[row].x[col] = True
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row) have X or I on col qubit
                X_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (stacked_symm_del[symm_idx, col] in (0, 1)
                            and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0):
                        X_or_I = False
                if X_or_I:
                    if ((stacked_symmetries[row, col] == 0
                         and stacked_symmetries[row, col + symm_shape[1] // 2] == 1)
                            or (stacked_symmetries[row, col] == 1
                                and stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2),
                                               np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = True
                        sq_paulis[row].x[col] = False
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row)  have Y or I on col qubit
                Y_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not ((stacked_symm_del[symm_idx, col] == 1
                             and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 1)
                            or (stacked_symm_del[symm_idx, col] == 0
                                and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0)):
                        Y_or_I = False
                if Y_or_I:
                    if ((stacked_symmetries[row, col] == 0
                         and stacked_symmetries[row, col + symm_shape[1] // 2] == 1)
                            or (stacked_symmetries[row, col] == 1
                                and stacked_symmetries[row, col + symm_shape[1] // 2] == 0)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2),
                                               np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = True
                        sq_paulis[row].x[col] = True
                        sq_list.append(col)
                        break

        return cls(pauli_symmetries, sq_paulis, sq_list, None)

    def taper(self, operator, tapering_values=None):
        """
        Taper an operator based on the z2_symmetries info and sector defined by `tapering_values`.
        The `tapering_values` will be stored into the resulted operator for a record.

        Args:
            operator (WeightedPauliOperator): the to-be-tapered operator.
            tapering_values (list[int], optional): if None, returns operators at each sector;
                                                   otherwise, returns
                                                   the operator located in that sector.
        Returns:
            list[WeightedPauliOperator] or WeightedPauliOperator: If
                tapering_values is None: [:class`WeightedPauliOperator`];
                otherwise, :class:`WeightedPauliOperator`

        Raises:
            AquaError: Z2 symmetries, single qubit pauli and single qubit list cannot be empty
        """
        if not self._symmetries or not self._sq_paulis or not self._sq_list:
            raise AquaError("Z2 symmetries, single qubit pauli and "
                            "single qubit list cannot be empty.")

        if operator.is_empty():
            logger.warning("The operator is empty, return the empty operator directly.")
            return operator

        for clifford in self.cliffords:
            operator = clifford * operator * clifford

        tapering_values = tapering_values if tapering_values is not None else self._tapering_values

        def _taper(op, curr_tapering_values):
            z2_symmetries = self.copy()
            z2_symmetries.tapering_values = curr_tapering_values
            operator_out = WeightedPauliOperator(paulis=[], z2_symmetries=z2_symmetries,
                                                 name=operator.name)
            for pauli_term in op.paulis:
                coeff_out = pauli_term[0]
                for idx, qubit_idx in enumerate(self._sq_list):
                    if not (not pauli_term[1].z[qubit_idx] and not pauli_term[1].x[qubit_idx]):
                        coeff_out = curr_tapering_values[idx] * coeff_out
                z_temp = np.delete(pauli_term[1].z.copy(), np.asarray(self._sq_list))
                x_temp = np.delete(pauli_term[1].x.copy(), np.asarray(self._sq_list))
                pauli_term_out = WeightedPauliOperator(paulis=[[coeff_out, Pauli(z_temp, x_temp)]])
                operator_out += pauli_term_out
            operator_out.chop(0.0)
            return operator_out

        if tapering_values is None:
            tapered_ops = []
            for coeff in itertools.product([1, -1], repeat=len(self._sq_list)):
                tapered_ops.append(_taper(operator, list(coeff)))
        else:
            tapered_ops = _taper(operator, tapering_values)

        return tapered_ops

    @staticmethod
    def two_qubit_reduction(operator, num_particles):
        """
        Eliminates the central and last qubit in a list of Pauli that has
        diagonal operators (Z,I) at those positions

        Chemistry specific method:
        It can be used to taper two qubits in parity and binary-tree mapped
        fermionic Hamiltonians when the spin orbitals are ordered in two spin
        sectors, (block spin order) according to the number of particles in the system.

        Args:
            operator (WeightedPauliOperator): the operator
            num_particles (Union(list, int)): number of particles, if it is a list,
                                              the first number is alpha
                                              and the second number if beta.

        Returns:
            WeightedPauliOperator: a new operator whose qubit number is reduced by 2.

        """
        if operator.is_empty():
            logger.info("Operator is empty, can not do two qubit reduction. "
                        "Return the empty operator back.")
            return operator

        if isinstance(num_particles, list):
            num_alpha = num_particles[0]
            num_beta = num_particles[1]
        else:
            num_alpha = num_particles // 2
            num_beta = num_particles // 2

        par_1 = 1 if (num_alpha + num_beta) % 2 == 0 else -1
        par_2 = 1 if num_alpha % 2 == 0 else -1
        tapering_values = [par_2, par_1]

        num_qubits = operator.num_qubits
        last_idx = num_qubits - 1
        mid_idx = num_qubits // 2 - 1
        sq_list = [mid_idx, last_idx]

        # build symmetries, sq_paulis:
        symmetries, sq_paulis = [], []
        for idx in sq_list:
            pauli_str = ['I'] * num_qubits

            pauli_str[idx] = 'Z'
            z_sym = Pauli.from_label(''.join(pauli_str)[::-1])
            symmetries.append(z_sym)

            pauli_str[idx] = 'X'
            sq_pauli = Pauli.from_label(''.join(pauli_str)[::-1])
            sq_paulis.append(sq_pauli)

        z2_symmetries = Z2Symmetries(symmetries, sq_paulis, sq_list, tapering_values)
        return z2_symmetries.taper(operator)

    def consistent_tapering(self, operator):
        """
        Tapering the `operator` with the same manner of how this tapered operator
        is created. i.e., using the same Cliffords and tapering values.

        Args:
            operator (WeightedPauliOperator): the to-be-tapered operator

        Returns:
            TaperedWeightedPauliOperator: the tapered operator

        Raises:
            AquaError: The given operator does not commute with the symmetry
        """
        if operator.is_empty():
            raise AquaError("Can not taper an empty operator.")

        for symmetry in self._symmetries:
            if not operator.commute_with(symmetry):
                raise AquaError("The given operator does not commute with "
                                "the symmetry, can not taper it.")

        return self.taper(operator)
