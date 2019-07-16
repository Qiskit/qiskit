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

import copy

from qiskit.aqua.operators.weighted_pauli_operator import WeightedPauliOperator
from qiskit.aqua.operators.pauli_graph import PauliGraph

def _post_format_conversion(grouped_paulis):
    # TODO: edit the codes without applying post formatting.
    basis = []
    paulis = []

    total_idx = 0
    for idx, tpb in enumerate(grouped_paulis):
        curr_basis = tpb[0][1]
        curr_paulis = tpb[1:]
        basis.append((curr_basis, list(range(total_idx, total_idx+len(curr_paulis)))))
        paulis.extend(curr_paulis)
        total_idx += len(curr_paulis)

    return basis, paulis


class TPBGroupedWeightedPauliOperator(WeightedPauliOperator):

    def __init__(self, paulis, basis, grouping_func=None, atol=1e-12, name=None, kwargs=None):
        super().__init__(paulis, basis, atol, name=name)
        self._grouping_func = grouping_func
        self._kwargs = kwargs or {}

    @property
    def num_groups(self):
        return len(self._basis)

    @property
    def grouping_func(self):
        return self._grouping_func

    # TODO: naming
    @classmethod
    def sorted_grouping(cls, paulis, method="largest-degree", name=None):
        p = PauliGraph(paulis, method)
        basis, paulis = _post_format_conversion(p.grouped_paulis)
        kwargs = {'method': method}
        return cls(paulis, basis, cls.sorted_grouping, name, kwargs)

    @classmethod
    def unsorted_grouping(cls, paulis, name=None):

        if len(paulis) == 0:
            return paulis

        temp_paulis = copy.deepcopy(paulis)
        n = paulis[0][1].numberofqubits
        grouped_paulis = []
        sorted_paulis = []

        def check_pauli_in_list(target, pauli_list):
            ret = False
            for pauli in pauli_list:
                if target[1] == pauli[1]:
                    ret = True
                    break
            return ret

        for i in range(len(temp_paulis)):
            p_1 = temp_paulis[i]
            if not check_pauli_in_list(p_1, sorted_paulis):
                paulis_temp = []
                # pauli_list_temp.extend(p_1) # this is going to signal the total
                # post-rotations of the set (set master)
                paulis_temp.append(p_1)
                paulis_temp.append(copy.deepcopy(p_1))
                paulis_temp[0][0] = 0.0  # zero coeff for HEADER
                indices = []
                for j in range(i + 1, len(temp_paulis)):
                    p_2 = temp_paulis[j]
                    if not check_pauli_in_list(p_2, sorted_paulis) and p_1[1] != p_2[1]:
                        j = 0
                        for i in range(n):
                            # p_2 is identity, p_1 is identity, p_1 and p_2 has same basis
                            if not ((not p_2[1].z[i] and not p_2[1].x[i]) or
                                    (not p_1[1].z[i] and not p_1[1].x[i]) or
                                    (p_2[1].z[i] == p_1[1].z[i] and
                                     p_2[1].x[i] == p_1[1].x[i])):
                                break
                            else:
                                # update master, if p_2 is not identity
                                if p_2[1].z[i] or p_2[1].x[i]:
                                    paulis_temp[0][1].update_z(p_2[1].z[i], i)
                                    paulis_temp[0][1].update_x(p_2[1].x[i], i)
                            j += 1
                        if j == n:
                            paulis_temp.append(p_2)
                            sorted_paulis.append(p_2)
                grouped_paulis.append(paulis_temp)

        basis, paulis = _post_format_conversion(grouped_paulis)
        return cls(paulis, basis, cls.unsorted_grouping, name=name)


    def __str__(self):
        """Overload str()."""
        curr_repr = 'tpb grouped paulis'
        length = len(self._paulis)
        name = "" if self._name is None else "{}: ".format(self._name)
        ret = "{}Representation: {}, qubits: {}, size: {}, group: {}".format(name, curr_repr,
                                                                             self.num_qubits, length, len(self._basis))
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
        for basis, indices in self._basis:
            ret = ''.join([ret, "TPB: {} ({})\n".format(basis.to_label(), len(indices))])
            for idx in indices:
                weight, pauli = self._paulis[idx]
                ret = ''.join([ret, "{}\t{}\n".format(pauli.to_label(), weight)])

        return ret

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

        TODO: is there any incremental approach for grouping?
        """
        # perform add or sub in paulis and then re-group it again
        ret_op = super()._add_or_sub(other, operation, copy)
        ret_op = ret_op.to_grouped_paulis(self._grouping_func, **self._kwargs)
        return ret_op

    def multiply(self, other):
        """
        Perform self * other.

        Args:
            other (WeightedPauliOperator): an operator

        Returns:
            WeightedPauliOperator: the multiplied operator
        """
        ret_op = super().multiply(other)
        ret_op = ret_op.to_grouped_paulis(self._grouping_func, **self._kwargs)
        return ret_op
