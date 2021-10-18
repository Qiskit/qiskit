# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Readout mitigator class based on the tensored A-matrix inversion method
"""

import logging
from typing import Optional, List, Tuple, Iterable, Callable, Union
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.result import Counts, marginal_counts, QuasiDistribution

logger = logging.getLogger(__name__)

class TensoredReadoutMitigator():
    """Tensored 1-qubit readout error mitigator.
    Mitigates expectation_value and quasi_probabilities.
    The mitigation_matrix should be calibrated using qiskit.experiments."""
    def __init__(self, cal_matrices: np.matrix,
                 substate_labels_list: list):
        self._cal_matrices = cal_matrices
        self._qubit_list_sizes = []
        self._indices_list = []
        self._substate_labels_list = []
        self.substate_labels_list = substate_labels_list

    @substate_labels_list.setter
    def substate_labels_list(self, new_substate_labels_list):
        """Return _substate_labels_list"""
        self._substate_labels_list = new_substate_labels_list

        # get the number of qubits in each subspace
        self._qubit_list_sizes = []
        for _, substate_label_list in enumerate(self._substate_labels_list):
            self._qubit_list_sizes.append(
                int(np.log2(len(substate_label_list))))

        # get the indices in the calibration matrix
        self._indices_list = []
        for _, sub_labels in enumerate(self._substate_labels_list):
            self._indices_list.append(
                {lab: ind for ind, lab in enumerate(sub_labels)})

    @property
    def nqubits(self):
        """Return the number of qubits. See also MeasurementFilter.apply() """
        return sum(self._qubit_list_sizes)

    def apply(self, data: Counts, method: Optional[str]='least_squares'):
        """
        Apply the calibration matrices to results.

        Args:
            data: The data to be corrected.
            method: fitting method. The following methods are supported:

                * 'pseudo_inverse': direct inversion of the cal matrices.

                * 'least_squares': constrained to have physical probabilities.

                * If `None`, 'least_squares' is used.

        Returns:
            dict or Result: The corrected data in the same form as raw_data

        Raises:
            QiskitError: if raw_data is not in a one of the defined forms.
        """

        all_states = count_keys(self.nqubits)
        num_of_states = 2**self.nqubits

        counts_list = np.zeros(num_of_states, dtype=float)
        for state, count in raw_data.items():
            stateidx = int(state, 2)
            counts_list[stateidx] = count

        if method == 'pseudo_inverse':
            mitigated_counts_list = self.apply_pseudo_inverse(counts_list)

        elif method == 'least_squares':
            mitigated_counts_list = self.apply_least_squares(counts_list)

        # convert back into a counts dictionary
        new_count_dict = {}
        for state_idx, state in enumerate(all_states):
            if mitigated_counts_list[state_idx] != 0:
                new_count_dict[state] = mitigated_counts_list[state_idx]

        return new_count_dict


        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == 'pseudo_inverse':
                inv_mat_dot_raw = np.zeros([num_of_states], dtype=float)
                for state1_idx, state1 in enumerate(all_states):
                    for state2_idx, state2 in enumerate(all_states):
                        if raw_data2[data_idx][state2_idx] == 0:
                            continue

                        product = 1.
                        end_index = self.nqubits
                        for p_ind, pinv_mat in enumerate(pinv_cal_matrices):

                            start_index = end_index - \
                                self._qubit_list_sizes[p_ind]

                            state1_as_int = \
                                self._indices_list[p_ind][
                                    state1[start_index:end_index]]

                            state2_as_int = \
                                self._indices_list[p_ind][
                                    state2[start_index:end_index]]

                            end_index = start_index
                            product *= \
                                pinv_mat[state1_as_int][state2_as_int]
                            if product == 0:
                                break
                        inv_mat_dot_raw[state1_idx] += \
                            (product * raw_data2[data_idx][state2_idx])
                raw_data2[data_idx] = inv_mat_dot_raw

    def apply_pseudo_inverse(self, counts_list):
        all_states = count_keys(self.nqubits)
        num_of_states = 2 ** self.nqubits
        pinv_cal_matrices = []
        for cal_mat in self._cal_matrices:
            pinv_cal_matrices.append(la.pinv(cal_mat))
        inv_mat_dot_raw = np.zeros([num_of_states], dtype=float)

        for state1_idx, state1 in enumerate(all_states):
            for state2_idx, state2 in enumerate(all_states):
                if counts_list[state2_idx] == 0:
                    continue

                product = 1.
                end_index = self.nqubits
                for p_ind, pinv_mat in enumerate(pinv_cal_matrices):

                    start_index = end_index - \
                                  self._qubit_list_sizes[p_ind]

                    state1_as_int = \
                        self._indices_list[p_ind][
                            state1[start_index:end_index]]

                    state2_as_int = \
                        self._indices_list[p_ind][
                            state2[start_index:end_index]]

                    end_index = start_index
                    product *= \
                        pinv_mat[state1_as_int][state2_as_int]
                    if product == 0:
                        break
                inv_mat_dot_raw[state1_idx] += \
                    (product * counts_list[state2_idx])
        return inv_mat_dot_raw

    def apply_least_squares(self, counts_list):
        all_states = count_keys(self.nqubits)
        num_of_states = 2 ** self.nqubits

        def fun(x):
            mat_dot_x = np.zeros([num_of_states], dtype=float)
            for state1_idx, state1 in enumerate(all_states):
                mat_dot_x[state1_idx] = 0.
                for state2_idx, state2 in enumerate(all_states):
                    if x[state2_idx] != 0:
                        product = 1.
                        end_index = self.nqubits
                        for c_ind, cal_mat in \
                                enumerate(self._cal_matrices):

                            start_index = end_index - \
                                          self._qubit_list_sizes[c_ind]

                            state1_as_int = \
                                self._indices_list[c_ind][
                                    state1[start_index:end_index]]

                            state2_as_int = \
                                self._indices_list[c_ind][
                                    state2[start_index:end_index]]

                            end_index = start_index
                            product *= \
                                cal_mat[state1_as_int][state2_as_int]
                            if product == 0:
                                break
                        mat_dot_x[state1_idx] += \
                            (product * x[state2_idx])
            return sum(
                (counts_list - mat_dot_x) ** 2)


        x0 = np.random.rand(num_of_states)
        x0 = x0 / sum(x0)
        nshots = sum(counts_list)
        cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
        bnds = tuple((0, nshots) for x in x0)
        res = minimize(fun, x0, method='SLSQP',
                       constraints=cons, bounds=bnds, tol=1e-6)
        return res.x