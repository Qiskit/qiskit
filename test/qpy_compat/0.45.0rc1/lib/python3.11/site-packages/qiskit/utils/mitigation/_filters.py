# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This code was originally copied from the qiskit-ignis see:
# https://github.com/Qiskit/qiskit-ignis/blob/b91066c72171bcd55a70e6e8993b813ec763cf41/qiskit/ignis/mitigation/measurement/filters.py
# it was migrated as qiskit-ignis is being deprecated

# pylint: disable=cell-var-from-loop


"""
Measurement correction filters.

"""

from typing import List
from copy import deepcopy

import numpy as np

import qiskit
from qiskit import QiskitError
from qiskit.tools import parallel_map
from qiskit.utils.mitigation.circuits import count_keys
from qiskit.utils.deprecation import deprecate_func


class MeasurementFilter:
    """
    Deprecated: Measurement error mitigation filter.

    Produced from a measurement calibration fitter and can be applied
    to data.

    """

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
    )
    def __init__(self, cal_matrix: np.matrix, state_labels: list):
        """
        Initialize a measurement error mitigation filter using the cal_matrix
        from a measurement calibration fitter.

        Args:
            cal_matrix: the calibration matrix for applying the correction
            state_labels: the states for the ordering of the cal matrix
        """

        self._cal_matrix = cal_matrix
        self._state_labels = state_labels

    @property
    def cal_matrix(self):
        """Return cal_matrix."""
        return self._cal_matrix

    @property
    def state_labels(self):
        """return the state label ordering of the cal matrix"""
        return self._state_labels

    @state_labels.setter
    def state_labels(self, new_state_labels):
        """set the state label ordering of the cal matrix"""
        self._state_labels = new_state_labels

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """Set cal_matrix."""
        self._cal_matrix = new_cal_matrix

    def apply(self, raw_data, method="least_squares"):
        """Apply the calibration matrix to results.

        Args:
            raw_data (dict or list): The data to be corrected. Can be in a number of forms:

                 Form 1: a counts dictionary from results.get_counts

                 Form 2: a list of counts of `length==len(state_labels)`

                 Form 3: a list of counts of `length==M*len(state_labels)` where M is an
                 integer (e.g. for use with the tomography data)

                 Form 4: a qiskit Result

            method (str): fitting method. If `None`, then least_squares is used.

                ``pseudo_inverse``: direct inversion of the A matrix

                ``least_squares``: constrained to have physical probabilities

        Returns:
            dict or list: The corrected data in the same form as `raw_data`

        Raises:
            QiskitError: if `raw_data` is not an integer multiple
                of the number of calibrated states.

        """
        from scipy.optimize import minimize
        from scipy import linalg as la

        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            for data_label in raw_data.keys():
                if data_label not in self._state_labels:
                    raise QiskitError(
                        f"Unexpected state label '{data_label}'."
                        " Verify the fitter's state labels correspond to the input data."
                    )

            data_format = 0
            # convert to form2
            raw_data2 = [np.zeros(len(self._state_labels), dtype=float)]
            for stateidx, state in enumerate(self._state_labels):
                raw_data2[0][stateidx] = raw_data.get(state, 0)

        elif isinstance(raw_data, list):
            size_ratio = len(raw_data) / len(self._state_labels)
            if len(raw_data) == len(self._state_labels):
                data_format = 1
                raw_data2 = [raw_data]
            elif int(size_ratio) == size_ratio:
                data_format = 2
                size_ratio = int(size_ratio)
                # make the list into chunks the size of state_labels for easier
                # processing
                raw_data2 = np.zeros([size_ratio, len(self._state_labels)])
                for i in range(size_ratio):
                    raw_data2[i][:] = raw_data[
                        i * len(self._state_labels) : (i + 1) * len(self._state_labels)
                    ]
            else:
                raise QiskitError(
                    "Data list is not an integer multiple of the number of calibrated states"
                )

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            new_counts_list = parallel_map(
                self._apply_correction,
                [resultidx for resultidx, _ in enumerate(raw_data.results)],
                task_args=(raw_data, method),
            )

            for resultidx, new_counts in new_counts_list:
                new_result.results[resultidx].data.counts = new_counts

            return new_result

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        if method == "pseudo_inverse":
            pinv_cal_mat = la.pinv(self._cal_matrix)

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == "pseudo_inverse":
                raw_data2[data_idx] = np.dot(pinv_cal_mat, raw_data2[data_idx])

            elif method == "least_squares":
                nshots = sum(raw_data2[data_idx])

                def fun(x):
                    return sum((raw_data2[data_idx] - np.dot(self._cal_matrix, x)) ** 2)

                x0 = np.random.rand(len(self._state_labels))
                x0 = x0 / sum(x0)
                cons = {"type": "eq", "fun": lambda x: nshots - sum(x)}
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bnds, tol=1e-6)
                raw_data2[data_idx] = res.x

            else:
                raise QiskitError("Unrecognized method.")

        if data_format == 2:
            # flatten back out the list
            raw_data2 = raw_data2.flatten()

        elif data_format == 0:
            # convert back into a counts dictionary
            new_count_dict = {}
            for stateidx, state in enumerate(self._state_labels):
                if raw_data2[0][stateidx] != 0:
                    new_count_dict[state] = raw_data2[0][stateidx]

            raw_data2 = new_count_dict
        else:
            # TODO: should probably change to:
            # raw_data2 = raw_data2[0].tolist()
            raw_data2 = raw_data2[0]
        return raw_data2

    def _apply_correction(self, resultidx, raw_data, method):
        """Wrapper to call apply with a counts dictionary."""
        new_counts = self.apply(raw_data.get_counts(resultidx), method=method)
        return resultidx, new_counts


class TensoredFilter:
    """
    Deprecated: Tensored measurement error mitigation filter.

    Produced from a tensored measurement calibration fitter and can be applied
    to data.
    """

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
    )
    def __init__(self, cal_matrices: np.matrix, substate_labels_list: list, mit_pattern: list):
        """
        Initialize a tensored measurement error mitigation filter using
        the cal_matrices from a tensored measurement calibration fitter.
        A simple usage this class is explained [here]
        (https://qiskit.org/documentation/tutorials/noise/3_measurement_error_mitigation.html).

        Args:
            cal_matrices: the calibration matrices for applying the correction.
            substate_labels_list: for each calibration matrix
                a list of the states (as strings, states in the subspace)
            mit_pattern: for each calibration matrix
                a list of the logical qubit indices (as int, states in the subspace)
        """

        self._cal_matrices = cal_matrices
        self._qubit_list_sizes = []
        self._indices_list = []
        self._substate_labels_list = []
        self.substate_labels_list = substate_labels_list
        self._mit_pattern = mit_pattern

    @property
    def cal_matrices(self):
        """Return cal_matrices."""
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        """Set cal_matrices."""
        self._cal_matrices = deepcopy(new_cal_matrices)

    @property
    def substate_labels_list(self):
        """Return _substate_labels_list"""
        return self._substate_labels_list

    @substate_labels_list.setter
    def substate_labels_list(self, new_substate_labels_list):
        """Return _substate_labels_list"""
        self._substate_labels_list = new_substate_labels_list

        # get the number of qubits in each subspace
        self._qubit_list_sizes = []
        for _, substate_label_list in enumerate(self._substate_labels_list):
            self._qubit_list_sizes.append(int(np.log2(len(substate_label_list))))

        # get the indices in the calibration matrix
        self._indices_list = []
        for _, sub_labels in enumerate(self._substate_labels_list):

            self._indices_list.append({lab: ind for ind, lab in enumerate(sub_labels)})

    @property
    def qubit_list_sizes(self):
        """Return _qubit_list_sizes."""
        return self._qubit_list_sizes

    @property
    def nqubits(self):
        """Return the number of qubits. See also MeasurementFilter.apply()"""
        return sum(self._qubit_list_sizes)

    def apply(
        self,
        raw_data,
        method="least_squares",
        meas_layout=None,
    ):
        """
        Apply the calibration matrices to results.

        Args:
            raw_data (dict or Result): The data to be corrected. Can be in one of two forms:

                * A counts dictionary from results.get_counts

                * A Qiskit Result

            method (str): fitting method. The following methods are supported:

                * 'pseudo_inverse': direct inversion of the cal matrices.
                    Mitigated counts can contain negative values
                    and the sum of counts would not equal to the shots.
                    Mitigation is conducted qubit wise:
                    For each qubit, mitigate the whole counts using the calibration matrices
                    which affect the corresponding qubit.
                    For example, assume we are mitigating the 3rd bit of the 4-bit counts
                    using '2\times 2' calibration matrix `A_3`.
                    When mitigating the count of '0110' in this step,
                    the following formula is applied:
                    `count['0110'] = A_3^{-1}[1, 0]*count['0100'] + A_3^{-1}[1, 1]*count['0110']`.

                    The total time complexity of this method is `O(m2^{n + t})`,
                    where `n` is the size of calibrated qubits,
                    `m` is the number of sets in `mit_pattern`,
                    and `t` is the size of largest set of mit_pattern.
                    If the `mit_pattern` is shaped like `[[0], [1], [2], ..., [n-1]]`,
                    which corresponds to the tensor product noise model without cross-talk,
                    then the time complexity would be `O(n2^n)`.
                    If the `mit_pattern` is shaped like `[[0, 1, 2, ..., n-1]]`,
                    which exactly corresponds to the complete error mitigation,
                    then the time complexity would be `O(2^(n+n)) = O(4^n)`.


                * 'least_squares': constrained to have physical probabilities.
                    Instead of directly applying inverse calibration matrices,
                    this method solve a constrained optimization problem to find
                    the closest probability vector to the result from 'pseudo_inverse' method.
                    Sequential least square quadratic programming (SLSQP) is used
                    in the internal process.
                    Every updating step in SLSQP takes `O(m2^{n+t})` time.
                    Since this method is using the SLSQP optimization over
                    the vector with lenght `2^n`, the mitigation for 8 bit counts
                    with the `mit_pattern = [[0], [1], [2], ..., [n-1]]` would
                    take 10 seconds or more.

                * If `None`, 'least_squares' is used.

            meas_layout (list of int): the mapping from classical registers to qubits

                * If you measure qubit `2` to clbit `0`, `0` to `1`, and `1` to `2`,
                    the list becomes `[2, 0, 1]`

                * If `None`, flatten(mit_pattern) is used.

        Returns:
            dict or Result: The corrected data in the same form as raw_data

        Raises:
            QiskitError: if raw_data is not in a one of the defined forms.
        """
        from scipy.optimize import minimize
        from scipy import linalg as la

        all_states = count_keys(self.nqubits)
        num_of_states = 2**self.nqubits

        if meas_layout is None:
            meas_layout = []
            for qubits in self._mit_pattern:
                meas_layout += qubits

        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            # convert to list
            raw_data2 = [np.zeros(num_of_states, dtype=float)]
            for state, count in raw_data.items():
                stateidx = int(state, 2)
                raw_data2[0][stateidx] = count

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            new_counts_list = parallel_map(
                self._apply_correction,
                [resultidx for resultidx, _ in enumerate(raw_data.results)],
                task_args=(raw_data, method, meas_layout),
            )

            for resultidx, new_counts in new_counts_list:
                new_result.results[resultidx].data.counts = new_counts

            return new_result

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        if method == "pseudo_inverse":
            pinv_cal_matrices = []
            for cal_mat in self._cal_matrices:
                pinv_cal_matrices.append(la.pinv(cal_mat))

        meas_layout = meas_layout[::-1]  # reverse endian
        qubits_to_clbits = [-1 for _ in range(max(meas_layout) + 1)]
        for i, qubit in enumerate(meas_layout):
            qubits_to_clbits[qubit] = i

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == "pseudo_inverse":
                for pinv_cal_mat, pos_qubits, indices in zip(
                    pinv_cal_matrices, self._mit_pattern, self._indices_list
                ):
                    inv_mat_dot_x = np.zeros([num_of_states], dtype=float)
                    pos_clbits = [qubits_to_clbits[qubit] for qubit in pos_qubits]
                    for state_idx, state in enumerate(all_states):
                        first_index = self.compute_index_of_cal_mat(state, pos_clbits, indices)
                        for i in range(len(pinv_cal_mat)):  # i is index of pinv_cal_mat
                            source_state = self.flip_state(state, i, pos_clbits)
                            second_index = self.compute_index_of_cal_mat(
                                source_state, pos_clbits, indices
                            )
                            inv_mat_dot_x[state_idx] += (
                                pinv_cal_mat[first_index, second_index]
                                * raw_data2[data_idx][int(source_state, 2)]
                            )
                    raw_data2[data_idx] = inv_mat_dot_x

            elif method == "least_squares":

                def fun(x):
                    mat_dot_x = deepcopy(x)
                    for cal_mat, pos_qubits, indices in zip(
                        self._cal_matrices, self._mit_pattern, self._indices_list
                    ):
                        res_mat_dot_x = np.zeros([num_of_states], dtype=float)
                        pos_clbits = [qubits_to_clbits[qubit] for qubit in pos_qubits]
                        for state_idx, state in enumerate(all_states):
                            second_index = self.compute_index_of_cal_mat(state, pos_clbits, indices)
                            for i in range(len(cal_mat)):
                                target_state = self.flip_state(state, i, pos_clbits)
                                first_index = self.compute_index_of_cal_mat(
                                    target_state, pos_clbits, indices
                                )
                                res_mat_dot_x[int(target_state, 2)] += (
                                    cal_mat[first_index, second_index] * mat_dot_x[state_idx]
                                )
                        mat_dot_x = res_mat_dot_x
                    return sum((raw_data2[data_idx] - mat_dot_x) ** 2)

                x0 = np.random.rand(num_of_states)
                x0 = x0 / sum(x0)
                nshots = sum(raw_data2[data_idx])
                cons = {"type": "eq", "fun": lambda x: nshots - sum(x)}
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bnds, tol=1e-6)
                raw_data2[data_idx] = res.x

            else:
                raise QiskitError("Unrecognized method.")

        # convert back into a counts dictionary
        new_count_dict = {}
        for state_idx, state in enumerate(all_states):
            if raw_data2[0][state_idx] != 0:
                new_count_dict[state] = raw_data2[0][state_idx]

        return new_count_dict

    def flip_state(self, state: str, mat_index: int, flip_poses: List[int]) -> str:
        """Flip the state according to the chosen qubit positions"""
        flip_poses = [pos for i, pos in enumerate(flip_poses) if (mat_index >> i) & 1]
        flip_poses = sorted(flip_poses)
        new_state = ""
        pos = 0
        for flip_pos in flip_poses:
            new_state += state[pos:flip_pos]
            new_state += str(int(state[flip_pos], 2) ^ 1)  # flip the state
            pos = flip_pos + 1
        new_state += state[pos:]
        return new_state

    def compute_index_of_cal_mat(self, state: str, pos_qubits: List[int], indices: dict) -> int:
        """Return the index of (pseudo inverse) calibration matrix for the input quantum state"""
        sub_state = ""
        for pos in pos_qubits:
            sub_state += state[pos]
        return indices[sub_state]

    def _apply_correction(
        self,
        resultidx,
        raw_data,
        method,
        meas_layout,
    ):
        """Wrapper to call apply with a counts dictionary."""
        new_counts = self.apply(
            raw_data.get_counts(resultidx), method=method, meas_layout=meas_layout
        )
        return resultidx, new_counts
