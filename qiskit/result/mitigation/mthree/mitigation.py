# This code is part of Mthree.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=no-name-in-module, invalid-name
"""Calibration data"""

import warnings
from time import perf_counter

import psutil
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import orjson

from .matrix import _reduced_cal_matrix, sdd_check
from .utils import counts_to_vector, vector_to_quasiprobs
from .norms import ainv_onenorm_est_lu, ainv_onenorm_est_iter
from .matvec import M3MatVec
from .exceptions import M3Error
from .classes import QuasiCollection

class M3Mitigation():
    """Main M3 calibration class."""
    def __init__(self, single_qubit_cals, iter_threshold=4096):
        """Main M3 calibration class.

        Parameters:
            single_qubit_cals (list): 1Q calibration matrices
            iter_threshold (int): Sets the bitstring count at which iterative mode
                                  is turned on (assuming reasonable error rates).
        """


        self.single_qubit_cals = single_qubit_cals
        self.num_qubits = len(self.single_qubit_cals)
        self.iter_threshold = iter_threshold
        self.cal_shots = None
        self.cal_method = 'balanced'
        self.rep_delay = None

    def _form_cals(self, qubits):
        """Form the 1D cals array from tensored cals data

        Parameters:
            qubits (array_like): The qubits to calibrate over.

        Returns:
            ndarray: 1D Array of float cals data.
        """
        qubits = np.asarray(qubits, dtype=int)
        cals = np.zeros(4*qubits.shape[0], dtype=float)

        # Reverse index qubits for easier indexing later
        for kk, qubit in enumerate(qubits[::-1]):
            cals[4*kk:4*kk+4] = self.single_qubit_cals[qubit].ravel()
        return cals

    def _check_sdd(self, counts, qubits, distance=None):
        """Checks if reduced A-matrix is SDD or not

        Parameters:
            counts (dict): Dictionary of counts.
            qubits (array_like): List of qubits.
            distance (int): Distance to compute over.

        Returns:
            bool: True if A-matrix is SDD, else False

        Raises:
            M3Error: Number of qubits supplied does not match bit-string length.
        """
        # If distance is None, then assume max distance.
        num_bits = len(qubits)
        if distance is None:
            distance = num_bits

        # check if len of bitstrings does not equal number of qubits passed.
        bitstring_len = len(next(iter(counts)))
        if bitstring_len != num_bits:
            raise M3Error('Bitstring length ({}) does not match'.format(bitstring_len) +
                          ' number of qubits ({})'.format(num_bits))
        cals = self._form_cals(qubits)
        return sdd_check(counts, cals, num_bits, distance)

    def cals_from_file(self, cals_file):
        """Generated the calibration data from a previous runs output

            cals_file (str): A string path to the saved counts file from an
                             earlier run.
        """
        with open(cals_file, 'r', encoding='utf-8') as fd:
            self.single_qubit_cals = [np.asarray(cal) if cal else None
                                      for cal in orjson.loads(fd.read())]



    def apply_correction(self, counts, qubits, distance=None,
                         method='auto',
                         max_iter=25, tol=1e-5,
                         return_mitigation_overhead=False,
                         details=False):
        """Applies correction to given counts.

        Parameters:
            counts (dict, list): Input counts dict or list of dicts.
            qubits (array_like): Qubits on which measurements applied.
            distance (int): Distance to correct for. Default=num_bits
            method (str): Solution method: 'auto', 'direct' or 'iterative'.
            max_iter (int): Max. number of iterations, Default=25.
            tol (float): Convergence tolerance of iterative method, Default=1e-5.
            return_mitigation_overhead (bool): Returns the mitigation overhead, default=False.
            details (bool): Return extra info, default=False.

        Returns:
            QuasiDistribution or QuasiCollection: Dictionary of quasiprobabilities if
                                                  input is a single dict, else a collection
                                                  of quasiprobabilities.

        Raises:
            M3Error: Bitstring length does not match number of qubits given.
        """
        if len(counts) == 0:
            raise M3Error('Input counts is any empty dict.')
        given_list = False
        if isinstance(counts, (list, np.ndarray)):
            given_list = True
        if not given_list:
            counts = [counts]

        if isinstance(qubits, dict):
            # If a mapping was given for qubits
            qubits = [list(qubits)]
        elif not any(isinstance(qq, (list, tuple, np.ndarray, dict)) for qq in qubits):
            qubits = [qubits]*len(counts)
        else:
            if isinstance(qubits[0], dict):
                # assuming passed a list of mappings
                qubits = [list(qu) for qu in qubits]

        if len(qubits) != len(counts):
            raise M3Error('Length of counts does not match length of qubits.')

        quasi_out = []

        for idx, cnts in enumerate(counts):

            quasi_out.append(
                self._apply_correction(cnts, qubits=qubits[idx],
                                       distance=distance,
                                       method=method,
                                       max_iter=max_iter, tol=tol,
                                       return_mitigation_overhead=return_mitigation_overhead,
                                       details=details)
                            )

        if not given_list:
            return quasi_out[0]
        return QuasiCollection(quasi_out)

    def _apply_correction(self, counts, qubits, distance=None,
                          method='auto',
                          max_iter=25, tol=1e-5,
                          return_mitigation_overhead=False,
                          details=False):
        """Applies correction to given counts.

        Parameters:
            counts (dict): Input counts dict.
            qubits (array_like): Qubits on which measurements applied.
            distance (int): Distance to correct for. Default=num_bits
            method (str): Solution method: 'auto', 'direct' or 'iterative'.
            max_iter (int): Max. number of iterations, Default=25.
            tol (float): Convergence tolerance of iterative method, Default=1e-5.
            return_mitigation_overhead (bool): Returns the mitigation overhead, default=False.
            details (bool): Return extra info, default=False.

        Returns:
            QuasiDistribution: Dictionary of quasiprobabilities.

        Raises:
            M3Error: Bitstring length does not match number of qubits given.
        """
        # This is needed because counts is a Counts object in Qiskit not a dict.
        counts = dict(counts)
        shots = sum(counts.values())

        # If distance is None, then assume max distance.
        num_bits = len(qubits)
        num_elems = len(counts)
        if distance is None:
            distance = num_bits

        # check if len of bitstrings does not equal number of qubits passed.
        bitstring_len = len(next(iter(counts)))
        if bitstring_len != num_bits:
            raise M3Error('Bitstring length ({}) does not match'.format(bitstring_len) +
                          ' number of qubits ({})'.format(num_bits))

        # Check if no cals done yet
        if self.single_qubit_cals is None:
            warnings.warn('No calibration data. Calibrating: {}'.format(qubits))
            self._grab_additional_cals(qubits, method=self.cal_method)

        # Check if one or more new qubits need to be calibrated.
        missing_qubits = [qq for qq in qubits if self.single_qubit_cals[qq] is None]
        if any(missing_qubits):
            warnings.warn('Computing missing calibrations for qubits: {}'.format(missing_qubits))
            self._grab_additional_cals(missing_qubits, method=self.cal_method)

        if method == 'auto':
            current_free_mem = psutil.virtual_memory().available / 1024**3
            # First check if direct method can be run
            if num_elems <= self.iter_threshold \
                    and ((num_elems**2+num_elems)*8/1024**3 < current_free_mem/2):
                method = 'direct'
            else:
                method = 'iterative'

        if method == 'direct':
            st = perf_counter()
            mit_counts, col_norms, gamma = self._direct_solver(counts, qubits, distance,
                                                               return_mitigation_overhead)
            dur = perf_counter()-st
            mit_counts.shots = shots
            if gamma is not None:
                mit_counts.mitigation_overhead = gamma * gamma
            if details:
                info = {'method': 'direct', 'time': dur, 'dimension': num_elems}
                info['col_norms'] = col_norms
                return mit_counts, info
            return mit_counts

        elif method == 'iterative':
            iter_count = np.zeros(1, dtype=int)

            def callback(_):
                iter_count[0] += 1

            if details:
                st = perf_counter()
                mit_counts, col_norms, gamma = self._matvec_solver(counts,
                                                                   qubits,
                                                                   distance,
                                                                   tol,
                                                                   max_iter,
                                                                   1,
                                                                   callback,
                                                                   return_mitigation_overhead)
                dur = perf_counter()-st
                mit_counts.shots = shots
                if gamma is not None:
                    mit_counts.mitigation_overhead = gamma * gamma
                info = {'method': 'iterative', 'time': dur, 'dimension': num_elems}
                info['iterations'] = iter_count[0]
                info['col_norms'] = col_norms
                return mit_counts, info
            # pylint: disable=unbalanced-tuple-unpacking
            mit_counts, gamma = self._matvec_solver(counts, qubits, distance, tol,
                                                    max_iter, 0, None,
                                                    return_mitigation_overhead)
            mit_counts.shots = shots
            if gamma is not None:
                mit_counts.mitigation_overhead = gamma * gamma
            return mit_counts

        else:
            raise M3Error('Invalid method: {}'.format(method))

    def reduced_cal_matrix(self, counts, qubits, distance=None):
        """Return the reduced calibration matrix used in the solution.

        Parameters:
            counts (dict): Input counts dict.
            qubits (array_like): Qubits on which measurements applied.
            distance (int): Distance to correct for. Default=num_bits

        Returns:
            ndarray: 2D array of reduced calibrations.
            dict: Counts in order they are displayed in matrix.

        Raises:
            M3Error: If bit-string length does not match passed number
                     of qubits.
        """
        counts = dict(counts)
        # If distance is None, then assume max distance.
        num_bits = len(qubits)
        if distance is None:
            distance = num_bits

        # check if len of bitstrings does not equal number of qubits passed.
        bitstring_len = len(next(iter(counts)))
        if bitstring_len != num_bits:
            raise M3Error('Bitstring length ({}) does not match'.format(bitstring_len) +
                          ' number of qubits ({})'.format(num_bits))

        cals = self._form_cals(qubits)
        A, counts, _ = _reduced_cal_matrix(counts, cals, num_bits, distance)
        return A, counts

    def _direct_solver(self, counts, qubits, distance=None,
                       return_mitigation_overhead=False):
        """Apply the mitigation using direct LU factorization.

        Parameters:
            counts (dict): Input counts dict.
            qubits (int): Qubits over which to calibrate.
            distance (int): Distance to correct for. Default=num_bits
            return_mitigation_overhead (bool): Returns the mitigation overhead, default=False.

        Returns:
            QuasiDistribution: dict of Quasiprobabilites
        """
        cals = self._form_cals(qubits)
        num_bits = len(qubits)
        A, sorted_counts, col_norms = _reduced_cal_matrix(counts, cals, num_bits, distance)
        vec = counts_to_vector(sorted_counts)
        LU = la.lu_factor(A, check_finite=False)
        x = la.lu_solve(LU, vec, check_finite=False)
        gamma = None
        if return_mitigation_overhead:
            gamma = ainv_onenorm_est_lu(A, LU)
        out = vector_to_quasiprobs(x, sorted_counts)
        return out, col_norms, gamma

    def _matvec_solver(self, counts, qubits, distance, tol=1e-5, max_iter=25,
                       details=0, callback=None,
                       return_mitigation_overhead=False):
        """Compute solution using GMRES and Jacobi preconditioning.

        Parameters:
            counts (dict): Input counts dict.
            qubits (int): Qubits over which to calibrate.
            tol (float): Tolerance to use.
            max_iter (int): Maximum number of iterations to perform.
            distance (int): Distance to correct for. Default=num_bits
            details (bool): Return col norms.
            callback (callable): Callback function to record iteration count.
            return_mitigation_overhead (bool): Returns the mitigation overhead, default=False.

        Returns:
            QuasiDistribution: dict of Quasiprobabilites

        Raises:
            M3Error: Solver did not converge.
        """
        cals = self._form_cals(qubits)
        M = M3MatVec(dict(counts), cals, distance)
        L = spla.LinearOperator((M.num_elems, M.num_elems),
                                matvec=M.matvec, rmatvec=M.rmatvec)
        diags = M.get_diagonal()

        def precond_matvec(x):
            out = x / diags
            return out

        P = spla.LinearOperator((M.num_elems, M.num_elems), precond_matvec)
        vec = counts_to_vector(M.sorted_counts)
        out, error = spla.gmres(L, vec, tol=tol, atol=tol, maxiter=max_iter,
                                M=P, callback=callback)
        if error:
            raise M3Error('GMRES did not converge: {}'.format(error))

        gamma = None
        if return_mitigation_overhead:
            gamma = ainv_onenorm_est_iter(M, tol=tol, max_iter=max_iter)

        quasi = vector_to_quasiprobs(out, M.sorted_counts)
        if details:
            return quasi, M.get_col_norms(), gamma
        return quasi, gamma

    def readout_fidelity(self, qubits=None):
        """Compute readout fidelity for calibrated qubits.

        Parameters:
            qubits (array_like): Qubits to compute over, default is all.

        Returns:
            list: List of qubit fidelities.

        Raises:
            M3Error: Mitigator is not calibrated.
            M3Error: Qubit indices out of range.
        """
        if self.single_qubit_cals is None:
            raise M3Error('Mitigator is not calibrated')

        if qubits is None:
            qubits = range(self.num_qubits)
        else:
            outliers = [kk for kk in qubits if kk >= self.num_qubits]
            if any(outliers):
                raise M3Error('One or more qubit indices out of range: {}'.format(outliers))
        fids = []
        for kk in qubits:
            qubit = self.single_qubit_cals[kk]
            if qubit is not None:
                fids.append(np.mean(qubit.diagonal()))
            else:
                fids.append(None)
        return fids
