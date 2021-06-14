# This code is part of Qiskit.
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

"""
Utilities and classes for compression of CNOT structures.
Note, CNOT structure is a circuit that consists of CNOT gates solely.
"""

import numpy as np
from .utils import check_num_qubits
from .cnot_structures import check_cnots

# Avoid excessive deprecation warnings in Qiskit on Linux system.
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)


class CNotCompressor:
    """
    Compresses CNOT structure as described in:
    "Equivalent Quantum Circuits", J.C.Garcia-Escartin and P.Chamorro-Posada,
    2011, sections "Rule V" to "Rule VII".
    """

    @staticmethod
    def compress(num_qubits: int, cnots: np.ndarray) -> np.ndarray:
        """
        Applies compression rules to the input CNOT structure and outputs
        a new, compressed one.
        """
        check_num_qubits(num_qubits)
        check_cnots(nqubits=num_qubits, cnots=cnots)

        compressed = cnots.copy()  # compressed CNOTs
        controls = np.zeros(4, dtype=cnots.dtype)  # temporary array of control bits
        targets = np.zeros(4, dtype=cnots.dtype)  # temporary array of target bits

        # Loop over CNOT list while changes are still happening ...
        modified = True
        while modified:
            modified = False

            # Loop through the list of CNOTs and find compressible patterns ...
            i = 0
            while True:
                # TODO: What is N here?
                N = compressed.shape[1]
                if i + 4 <= N:
                    # Copy a fragment of circuit to avoid side effects.
                    controls[0:4] = compressed[0, i : i + 4]  # 4 control bits
                    targets[0:4] = compressed[1, i : i + 4]  # 4 target bits

                    # Try to compress 4 consecutive CNOTs.
                    cnots_new = CNotCompressor._compress4(
                        controls[0:4], targets[0:4], i, compressed
                    )
                    if compressed.size > cnots_new.size:
                        compressed = cnots_new
                        modified = True
                        continue

                if i + 3 <= N:
                    # Copy a fragment of circuit to avoid side effects.
                    controls[0:3] = compressed[0, i : i + 3]  # 3 control bits
                    targets[0:3] = compressed[1, i : i + 3]  # 3 target bits

                    # Try to compress 3 consecutive CNOTs.
                    cnots_new = CNotCompressor._compress3(
                        controls[0:3], targets[0:3], i, compressed
                    )
                    if compressed.size > cnots_new.size:
                        compressed = cnots_new
                        modified = True
                        continue

                if i + 2 <= N:
                    # Consecutive CNOTs acting on same qubits cancel each other.
                    if np.all(compressed[:, i] == compressed[:, i + 1]):
                        compressed = np.delete(compressed, [i, i + 1], 1)
                        modified = True
                        continue
                else:
                    break
                i += 1

        return compressed

    @staticmethod
    def _compress4(c: np.ndarray, t: np.ndarray, i: int, cnots: np.ndarray) -> np.ndarray:
        """
        Compresses 4 consecutive CNOTs.
        """
        #                [0] [1] [2] [3]
        # ---o---      ---o-------o-------
        #    |            |       |
        # ---|---  ==  ---X---o---X---o---
        #    |                |       |
        # ---X---      -------X-------X---
        if c[0] == c[2] and t[1] == t[3] and c[0] != t[1] and t[0] == c[1] == t[2] == c[3]:
            cnots[:, i] = [c[0], t[1]]
            cnots_new = np.delete(cnots, [i + 1, i + 2, i + 3], 1)
            return cnots_new

        #                [0] [1] [2] [3]
        # ---o---      -------o-------o---
        #    |                |       |
        # ---|---  ==  ---o---X---o---X---
        #    |            |       |
        # ---X---      ---X-------X-------
        if c[1] == c[3] and t[0] == t[2] and c[1] != t[0] and c[0] == t[1] == c[2] == t[3]:
            cnots[:, i] = [c[1], t[0]]
            cnots_new = np.delete(cnots, [i + 1, i + 2, i + 3], 1)
            return cnots_new

        return cnots

    @staticmethod
    def _compress3(c: np.ndarray, t: np.ndarray, i: int, cnots: np.ndarray) -> np.ndarray:
        """
        Compresses 3 consecutive CNOTs.
        """
        #                    [0] [1] [2]
        # ---o---o---      -------o-------
        #    |   |                |
        # ---X---|---  ==  ---o---X---o---
        #        |            |       |
        # -------X---      ---X-------X---
        if c[0] == t[1] == c[2] and t[0] == t[2] and c[1] != t[0]:
            cnots[:, i + 0] = [c[1], t[1]]
            cnots[:, i + 1] = [c[1], t[2]]
            cnots_new = np.delete(cnots, [i + 2], 1)
            return cnots_new

        #                    [0] [1] [2]
        # ---o-------      ---o-------o---
        #    |                |       |
        # ---X---o---  ==  ---|---o---X---
        #        |            |   |
        # -------X---      ---X---X-------
        if c[0] == c[2] and c[1] == t[2] and t[0] == t[1]:
            cnots[:, i + 0] = [c[2], t[2]]
            cnots[:, i + 1] = [c[1], t[1]]
            cnots_new = np.delete(cnots, [i + 2], 1)
            return cnots_new

        #                    [0] [1] [2]
        # ---o-------      -------o---o---
        #    |                    |   |
        # ---X---o---  ==  ---o---X---|---
        #        |            |       |
        # -------X---      ---X-------X---
        if c[0] == t[1] and c[1] == c[2] and t[0] == t[2]:
            cnots[:, i + 0] = [c[1], t[1]]
            cnots[:, i + 1] = [c[0], t[0]]
            cnots_new = np.delete(cnots, [i + 2], 1)
            return cnots_new

        #                    [0] [1] [2]
        # -------o---      ---o---o-------
        #        |            |   |
        # ---o---X---  ==  ---|---X---o---
        #    |                |       |
        # ---X-------      ---X-------X---
        if c[0] == c[1] and t[1] == c[2] and t[0] == t[2]:
            cnots[:, i + 0] = [c[2], t[2]]
            cnots[:, i + 1] = [c[1], t[1]]
            cnots_new = np.delete(cnots, [i + 2], 1)
            return cnots_new

        #                    [0] [1] [2]
        # -------o---      ---o-------o---
        #        |            |       |
        # ---o---X---  ==  ---X---o---|---
        #    |                    |   |
        # ---X-------      -------X---X---
        if c[0] == c[2] and t[0] == c[1] and t[1] == t[2]:
            cnots[:, i + 0] = [c[1], t[1]]
            cnots[:, i + 1] = [c[0], t[0]]
            cnots_new = np.delete(cnots, [i + 2], 1)
            return cnots_new

        return cnots


class CNotSynthesis:
    """
    Class implements the CNOT compression algorithm described in:
    "Optimal synthesis of linear reversible circuits", Patel, Markov and Hayes,
    Quantum Information and Computation, Vol. 8, No. 3&4 (2008) 0282–0294.
    Note, this is a lite, non-optimized version with O(n^2) complexity.
    """

    def __init__(self):
        pass

    @staticmethod
    def synthesis(num_qubits: int, cnots: np.ndarray, choose_shortest: bool = True) -> np.ndarray:
        """
        Runs Synthesis algorithm for CNOT structure compression.
        N O T E: the algorithm does not guarantee shorter CNOT structure after
        compression (although this is typically the case). Parameter
        choose_shortest enforces selection of the best structure: either
        the compressed one or the original.
        Args:
            num_qubits: number of qubits.
            cnots: CNOT structure of size 2xN subject to compression.
            choose_shortest: if True, the shortest structure is selected,
                             either the compressed or the original one.
        Returns:
            compressed structure with possibly fewer CNOTs.
        """
        assert isinstance(choose_shortest, bool)
        depth = cnots.shape[1]
        m = min(num_qubits - 1, min(max(1, int(round(float(np.log2(num_qubits))))), 8))
        A = CNotSynthesis._build_circuit_matrix(num_qubits, cnots)
        lower_circuit = CNotSynthesis._lower_cnot_synth(num_qubits, m, A, depth)
        A = A.T.copy()
        upper_circuit = CNotSynthesis._lower_cnot_synth(num_qubits, m, A, depth)

        # Concatenate upper/lower results. Note, "upper_circuit" is flipped
        # over both (!) axes.
        synth_cnots = np.concatenate((np.flip(upper_circuit), lower_circuit), axis=1)

        synth_cnots += 1  # 1-based index
        # Check correctness of compression at the end.
        # assert CNotSynthesis.compare_cnot_circuits(nqubits, cnots, synth_cnots)
        if choose_shortest:
            return synth_cnots if synth_cnots.size < cnots.size else cnots
        else:
            return synth_cnots

    @staticmethod
    def compare_cnot_circuits(num_qubits: int, cnots1: np.ndarray, cnots2: np.ndarray) -> bool:
        """
        Returns True, if two CNOT structures implement the same circuit
        albeit the different number of CNOTs.
        """
        M1 = CNotSynthesis._build_circuit_matrix(num_qubits, cnots1)
        M2 = CNotSynthesis._build_circuit_matrix(num_qubits, cnots2)
        assert M1.dtype == np.int64 and M2.dtype == np.int64
        return np.all(M1 == M2)

    @staticmethod
    def _cnot(control: int, target: int, G: np.ndarray):
        """
        Initializes CNOT matrix acting on particular qubits.
        N O T E, this is not a gate matrix, which would have 2^n-x-2^n size.
        Rather, this is a matrix defined over F2 field as detailed in the paper.
        """
        assert control != target
        G.fill(0)
        np.fill_diagonal(G, 1)
        G[target, control] = 1

    @staticmethod
    def _build_circuit_matrix(n: int, cnots: np.ndarray) -> np.ndarray:
        """
        Builds n-x-n matrix equivalent to the sequence of CNOTs.
        N O T E, this is not a gate matrix, which would have 2^n-x-2^n size.
        Rather, this is a matrix defined over F2 field as detailed in the paper.
        Args:
            n: number of qubits.
            cnots: CNOT structure defined as Numpy array of size (2, N).
        Returns:
            n-x-n circuit matrix of zeros and ones (field F2).
        """
        check_num_qubits(n)
        check_cnots(n, cnots)

        A = np.eye(n, dtype=np.int64)  # circuit matrix
        G = np.zeros_like(A)  # CNOT-gate matrix
        T = np.zeros_like(A)  # temporary matrix

        for i in range(cnots.shape[1]):
            CNotSynthesis._cnot(control=cnots[0, i] - 1, target=cnots[1, i] - 1, G=G)
            np.mod(np.dot(G, A, out=T), 2, out=A)  # A = mod(G @ A, 2)

        assert A.dtype == np.int64
        return A

    @staticmethod
    def _lower_cnot_synth(n: int, m: int, A: np.ndarray, depth: int) -> np.ndarray:
        """
        Reduces lower-triangular matrix to all-zeros below the main diagonal.
        """
        # m is restricted to 8 to avoid large sections and
        # because np.packbits() operates on uint8 numbers only.
        assert isinstance(m, int) and 1 <= m <= 8 and m < n

        # To avoid often reallocation, we reserve extra memory initially:
        circuit = np.zeros((2, 8 * (n + depth)), np.int64)
        # Counter of inserted CNOTs:
        count = 0

        # There are 2^m possible 0/1 patterns in a section of size m:
        pattern = np.zeros(2 ** m, dtype=np.int64)
        # Marker of previously unseen pattern:
        # TODO: Why we need this?
        not_found = -1

        # Iterate over column sections.
        for sec in range((n + m - 1) // m):  # range(ceil(n / m))
            pattern.fill(not_found)

            # Remove duplicate sub-rows in section "sec".
            for row in range(sec * m, n):
                sub_row_patt = int(
                    np.packbits(A[row, sec * m : (sec + 1) * m], bitorder="little")[0]
                )
                assert 0 <= sub_row_patt < pattern.size
                patt = pattern[sub_row_patt]

                if patt == not_found:
                    # Memorise row index where this pattern was seen first time.
                    pattern[sub_row_patt] = row
                else:
                    # If we saw the same pattern before, we can eliminate
                    # this row completely.
                    A[row, :] += A[patt, :]
                    A[row, :] %= 2
                    circuit[:, count] = [patt, row]
                    count += 1

            # Gaussian elimination for remaining entries in column section.
            for col in range(sec * m, min((sec + 1) * m, n)):
                # Check for 1 on diagonal.
                diag_one = A[col, col] != 0

                # Add CNOT that removes ones in rows below column col.
                for row in range(col + 1, n):
                    if A[row, col] == 1:
                        if not diag_one:
                            # This branch of code is executed only once during
                            # the first call to this function. Afterwards,
                            # the main diagonal will be set to all ones.
                            diag_one = True
                            A[col, :] += A[row, :]
                            A[col, :] %= 2
                            circuit[:, count] = [row, col]
                            count += 1

                        A[row, :] += A[col, :]
                        A[row, :] %= 2
                        circuit[:, count] = [col, row]
                        count += 1

        # Note, we append CNOTs instead of prepending, then we flip at the end.
        circuit = np.flip(circuit[:, 0:count], axis=1)
        return circuit


def compress_cnots(
    num_qubits: int, cnots: np.ndarray, choose_shortest: bool = True, check_correctness: bool = True
) -> np.ndarray:
    """
    Compresses CNOT structure by applying Synthesis algorithm
    prepended/followed by rule based compressor as a pre/post-processing step.
    N O T E: the first CNOT (cnots[:, 0]) is the one applied first to a quantum
    state vector. It appears on the left side on the graphical representation
    of a circuit.
    N O T E: the algorithm does not guarantee shorter CNOT structure after
    compression (although this is typically the case). Parameter
    choose_shortest enforces selection of the best structure: either
    the compressed one or the original.

    Args:
        num_qubits: number of qubits.
        cnots: CNOT structure of size 2xN subject to compression.
        choose_shortest: if True, the shortest structure is selected,
                         either the compressed or the original one.
        check_correctness: flag enforces verification of compression.

    Returns:
        compressed structure with possibly fewer CNOTs.
    """
    assert isinstance(choose_shortest, bool)
    assert isinstance(check_correctness, bool)
    compressed_by_rules = CNotCompressor.compress(num_qubits, cnots)
    synth_cnots = CNotSynthesis.synthesis(num_qubits, compressed_by_rules, choose_shortest)
    compressed_by_rules = CNotCompressor.compress(num_qubits, synth_cnots)
    if check_correctness:
        assert CNotSynthesis.compare_cnot_circuits(num_qubits, cnots, compressed_by_rules)
    return compressed_by_rules


# Simplified version:
#
# class _CNotSynthesis:
#     """
#     Class implements the CNOT compression algorithm described in:
#     "Optimal synthesis of linear reversible circuits", Patel, Markov and Hayes,
#     Quantum Information and Computation, Vol. 8, No. 3&4 (2008) 0282–0294.
#     Note, this is a lite, non-optimized version with O(n^2) complexity.
#     """
#
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def synthesis(nqubits: int, cnots: np.ndarray) -> np.ndarray:
#         depth = cnots.shape[1]
#         A = CNotSynthesis._build_circuit_matrix(nqubits, cnots)
#         lower_circuit = CNotSynthesis._lower_cnot_synth(nqubits, A, depth)
#         A = A.T.copy()
#         upper_circuit = CNotSynthesis._lower_cnot_synth(nqubits, A, depth)
#         synth_cnots = np.concatenate(
#             (np.flip(np.flip(upper_circuit, axis=0), axis=1),
#              lower_circuit),
#             axis=1)
#         check_cnots(nqubits, synth_cnots)
#         return synth_cnots
#
#     @staticmethod
#     def compare_cnot_circuits(nqubits: int, cnots1: np.ndarray,
#                                             cnots2: np.ndarray) -> bool:
#         """
#         Returns True, if two CNOT structures implement the same circuit
#         albeit the different number of CNOTs.
#         """
#         M1 = CNotSynthesis._build_circuit_matrix(nqubits, cnots1)
#         M2 = CNotSynthesis._build_circuit_matrix(nqubits, cnots2)
#         assert M1.dtype == np.int64 and M2.dtype == np.int64
#         return np.all(M1 == M2)
#
#     @staticmethod
#     def _cnot(control: int, target: int, G: np.ndarray):
#         assert control != target
#         G.fill(0)
#         np.fill_diagonal(G, 1)
#         G[target, control] = 1
#
#     @staticmethod
#     def _build_circuit_matrix(n: int, cnots: np.ndarray) -> np.ndarray:
#         """
#         Builds n-x-n matrix equivalent to the sequence of CNOTs.
#         N O T E, this is not a gate matrix, which would have 2^n-x-2^n size.
#         Rather, this is a matrix defined over F2 field as detailed in the paper.
#         Args:
#             n: number of qubits.
#             cnots: CNOT structure defined as Numpy array of size (2, N).
#         Returns:
#             n-x-n circuit matrix of zeros and ones (field F2).
#         """
#         check_num_qubits(n)
#         check_cnots(n, cnots)
#
#         A = np.eye(n, dtype=np.int64)                       # circuit matrix
#         G = np.zeros_like(A)                                # CNOT-gate matrix
#         T = np.zeros_like(A)                                # temporary matrix
#
#         for i in range(cnots.shape[1]):
#             CNotSynthesis._cnot(control=cnots[0, i] - 1,
#                                 target =cnots[1, i] - 1, G=G)
#             np.mod(np.dot(G, A, out=T), 2, out=A)           # A = mod(G @ A, 2)
#
#         assert A.dtype == np.int64
#         return A
#
#     @staticmethod
#     def _lower_cnot_synth(n: int, A: np.ndarray, depth: int) -> np.ndarray:
#         """
#         """
#         circuit = np.zeros((2, 4 * depth), np.int64)    # extra deep initially
#         count = 0                                       # CNOT counter
#
#         # Gaussian elimination column after column.
#         for col in range(n):
#             # Check for 1 on diagonal.
#             diag_one = (A[col, col] != 0)
#
#             # Add CNOT that removes ones in rows below column col.
#             for row in range(col + 1, n):
#                 if A[row, col] == 1:
#                     if not diag_one:
#                         # This branch of code is executed only one during the
#                         # first call to this function .... TODO: finish comment
#                         diag_one = True
#                         A[col, :] += A[row, :]
#                         A[col, :] %= 2
#                         circuit[:, count] = [row, col]
#                         count += 1
#
#                     A[row, :] += A[col, :]
#                     A[row, :] %= 2
#                     circuit[:, count] = [col, row]
#                     count += 1
#
#         # Note, we append CNOTs instead of prepending, then we flip at the end.
#         circuit = np.flip(circuit[:, 0 : count], axis=1)
#         circuit += 1        # 1-based index
#         return circuit
#
#
