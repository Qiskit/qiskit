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
"""Circuit compressor classes."""
import logging
from abc import abstractmethod, ABC
from typing import Tuple, List, Dict

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, QiskitError
from qiskit.quantum_info import Operator, OneQubitEulerDecomposer
from .parametric_circuit import ParametricCircuit

logger = logging.getLogger(__name__)


class CompressorBase(ABC):
    """A base interface for a circuit compressor."""

    @abstractmethod
    def compress(self, circuit: ParametricCircuit) -> ParametricCircuit:
        """
        Compresses a parametric circuit.

        Args:
            circuit: parametric circuit to be compressed.

        Returns:
            compressed circuit.
        """
        raise NotImplementedError("Abstract method is called!")


class EulerCompressor(CompressorBase):
    """
    todo: reformat as in Qiskit
    This is the first implementation by Andrea Simonetto where two strategies have been used.
    First, consecutive bare CNOTs (that stem from block with all zero thetas) are compressed by
    using mirror relation as described in "Equivalent Quantum Circuits", Garcia-Escartin and
    Posada, 2011. Another variant of bare CNOT compression (aka "synthesis") borrows its idea from:
    "Optimal synthesis of linear reversible circuits", Patel et all, 2008. Second, by Euler
    decomposition of consecutive 1-qubit gates this approach extracts Rz and Rx gates that can be
    flipped over CNOT one in hope to merge them with 1-qubit gates of preceding block. Flip trick
    is detailed in: "Minimal Universal Two-Qubit CNOT-based Circuits", Shende et all, 2004.
    """

    def __init__(self, eps: float = 0.04, niter: int = 100, synth: bool = False, verbose: int = 0):
        """
        Args:
            eps: zero threshold.
            niter: how many sweeps of the commutation/mirror rules.
            synth: Bool, true or false if we apply synthesis or not.
            verbose: verbosity level.
        """
        super().__init__()
        assert isinstance(eps, float)
        assert isinstance(niter, int)
        assert isinstance(synth, bool)
        assert isinstance(verbose, int)
        self._eps = eps
        self._niter = niter
        self._synth = synth
        self._verbose = verbose

    def compress(self, circuit: ParametricCircuit) -> ParametricCircuit:
        """
        See base class description.
        Function to compress a given circuit based on its rotation gate sparsity pattern,
        via commutation & mirror rules plus the synthesis algorithm (if connectivity is Full)
        Note: synthesis does not keep connectivity, the rest does

        # applies cnot_synthesis and three_qubit_shuffle to the sparsity pattern
        # synth=False - compliant with coupling map, otherwise (True) may introduce connections
        # between qubits, breaking coupling map, use for full connectivity only

        Args:
            circuit: parametric circuit to be compressed.

        Returns:
            compressed parametric circuit.
        """

        # (1) sparsity determines which CNOT units are bare (i.e. have rotations set to zero)
        # (2) zero_runs determines which CNOTs are consecutive (i.e. have all rotations between set to zero)
        # For each word of consecutive CNOTs:
        # (3) synthesis computes an equivalent word
        #     synthesis currently does not respect hardware connectivity
        # (4) three_qubit_shuffle computes an equivalent word by
        #     alternating between random allowable commutations and up to three qubit reductions
        #     (a) commute randomly chooses a pair of commutable CNOTs to commute
        #     (b) three_qubit_reduce applies three qubit reductions
        #     (c) two_qubit_reduce applies two qubit reductions (i.e. identical CNOTs cancel)
        # (5) If the new word is shorter, it replaces the old word and
        #     extra_angles shifts the leftover angles from the original word to the left
        # A new CNOT structure is returned along with angles for warm-starting the optimization

        # Liam: I changed variable names and commented
        # Liam: I added an if statement to only replace sections with smaller sections
        # Liam: I replaced np.shape(.)[0] with (.).size
        # Liam: I renamed compress_equals to two_qubit_reduce
        # Liam: I renamed reduce_for_minimal to three_qubit_reduce
        # Liam: I renamed minimal to three_qubit_shuffle (a dance it seems)
        # Liam: I finished changed variable names and commented for:
        #       sparsity, zero_runs, cnot_synthesis, three_qubit_shuffle,
        #       commute, two_qubit_reduce, and three_qubit_reduce

        assert isinstance(circuit, ParametricCircuit)

        n = circuit.num_qubits
        cnots = circuit.cnots
        compressed_cnots = np.array(cnots)

        # Initialize the compressed theta for book-keeping/the warm-start
        compressed_thetas = np.array(circuit.thetas)

        # Determine the sparsity pattern:
        # vector that tells where four angles after a CNOT are 0s, of dimension num_cnots - 1
        sparsity = self._sparsity(circuit.cnots, circuit.thetas, self._eps)

        # Determine which cnot indices are to be affected by the compression:
        # [[start1, end1], [start2, end2], ... ]
        ranges = self._zero_runs(sparsity)
        num_ranges = np.shape(ranges)[0]

        for word_index in range(num_ranges):
            # the cnot indices of beginning and end of word to be reduced
            cnot_index1 = ranges[word_index, 0]
            cnot_index2 = ranges[word_index, 1] + 1
            # Slice the section of cnots that will be affected by the compression rules
            section = np.array(compressed_cnots[0:2, cnot_index1:cnot_index2])

            # Apply synthesis if full connectivity
            if self._synth:
                # TODO: check if this works
                new_section = self._cnot_synthesis(n, section)
            else:
                new_section = section

            # Apply three_qubit_shuffle
            # If there is still a CNOT:
            if new_section.size != 0:
                if np.shape(new_section)[1] > 2:
                    new_section = self._three_qubit_shuffle(new_section, self._niter)

            # if the new_section is strictly better than section, then replace the latter with the former
            # both cnot_synthesis and three_qubit_shuffle only return new words if they are strictly shorter
            if new_section.size < section.size:

                num_cnots1 = np.shape(compressed_cnots)[1]

                # keep the rightmost cnot that is deleted
                old_cnots = np.array(compressed_cnots[:, cnot_index2 - 1])
                logger.debug("Old cnots: %s", old_cnots)

                compressed_cnots = np.delete(compressed_cnots, np.arange(cnot_index1, cnot_index2), 1)
                logger.debug("Compressed cnots: %s", compressed_cnots)

                # keep the angles corresponding to old_cnot
                old_thetas = np.array(compressed_thetas[4 * (cnot_index2 - 1) : 4 * cnot_index2])
                logger.debug("Old thetas: %s", old_thetas)

                # delete angles corresponding to deleted cnots
                compressed_thetas = np.delete(compressed_thetas, np.arange(4 * cnot_index1, 4 * cnot_index2))
                logger.debug("After deletion: %s", compressed_thetas)

                # If there is still a CNOT:
                if new_section.size != 0:
                    compressed_cnots = np.insert(compressed_cnots, cnot_index1, np.transpose(new_section), 1)
                    new_section_size = np.shape(new_section)[1]

                    # insert zero angles for inserted cnots
                    compressed_thetas = np.insert(compressed_thetas, 4 * cnot_index1, np.zeros(4 * new_section_size))

                # print('cmp_cnot = ', cmp_cnots)
                # print('After =', (cmp_thetas))

                num_cnots2 = np.shape(compressed_cnots)[1]

                # print('L2 = ', L2)
                # print('ranges = ', ranges)
                # Redefine ranges in order to shift the indices depending on the compression
                ranges += num_cnots2 - num_cnots1
                # print('ranges = ', ranges)
                L2n = ranges[word_index, 1] + 1
                # print('Bookkeeping : ', cmp_cnots, cmp_thetas, old_cnot, old_thetas, L2n)

                # update angles
                compressed_thetas = self._extra_angles(
                    compressed_cnots, L2n, compressed_thetas, old_cnots, old_thetas, n
                )
                logger.debug("Compressed thetas: %s", compressed_thetas)

        logger.debug("Sparsity pattern: %s", sparsity)

        return ParametricCircuit(num_qubits=n, cnots=compressed_cnots, thetas=compressed_thetas)
        # return cmp_cnots, cmp_thetas, spar  # added cmp_thetas as an output

    def _extra_angles(self, lcnots, L, thetas, rcnot, rthetas, num_qubits, checks=True):
        # Internal routines to identify naked CNOT units and compress the circuit

        # 1. Build a qiskit circuit,
        v_base = ParametricCircuit(num_qubits=num_qubits, cnots=lcnots, thetas=thetas)
        # todo: why is tol=-1. ?
        qc = v_base.to_circuit(reverse=True, tol=-1.0)
        # Vbase.Plot(qc)

        qc2add = QuantumCircuit(num_qubits, 1)
        qc2add.ry(rthetas[0], rcnot[0] - 1)
        qc2add.rz(rthetas[1], rcnot[0] - 1)
        qc2add.ry(rthetas[2], rcnot[1] - 1)
        qc2add.rx(rthetas[3], rcnot[1] - 1)

        p = 3 * num_qubits + 5 * L

        # print(qc.data)
        # print(len(qc.data), p, L)
        instructions = qc.data[0:p] + qc2add.data[0:4]

        # print(instr)

        q = QuantumRegister(num_qubits, name="q")
        qcirc = QuantumCircuit(q)
        for ops in instructions:
            qcirc.append(ops[0], ops[1], ops[2])

        # qcirc.draw(output='mpl')
        # plt.show()

        # 2. Transform it into CNOT unit structure and extract thetas
        qbit_map = {}
        for ii in range(num_qubits):
            qbit_map[ii] = ii
        _, cnots, gate_list = self._reduce(qcirc.data, qbit_map)
        th_in, qct_in = self._extract_thetas(cnots, gate_list)

        if len(qc.data) > p:
            instructions = qct_in.data + qc.data[p:]
            q = QuantumRegister(num_qubits, name="q")
            qct = QuantumCircuit(q)
            for ops in instructions:
                qct.append(ops[0], ops[1], ops[2])
            _, cnots, gate_list = self._reduce(qct.data, qbit_map)
            # th = np.zeros(3*n + 4*np.shape(cnots)[1])
            # print('Patching Circuits and Angles --- ')
            # print(th_in, len(th_in), L)
            # print(thetas, thetas[3*n + 4*L:], len(thetas))
            th = np.zeros([3 * num_qubits + 4 * np.shape(cnots)[1]])
            # print(len(th))
            th[0 : len(th_in) - 3 * num_qubits] = th_in[: len(th_in) - 3 * num_qubits]
            th[len(th_in) - 3 * num_qubits : -3 * num_qubits] = thetas[4 * L : -3 * num_qubits]
            th[-3 * num_qubits :] = th_in[-3 * num_qubits :]
            # print(len(th), 3*n + 4*np.shape(cnots)[1])

            # pcpc
            for ops in qc.data[p:]:
                qcirc.append(ops[0], ops[1], ops[2])

        else:
            qct = qct_in
            th = th_in

        # qct.draw(output='mpl')
        # plt.show()

        # Internal checks
        if checks:
            u1 = Operator(qcirc).data  # todo: what is u1? (was U1)
            v1 = Operator(qct).data  # todo: what is v1? (was V1)
            alpha = np.angle(np.trace(np.dot(v1.conj().T, u1)))
            u_new1 = np.exp(-1j * alpha) * u1
            diff = np.linalg.norm(u_new1 - v1)

            if np.abs(diff) > 0.001:
                raise QiskitError("Internal inconsistency: failed compression, difference: " + diff)

        return th

    @staticmethod
    def _sparsity(cnots: np.ndarray, thetas: np.ndarray, eps: float) -> np.ndarray:
        # there are (num_cnots - 1) groups of 4 angles between each pair of consecutive cnots
        # sparsity is 0 corresponding to groups that are zero
        num_cnots = np.shape(cnots)[1]
        angle_sums = np.zeros(num_cnots - 1)
        for cnot_index in range(num_cnots - 1):
            angle_sums[i] = np.sum(np.abs(thetas[4 * cnot_index : 4 * (cnot_index + 1)]))
        sparsity = np.zeros(num_cnots - 1)
        nonzero_indices = angle_sums > eps
        sparsity[nonzero_indices] = 1
        sparsity = sparsity.astype(int)
        return sparsity

    @staticmethod
    def _zero_runs(vector):
        # determines the indices of vector that define runs of zeros
        # from https://stackoverflow.com/a/24892274
        # Create an array that is 1 where vector is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(vector, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    @staticmethod
    def _cnot_synthesis(num_qubits: int, cnots: np.ndarray):
        """
        synthesis algorithm from paper
        """
        # given a word of CNOTs, perform Guassian elimination on LT(n,Z2) to compute
        # an equivalent word that uses each CNOT at most once

        num_cnots = np.shape(cnots)[1]
        cnot_matrix_product = np.eye(num_qubits)
        for cnot_index in range(num_cnots):
            # indices of control and target qubit of new CNOT to multiply
            j = cnots[0, cnot_index] - 1
            k = cnots[1, cnot_index] - 1

            # new CNOT matrix to multiply into product
            cnot_matrix = np.eye(num_qubits)
            cnot_matrix[k, j] = 1
            cnot_matrix_product = np.dot(cnot_matrix, cnot_matrix_product)
            cnot_matrix_product = np.mod(cnot_matrix_product, 2)
        cnot_matrix_product = cnot_matrix_product.astype(int)
        synthesized_cnots = np.array([[0], [0]])

        # perform Gaussian elimination on cnot_matrix_product
        # cnot_matrix_product is lower triangular, so j > k
        # one step of Gaussian elimination does not have to actually be computed
        # instead, the step is just read off of the matrix:
        # the entries are iterated through in order,
        # each corresponding to a CNOT,
        # if the entry is 1 instead of 0,
        # the CNOT is added to the equivalent word
        for j in range(1, num_qubits):
            for k in range(j):
                if cnot_matrix_product[j, k] == 1:
                    synthesized_cnots = np.insert(synthesized_cnots, 0, np.array([[k + 1], [j + 1]]).T, 1)
        synthesized_cnots = synthesized_cnots[:, 0 : np.shape(synthesized_cnots)[1] - 1]

        # only replace with synthesized_cnots if it is shorter
        if np.shape(synthesized_cnots)[0] != 0:
            if np.shape(synthesized_cnots)[1] >= num_cnots:
                synthesized_cnots = np.array(cnots)
        return synthesized_cnots

    @staticmethod
    def _commute(cnots: np.ndarray):
        # randomly permutes one pair of consecutive cnots that can commute
        commuted_cnots = np.array(cnots)
        num_cnots = np.shape(cnots)[1]
        perm = np.random.permutation(num_cnots - 1)
        for p in perm:
            # control and target qubit of first CNOT
            i = cnots[0, p]
            j = cnots[1, p]

            # control and target qubit of second CNOT
            k = cnots[0, p + 1]
            m = cnots[1, p + 1]

            # if they can be permuted, then permute them
            if i != m and j != k:
                commuted_cnots[0:2, p : p + 2] = np.array([[k, i], [m, j]])
                break
        return commuted_cnots

    @staticmethod
    def _three_qubit_reduce(cnots: np.ndarray):
        # makes mirror reductions in consecutive cnots
        reduced_cnots = np.array(cnots)
        num_cnots = np.shape(reduced_cnots)[1]

        # check each three consecutive cnots for possible reductions
        for cnot_index in range(num_cnots - 2):
            if np.shape(reduced_cnots)[1] > cnot_index + 2:
                section = reduced_cnots[0:2, cnot_index : cnot_index + 3]
                indices = np.unique(section)
                num_indices = np.shape(indices)[0]
                reduced_section = section
                if num_indices == 3:
                    # the control/target qubits of the three CNOTs
                    j = section[0, 0]
                    k = section[1, 0]
                    v = section[0, 1]
                    w = section[1, 1]
                    p = section[0, 2]
                    q = section[1, 2]

                    # Define the 4 possible mirror configurations
                    # 1) long on the left
                    if k - j > 1:
                        # 1.1) up then down
                        if w == p and j == v and k == q:
                            reduced_section = np.array([[p, v], [q, w]])
                        # 1.2) down then up
                        if v == q and j == p and k == w:
                            reduced_section = np.array([[p, v], [q, w]])

                    # 2) long on the right
                    if q - p > 1:
                        # 2.1) up then down
                        if k == v and j == p and w == q:
                            reduced_section = np.array([[v, j], [w, k]])
                        # 2.2) down then up
                        if j == w and v == p and k == q:
                            reduced_section = np.array([[v, w], [j, k]])

                    reduced_cnots = np.delete(reduced_cnots, np.arange(cnot_index, cnot_index + 3), 1)
                    reduced_cnots = np.insert(reduced_cnots, cnot_index, np.transpose(reduced_sec), 1)
        return reduced_cnots

    @staticmethod
    def _two_qubit_reduce(cnots: np.ndarray):
        # compress equal consecutive cnots
        reduced_cnots = np.array(cnots)
        num_cnots = np.shape(reduced_cnots)[1]

        # check each pair of consecutive CNOTs to see if they are identical
        for cnot_index in range(num_cnots - 1):
            if np.shape(reduced_cnots)[1] > cnot_index + 1:
                section = reduced_cnots[0:2, cnot_index : cnot_index + 2]
                indices = np.unique(section)
                num_indices = np.shape(indices)[0]
                reduced_section = section
                if num_indices == 2:
                    # control/target qubits of the two CNOTs
                    j = section[0, 0]
                    k = section[1, 0]
                    v = section[0, 1]
                    w = section[1, 1]

                    # delete both if they are identical
                    if j == v and k == w:
                        reduced_section = [[], []]

                    reduced_cnots = np.delete(reduced_cnots, np.arange(cnot_index, cnot_index + 2), 1)
                    reduced_cnots = np.insert(reduced_cnots, cnot_index, np.transpose(reduced_section), 1)
        return red_cnots

    @staticmethod
    def _three_qubit_shuffle(cnots: np.ndarray, niter: int):
        # alternates between commute and two_qubit_reduce/three_qubit_reduce
        num_cnots = np.shape(cnots)[1]
        reduced_cnots = np.array(cnots)
        for _ in range(niter):
            for _ in range(num_cnots):
                reduced_cnots = EulerCompressor._three_qubit_reduce(reduced_cnots)
            for _ in range(num_cnots):
                reduced_cnots = EulerCompressor._two_qubit_reduce(reduced_cnots)
            reduced_cnots = EulerCompressor._commute(reduced_cnots)

        # only replace if reduced_cnots is shorter than cnots
        if np.shape(reduced_cnots)[0] != 0:
            if np.shape(reduced_cnots)[1] >= num_cnots:
                reduced_cnots = np.array(cnots)
        return reduced_cnots

    @staticmethod
    def _reduce(qc_data, qubit_map) -> Tuple[QuantumCircuit, List, Dict]:
        """
        Transforms to a QC that is in the same fashion with CNOT structures.
        Note, exactly the same function remains in transformations.py
        where it is used for transpilation.

        Args:
            qc_data:
            qubit_map:

        Returns:
            a tuple of quantum circuit, a list of two cnot structures, a dictionary of gates
        """
        qc = QuantumCircuit(len(qubit_map), 1)
        cnots_u = []
        cnots_d = []
        gate_list = dict()
        for ii in range(len(qubit_map)):
            gate_list[ii] = []
        for operations in qc_data:
            q = []
            for elements in operations[1]:
                # print(elements.index, qubit_map)
                if isinstance(qubit_map, list):
                    q += [qubit_map.index(elements.index)]
                else:
                    q += [elements.index]
                # print(elements.index, qubit_map.index(elements.index))
            qc.append(operations[0], q)
            if len(q) > 1:
                cnots_u += [q[0] + 1]
                cnots_d += [q[1] + 1]
                gate_list[q[0]] += ["xu"]
                gate_list[q[1]] += ["xd"]
            else:
                gate_list[q[0]] += [operations[0]]
        cnots = [cnots_u, cnots_d]

        return qc, cnots, gate_list

    @staticmethod
    def _extract_thetas(cnots, gate_list) -> Tuple[np.ndarray, QuantumCircuit]:
        """
        TODO: add description

        Args:
            cnots:
            gate_list:

        Returns:
            theta for parametric circuit and a qiskit quantum circuit.
        """

        thetas = dict()
        for elements in gate_list:
            thetas[elements] = []

        for qbits in gate_list:
            op = []
            for ii in range(len(gate_list[qbits])):
                nn = len(gate_list[qbits]) - ii - 1

                if gate_list[qbits][nn] == "xd":
                    if not op:  # TODO: originally op == []
                        thetas[qbits] += [0.0, 0.0]
                    else:
                        # RxRyRx decomposition on the lower layer
                        q = QuantumRegister(1, name="q")
                        qcirc = QuantumCircuit(q)
                        for ops in list(reversed(op)):
                            try:
                                qcirc.append(ops, [0])
                            except (AttributeError, TypeError):
                                if ops[1] == "rz":
                                    qcirc.rz(ops[0], 0)
                                else:
                                    qcirc.rx(ops[0], 0)

                        angles = OneQubitEulerDecomposer(basis="XYX").angles(Operator(qcirc).data)
                        # TODO: remove
                        # [angles[1], angles[0], angles[2]]
                        # this is how qiskit output the angles for XYX
                        alpha = angles[2]
                        thetas[qbits] += [angles[0], angles[1]]

                        op = [[alpha, "rx"]]  # rotation gate x!

                elif gate_list[qbits][nn] == "xu":
                    if not op:  # TODO: originally op == []
                        thetas[qbits] += [0.0, 0.0]
                    else:
                        # print('OP ', op[0].params, op[0].definition, op[0].decompositions)
                        # RzRyRz decomposition on the upper layer
                        q = QuantumRegister(1, name="q")
                        qcirc = QuantumCircuit(q)
                        for ops in list(reversed(op)):
                            try:
                                qcirc.append(ops, [0])
                            except (AttributeError, TypeError):
                                if ops[1] == "rz":
                                    qcirc.rz(ops[0], 0)
                                else:
                                    qcirc.rx(ops[0], 0)

                        # print(Operator(qcirc).data)
                        angles = OneQubitEulerDecomposer(basis="ZYZ").angles(Operator(qcirc).data)
                        # TODO: remove
                        # [angles[1], angles[0], angles[2]]
                        # this is how qiskit output the angles for ZYZ
                        alpha = angles[2]
                        thetas[qbits] += [angles[0], angles[1]]

                        op = [[alpha, "rz"]]  # rotation gate z!

                else:
                    op += [gate_list[qbits][nn]]

            if not op:
                thetas[qbits] += [0.0, 0.0, 0.0]
            else:
                q = QuantumRegister(1, name="q")
                qcirc = QuantumCircuit(q)
                for ops in list(reversed(op)):
                    try:
                        qcirc.append(ops, [0])
                    except (AttributeError, TypeError):
                        # print('OPS ', ops)
                        if ops[1] == "rz":
                            qcirc.rz(ops[0], 0)
                        else:
                            qcirc.rx(ops[0], 0)

                # TODO: decomposition (for now zyz)
                angles = OneQubitEulerDecomposer(basis="ZYZ").angles(Operator(qcirc).data)
                thetas[qbits] += [
                    angles[2],
                    angles[0],
                    angles[1],
                ]  # this is how qiskit output the angles for ZYZ

        num_thetas = np.shape(cnots)[1]
        n = len(gate_list)
        p = 3 * n + 4 * num_thetas
        thetas_t = np.zeros(p)
        qct = QuantumCircuit(n, 1)

        for k in range(n):
            p = 3 * k + 4 * num_thetas
            thetas_t[0 + p] = thetas[k][-1]
            thetas_t[1 + p] = thetas[k][-2]
            thetas_t[2 + p] = thetas[k][-3]

            qct.rz(thetas[k][-3], k)
            qct.ry(thetas[k][-2], k)
            qct.rz(thetas[k][-1], k)

        kk = dict()
        for k in range(n):
            kk[k] = 0

        for i in range(num_thetas):
            p = 4 * i
            q1 = cnots[0][i] - 1
            q2 = cnots[1][i] - 1

            thetas_t[0 + p] = thetas[q1][-5 - kk[q1]]
            thetas_t[1 + p] = thetas[q1][-4 - kk[q1]]
            thetas_t[2 + p] = thetas[q2][-5 - kk[q2]]
            thetas_t[3 + p] = thetas[q2][-4 - kk[q2]]

            qct.cx(q1, q2)
            qct.ry(thetas[q1][-5 - kk[q1]], q1)
            qct.rz(thetas[q1][-4 - kk[q1]], q1)
            qct.ry(thetas[q2][-5 - kk[q2]], q2)
            qct.rx(thetas[q2][-4 - kk[q2]], q2)

            kk[q1] += 2
            kk[q2] += 2

        return thetas_t, qct
