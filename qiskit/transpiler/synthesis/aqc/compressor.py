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
from qiskit import QuantumCircuit, QuantumRegister
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

        # applies cnot_synthesis and minimal to the sparsity pattern
        # synth=False - compliant with coupling map, otherwise (True) may introduce connections
        # between qubits, breaking coupling map, use for full connectivity only

        Args:
            circuit: parametric circuit to be compressed.

        Returns:
            compressed parametric circuit.
        """
        assert isinstance(circuit, ParametricCircuit)

        n = circuit.num_qubits
        cnots = circuit.cnots
        cmp_cnots = np.array(cnots)

        # Initialize the compressed theta for bookeeping/the warm start
        cmp_thetas = np.array(circuit.thetas)

        # Determine the sparsity pattern:
        # vector that tells where four angles after a CNOT are 0s, of dimension L-1
        spar = self._sparsity(circuit.cnots, circuit.thetas, self._eps)

        # Determine which cnot indices are to be affected by the compression:
        # [[start1, end1], [start2, end2], ... ]
        ranges = self._zero_runs(spar)
        num_ranges = np.shape(ranges)[0]

        for z in range(num_ranges):
            a = ranges[z, 0]
            b = ranges[z, 1] + 1
            # Slice the section of cnots that will be affected by the compression rules
            sec = np.array(cmp_cnots[0:2, a:b])

            # Apply synthesis or not
            if self._synth:
                # TODO: check if this works
                syn_sec = self._cnot_synthesis(n, sec)
            else:
                syn_sec = sec

            # Apply commutation and mirror rules
            # If there is still a CNOT:
            if np.shape(syn_sec)[0] != 0:
                if np.shape(syn_sec)[1] > 2:
                    syn_sec = self._minimal(syn_sec, self._niter)

            num_cnots1 = np.shape(cmp_cnots)[1]

            old_cnots = np.array(cmp_cnots[:, b - 1])  # keep the rightmost cnot that is deleted
            logger.debug("old cnot = %s", old_cnots)
            # print('old_cnot = ', old_cnot)

            cmp_cnots = np.delete(cmp_cnots, np.arange(a, b), 1)

            # print('cmp_cnot = ', cmp_cnots)

            old_thetas = np.array(
                cmp_thetas[4 * (b - 1) : 4 * b]
            )  # keep the angles corresponding to old_cnot
            # print('Before =', (old_thetas))
            cmp_thetas = np.delete(
                cmp_thetas, np.arange(4 * a, 4 * b)
            )  # delete angles corresponding to deleted cnots
            # print('After =', (cmp_thetas))

            # c = 0  # length of syn_sec

            # If there is still a CNOT:
            if np.shape(syn_sec)[0] != 0:
                cmp_cnots = np.insert(cmp_cnots, a, np.transpose(syn_sec), 1)
                c = np.shape(syn_sec)[1]  # length of syn_sec
                cmp_thetas = np.insert(
                    cmp_thetas, 4 * a, np.zeros(4 * c)
                )  # insert zero angles for inserted cnots

            # print('cmp_cnot = ', cmp_cnots)
            # print('After =', (cmp_thetas))

            num_cnots2 = np.shape(cmp_cnots)[1]

            # print('L2 = ', L2)
            # print('ranges = ', ranges)
            # Redefine ranges in order to shift the indices depending on the compression
            ranges += num_cnots2 - num_cnots1
            # print('ranges = ', ranges)
            L2n = ranges[z, 1] + 1
            # print('Bookkeeping : ', cmp_cnots, cmp_thetas, old_cnot, old_thetas, L2n)

            #
            cmp_thetas = self._extra_angles(
                cmp_cnots, L2n, cmp_thetas, old_cnots, old_thetas, n
            )  # update angles
            # print(cmp_thetas)
            # cpcp
        if self._verbose >= 1:
            print("Sparsity pattern: {}".format(spar))
        return ParametricCircuit(num_qubits=n, cnots=cmp_cnots, thetas=cmp_thetas)
        # return cmp_cnots, cmp_thetas, spar  # added cmp_thetas as an output

    def _extra_angles(self, lcnots, L, thetas, rcnot, rthetas, n, checks=True):
        # Internal routines to identify naked CNOT units and compress the circuit

        # 1. Build a qiskit circuit,
        v_base = ParametricCircuit(num_qubits=n, cnots=lcnots, thetas=thetas)
        # todo: why is tol=-1. ?
        qc = v_base.to_circuit(reverse=True, tol=-1.0)
        # Vbase.Plot(qc)

        qc2add = QuantumCircuit(n, 1)
        qc2add.ry(rthetas[0], rcnot[0] - 1)
        qc2add.rz(rthetas[1], rcnot[0] - 1)
        qc2add.ry(rthetas[2], rcnot[1] - 1)
        qc2add.rx(rthetas[3], rcnot[1] - 1)

        p = 3 * n + 5 * L

        # print(qc.data)
        # print(len(qc.data), p, L)
        instructions = qc.data[0:p] + qc2add.data[0:4]

        # print(instr)

        q = QuantumRegister(n, name="q")
        qcirc = QuantumCircuit(q)
        for ops in instructions:
            qcirc.append(ops[0], ops[1], ops[2])

        # qcirc.draw(output='mpl')
        # plt.show()

        # 2. Transform it into CNOT unit structure and extract thetas
        qbit_map = dict()
        for ii in range(n):
            qbit_map[ii] = ii
        _, cnots, gate_list = self._reduce(qcirc.data, qbit_map)
        th_in, qct_in = self._extract_thetas(cnots, gate_list)

        if len(qc.data) > p:
            instructions = qct_in.data + qc.data[p:]
            q = QuantumRegister(n, name="q")
            qct = QuantumCircuit(q)
            for ops in instructions:
                qct.append(ops[0], ops[1], ops[2])
            _, cnots, gate_list = self._reduce(qct.data, qbit_map)
            # th = np.zeros(3*n + 4*np.shape(cnots)[1])
            # print('Patching Circuits and Angles --- ')
            # print(th_in, len(th_in), L)
            # print(thetas, thetas[3*n + 4*L:], len(thetas))
            th = np.zeros([3 * n + 4 * np.shape(cnots)[1]])
            # print(len(th))
            th[0 : len(th_in) - 3 * n] = th_in[: len(th_in) - 3 * n]
            th[len(th_in) - 3 * n : -3 * n] = thetas[4 * L : -3 * n]
            th[-3 * n :] = th_in[-3 * n :]
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
            U1 = Operator(qcirc).data
            V1 = Operator(qct).data
            alpha = np.angle(np.trace(np.dot(V1.conj().T, U1)))
            u_new1 = np.exp(-1j * alpha) * U1
            diff = np.linalg.norm(u_new1 - V1)

            if np.abs(diff) > 0.001:
                print("Internal inconsistency: failed compression")

        return th

    @staticmethod
    def _sparsity(cnots, thetas, eps):
        # if the 4 angles between 2 CNOT units are all equal to zero, then spar is 0 there
        L = np.shape(cnots)[1]
        sums = np.zeros(L - 1)
        for l in range(L - 1):
            sums[l] = np.sum(np.abs(thetas[4 * l : 4 * (l + 1)]))
        spar = np.zeros(L - 1)
        idx = sums > eps
        spar[idx] = 1
        spar = spar.astype(int)
        return spar

    @staticmethod
    def _zero_runs(a):
        # from https://stackoverflow.com/a/24892274
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    @staticmethod
    def _cnot_synthesis(num_qubits: int, cnots: np.ndarray):
        """
        synthesis algorithm from paper
        """
        num_cnots = np.shape(cnots)[1]
        A = np.eye(num_qubits)
        for cnot_index in range(num_cnots):
            j = cnots[0, cnot_index] - 1
            k = cnots[1, cnot_index] - 1
            # todo: what is B?
            B = np.eye(num_qubits)
            B[k, j] = 1
            A = np.dot(B, A)
            A = np.mod(A, 2)
        A = A.astype(int)
        syn_cnots = np.array([[0], [0]])
        for j in range(1, num_qubits):
            for k in range(j):
                if A[j, k] == 1:
                    syn_cnots = np.insert(syn_cnots, 0, np.array([[k + 1], [j + 1]]).T, 1)
        syn_cnots = syn_cnots[:, 0 : np.shape(syn_cnots)[1] - 1]
        if np.shape(syn_cnots)[0] != 0:
            if np.shape(syn_cnots)[1] >= num_cnots:
                syn_cnots = np.array(cnots)
        return syn_cnots

    @staticmethod
    def _commute(cnots: np.ndarray):
        # randomly permutes one pair of consecutive cnots that can commute
        com_cnots = np.array(cnots)
        num_cnots = np.shape(cnots)[1]
        perm = np.random.permutation(num_cnots - 1)
        for p in perm:
            j = cnots[0, p]
            k = cnots[1, p]
            l = cnots[0, p + 1]  # todo: bad index: l
            m = cnots[1, p + 1]
            if j != m and k != l:
                com_cnots[0:2, p : p + 2] = np.array([[l, j], [m, k]])
                break
        return com_cnots

    @staticmethod
    def _reduce_for_minimal(cnots: np.ndarray):
        """
        This is reduce() function that is called in minimal().
        Do not confuse it with another reduce().
        """
        # makes mirror reductions in consecutive cnots
        red_cnots = np.array(cnots)
        num_cnots = np.shape(red_cnots)[1]

        for i in range(num_cnots - 2):
            if np.shape(red_cnots)[1] > i + 2:
                # print('** In reduce i = ', i)
                sec = red_cnots[0:2, i : i + 3]
                indices = np.unique(sec)
                num_indices = np.shape(indices)[0]
                red_sec = sec
                # TODO: remove
                # print('parameters: ', indices, num_indices, sec, red_cnots,
                #       L, i, np.shape(red_cnots)[1])
                if num_indices == 3:
                    j = sec[0, 0]
                    k = sec[1, 0]
                    l = sec[0, 1]  # todo: bad index: l
                    m = sec[1, 1]
                    p = sec[0, 2]
                    q = sec[1, 2]

                    # Define the 4 possible mirror configurations
                    # 1) long on the left
                    if k - j > 1:
                        # 1.1) up then down
                        if m == p and j == l and k == q:
                            red_sec = np.array([[p, l], [q, m]])
                        # 1.2) down then up
                        if l == q and j == p and k == m:
                            red_sec = np.array([[p, l], [q, m]])

                    # 2) long on the right
                    if q - p > 1:
                        # 2.1) up then down
                        if k == l and j == p and m == q:
                            red_sec = np.array([[l, j], [m, k]])
                        # 2.2) down then up
                        if j == m and l == p and k == q:
                            red_sec = np.array([[l, m], [j, k]])

                    # print('1, ', red_cnots)
                    red_cnots = np.delete(red_cnots, np.arange(i, i + 3), 1)
                    # print('2, ', red_cnots)
                    red_cnots = np.insert(red_cnots, i, np.transpose(red_sec), 1)
                    # print('3, ', red_cnots)

        # todo: why this is needed? we don't use num_cnots
        # new_num_cnots = np.shape(red_cnots)[1]
        # if new_num_cnots != num_cnots:
        #     num_cnots = new_num_cnots
        return red_cnots

    @staticmethod
    def _compress_equals(cnots: np.ndarray):
        # compress equal consecutive cnots
        red_cnots = np.array(cnots)
        num_cnots = np.shape(red_cnots)[1]

        for i in range(num_cnots - 1):
            if np.shape(red_cnots)[1] > i + 1:
                # print('In eq compress i = ', i)
                sec = red_cnots[0:2, i : i + 2]
                indices = np.unique(sec)
                num_indices = np.shape(indices)[0]
                red_sec = sec
                if num_indices == 2:
                    j = sec[0, 0]
                    k = sec[1, 0]
                    l = sec[0, 1]  # todo: bad index: l
                    m = sec[1, 1]

                    if j == l and k == m:
                        red_sec = [[], []]

                    # print('1, ', red_cnots)
                    red_cnots = np.delete(red_cnots, np.arange(i, i + 2), 1)
                    # print('2, ', red_cnots)
                    red_cnots = np.insert(red_cnots, i, np.transpose(red_sec), 1)
                    # print('3, ', red_cnots)

        # # todo: why this is needed? we don't use num_cnots
        # new_num_cnots = np.shape(red_cnots)[1]
        # if new_num_cnots != num_cnots:
        #     num_cnots = new_num_cnots
        return red_cnots

    @staticmethod
    def _minimal(cnots: np.ndarray, niter: int):
        # alternates between commute and reduce
        min_cnots = np.array(cnots)
        for _ in range(niter):
            # print('External i = ', i, ' cnots = ', min_cnots)
            for _ in range(np.shape(cnots)[1]):
                min_cnots = EulerCompressor._reduce_for_minimal(min_cnots)
            # print('reduce ', min_cnots)
            for _ in range(np.shape(cnots)[1]):
                min_cnots = EulerCompressor._compress_equals(min_cnots)
            # print('compress equals ', min_cnots)
            min_cnots = EulerCompressor._commute(min_cnots)
            # print('commute ', min_cnots)
        return min_cnots

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

        L = np.shape(cnots)[1]
        n = len(gate_list)
        p = 3 * n + 4 * L
        thetas_t = np.zeros(p)
        qct = QuantumCircuit(n, 1)

        for k in range(n):
            p = 3 * k + 4 * L
            thetas_t[0 + p] = thetas[k][-1]
            thetas_t[1 + p] = thetas[k][-2]
            thetas_t[2 + p] = thetas[k][-3]

            qct.rz(thetas[k][-3], k)
            qct.ry(thetas[k][-2], k)
            qct.rz(thetas[k][-1], k)

        kk = dict()
        for k in range(n):
            kk[k] = 0

        for l in range(L):
            p = 4 * l
            q1 = cnots[0][l] - 1
            q2 = cnots[1][l] - 1

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
