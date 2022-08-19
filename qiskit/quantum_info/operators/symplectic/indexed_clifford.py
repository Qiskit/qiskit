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
Utilities for using the Clifford group in randomized benchmarking
"""

from functools import lru_cache

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic import StabilizerTable


@lru_cache(maxsize=24)
def clifford_1_qubit_circuit(num):
    """Return the 1-qubit clifford circuit corresponding to `num`
    where `num` is between 0 and 23.
    """
    unpacked = _unpack_num(num, (2, 3, 4))
    i, j, p = unpacked[0], unpacked[1], unpacked[2]
    qc = QuantumCircuit(1)
    if i == 1:
        qc.h(0)
    if j == 1:
        qc.sxdg(0)
    if j == 2:
        qc.s(0)
    if p == 1:
        qc.x(0)
    if p == 2:
        qc.y(0)
    if p == 3:
        qc.z(0)
    return qc


_CLIFFORD_2_QUBIT_SIGS = (
    (2, 2, 3, 3, 4, 4),
    (2, 2, 3, 3, 3, 3, 4, 4),
    (2, 2, 3, 3, 3, 3, 4, 4),
    (2, 2, 3, 3, 4, 4),
)


@lru_cache(maxsize=11520)
def clifford_2_qubit_circuit(num):
    """Return the 2-qubit clifford circuit corresponding to `num`
    where `num` is between 0 and 11519.
    """
    vals = _unpack_num_multi_sigs(num, _CLIFFORD_2_QUBIT_SIGS)
    qc = QuantumCircuit(2)
    if vals[0] == 0 or vals[0] == 3:
        (form, i0, i1, j0, j1, p0, p1) = vals
    else:
        (form, i0, i1, j0, j1, k0, k1, p0, p1) = vals
    if i0 == 1:
        qc.h(0)
    if i1 == 1:
        qc.h(1)
    if j0 == 1:
        qc.sxdg(0)
    if j0 == 2:
        qc.s(0)
    if j1 == 1:
        qc.sxdg(1)
    if j1 == 2:
        qc.s(1)
    if form in (1, 2, 3):
        qc.cx(0, 1)
    if form in (2, 3):
        qc.cx(1, 0)
    if form == 3:
        qc.cx(0, 1)
    if form in (1, 2):
        if k0 == 1:  # V gate
            qc.sdg(0)
            qc.h(0)
        if k0 == 2:  # W gate
            qc.h(0)
            qc.s(0)
        if k1 == 1:  # V gate
            qc.sdg(1)
            qc.h(1)
        if k1 == 2:  # W gate
            qc.h(1)
            qc.s(1)
    if p0 == 1:
        qc.x(0)
    if p0 == 2:
        qc.y(0)
    if p0 == 3:
        qc.z(0)
    if p1 == 1:
        qc.x(1)
    if p1 == 2:
        qc.y(1)
    if p1 == 3:
        qc.z(1)
    return qc


def _unpack_num(num, sig):
    r"""Returns a tuple :math:`(a_1, \ldots, a_n)` where
    :math:`0 \le a_i \le \sigma_i` where
    sig=:math:`(\sigma_1, \ldots, \sigma_n)` and num is the sequential
    number of the tuple
    """
    res = []
    for k in sig:
        res.append(num % k)
        num //= k
    return res


def _unpack_num_multi_sigs(num, sigs):
    """Returns the result of `_unpack_num` on one of the
    signatures in `sigs`
    """
    for i, sig in enumerate(sigs):
        sig_size = 1
        for k in sig:
            sig_size *= k
        if num < sig_size:
            return [i] + _unpack_num(num, sig)
        num -= sig_size
    return None


class IndexedClifford(Clifford):
    """Indexed 1- or 2-qubit Clifford"""

    def __init__(self, num_qubits: int, index: int):
        """Initialize an operator object."""
        if num_qubits > 2:
            raise QiskitError("num_qubits must be 1 or 2 for IndexedClifford")

        # pylint: disable=bad-super-call
        super(Clifford, self).__init__(num_qubits=num_qubits)
        self._index = index
        self._table = None

    @property
    def index(self):
        """Index of the Clifford."""
        return self._index

    def __repr__(self):
        return f"IndexedClifford(num_qubits={self.num_qubits}, index={self.index})"

    def __str__(self):
        return f"IndexedClifford: num_qubits={self.num_qubits}, index={self.index}"

    def __hash__(self):
        return hash((self.num_qubits, self._index))

    def __eq__(self, other):
        if isinstance(other, IndexedClifford):
            return self.num_qubits == other.num_qubits and self._index == other._index
        return super().__eq__(other)

    def _create_table(self):
        self._table = Clifford.from_circuit(self.to_circuit()).table

    @property
    def table(self):
        """Return StabilizerTable"""
        if self._table is None:
            self._create_table()
        return self._table

    @table.setter
    def table(self, value):
        """Set the stabilizer table"""
        raise NotImplementedError("Setter for table attribute is not supported")

    @property
    def stabilizer(self):
        """Return the stabilizer block of the StabilizerTable."""
        return StabilizerTable(self.table[self.num_qubits : 2 * self.num_qubits])

    @stabilizer.setter
    def stabilizer(self, value):
        """Set the value of stabilizer block of the StabilizerTable"""
        raise NotImplementedError("Setter for stabilizer attribute is not supported")

    @property
    def destabilizer(self):
        """Return the destabilizer block of the StabilizerTable."""
        return StabilizerTable(self.table[0 : self.num_qubits])

    @destabilizer.setter
    def destabilizer(self, value):
        """Set the value of destabilizer block of the StabilizerTable"""
        raise NotImplementedError("Setter for destabilizer attribute is not supported")

    def conjugate(self):
        self._create_table()
        return super().conjugate()

    def adjoint(self):
        self._create_table()
        return super().adjoint()

    def transpose(self):
        self._create_table()
        return super().transpose()

    def tensor(self, other):
        self._create_table()
        return super().tensor(other)

    def expand(self, other):
        self._create_table()
        return super().expand(other)

    @classmethod
    def _tensor(cls, a, b):
        return Clifford._tensor(a, b)

    def compose(self, other, qargs=None, front=False):
        self._create_table()
        return super().compose(other, qargs, front)

    INDEX_TO_CIRCUIT = {
        1: clifford_1_qubit_circuit,
        2: clifford_2_qubit_circuit,
    }

    def to_circuit(self):
        """Return a QuantumCircuit implementing the Clifford."""
        circ = self.INDEX_TO_CIRCUIT[self.num_qubits](self.index)
        circ.name = str(self)
        return circ
