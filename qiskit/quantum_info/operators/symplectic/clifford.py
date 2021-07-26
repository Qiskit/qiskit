# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Clifford operator class.
"""
import re
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate, HGate, SGate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.synthesis.clifford_decompose import decompose_clifford
from qiskit.quantum_info.operators.mixins import generate_apidocs, AdjointMixin
from .stabilizer_table import StabilizerTable
from .clifford_circuits import _append_circuit


class Clifford(BaseOperator, AdjointMixin):
    """An N-qubit unitary operator from the Clifford group.

    **Representation**

    An *N*-qubit Clifford operator is stored as a length *2N*
    :class:`~qiskit.quantum_info.StabilizerTable` using the convention
    from reference [1].

    * Rows 0 to *N-1* are the *destabilizer* group generators
    * Rows *N* to *2N-1* are the *stabilizer* group generators.

    The internal :class:`~qiskit.quantum_info.StabilizerTable` for the Clifford
    can be accessed using the :attr:`table` attribute. The destabilizer or
    stabilizer rows can each be accessed as a length-N Stabilizer table using
    :attr:`destabilizer` and :attr:`stabilizer` attributes.

    A more easily human readable representation of the Clifford operator can
    be obtained by calling the :meth:`to_dict` method. This representation is
    also used if a Clifford object is printed as in the following example

    .. jupyter-execute::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Clifford

        # Bell state generation circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cliff = Clifford(qc)

        # Print the Clifford
        print(cliff)

        # Print the Clifford destabilizer rows
        print(cliff.destabilizer)

        # Print the Clifford stabilizer rows
        print(cliff.stabilizer)

    **Circuit Conversion**

    Clifford operators can be initialized from circuits containing *only* the
    following Clifford gates: :class:`~qiskit.circuit.library.IGate`,
    :class:`~qiskit.circuit.library.XGate`, :class:`~qiskit.circuit.library.YGate`,
    :class:`~qiskit.circuit.library.ZGate`, :class:`~qiskit.circuit.library.HGate`,
    :class:`~qiskit.circuit.library.SGate`, :class:`~qiskit.circuit.library.SdgGate`,
    :class:`~qiskit.circuit.library.CXGate`, :class:`~qiskit.circuit.library.CZGate`,
    :class:`~qiskit.circuit.library.SwapGate`.
    They can be converted back into a :class:`~qiskit.circuit.QuantumCircuit`,
    or :class:`~qiskit.circuit.Gate` object using the :meth:`~Clifford.to_circuit`
    or :meth:`~Clifford.to_instruction` methods respectively. Note that this
    decomposition is not necessarily optimal in terms of number of gates.

    .. note::

        A minimally generating set of gates for Clifford circuits is
        the :class:`~qiskit.circuit.library.HGate` and
        :class:`~qiskit.circuit.library.SGate` gate and *either* the
        :class:`~qiskit.circuit.library.CXGate` or
        :class:`~qiskit.circuit.library.CZGate` two-qubit gate.

    Clifford operators can also be converted to
    :class:`~qiskit.quantum_info.Operator` objects using the
    :meth:`to_operator` method. This is done via decomposing to a circuit, and then
    simulating the circuit as a unitary operator.

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    def __array__(self, dtype=None):
        if dtype:
            return np.asarray(self.to_matrix(), dtype=dtype)
        return self.to_matrix()

    def __init__(self, data, validate=True):
        """Initialize an operator object."""

        # Initialize from another Clifford by sharing the underlying
        # StabilizerTable
        if isinstance(data, Clifford):
            self._table = data._table

        # Initialize from ScalarOp as N-qubit identity discarding any global phase
        elif isinstance(data, ScalarOp):
            if not data.num_qubits or not data.is_unitary():
                raise QiskitError("Can only initialize from N-qubit identity ScalarOp.")
            self._table = StabilizerTable(np.eye(2 * data.num_qubits, dtype=bool))

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._table = Clifford.from_circuit(data)._table

        # Initialize StabilizerTable directly from the data
        else:
            self._table = StabilizerTable(data)

            # Validate table is a symplectic matrix
            if validate and not Clifford._is_symplectic(self._table.array):
                raise QiskitError(
                    "Invalid Clifford. Input StabilizerTable is not a valid" " symplectic matrix."
                )

        # Initialize BaseOperator
        super().__init__(num_qubits=self._table.num_qubits)

    def __repr__(self):
        return f"Clifford({repr(self.table)})"

    def __str__(self):
        return "Clifford: Stabilizer = {}, Destabilizer = {}".format(
            str(self.stabilizer.to_labels()), str(self.destabilizer.to_labels())
        )

    def __eq__(self, other):
        """Check if two Clifford tables are equal"""
        return super().__eq__(other) and self._table == other._table

    # ---------------------------------------------------------------------
    # Attributes
    # ---------------------------------------------------------------------
    def __getitem__(self, key):
        """Return a stabilizer Pauli row"""
        return self._table.__getitem__(key)

    def __setitem__(self, key, value):
        """Set a stabilizer Pauli row"""
        self._table.__setitem__(key, value)

    @property
    def table(self):
        """Return StabilizerTable"""
        return self._table

    @table.setter
    def table(self, value):
        """Set the stabilizer table"""
        # Note this setter cannot change the size of the Clifford
        # It can only replace the contents of the StabilizerTable with
        # another StabilizerTable of the same size.
        if not isinstance(value, StabilizerTable):
            value = StabilizerTable(value)
        self._table._array[:, :] = value._table._array
        self._table._phase[:] = value._table._phase

    @property
    def stabilizer(self):
        """Return the stabilizer block of the StabilizerTable."""
        return StabilizerTable(self._table[self.num_qubits : 2 * self.num_qubits])

    @stabilizer.setter
    def stabilizer(self, value):
        """Set the value of stabilizer block of the StabilizerTable"""
        inds = slice(self.num_qubits, 2 * self.num_qubits)
        self._table.__setitem__(inds, value)

    @property
    def destabilizer(self):
        """Return the destabilizer block of the StabilizerTable."""
        return StabilizerTable(self._table[0 : self.num_qubits])

    @destabilizer.setter
    def destabilizer(self, value):
        """Set the value of destabilizer block of the StabilizerTable"""
        inds = slice(0, self.num_qubits)
        self._table.__setitem__(inds, value)

    # ---------------------------------------------------------------------
    # Utility Operator methods
    # ---------------------------------------------------------------------

    def is_unitary(self):
        """Return True if the Clifford table is valid."""
        # A valid Clifford is always unitary, so this function is really
        # checking that the underlying Stabilizer table array is a valid
        # Clifford array.
        return Clifford._is_symplectic(self.table.array)

    # ---------------------------------------------------------------------
    # BaseOperator Abstract Methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        return Clifford._conjugate_transpose(self, "C")

    def adjoint(self):
        return Clifford._conjugate_transpose(self, "A")

    def transpose(self):
        return Clifford._conjugate_transpose(self, "T")

    def tensor(self, other):
        if not isinstance(other, Clifford):
            other = Clifford(other)
        return self._tensor(self, other)

    def expand(self, other):
        if not isinstance(other, Clifford):
            other = Clifford(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        # Pad stabilizers and destabilizers
        destab = b.destabilizer.expand(a.num_qubits * "I") + a.destabilizer.tensor(
            b.num_qubits * "I"
        )
        stab = b.stabilizer.expand(a.num_qubits * "I") + a.stabilizer.tensor(b.num_qubits * "I")

        # Add the padded table
        return Clifford(destab + stab, validate=False)

    def compose(self, other, qargs=None, front=False):
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        # If other is a QuantumCircuit we can more efficiently compose
        # using the _append_circuit method to update each gate recursively
        # to the current Clifford, rather than converting to a Clifford first
        # and then doing the composition of tables.
        if not front and isinstance(other, (QuantumCircuit, Instruction)):
            ret = self.copy()
            _append_circuit(ret, other, qargs=qargs)
            return ret

        if not isinstance(other, Clifford):
            other = Clifford(other)

        # Validate compose dimensions
        self._op_shape.compose(other._op_shape, qargs, front)

        # Pad other with identities if composing on subsystem
        other = self._pad_with_identity(other, qargs)

        if front:
            table1 = self.table
            table2 = other.table
        else:
            table1 = other.table
            table2 = self.table

        num_qubits = self.num_qubits

        array1 = table1.array.astype(int)
        phase1 = table1.phase.astype(int)

        array2 = table2.array.astype(int)
        phase2 = table2.phase.astype(int)

        # Update Pauli table
        pauli = StabilizerTable(array2.dot(array1) % 2)

        # Add phases
        phase = np.mod(array2.dot(phase1) + phase2, 2)

        # Correcting for phase due to Pauli multiplication
        ifacts = np.zeros(2 * num_qubits, dtype=int)

        for k in range(2 * num_qubits):

            row2 = array2[k]
            x2 = table2.X[k]
            z2 = table2.Z[k]

            # Adding a factor of i for each Y in the image of an operator under the
            # first operation, since Y=iXZ

            ifacts[k] += np.sum(x2 & z2)

            # Adding factors of i due to qubit-wise Pauli multiplication

            for j in range(num_qubits):
                x = 0
                z = 0
                for i in range(2 * num_qubits):
                    if row2[i]:
                        x1 = array1[i, j]
                        z1 = array1[i, j + num_qubits]
                        if (x | z) & (x1 | z1):
                            val = np.mod(np.abs(3 * z1 - x1) - np.abs(3 * z - x) - 1, 3)
                            if val == 0:
                                ifacts[k] += 1
                            elif val == 1:
                                ifacts[k] -= 1
                        x = np.mod(x + x1, 2)
                        z = np.mod(z + z1, 2)

        p = np.mod(ifacts, 4) // 2

        phase = np.mod(phase + p, 2)

        return Clifford(StabilizerTable(pauli, phase), validate=False)

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def to_dict(self):
        """Return dictionary representation of Clifford object."""
        return {
            "stabilizer": self.stabilizer.to_labels(),
            "destabilizer": self.destabilizer.to_labels(),
        }

    @staticmethod
    def from_dict(obj):
        """Load a Clifford from a dictionary"""
        destabilizer = StabilizerTable.from_labels(obj.get("destabilizer"))
        stabilizer = StabilizerTable.from_labels(obj.get("stabilizer"))
        return Clifford(destabilizer + stabilizer)

    def to_matrix(self):
        """Convert operator to Numpy matrix."""
        return self.to_operator().data

    def to_operator(self):
        """Convert to an Operator object."""
        return Operator(self.to_instruction())

    def to_circuit(self):
        """Return a QuantumCircuit implementing the Clifford.

        For N <= 3 qubits this is based on optimal CX cost decomposition
        from reference [1]. For N > 3 qubits this is done using the general
        non-optimal compilation routine from reference [2].

        Return:
            QuantumCircuit: a circuit implementation of the Clifford.

        References:
            1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
               structure of the Clifford group*,
               `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_

            2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
               Phys. Rev. A 70, 052328 (2004).
               `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
        """
        return decompose_clifford(self)

    def to_instruction(self):
        """Return a Gate instruction implementing the Clifford."""
        return self.to_circuit().to_gate()

    @staticmethod
    def from_circuit(circuit):
        """Initialize from a QuantumCircuit or Instruction.

        Args:
            circuit (QuantumCircuit or ~qiskit.circuit.Instruction):
                instruction to initialize.

        Returns:
            Clifford: the Clifford object for the instruction.

        Raises:
            QiskitError: if the input instruction is non-Clifford or contains
                         classical register instruction.
        """
        if not isinstance(circuit, (QuantumCircuit, Instruction)):
            raise QiskitError("Input must be a QuantumCircuit or Instruction")

        # Convert circuit to an instruction
        if isinstance(circuit, QuantumCircuit):
            circuit = circuit.to_instruction()

        # Initialize an identity Clifford
        clifford = Clifford(np.eye(2 * circuit.num_qubits), validate=False)
        _append_circuit(clifford, circuit)
        return clifford

    @staticmethod
    def from_label(label):
        """Return a tensor product of single-qubit Clifford gates.

        Args:
            label (string): single-qubit operator string.

        Returns:
            Clifford: The N-qubit Clifford operator.

        Raises:
            QiskitError: if the label contains invalid characters.

        Additional Information:
            The labels correspond to the single-qubit Cliffords are

            * - Label
              - Stabilizer
              - Destabilizer
            * - ``"I"``
              - +Z
              - +X
            * - ``"X"``
              - -Z
              - +X
            * - ``"Y"``
              - -Z
              - -X
            * - ``"Z"``
              - +Z
              - -X
            * - ``"H"``
              - +X
              - +Z
            * - ``"S"``
              - +Z
              - +Y
        """
        # Check label is valid
        label_gates = {
            "I": IGate(),
            "X": XGate(),
            "Y": YGate(),
            "Z": ZGate(),
            "H": HGate(),
            "S": SGate(),
        }
        if re.match(r"^[IXYZHS\-+]+$", label) is None:
            raise QiskitError("Label contains invalid characters.")
        # Initialize an identity matrix and apply each gate
        num_qubits = len(label)
        op = Clifford(np.eye(2 * num_qubits, dtype=bool))
        for qubit, char in enumerate(reversed(label)):
            _append_circuit(op, label_gates[char], qargs=[qubit])
        return op

    # ---------------------------------------------------------------------
    # Internal helper functions
    # ---------------------------------------------------------------------

    @staticmethod
    def _is_symplectic(mat):
        """Return True if input is symplectic matrix."""
        # Condition is
        # table.T * [[0, 1], [1, 0]] * table = [[0, 1], [1, 0]]
        # where we are block matrix multiplying using symplectic product

        dim = len(mat) // 2
        if mat.shape != (2 * dim, 2 * dim):
            return False

        one = np.eye(dim, dtype=int)
        zero = np.zeros((dim, dim), dtype=int)
        seye = np.block([[zero, one], [one, zero]])
        arr = mat.astype(int)
        return np.array_equal(np.mod(arr.T.dot(seye).dot(arr), 2), seye)

    @staticmethod
    def _conjugate_transpose(clifford, method):
        """Return the adjoint, conjugate, or transpose of the Clifford.

        Args:
            clifford (Clifford): a clifford object.
            method (str): what function to apply 'A', 'C', or 'T'.

        Returns:
            Clifford: the modified clifford.
        """
        ret = clifford.copy()
        if method in ["A", "T"]:
            # Apply inverse
            # Update table
            tmp = ret.destabilizer.X.copy()
            ret.destabilizer.X = ret.stabilizer.Z.T
            ret.destabilizer.Z = ret.destabilizer.Z.T
            ret.stabilizer.X = ret.stabilizer.X.T
            ret.stabilizer.Z = tmp.T
            # Update phase
            ret.table.phase ^= clifford.dot(ret).table.phase
        if method in ["C", "T"]:
            # Apply conjugate
            ret.table.phase ^= np.mod(np.sum(ret.table.X & ret.table.Z, axis=1), 2).astype(bool)
        return ret

    def _pad_with_identity(self, clifford, qargs):
        """Pad Clifford with identities on other subsystems."""
        if qargs is None:
            return clifford

        padded = Clifford(StabilizerTable(np.eye(2 * self.num_qubits, dtype=bool)), validate=False)

        inds = list(qargs) + [self.num_qubits + i for i in qargs]

        # Pad Pauli array
        pauli = clifford.table.array
        for i, pos in enumerate(qargs):
            padded.table.array[inds, pos] = pauli[:, i]
            padded.table.array[inds, self.num_qubits + pos] = pauli[:, clifford.num_qubits + i]

        # Pad phase
        padded.table.phase[inds] = clifford.table.phase

        return padded


# Update docstrings for API docs
generate_apidocs(Clifford)
