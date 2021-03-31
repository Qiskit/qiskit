# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Stabilizer state class.
"""

import copy
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_x


class StabilizerState(QuantumState):
    """StabilizerState class. Based on the internal class:
    :class:`~qiskit.quantum_info.Clifford`

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    def __init__(self, data):
        """Initialize a StabilizerState object.

        Args:
            data (StabilizerState or Clifford or Pauli or QuantumCircuit or
                  qiskit.circuit.Instruction):
                Data from which the stabilizer state can be constructed.

        Raises:
            QiskitError: if input data is not a valid stabilizer state (not a valid Clifford).
        """

        # Initialize from a Clifford
        if isinstance(data, Clifford):
            self._data = data

        # Initialize from a Pauli
        elif isinstance(data, Pauli):
            self._data = Clifford(data.to_instruction())

        # Initialize from another StabilizerState
        elif isinstance(data, StabilizerState):
            self._data = data._data

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._data = Clifford.from_circuit(data)

        # Validate that Clifford table is a symplectic matrix
        if not self.is_valid():
            raise QiskitError(
                'Invalid StabilizerState. Input is not a valid Clifford.')

        # Initialize
        super().__init__()

    def __eq__(self, other):
        return self._data == other._data

    def __repr__(self):
        return 'StabilizerState({})'.format(repr(self._data))

    @property
    def data(self):
        """Return StabilizerState data"""
        return self._data

    def is_valid(self, atol=None, rtol=None):
        """Return True if a valid StabilizerState."""
        return Clifford.is_unitary(self.data)

    def _add(self, other):
        raise NotImplementedError(
            "{} does not support addition".format(type(self)))

    def _multiply(self, other):
        raise NotImplementedError(
            "{} does not support scalar multiplication".format(type(self)))

    def trace(self):
        """Return the trace of the stabilizer state as a density matrix,
        which equals to 1, since it is always a pure state."""
        return 1

    def purity(self):
        """Return the purity of the quantum state,
        which equals to 1, since it is always a pure state."""
        return 1

    def to_operator(self):
        """Convert state to matrix operator class"""
        return Clifford(self.data).to_operator()

    def conjugate(self):
        """Return the conjugate of the operator."""
        return StabilizerState(Clifford.conjugate(self.data))

    def transpose(self):
        """Return the transpose of the operator."""
        return StabilizerState(Clifford.transpose(self.data))

    def tensor(self, other):
        """Return the tensor product stabilzier state self ⊗ other.

        Args:
            other (StabilizerState): a stabilizer state object.

        Returns:
            StabilizerState: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a StabilizerState.
        """
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        return StabilizerState((self.data).tensor(other.data))

    def expand(self, other):
        """Return the tensor product stabilzier state other ⊗ self.

        Args:
            other (StabilizerState): a stabilizer state object.

        Returns:
            StabilizerState: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other is not a StabilizerState.
        """
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        return StabilizerState((self.data).expand(other.data))

    def evolve(self, other, qargs=None):
        """Evolve a stabilizer state by a Clifford operator.

        Args:
            other (Clifford or QuantumCircuit or qiskit.circuit.Instruction):
                The Clifford operator to evolve by.
            qargs (list): a list of stabilizer subsystem positions to apply the operator on.

        Returns:
            StabilizerState: the output stabilizer state.

        Raises:
            QiskitError: if other is not a StabilizerState.
            QiskitError: if the operator dimension does not match the
                         specified StabilizerState subsystem dimensions.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        return StabilizerState((self.data).compose(other.data, qargs))

    def expectation_value(self, oper, qargs=None):
        """Compute the expectation value of an operator.

        Args:
            oper (BaseOperator): an operator to evaluate expval.
            qargs (None or list): subsystems to apply the operator on.

        Returns:
            complex: the expectation value.
        """
        pass

    def probabilities(self, qargs=None, decimals=None):
        """Return the subsystem measurement probability vector.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            np.array: The Numpy vector array of probabilities.
        """
        pass

    def probabilities_dict(self, qargs=None, decimals=None):
        """Return the subsystem measurement probability dictionary.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        This dictionary representation uses a Ket-like notation where the
        dictionary keys are qudit strings for the subsystem basis vectors.
        If any subsystem has a dimension greater than 10 comma delimiters are
        inserted between integers so that subsystems can be distinguished.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            dict: The measurement probabilities in dict (ket) form.
        """
        pass

    def reset(self, qargs=None):
        """Reset state or subsystems to the 0-state.

        Args:
            qargs (list or None): subsystems to reset, if None all
                                  subsystems will be reset to their 0-state
                                  (Default: None).

        Returns:
            StabilizerState: the reset state.

        Additional Information:
            If all subsystems are reset this will return the ground state
            on all subsystems. If only some subsystems are reset this
            function will perform a measurement on those subsystems and
            evolve the subsystems so that the collapsed post-measurement
            states are rotated to the 0-state. The RNG seed for this
            sampling can be set using the :meth:`seed` method.
        """
        # Resetting all qubits does not require sampling or RNG
        if qargs is None:
            return StabilizerState(Clifford((np.eye(2 * self.data.num_qubits))))

        for qubit in qargs:
            # Apply measurement and get classical outcome
            outcome = self._measure_and_update(qubit, 0)

            # Use the outcome to apply X gate to any qubits left in the
            # |1> state after measure, then discard outcome.
            if outcome == 1:
                _append_x(self.data, qubit)

        return self

    def measure(self, qargs=None):
        """Measure subsystems and return outcome and post-measure state.

        Note that this function uses the QuantumStates internal random
        number generator for sampling the measurement outcome. The RNG
        seed can be set using the :meth:`seed` method.

        Args:
            qargs (list or None): subsystems to sample measurements for,
                                  if None sample measurement of all
                                  subsystems (Default: None).

        Returns:
            tuple: the pair ``(outcome, state)`` where ``outcome`` is the
                   measurement outcome string label, and ``state`` is the
                   collapsed post-measurement stabilizer state for the
                   corresponding outcome.
        """
        if qargs is None:
            qargs = range(self.data.num_qubits)

        outcome = ''
        for qubit in qargs:
            randbit = self._rng.randint(2)
            outcome = str(self._measure_and_update(qubit, randbit)) + outcome
        return outcome, self

    def sample_memory(self, shots, qargs=None):
        """Sample a list of qubit measurement outcomes in the computational basis.

        Args:
            shots (int): number of samples to generate.
            qargs (None or list): subsystems to sample measurements for,
                                if None sample measurement of all
                                subsystems (Default: None).

        Returns:
            np.array: list of sampled counts if the order sampled.

        Additional Information:

            This function implements the measurement :meth:`measure` method.

            The seed for random number generator used for sampling can be
            set to a fixed value by using the stats :meth:`seed` method.
        """
        memory = []
        for _ in range(shots):
            # copy the StabilizerState since measure updates it
            stab = copy.deepcopy(self)
            memory.append(stab.measure(qargs)[0])
        return memory

    # -----------------------------------------------------------------------
    # Helper functions for calculating the measurement
    # -----------------------------------------------------------------------
    def _measure_and_update(self, qubit, randbit):
        """ Measure a single qubit and return outcome and post-measure state.

        Note that this function uses the QuantumStates internal random
        number generator for sampling the measurement outcome. The RNG
        seed can be set using the :meth:`seed` method.

        Note that stabilizer state measurements only have three probabilities:
        (p0, p1) = (0.5, 0.5), (1, 0), or (0, 1)
        The random case happens if there is a row anti-commuting with Z[qubit]
        """

        num_qubits = self.data.num_qubits

        # Check if there exists stabilizer anticommuting with Z[qubit]
        # in this case the measurement outcome is random
        z_anticommuting = np.nonzero(self.data.stabilizer.X[:, qubit])[0]

        # Non-deterministic outcome
        if len(z_anticommuting) != 0:
            p_qubit = np.min(np.nonzero(self.data.stabilizer.X[:, qubit]))
            p_qubit += num_qubits
            outcome = randbit

            # Updating the StabilizerState
            for i in range(2 * num_qubits):
                # the last condition is not in the AG paper but we seem to need it
                if (self.data.table.X[i][qubit]) and (i != p_qubit) and \
                        (i != (p_qubit - num_qubits)):
                    self._rowsum_nondeterministic(i, p_qubit)

            self.data.table[p_qubit - num_qubits] = copy.deepcopy(self.data.table[p_qubit])
            self.data.table.X[p_qubit] = np.zeros(num_qubits)
            self.data.table.Z[p_qubit] = np.zeros(num_qubits)
            self.data.table.Z[p_qubit][qubit] = True
            self.data.table.phase[p_qubit] = outcome
            return outcome

        # Deterministic outcome - measuring it will not change the StabilizerState
        aux_pauli = Pauli(num_qubits * 'I')
        for i in range(num_qubits):
            if self.data.table.X[i][qubit]:
                aux_pauli = self._rowsum_deterministic(aux_pauli, i + num_qubits)
        outcome = aux_pauli.phase
        return outcome

    @staticmethod
    def _phase_exponent(x1, z1, x2, z2):
        """ Exponent g of i such that Pauli(x1,z1) * Pauli(x2,z2) = i^g Pauli(x1+x2,z1+z2) """

        phase = (x2 * z1 * (1 + 2 * z2 + 2 * x1) - x1 * z2 * (1 + 2 * z1 + 2 * x2)) % 4
        if phase < 0:
            phase += 4  # now phase in {0, 1, 3}

        if phase == 2:
            raise QiskitError(
                'Invalid rowsum phase exponent in measurement calculation.')
        return phase

    def _rowsum(self, accum_pauli, accum_phase, row_pauli, row_phase):
        """ Aaronson-Gottesman rowsum helper function """

        newr = 2 * row_phase + 2 * accum_phase

        for qubit in range(self.data.num_qubits):
            newr += self._phase_exponent(row_pauli.x[qubit],
                                         row_pauli.z[qubit],
                                         accum_pauli.x[qubit],
                                         accum_pauli.z[qubit])
        newr %= 4
        if (newr != 0) & (newr != 2):
            raise QiskitError(
                'Invalid rowsum in measurement calculation.')

        accum_phase = int((newr == 2))
        accum_pauli.x += row_pauli.x
        accum_pauli.z += row_pauli.z
        return accum_pauli, accum_phase

    def _rowsum_nondeterministic(self, accum, row):
        """ Updating StabilizerState Clifford table in the
        non-deterministic rowsum calculation.
        row and accum are rows in the StabilizerState Clifford table."""

        row_phase = self.data.table.phase[row]
        accum_phase = self.data.table.phase[accum]

        row_pauli = self.data.table.pauli[row]
        accum_pauli = self.data.table.pauli[accum]
        row_pauli = Pauli(row_pauli.to_labels()[0])
        accum_pauli = Pauli(accum_pauli.to_labels()[0])

        accum_pauli, accum_phase = self._rowsum(accum_pauli, accum_phase,
                                                row_pauli, row_phase)

        self.data.table.phase[accum] = accum_phase
        self.data.table.X[accum] = accum_pauli.x
        self.data.table.Z[accum] = accum_pauli.z

    def _rowsum_deterministic(self, aux_pauli, row):
        """ Updating an auxilary Pauli aux_pauli in the
        deterministic rowsum calculation.
        The StabilizerState itself is not updated. """

        row_phase = self.data.table.phase[row]
        accum_phase = aux_pauli.phase

        accum_pauli = aux_pauli
        row_pauli = self.data.table.pauli[row]
        row_pauli = Pauli(row_pauli.to_labels()[0])

        accum_pauli, accum_phase = self._rowsum(accum_pauli, accum_phase,
                                                row_pauli, row_phase)

        aux_pauli = accum_pauli
        aux_pauli.phase = accum_phase
        return aux_pauli

    # -----------------------------------------------------------------------
    # Helper functions for calculating the probabilities
    # -----------------------------------------------------------------------
