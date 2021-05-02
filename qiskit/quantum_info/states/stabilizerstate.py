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
    """StabilizerState class.
    Stabilizer simulator using the convention from reference [1].
    Based on the internal class :class:`~qiskit.quantum_info.Clifford`.

    .. jupyter-execute::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import StabilizerState, Pauli

        # Bell state generation circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        stab = StabilizerState(qc)

        # Print the StabilizerState
        print(stab)

        # Calculate the StabilizerState measurement probabilities dictionary
        print (stab.probabilities_dict())

        # Calculate expectation value of the StabilizerState
        print (stab.expectation_value(Pauli('ZZ')))

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
        return 'StabilizerState({})'.format(self._data.stabilizer)

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
        which equals to 1, since it is always a pure state.

        Returns:
            double: the trace (should equal 1).

        Raises:
            QiskitError: if input is not a StabilizerState.
        """
        if not self.is_valid():
            raise QiskitError(
                'Invalid StabilizerState. Input is not a valid Clifford.')
        return 1.0

    def purity(self):
        """Return the purity of the quantum state,
        which equals to 1, since it is always a pure state.

        Returns:
            double: the purity (should equal 1).

        Raises:
            QiskitError: if input is not a StabilizerState.
        """
        if not self.is_valid():
            raise QiskitError(
                'Invalid StabilizerState. Input is not a valid Clifford.')
        return 1.0

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
            complex: the expectation value (only 0 or 1 or -1).
        """
        num_qubits = self.data.num_qubits
        if qargs is None:
            qubits = range(num_qubits)
        else:
            qubits = qargs

        # Construct Pauli on num_qubits
        pauli = Pauli(num_qubits * 'I')
        phase = 0

        for pos, qubit in enumerate(qubits):
            pauli_pos = (oper.to_label())[len(oper) - 1 - pos]
            if pauli_pos == 'X':
                pauli.x[qubit] = 1
            elif pauli_pos == 'Y':
                pauli.x[qubit] = 1
                pauli.z[qubit] = 1
                phase += 1
            elif pauli_pos == 'Z':
                pauli.z[qubit] = 1
            else:
                pass

        # Check if there is a stabilizer that anti-commutes with an odd number of qubits
        # If so the expectation value is 0
        for p in range(num_qubits):
            stab = self.data.stabilizer
            num_anti = 0
            num_anti += np.count_nonzero(pauli.z & stab.X[p])
            num_anti += np.count_nonzero(pauli.x & stab.Z[p])
            if num_anti % 2 == 1:
                return 0

        # Otherwise pauli is (-1)^a prod_j S_j^b_j for Clifford stabilizers
        # If pauli anti-commutes with D_j then b_j = 1.
        # Multiply pauli by stabilizers with anti-commuting destabilizers
        pauli_z = (pauli.z).copy()  # Make a copy of pauli.z
        for p in range(num_qubits):
            # Check if destabilizer anti-commutes
            destab = self.data.destabilizer
            num_anti = 0
            num_anti += np.count_nonzero(pauli.z & destab.X[p])
            num_anti += np.count_nonzero(pauli.x & destab.Z[p])
            if num_anti % 2 == 0:
                continue

            # If anti-commutes multiply Pauli by stabilizer
            stab = self.data.stabilizer
            phase += 2 * self.data.table.phase[p + num_qubits]
            phase += np.count_nonzero(stab.Z[p] & stab.X[p])
            phase += 2 * np.count_nonzero(pauli_z & stab.X[p])
            pauli_z = pauli_z ^ stab.Z[p]

        if phase % 4 != 0:
            return -1

        return 1

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
        probs_dict = self.probabilities_dict(qargs, decimals)
        if qargs is None:
            qargs = range(self.data.num_qubits)
        probs = np.zeros(2 ** len(qargs))

        for key, value in probs_dict.items():
            place = int(key, 2)
            probs[place] = value

        return probs

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
        if qargs is None:
            qubits = range(self.data.num_qubits)
        else:
            qubits = qargs

        outcome = ['X'] * len(qubits)
        outcome_prob = 1.0
        probs = {}  # probabilities dictionary

        self._get_probablities(qubits, outcome, outcome_prob, probs)

        if decimals is not None:
            for key, value in probs.items():
                probs[key] = round(value, decimals)

        return probs

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
        """Measure a single qubit and return outcome and post-measure state.

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
        """Exponent g of i such that Pauli(x1,z1) * Pauli(x2,z2) = i^g Pauli(x1+x2,z1+z2) """
        # pylint: disable=invalid-name

        phase = (x2 * z1 * (1 + 2 * z2 + 2 * x1) - x1 * z2 * (1 + 2 * z1 + 2 * x2)) % 4
        if phase < 0:
            phase += 4  # now phase in {0, 1, 3}

        if phase == 2:
            raise QiskitError(
                'Invalid rowsum phase exponent in measurement calculation.')
        return phase

    def _rowsum(self, accum_pauli, accum_phase, row_pauli, row_phase):
        """Aaronson-Gottesman rowsum helper function """

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
        accum_pauli.x ^= row_pauli.x
        accum_pauli.z ^= row_pauli.z
        return accum_pauli, accum_phase

    def _rowsum_nondeterministic(self, accum, row):
        """Updating StabilizerState Clifford table in the
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
        """Updating an auxilary Pauli aux_pauli in the
        deterministic rowsum calculation.
        The StabilizerState itself is not updated."""

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
    def _get_probablities(self, qubits, outcome, outcome_prob, probs):
        """Recursive helper function for calculating the probabilities"""

        qubit_for_branching = -1

        for i in range(len(qubits)):
            qubit = qubits[len(qubits) - i - 1]
            if outcome[i] == 'X':
                is_deterministic = not any(self.data.stabilizer.X[:, qubit])
                if is_deterministic:
                    single_qubit_outcome = self._measure_and_update(qubit, 0)
                    if single_qubit_outcome:
                        outcome[i] = '1'
                    else:
                        outcome[i] = '0'
                else:
                    qubit_for_branching = i

        if qubit_for_branching == -1:
            str_outcome = ''.join(outcome)
            probs[str_outcome] = outcome_prob
            return

        for single_qubit_outcome in range(0, 2):
            new_outcome = outcome.copy()
            if single_qubit_outcome:
                new_outcome[qubit_for_branching] = '1'
            else:
                new_outcome[qubit_for_branching] = '0'

            stab_cpy = self.copy()
            stab_cpy._measure_and_update(
                qubits[len(qubits) - qubit_for_branching - 1], single_qubit_outcome)
            stab_cpy._get_probablities(qubits, new_outcome, 0.5 * outcome_prob, probs)
