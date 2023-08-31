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

from __future__ import annotations
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli, PauliList
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_x
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit import QuantumCircuit, Instruction


class StabilizerState(QuantumState):
    """StabilizerState class.
    Stabilizer simulator using the convention from reference [1].
    Based on the internal class :class:`~qiskit.quantum_info.Clifford`.

    .. code-block::

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

    .. parsed-literal::

        StabilizerState(StabilizerTable: ['+XX', '+ZZ'])
        {'00': 0.5, '11': 0.5}
        1

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    def __init__(
        self,
        data: StabilizerState | Clifford | Pauli | QuantumCircuit | Instruction,
        validate: bool = True,
    ):
        """Initialize a StabilizerState object.

        Args:
            data (StabilizerState or Clifford or Pauli or QuantumCircuit or
                  qiskit.circuit.Instruction):
                Data from which the stabilizer state can be constructed.
            validate (boolean): validate that the stabilizer state data is
                a valid Clifford.
        """

        # Initialize from another StabilizerState
        if isinstance(data, StabilizerState):
            self._data = data._data
        # Initialize from a Pauli
        elif isinstance(data, Pauli):
            self._data = Clifford(data.to_instruction())
        # Initialize from a Clifford, QuantumCircuit or Instruction
        else:
            self._data = Clifford(data, validate)

        # Initialize
        super().__init__(op_shape=OpShape.auto(num_qubits_r=self._data.num_qubits, num_qubits_l=0))

    def __eq__(self, other):
        return (self._data.stab == other._data.stab).all()

    def __repr__(self):
        return f"StabilizerState({self._data.stabilizer})"

    @property
    def clifford(self):
        """Return StabilizerState Clifford data"""
        return self._data

    def is_valid(self, atol=None, rtol=None):
        """Return True if a valid StabilizerState."""
        return self._data.is_unitary()

    def _add(self, other):
        raise NotImplementedError(f"{type(self)} does not support addition")

    def _multiply(self, other):
        raise NotImplementedError(f"{type(self)} does not support scalar multiplication")

    def trace(self) -> float:
        """Return the trace of the stabilizer state as a density matrix,
        which equals to 1, since it is always a pure state.

        Returns:
            float: the trace (should equal 1).

        Raises:
            QiskitError: if input is not a StabilizerState.
        """
        if not self.is_valid():
            raise QiskitError("StabilizerState is not a valid quantum state.")
        return 1.0

    def purity(self) -> float:
        """Return the purity of the quantum state,
        which equals to 1, since it is always a pure state.

        Returns:
            float: the purity (should equal 1).

        Raises:
            QiskitError: if input is not a StabilizerState.
        """
        if not self.is_valid():
            raise QiskitError("StabilizerState is not a valid quantum state.")
        return 1.0

    def to_operator(self) -> Operator:
        """Convert state to matrix operator class"""
        return Clifford(self.clifford).to_operator()

    def conjugate(self):
        """Return the conjugate of the operator."""
        ret = self.copy()
        ret._data = ret._data.conjugate()
        return ret

    def tensor(self, other: StabilizerState) -> StabilizerState:
        """Return the tensor product stabilizer state self ⊗ other.

        Args:
            other (StabilizerState): a stabilizer state object.

        Returns:
            StabilizerState: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a StabilizerState.
        """
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        ret = self.copy()
        ret._data = self.clifford.tensor(other.clifford)
        return ret

    def expand(self, other: StabilizerState) -> StabilizerState:
        """Return the tensor product stabilizer state other ⊗ self.

        Args:
            other (StabilizerState): a stabilizer state object.

        Returns:
            StabilizerState: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other is not a StabilizerState.
        """
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        ret = self.copy()
        ret._data = self.clifford.expand(other.clifford)
        return ret

    def evolve(
        self, other: Clifford | QuantumCircuit | Instruction, qargs: list | None = None
    ) -> StabilizerState:
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
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        ret = self.copy()
        ret._data = self.clifford.compose(other.clifford, qargs=qargs)
        return ret

    def expectation_value(self, oper: Pauli, qargs: None | list = None) -> complex:
        """Compute the expectation value of a Pauli operator.

        Args:
            oper (Pauli): a Pauli operator to evaluate expval.
            qargs (None or list): subsystems to apply the operator on.

        Returns:
            complex: the expectation value (only 0 or 1 or -1 or i or -i).

        Raises:
            QiskitError: if oper is not a Pauli operator.
        """
        if not isinstance(oper, Pauli):
            raise QiskitError("Operator for expectation value is not a Pauli operator.")

        num_qubits = self.clifford.num_qubits
        if qargs is None:
            qubits = range(num_qubits)
        else:
            qubits = qargs

        # Construct Pauli on num_qubits
        pauli = Pauli(num_qubits * "I")
        phase = 0
        pauli_phase = (-1j) ** oper.phase if oper.phase else 1

        for pos, qubit in enumerate(qubits):
            pauli.x[qubit] = oper.x[pos]
            pauli.z[qubit] = oper.z[pos]
            phase += pauli.x[qubit] & pauli.z[qubit]

        # Check if there is a stabilizer that anti-commutes with an odd number of qubits
        # If so the expectation value is 0
        for p in range(num_qubits):
            num_anti = 0
            num_anti += np.count_nonzero(pauli.z & self.clifford.stab_x[p])
            num_anti += np.count_nonzero(pauli.x & self.clifford.stab_z[p])
            if num_anti % 2 == 1:
                return 0

        # Otherwise pauli is (-1)^a prod_j S_j^b_j for Clifford stabilizers
        # If pauli anti-commutes with D_j then b_j = 1.
        # Multiply pauli by stabilizers with anti-commuting destabilizers
        pauli_z = (pauli.z).copy()  # Make a copy of pauli.z
        for p in range(num_qubits):
            # Check if destabilizer anti-commutes
            num_anti = 0
            num_anti += np.count_nonzero(pauli.z & self.clifford.destab_x[p])
            num_anti += np.count_nonzero(pauli.x & self.clifford.destab_z[p])
            if num_anti % 2 == 0:
                continue

            # If anti-commutes multiply Pauli by stabilizer
            phase += 2 * self.clifford.stab_phase[p]
            phase += np.count_nonzero(self.clifford.stab_z[p] & self.clifford.stab_x[p])
            phase += 2 * np.count_nonzero(pauli_z & self.clifford.stab_x[p])
            pauli_z = pauli_z ^ self.clifford.stab_z[p]

        # For valid stabilizers, `phase` can only be 0 (= 1) or 2 (= -1) at this point.
        if phase % 4 != 0:
            return -pauli_phase

        return pauli_phase

    def equiv(self, other: StabilizerState) -> bool:
        """Return True if the two generating sets generate the same stabilizer group.

        Args:
            other (StabilizerState): another StabilizerState.

        Returns:
            bool: True if other has a generating set that generates the same StabilizerState.
        """
        if not isinstance(other, StabilizerState):
            try:
                other = StabilizerState(other)
            except QiskitError:
                return False

        num_qubits = self.num_qubits
        if other.num_qubits != num_qubits:
            return False

        pauli_orig = PauliList.from_symplectic(
            self._data.stab_z, self._data.stab_x, 2 * self._data.stab_phase
        )
        pauli_other = PauliList.from_symplectic(
            other._data.stab_z, other._data.stab_x, 2 * other._data.stab_phase
        )

        #  Check that each stabilizer from the original set commutes with each stabilizer
        #  from the other set
        if not np.all([pauli.commutes(pauli_other) for pauli in pauli_orig]):
            return False

        # Compute the expected value of each stabilizer from the original set on the stabilizer state
        # determined by the other set. The two stabilizer states coincide if and only if the
        # expected value is +1 for each stabilizer
        for i in range(num_qubits):
            exp_val = self.expectation_value(pauli_other[i])
            if exp_val != 1:
                return False

        return True

    def probabilities(self, qargs: None | list = None, decimals: None | int = None) -> np.ndarray:
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
            qargs = range(self.clifford.num_qubits)
        probs = np.zeros(2 ** len(qargs))

        for key, value in probs_dict.items():
            place = int(key, 2)
            probs[place] = value

        return probs

    def probabilities_dict(self, qargs: None | list = None, decimals: None | int = None) -> dict:
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
            qubits = range(self.clifford.num_qubits)
        else:
            qubits = qargs

        outcome = ["X"] * len(qubits)
        outcome_prob = 1.0
        probs = {}  # probabilities dictionary

        self._get_probablities(qubits, outcome, outcome_prob, probs)

        if decimals is not None:
            for key, value in probs.items():
                probs[key] = round(value, decimals)

        return probs

    def reset(self, qargs: list | None = None) -> StabilizerState:
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
            return StabilizerState(Clifford(np.eye(2 * self.clifford.num_qubits)))

        randbits = self._rng.integers(2, size=len(qargs))
        ret = self.copy()

        for bit, qubit in enumerate(qargs):
            # Apply measurement and get classical outcome
            outcome = ret._measure_and_update(qubit, randbits[bit])

            # Use the outcome to apply X gate to any qubits left in the
            # |1> state after measure, then discard outcome.
            if outcome == 1:
                _append_x(ret.clifford, qubit)

        return ret

    def measure(self, qargs: list | None = None) -> tuple:
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
            qargs = range(self.clifford.num_qubits)

        randbits = self._rng.integers(2, size=len(qargs))
        ret = self.copy()

        outcome = ""
        for bit, qubit in enumerate(qargs):
            outcome = str(ret._measure_and_update(qubit, randbits[bit])) + outcome

        return outcome, ret

    def sample_memory(self, shots: int, qargs: None | list = None) -> np.ndarray:
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
            stab = self.copy()
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

        num_qubits = self.clifford.num_qubits
        clifford = self.clifford
        stab_x = self.clifford.stab_x

        # Check if there exists stabilizer anticommuting with Z[qubit]
        # in this case the measurement outcome is random
        z_anticommuting = np.any(stab_x[:, qubit])

        if z_anticommuting == 0:
            # Deterministic outcome - measuring it will not change the StabilizerState
            aux_pauli = Pauli(num_qubits * "I")
            for i in range(num_qubits):
                if clifford.x[i][qubit]:
                    aux_pauli = self._rowsum_deterministic(clifford, aux_pauli, i + num_qubits)
            outcome = aux_pauli.phase
            return outcome

        else:
            # Non-deterministic outcome
            outcome = randbit
            p_qubit = np.min(np.nonzero(stab_x[:, qubit]))
            p_qubit += num_qubits

            # Updating the StabilizerState
            for i in range(2 * num_qubits):
                # the last condition is not in the AG paper but we seem to need it
                if (clifford.x[i][qubit]) and (i != p_qubit) and (i != (p_qubit - num_qubits)):
                    self._rowsum_nondeterministic(clifford, i, p_qubit)

            clifford.destab[p_qubit - num_qubits] = clifford.stab[p_qubit - num_qubits].copy()
            clifford.x[p_qubit] = np.zeros(num_qubits)
            clifford.z[p_qubit] = np.zeros(num_qubits)
            clifford.z[p_qubit][qubit] = True
            clifford.phase[p_qubit] = outcome
            return outcome

    @staticmethod
    def _phase_exponent(x1, z1, x2, z2):
        """Exponent g of i such that Pauli(x1,z1) * Pauli(x2,z2) = i^g Pauli(x1+x2,z1+z2)"""
        # pylint: disable=invalid-name

        phase = (x2 * z1 * (1 + 2 * z2 + 2 * x1) - x1 * z2 * (1 + 2 * z1 + 2 * x2)) % 4
        if phase < 0:
            phase += 4  # now phase in {0, 1, 3}

        if phase == 2:
            raise QiskitError("Invalid rowsum phase exponent in measurement calculation.")
        return phase

    @staticmethod
    def _rowsum(accum_pauli, accum_phase, row_pauli, row_phase):
        """Aaronson-Gottesman rowsum helper function"""

        newr = 2 * row_phase + 2 * accum_phase

        for qubit in range(row_pauli.num_qubits):
            newr += StabilizerState._phase_exponent(
                row_pauli.x[qubit], row_pauli.z[qubit], accum_pauli.x[qubit], accum_pauli.z[qubit]
            )
        newr %= 4
        if (newr != 0) & (newr != 2):
            raise QiskitError("Invalid rowsum in measurement calculation.")

        accum_phase = int(newr == 2)
        accum_pauli.x ^= row_pauli.x
        accum_pauli.z ^= row_pauli.z
        return accum_pauli, accum_phase

    @staticmethod
    def _rowsum_nondeterministic(clifford, accum, row):
        """Updating StabilizerState Clifford in the
        non-deterministic rowsum calculation.
        row and accum are rows in the StabilizerState Clifford."""

        row_phase = clifford.phase[row]
        accum_phase = clifford.phase[accum]

        z = clifford.z
        x = clifford.x
        row_pauli = Pauli((z[row], x[row]))
        accum_pauli = Pauli((z[accum], x[accum]))

        accum_pauli, accum_phase = StabilizerState._rowsum(
            accum_pauli, accum_phase, row_pauli, row_phase
        )

        clifford.phase[accum] = accum_phase
        x[accum] = accum_pauli.x
        z[accum] = accum_pauli.z

    @staticmethod
    def _rowsum_deterministic(clifford, aux_pauli, row):
        """Updating an auxilary Pauli aux_pauli in the
        deterministic rowsum calculation.
        The StabilizerState itself is not updated."""

        row_phase = clifford.phase[row]
        accum_phase = aux_pauli.phase

        accum_pauli = aux_pauli
        row_pauli = Pauli((clifford.z[row], clifford.x[row]))

        accum_pauli, accum_phase = StabilizerState._rowsum(
            accum_pauli, accum_phase, row_pauli, row_phase
        )

        aux_pauli = accum_pauli
        aux_pauli.phase = accum_phase
        return aux_pauli

    # -----------------------------------------------------------------------
    # Helper functions for calculating the probabilities
    # -----------------------------------------------------------------------
    def _get_probablities(self, qubits, outcome, outcome_prob, probs):
        """Recursive helper function for calculating the probabilities"""

        qubit_for_branching = -1
        ret = self.copy()

        for i in range(len(qubits)):
            qubit = qubits[len(qubits) - i - 1]
            if outcome[i] == "X":
                is_deterministic = not any(ret.clifford.stab_x[:, qubit])
                if is_deterministic:
                    single_qubit_outcome = ret._measure_and_update(qubit, 0)
                    if single_qubit_outcome:
                        outcome[i] = "1"
                    else:
                        outcome[i] = "0"
                else:
                    qubit_for_branching = i

        if qubit_for_branching == -1:
            str_outcome = "".join(outcome)
            probs[str_outcome] = outcome_prob
            return

        for single_qubit_outcome in range(0, 2):
            new_outcome = outcome.copy()
            if single_qubit_outcome:
                new_outcome[qubit_for_branching] = "1"
            else:
                new_outcome[qubit_for_branching] = "0"

            stab_cpy = ret.copy()
            stab_cpy._measure_and_update(
                qubits[len(qubits) - qubit_for_branching - 1], single_qubit_outcome
            )
            stab_cpy._get_probablities(qubits, new_outcome, 0.5 * outcome_prob, probs)
