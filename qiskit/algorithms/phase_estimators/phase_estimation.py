# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""The Quantum Phase Estimation Algorithm."""


from typing import Optional, Union
import numpy
from qiskit.circuit import QuantumCircuit
import qiskit
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance
from qiskit.result import Result
from .phase_estimation_result import PhaseEstimationResult, _sort_phases
from .phase_estimator import PhaseEstimator


class PhaseEstimation(PhaseEstimator):
    """Run the Quantum Phase Estimation (QPE) algorithm.

    This runs a version of QPE with a multi-qubit register for reading the phase [1]. The main
    inputs are the number of qubits in the phase-reading register, a state preparation circuit to
    prepare an input state, and either
    1) A unitary that will act on the the input state, or
    2) A quantum-phase-estimation circuit in which the unitary is already embedded.
    In case 1), an instance of `qiskit.circuit.PhaseEstimation`, a QPE circuit, containing the input
    unitary will be constructed. After construction, the QPE circuit is run on a backend via the
    `run` method, and the frequencies or counts of the phases represented by bitstrings are
    recorded. The results are returned as an instance of
    :class:`~qiskit.algorithms.phase_estimator_result.PhaseEstimationResult`.

    If the input state is an eigenstate of the unitary, then in the ideal case, all probability is
    concentrated on the bitstring corresponding to the eigenvalue of the input state. If the input
    state is a superposition of eigenstates, then each bitstring is measured with a probability
    corresponding to its weight in the superposition. In addition, if the phase is not representable
    exactly by the phase-reading register, the probability will be spread across bitstrings, with an
    amplitude that decreases with distance from the bitstring most closely approximating the phase.

    **Reference:**

    [1]: Michael A. Nielsen and Isaac L. Chuang. 2011.
         Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.).
         Cambridge University Press, New York, NY, USA.

    """

    def __init__(self,
                 num_evaluation_qubits: int,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:

        """Args:
            num_evaluation_qubits: The number of qubits used in estimating the phase. The phase will
                                   be estimated as a binary string with this many bits.
            quantum_instance: The quantum instance on which the circuit will be run.

           Raises:
               ValueError: unless only one of `unitary` and `pe_circuit` is `None`.
                           `num_unitary_qubits` disagrees with size of `unitary`.
        """

        self._measurements_added = False
        if num_evaluation_qubits is not None:
            self._num_evaluation_qubits = num_evaluation_qubits

        self._quantum_instance = quantum_instance

    def _add_classical_register(self) -> None:
        """Add measurement instructions only if we are not using a state vector simulator."""
        if not self._quantum_instance.is_statevector and not self._measurements_added:
            # Measure only the evaluation qubits.
            regname = 'meas'
            circ = self._pe_circuit
            creg = ClassicalRegister(self._num_evaluation_qubits, regname)
            circ.add_register(creg)
            circ.barrier()
            circ.measure(range(self._num_evaluation_qubits), range(self._num_evaluation_qubits))
            self._measurements_added = True

    def construct_circuit(self) -> QuantumCircuit:
        """Return the circuit to be executed to estimate phases.

        This circuit includes as sub-circuits the core phase estimation circuit,
        with the addition of the state-preparation circuit and possibly measurement instructions.
        """
        self._add_classical_register()
        return self._pe_circuit

    def _compute_phases(self, circuit_result: Result) -> Union[numpy.ndarray,
                                                               qiskit.result.Counts]:
        """Compute frequencies/counts of phases from the result of running the QPE circuit.

        How the frequencies are computed depends on whether the backend computes amplitude or
        samples outcomes.

        1) If the backend is a statevector simulator, then the reduced density matrix of the
        phase-reading register is computed from the combined phase-reading- and input-state
        registers. The elements of the diagonal :math:`(i, i)` give the probability to measure the
        each of the states `i`. The index `i` expressed as a binary integer with the LSB rightmost
        gives the state of the phase-reading register with the LSB leftmost when interpreted as a
        phase. In order to maintain the compact representation, the phases are maintained as decimal
        integers.  They may be converted to other forms via the results object,
        `PhaseEstimationResult` or `HamiltonianPhaseEstimationResult`.

         2) If the backend samples bitstrings, then the counts are first retrieved as a dict.  The
        binary strings (the keys) are then reversed so that the LSB is rightmost and the counts are
        converted to frequencies. Then the keys are sorted according to increasing phase, so that
        they can be easily understood when displaying or plotting a histogram.

        Args:
            circuit_result: the result object returned by the backend that ran the QPE circuit.

        Returns:
               Either a dict or numpy.ndarray representing the frequencies of the phases.

        """
        if self._quantum_instance.is_statevector:
            state_vec = circuit_result.get_statevector()
            evaluation_density_matrix = qiskit.quantum_info.partial_trace(
                state_vec,
                range(self._num_evaluation_qubits,
                      self._num_evaluation_qubits + self._num_unitary_qubits)
            )
            phases = evaluation_density_matrix.probabilities()
        else:
            # return counts with keys sorted numerically
            num_shots = circuit_result.results[0].shots
            counts = circuit_result.get_counts()
            phases = {k[::-1]: counts[k] / num_shots for k in counts.keys()}
            phases = _sort_phases(phases)
            phases = qiskit.result.Counts(phases, memory_slots=counts.memory_slots,
                                          creg_sizes=counts.creg_sizes)

        return phases

    def estimate(self,
                 num_evaluation_qubits: Optional[int] = None,
                 unitary: Optional[QuantumCircuit] = None,
                 state_preparation: Optional[QuantumCircuit] = None,
                 pe_circuit: Optional[QuantumCircuit] = None,
                 num_unitary_qubits: Optional[int] = None) -> PhaseEstimationResult:
        """Run the circuit and return and return `PhaseEstimationResult`.

           Args:
            num_evaluation_qubits: The number of qubits used in estimating the phase. The phase will
                                   be estimated as a binary string with this many bits.
            unitary: The circuit representing the unitary operator whose eigenvalues (via phase)
                     will be measured. Exactly one of `pe_circuit` and `unitary` must be passed.
            state_preparation: The circuit that prepares the state whose eigenphase will be
                                 measured.  If this parameter is omitted, no preparation circuit
                                 will be run and input state will be the all-zero state in the
                                 computational basis.
            pe_circuit: The phase estimation circuit.
            num_unitary_qubits: Must agree with the number of qubits in the unitary in `pe_circuit`
                                if `pe_circuit` is passed. This parameter will be set from `unitary`
                                if `unitary` is passed.

        Returns:
               An instance of qiskit.algorithms.phase_estimator_result.PhaseEstimationResult.
        """
        super().estimate(num_evaluation_qubits, unitary, state_preparation,
                         pe_circuit, num_unitary_qubits)
        if hasattr(self._quantum_instance, 'execute'):
            circuit_result = self._quantum_instance.execute(self.construct_circuit())
        else:
            circuit_result = self._quantum_instance.run(self.construct_circuit())
        phases = self._compute_phases(circuit_result)
        return self._result(circuit_result, phases)

    def _result(self, circuit_result, phases) -> PhaseEstimationResult:
        return PhaseEstimationResult(self._num_evaluation_qubits, circuit_result=circuit_result,
                                     phases=phases)
