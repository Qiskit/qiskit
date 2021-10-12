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

"""Phase estimation for the spectrum of a Hamiltonian"""

from typing import Optional, Union
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.opflow import (
    EvolutionBase,
    PauliTrotterEvolution,
    OperatorBase,
    SummedOp,
    PauliOp,
    MatrixOp,
    PauliSumOp,
    StateFn,
)
from qiskit.providers import BaseBackend
from .phase_estimation import PhaseEstimation
from .hamiltonian_phase_estimation_result import HamiltonianPhaseEstimationResult
from .phase_estimation_scale import PhaseEstimationScale


class HamiltonianPhaseEstimation:
    r"""Run the Quantum Phase Estimation algorithm to find the eigenvalues of a Hermitian operator.

    This class is nearly the same as :class:`~qiskit.algorithms.PhaseEstimation`, differing only
    in that the input in that class is a unitary operator, whereas here the input is a Hermitian
    operator from which a unitary will be obtained by scaling and exponentiating. The scaling is
    performed in order to prevent the phases from wrapping around :math:`2\pi`.
    The problem of estimating eigenvalues :math:`\lambda_j` of the Hermitian operator
    :math:`H` is solved by running a circuit representing

    .. math::

        \exp(i b H) |\psi\rangle = \sum_j \exp(i b \lambda_j) c_j |\lambda_j\rangle,

    where the input state is

    .. math::

        |\psi\rangle = \sum_j c_j |\lambda_j\rangle,

    and :math:`\lambda_j` are the eigenvalues of :math:`H`.

    Here, :math:`b` is a scaling factor sufficiently large to map positive :math:`\lambda` to
    :math:`[0,\pi)` and negative :math:`\lambda` to :math:`[\pi,2\pi)`. Each time the circuit is
    run, one measures a phase corresponding to :math:`lambda_j` with probability :math:`|c_j|^2`.

    If :math:`H` is a Pauli sum, the bound :math:`b` is computed from the sum of the absolute
    values of the coefficients of the terms. There is no way to reliably recover eigenvalues
    from phases very near the endpoints of these intervals. Because of this you should be aware
    that for degenerate cases, such as :math:`H=Z`, the eigenvalues :math:`\pm 1` will be
    mapped to the same phase, :math:`\pi`, and so cannot be distinguished. In this case, you need
    to specify a larger bound as an argument to the method ``estimate``.

    This class uses and works together with :class:`~qiskit.algorithms.PhaseEstimationScale` to
    manage scaling the Hamiltonian and the phases that are obtained by the QPE algorithm. This
    includes setting, or computing, a bound on the eigenvalues of the operator, using this
    bound to obtain a scale factor, scaling the operator, and shifting and scaling the measured
    phases to recover the eigenvalues.

    Note that, although we speak of "evolving" the state according the the Hamiltonian, in the
    present algorithm, we are not actually considering time evolution. Rather, the role of time is
    played by the scaling factor, which is chosen to best extract the eigenvalues of the
    Hamiltonian.

    A few of the ideas in the algorithm may be found in Ref. [1].

    **Reference:**

    [1]: Quantum phase estimation of multiple eigenvalues for small-scale (noisy) experiments
         T.E. O'Brien, B. Tarasinski, B.M. Terhal
         `arXiv:1809.09697 <https://arxiv.org/abs/1809.09697>`_

    """

    def __init__(
        self,
        num_evaluation_qubits: int,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None,
    ) -> None:
        """
        Args:
            num_evaluation_qubits: The number of qubits used in estimating the phase. The phase will
                be estimated as a binary string with this many bits.
            quantum_instance: The quantum instance on which the circuit will be run.
        """
        self._phase_estimation = PhaseEstimation(
            num_evaluation_qubits=num_evaluation_qubits, quantum_instance=quantum_instance
        )

    def _get_scale(self, hamiltonian, bound=None) -> None:
        if bound is None:
            return PhaseEstimationScale.from_pauli_sum(hamiltonian)

        return PhaseEstimationScale(bound)

    def _get_unitary(self, hamiltonian, pe_scale, evolution) -> QuantumCircuit:
        """Evolve the Hamiltonian to obtain a unitary.

        Apply the scaling to the Hamiltonian that has been computed from an eigenvalue bound
        and compute the unitary by applying the evolution object.
        """
        # scale so that phase does not wrap.
        scaled_hamiltonian = -pe_scale.scale * hamiltonian
        unitary = evolution.convert(scaled_hamiltonian.exp_i())
        if not isinstance(unitary, QuantumCircuit):
            unitary_circuit = unitary.to_circuit()
        else:
            unitary_circuit = unitary

        # Decomposing twice allows some 1Q Hamiltonians to give correct results
        # when using MatrixEvolution(), that otherwise would give incorrect results.
        # It does not break any others that we tested.
        return unitary_circuit.decompose().decompose()

    # pylint: disable=arguments-differ
    def estimate(
        self,
        hamiltonian: OperatorBase,
        state_preparation: Optional[StateFn] = None,
        evolution: Optional[EvolutionBase] = None,
        bound: Optional[float] = None,
    ) -> HamiltonianPhaseEstimationResult:
        """Run the Hamiltonian phase estimation algorithm.

        Args:
            hamiltonian: A Hermitian operator.
            state_preparation: The ``StateFn`` to be prepared, whose eigenphase will be
                measured. If this parameter is omitted, no preparation circuit will be run and
                input state will be the all-zero state in the computational basis.
            evolution: An evolution converter that generates a unitary from ``hamiltonian``. If
                ``None``, then the default ``PauliTrotterEvolution`` is used.
            bound: An upper bound on the absolute value of the eigenvalues of
                ``hamiltonian``. If omitted, then ``hamiltonian`` must be a Pauli sum, or a
                ``PauliOp``, in which case a bound will be computed. If ``hamiltonian``
                is a ``MatrixOp``, then ``bound`` may not be ``None``. The tighter the bound,
                the higher the resolution of computed phases.

        Returns:
            HamiltonianPhaseEstimationResult instance containing the result of the estimation
            and diagnostic information.

        Raises:
            ValueError: If ``bound`` is ``None`` and ``hamiltonian`` is not a Pauli sum, i.e. a
                ``PauliSumOp`` or a ``SummedOp`` whose terms are of type ``PauliOp``.
            TypeError: If ``evolution`` is not of type ``EvolutionBase``.
        """
        if evolution is None:
            evolution = PauliTrotterEvolution()
        elif not isinstance(evolution, EvolutionBase):
            raise TypeError(f"Expecting type EvolutionBase, got {type(evolution)}")

        if isinstance(hamiltonian, PauliSumOp):
            hamiltonian = hamiltonian.to_pauli_op()
        elif isinstance(hamiltonian, PauliOp):
            hamiltonian = SummedOp([hamiltonian])

        if isinstance(hamiltonian, SummedOp):
            # remove identitiy terms
            # The term propto the identity is removed from hamiltonian.
            # This is done for three reasons:
            # 1. Work around an unknown bug that otherwise causes the energies to be wrong in some
            #    cases.
            # 2. Allow working with a simpler Hamiltonian, one with fewer terms.
            # 3. Tighten the bound on the eigenvalues so that the spectrum is better resolved, i.e.
            #   occupies more of the range of values representable by the qubit register.
            # The coefficient of this term will be added to the eigenvalues.
            id_coefficient, hamiltonian_no_id = _remove_identity(hamiltonian)

            # get the rescaling object
            pe_scale = self._get_scale(hamiltonian_no_id, bound)

            # get the unitary
            unitary = self._get_unitary(hamiltonian_no_id, pe_scale, evolution)

        elif isinstance(hamiltonian, MatrixOp):
            if bound is None:
                raise ValueError("bound must be specified if Hermitian operator is MatrixOp")

            # Do not subtract an identity term from the matrix, so do not compensate.
            id_coefficient = 0.0
            pe_scale = self._get_scale(hamiltonian, bound)
            unitary = self._get_unitary(hamiltonian, pe_scale, evolution)
        else:
            raise TypeError(f"Hermitian operator of type {type(hamiltonian)} not supported.")

        if state_preparation is not None:
            state_preparation = state_preparation.to_circuit_op().to_circuit()
        # run phase estimation
        phase_estimation_result = self._phase_estimation.estimate(
            unitary=unitary, state_preparation=state_preparation
        )

        return HamiltonianPhaseEstimationResult(
            phase_estimation_result=phase_estimation_result,
            id_coefficient=id_coefficient,
            phase_estimation_scale=pe_scale,
        )


def _remove_identity(pauli_sum):
    """Remove any identity operators from `pauli_sum`. Return
    the sum of the coefficients of the identities and the new operator.
    """
    idcoeff = 0.0
    ops = []
    for op in pauli_sum:
        p = op.primitive
        if p.x.any() or p.z.any():
            ops.append(op)
        else:
            idcoeff += op.coeff

    return idcoeff, SummedOp(ops)
