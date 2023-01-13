# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QGT with the classically efficient reverse mode."""

from __future__ import annotations
from collections.abc import Sequence
import logging

import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Statevector

from ..base_qfi import BaseQFI
from ..qfi_result import QFIResult
from ..utils import DerivativeType

logger = logging.getLogger(__name__)


from surfer.tools.gradient_lookup import extract_single_parameter

from .split_circuits import split
from .bind import bind
from .derive_circuit import derive_circuit


class ReverseQGT(BaseQFI):
    """QGT calculation with the classically efficient reverse mode.

    .. note::

        This QGT implementation is based on statevector manipulations and scales exponentially
        with the number of qubits. However, for small system sizes it can be very fast
        compared to circuit-based gradients.

    This class implements the calculation of the QGT as described in [1].
    By keeping track of three statevectors and iteratively sweeping through each parameterized
    gate, this method scales only quadratically with the number of parameters.

    **References:**

        [1]: Jones, T. "Efficient classical calculation of the Quantum Natural Gradient" (2020).
             `arXiv:2011.02991 <https://arxiv.org/abs/2011.02991>`_.

    """

    SUPPORTED_GATES = ["rx", "ry", "rz", "cp", "crx", "cry", "crz"]

    def __init__(
        self, phase_fix: bool = True, derivative_type: DerivativeType = DerivativeType.COMPLEX
    ):
        """
        Args:
            phase_fix: Whether or not to include the phase fix.
            derivative_type: Determines whether the complex QGT or only the real or imaginary
                parts are calculated.
        """
        super().__init__()
        self.phase_fix = phase_fix
        self._derivative_type = derivative_type

    # TODO this should be in the base class of QGT
    @property
    def derivative_type(self) -> DerivativeType:
        """The derivative type."""
        return self._derivative_type

    @derivative_type.setter
    def derivative_type(self, derivative_type: DerivativeType) -> None:
        """Set the derivative type."""
        self._derivative_type = derivative_type

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> QFIResult:
        # cast to array, if values are a list
        num_qgts = len(circuits)
        qgts = []
        metadata = []

        for k in range(num_qgts):

            # TODO unrolling done by QGT
            # circuit = self.unroller(circuit)

            values = np.asarray(parameter_values[k])
            circuit = circuits[k]

            # TODO parameters is None captured by QGT
            if parameters[k] is None:
                parameters[k] = circuit.parameters

            num_parameters = len(parameters[k])
            original_parameter_order = [p for p in circuit.parameters if p in parameters]

            metadata.append(
                {
                    "parameters": original_parameter_order,
                    "derivative_type": self.derivative_type,
                }
            )

            unitaries, paramlist = split(circuit, parameters=parameters)

            # initialize the phase fix vector and the hessian part ``lis``
            num_parameters = len(unitaries)
            phase_fixes = np.zeros(num_parameters, dtype=complex)
            lis = np.zeros((num_parameters, num_parameters), dtype=complex)

            # initialize the state variables -- naming convention is the same as the paper
            parameter_binds = dict(zip(circuit.parameters, values))
            bound_unitaries = bind(unitaries, parameter_binds)

            chi = Statevector(bound_unitaries[0])
            psi = chi.copy()
            phi = Statevector.from_int(0, (2,) * circuit.num_qubits)

            # get the analytic gradient of the first unitary
            deriv = derive_circuit(unitaries[0], paramlist[0][0])
            for _, gate in deriv:
                bind(gate, parameter_binds, inplace=True)

            grad_coeffs = [coeff for coeff, _ in deriv]
            grad_states = [phi.evolve(gate) for _, gate in deriv]

            # compute phase fix (optional) and the hessian part
            if self.phase_fix:
                phase_fixes[0] = _phasefix_term(chi, grad_coeffs, grad_states)

            lis[0, 0] = _l_term(grad_coeffs, grad_states, grad_coeffs, grad_states)

            for j in range(1, num_parameters):
                lam = psi.copy()
                phi = psi.copy()

                # get the analytic gradient d U_j / d p_j and apply it
                deriv = derive_circuit(unitaries[j], paramlist[j][0])

                for _, gate in deriv:
                    bind(gate, parameter_binds, inplace=True)

                # compute |phi> (in general it's a sum of states and coeffs)
                grad_coeffs = [coeff for coeff, _ in deriv]
                grad_states = [phi.evolve(gate.decompose()) for _, gate in deriv]

                # compute the digaonal element L_{j, j}
                lis[j, j] += _l_term(grad_coeffs, grad_states, grad_coeffs, grad_states)

                # compute the off diagonal elements L_{i, j}
                for i in reversed(range(j)):
                    # apply U_{i + 1}_dg
                    unitary_ip_inv = bound_unitaries[i + 1].inverse()
                    grad_states = [state.evolve(unitary_ip_inv) for state in grad_states]

                    lam = lam.evolve(bound_unitaries[i].inverse())

                    # get the gradient d U_i / d p_i and apply it
                    deriv = derive_circuit(unitaries[i], paramlist[i][0])
                    for _, gate in deriv:
                        bind(gate, parameter_binds, inplace=True)

                    grad_coeffs_mu = [coeff for coeff, _ in deriv]
                    grad_states_mu = [lam.evolve(gate) for _, gate in deriv]

                    lis[i, j] += _l_term(grad_coeffs_mu, grad_states_mu, grad_coeffs, grad_states)

                if self.phase_fix:
                    phase_fixes[j] += _phasefix_term(chi, grad_coeffs, grad_states)

                psi = psi.evolve(bound_unitaries[j])

            # stack quantum geometric tensor together and take into account the original
            # order of parameters
            param_to_circuit = {
                param: index for index, param in enumerate(original_parameter_order)
            }
            remap = {
                index: param_to_circuit[extract_single_parameter(plist[0])]
                for index, plist in enumerate(paramlist)
            }

            qgt = np.zeros((num_parameters, num_parameters), dtype=complex)
            for i in range(num_parameters):
                iloc = remap[i]
                for j in range(num_parameters):
                    jloc = remap[j]
                    if i <= j:
                        qgt[iloc, jloc] += lis[i, j]
                    else:
                        qgt[iloc, jloc] += np.conj(lis[j, i])

                    qgt[iloc, jloc] -= np.conj(phase_fixes[i]) * phase_fixes[j]

            if self.derivative_type == DerivativeType.REAL:
                qgt = np.real(qgt)
            elif self.derivative_type == DerivativeType.IMAG:
                qgt = np.imag(qgt)

            qgts.append(self._to_derivtype(qgt))

        result = QFIResult(qgts, metadata, options=None)
        return result

    def _to_derivtype(self, qgt):
        # TODO remove factor 4 once the QGT interface is there
        if self.derivative_type == DerivativeType.REAL:
            return 4 * np.real(qgt)
        if self.derivative_type == DerivativeType.IMAG:
            return 4 * np.imag(qgt)

        return 4 * gradient


def _l_term(coeffs_i, states_i, coeffs_j, states_j):
    return sum(
        sum(
            np.conj(c_i) * c_j * np.conj(state_i.data).dot(state_j.data)
            for c_i, state_i in zip(coeffs_i, states_i)
        )
        for c_j, state_j in zip(coeffs_j, states_j)
    )


def _phasefix_term(chi, coeffs, states):
    return sum(c_i * np.conj(chi.data).dot(state_i.data) for c_i, state_i in zip(coeffs, states))


def _evolve_by_operator(operator, state):
    """Evolve the Statevector state by operator."""

    # try casting to sparse matrix and use sparse matrix-vector multiplication, which is
    # a lot faster than using Statevector.evolve
    if isinstance(operator, PauliSumOp):
        operator = operator.primitive * operator.coeff

    try:
        spmatrix = operator.to_matrix(sparse=True)
        evolved = spmatrix @ state.data
        return Statevector(evolved)
    except AttributeError:
        logger.info("Operator is not castable to a sparse matrix, using Statevector.evolve.")

    return state.evolve(operator)
