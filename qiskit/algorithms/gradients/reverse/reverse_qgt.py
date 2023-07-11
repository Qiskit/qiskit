# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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
from qiskit.quantum_info import Statevector
from qiskit.providers import Options
from qiskit.primitives import Estimator

from ..base.base_qgt import BaseQGT
from ..base.qgt_result import QGTResult
from ..utils import DerivativeType

from .split_circuits import split
from .bind import bind
from .derive_circuit import derive_circuit

logger = logging.getLogger(__name__)


class ReverseQGT(BaseQGT):
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
        dummy_estimator = Estimator()  # this method does not need an estimator
        super().__init__(dummy_estimator, phase_fix, derivative_type)

    @property
    def options(self) -> Options:
        """There are no options for the reverse QGT, returns an empty options dict.

        Returns:
            Empty options.
        """
        return Options()

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        **options,
    ) -> QGTResult:
        """Compute the QGT on the given circuits."""
        g_circuits, g_parameter_values, g_parameter_sets = self._preprocess(
            circuits, parameter_values, parameter_sets, self.SUPPORTED_GATES
        )
        results = self._run_unique(g_circuits, g_parameter_values, g_parameter_sets, **options)
        return self._postprocess(results, circuits, parameter_values, parameter_sets)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        **options,  # pylint: disable=unused-argument
    ) -> QGTResult:
        num_qgts = len(circuits)
        qgts = []
        metadata = []

        for k in range(num_qgts):
            values = np.asarray(parameter_values[k])
            circuit = circuits[k]
            parameters = list(parameter_sets[k])

            num_parameters = len(parameters)
            original_parameter_order = [p for p in circuit.parameters if p in parameters]
            metadata.append({"parameters": original_parameter_order})

            unitaries, paramlist = split(circuit, parameters=parameters)

            # initialize the phase fix vector and the hessian part ``metric``
            num_parameters = len(unitaries)
            phase_fixes = np.zeros(num_parameters, dtype=complex)
            metric = np.zeros((num_parameters, num_parameters), dtype=complex)

            # initialize the state variables -- naming convention is the same as the paper
            parameter_binds = dict(zip(circuit.parameters, values))
            bound_unitaries = bind(unitaries, parameter_binds)

            chi = Statevector(bound_unitaries[0])
            psi = chi.copy()
            phi = Statevector.from_int(0, (2,) * circuit.num_qubits)

            # Get the analytic gradient of the first unitary
            # Note: We currently only support gates with a single parameter -- which is reflected
            # in self.SUPPORTED_GATES -- but generally we could also support gates with multiple
            # parameters per gate. This is the reason for the second 0-index.
            deriv = derive_circuit(unitaries[0], paramlist[0][0])
            for _, gate in deriv:
                bind(gate, parameter_binds, inplace=True)

            grad_coeffs = [coeff for coeff, _ in deriv]
            grad_states = [phi.evolve(gate) for _, gate in deriv]

            # compute phase fix (optional) and the hessian part
            if self._phase_fix:
                phase_fixes[0] = _phasefix_term(chi, grad_coeffs, grad_states)

            metric[0, 0] = _l_term(grad_coeffs, grad_states, grad_coeffs, grad_states)

            for j in range(1, num_parameters):
                lam = psi.copy()
                phi = psi.copy()

                # get the analytic gradient d U_j / d p_j and apply it
                deriv = derive_circuit(unitaries[j], paramlist[j][0])

                for _, gate in deriv:
                    bind(gate, parameter_binds, inplace=True)

                # compute |phi> (in general it's a sum of states and coeffs)
                grad_coeffs = [coeff for coeff, _ in deriv]
                grad_states = [phi.evolve(gate) for _, gate in deriv]

                # compute the digaonal element L_{j, j}
                metric[j, j] += _l_term(grad_coeffs, grad_states, grad_coeffs, grad_states)

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

                    metric[i, j] += _l_term(
                        grad_coeffs_mu, grad_states_mu, grad_coeffs, grad_states
                    )

                if self._phase_fix:
                    phase_fixes[j] += _phasefix_term(chi, grad_coeffs, grad_states)

                psi = psi.evolve(bound_unitaries[j])

            # The following code stacks the QGT together and maps the values into the
            # correct original order of parameters

            # map circuit parameter to global index in the circuit
            param_to_circuit = {
                param: index for index, param in enumerate(original_parameter_order)
            }
            # map global index to the local index used in the calculation, the new index can
            # now be accessed by remap[index]
            remap = {
                index: param_to_circuit[_extract_parameter(plist[0])]
                for index, plist in enumerate(paramlist)
            }

            qgt = np.zeros((num_parameters, num_parameters), dtype=complex)
            for i in range(num_parameters):
                iloc = remap[i]
                for j in range(num_parameters):
                    jloc = remap[j]
                    if i <= j:
                        qgt[iloc, jloc] += metric[i, j]
                    else:
                        qgt[iloc, jloc] += np.conj(metric[j, i])

                    qgt[iloc, jloc] -= np.conj(phase_fixes[i]) * phase_fixes[j]

            # append and cast to real/imag if required
            qgts.append(self._to_derivtype(qgt))

        result = QGTResult(qgts, self.derivative_type, metadata, options=None)
        return result

    def _to_derivtype(self, qgt):
        if self.derivative_type == DerivativeType.REAL:
            return np.real(qgt)
        if self.derivative_type == DerivativeType.IMAG:
            return np.imag(qgt)

        return qgt


def _l_term(coeffs_i, states_i, coeffs_j, states_j):
    return sum(
        sum(
            np.conj(coeff_i) * coeff_j * np.conj(state_i.data).dot(state_j.data)
            for coeff_i, state_i in zip(coeffs_i, states_i)
        )
        for coeff_j, state_j in zip(coeffs_j, states_j)
    )


def _phasefix_term(chi, coeffs, states):
    return sum(
        coeff_i * np.conj(chi.data).dot(state_i.data) for coeff_i, state_i in zip(coeffs, states)
    )


def _extract_parameter(expression):
    if isinstance(expression, Parameter):
        return expression

    if len(expression.parameters) > 1:
        raise ValueError("Expression has more than one parameter.")

    return list(expression.parameters)[0]
