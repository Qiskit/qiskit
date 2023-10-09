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

"""Unitary overlap circuit."""

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.exceptions import CircuitError


class UnitaryOverlap(QuantumCircuit):
    r"""Circuit that returns the overlap between two unitaries :math:`V^{\dag} U`.

    The input quantum circuits must represent unitary operations, since they must be invertible.
    If the inputs will have parameters, they are replaced by :class:`.ParameterVector`\s with
    names `"a"` (for circuit ``U``) and `"b"` (for circuit ``V``) in the output circut.

    This circuit is usually employed in computing the fidelity::

        .. math::

            \left|\langle 0| V^{\dag} U|0\rangle\right|^{2}

    by computing the probability of being in the all-zeros bit-string, or equivalently,
    the expectation value of projector :math:`|0\rangle\langle 0|`.

    Example::

        import numpy as np
        from qiskit.circuit.library import EfficientSU2, UnitaryOverlap
        from qiskit.primitives import Sampler

        # get two circuit to prepare states of which we comput the overlap
        circuit = EfficientSU2(2, reps=1)
        U = circuit.assign_parameters(np.random.random(circuit.num_parameters))
        V = circuit.assign_parameters(np.random.random(circuit.num_parameters))

        # create the overlap circuit
        overlap = UnitaryOverap(U, V)

        # sample from the overlap
        sampler = Sampler(options={"shots": 100})
        result = sampler.run(overlap).result()

        # the fidelity is the probability to measure 0
        fidelity = result.quasi_dists[0].get(0, 0)

    """

    def __init__(self, U: QuantumCircuit, V: QuantumCircuit):
        """
        Args:
            U: Unitary acting on the ket vector.
            V: Unitary whose inverse operates on the bra vector.

        Raises:
            CircuitError: Number of qubits in ``U`` and ``V`` does not match.
            CircuitError: Inputs contain measurements and/or resets.
        """
        # check inputs are valid
        if U.num_qubits != V.num_qubits:
            raise CircuitError(
                f"Number of qubits in unitaries does "
                f"not match: {U.num_qubits} != {V.num_qubits}."
            )
        _check_unitary(U)
        _check_unitary(V)

        # Vectors of new parameters, if any
        a_vec = ParameterVector("a", U.num_parameters)
        b_vec = ParameterVector("b", V.num_parameters)

        # Assign new labels so that alphabetical order matches insertion order
        circ1 = U.assign_parameters(a_vec)
        circ2 = V.assign_parameters(b_vec)

        # Generate the actual overlap circuit
        super().__init__(*circ1.qregs, name="UnitaryOverlap")
        self.compose(circ1, inplace=True)
        self.compose(circ2.inverse(), inplace=True)


def _check_unitary(circuit):
    """Check a circuit is unitary by checking if all operations are of type ``Gate``."""

    for instruction in circuit.data:
        if not isinstance(instruction.operation, Gate):
            raise CircuitError(
                (
                    "One or more instructions cannot be converted to"
                    ' a gate. "{}" is not a gate instruction'
                ).format(instruction.operation.name)
            )
