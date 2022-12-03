# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Global Phase Gate"""

from typing import Optional  # , Union
import math
import numpy
from qiskit.qasm import pi

# from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister

# from qiskit.circuit._utils import _compute_control_matrix
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType

# from qiskit.circuit.exceptions import CircuitError


class GlobalPhaseGate(Gate):
    r"""The global phase gate (:math:`e^{i\theta}`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.gphase` method.

    **Mathamatical Representation:**


    .. math::
        \text{GlobalPhaseGate}\ =
            \begin{pmatrix}
                e^{i\theta}
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::
    """

    def __init__(self, phase: ParameterValueType, label: Optional[str] = None):
        """Create new globalphase gate.
        Args:
            num_qubits: The number of qubits the gate acts on.
            label: An optional label for the gate.
        """
        super().__init__("gphase", 0, [phase], label=label)

    def _define(self):
        # pylint: disable=cyclic-import

        q = QuantumRegister(0, "q")
        qc = QuantumCircuit(
            q, name=self.name, global_phase=self.params[0]
        )  # pylint: disable=no-member

        self.definition = qc

    def inverse(self):
        r"""Return inverted GLobalPhaseGate gate.

        :math:`GLobalPhaseGate(\lambda){\dagger} = GLobalPhaseGate(-\lambda)`
        """
        return GlobalPhaseGate(-self.params[0])

    def __array__(self, dtype=complex):
        """Return a numpy.array for the gphase gate."""
        cos = math.cos(self.params[0])
        sin = math.sin(self.params[0])
        return numpy.array([[cos + 1j * sin]], dtype=dtype)
