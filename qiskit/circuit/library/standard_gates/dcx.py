# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Double-CNOT gate."""

from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array
from qiskit._accelerate.circuit import StandardGate


@with_gate_array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
class DCXGate(SingletonGate):
    r"""Double-CNOT gate.

    A 2-qubit Clifford gate consisting of two back-to-back
    CNOTs with alternate controls.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.dcx` method.

    .. code-block:: text

                  ┌───┐
        q_0: ──■──┤ X ├
             ┌─┴─┐└─┬─┘
        q_1: ┤ X ├──■──
             └───┘

    This is a classical logic gate, equivalent to a CNOT-SWAP (CNS) sequence,
    and locally equivalent to an iSWAP.

    .. math::

        DCX\ q_0, q_1 =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0
            \end{pmatrix}
    """

    _standard_gate = StandardGate.DCX

    def __init__(self, label=None):
        """Create new DCX gate."""
        super().__init__("dcx", 2, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit

        #           ┌───┐
        # q_0: ──■──┤ X ├
        #      ┌─┴─┐└─┬─┘
        # q_1: ┤ X ├──■──
        #      └───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.DCX._get_definition(self.params), add_regs=True, name=self.name
        )

    def __eq__(self, other):
        return isinstance(other, DCXGate)
