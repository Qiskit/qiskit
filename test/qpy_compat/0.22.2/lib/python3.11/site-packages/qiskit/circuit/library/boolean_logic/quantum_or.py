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


"""Implementations of boolean logic quantum circuits."""

from typing import List, Optional

from qiskit.circuit import QuantumRegister, QuantumCircuit, AncillaRegister
from qiskit.circuit.library.standard_gates import MCXGate


class OR(QuantumCircuit):
    r"""A circuit implementing the logical OR operation on a number of qubits.

    For the OR operation the state :math:`|1\rangle` is interpreted as ``True``. The result
    qubit is flipped, if the state of any variable qubit is ``True``. The OR is implemented using
    a multi-open-controlled X gate (i.e. flips if the state is :math:`|0\rangle`) and
    applying an X gate on the result qubit.
    Using a list of flags, qubits can be skipped or negated.

    The OR gate without special flags:

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import OR
        import qiskit.tools.jupyter
        circuit = OR(5)
        %circuit_library_info circuit

    Using flags we can negate qubits or skip them. For instance, if we have 5 qubits and want to
    return ``True`` if the first qubit is ``False`` or one of the last two are ``True`` we use the
    flags ``[-1, 0, 0, 1, 1]``.

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import OR
        import qiskit.tools.jupyter
        circuit = OR(5, flags=[-1, 0, 0, 1, 1])
        %circuit_library_info circuit

    """

    def __init__(
        self,
        num_variable_qubits: int,
        flags: Optional[List[int]] = None,
        mcx_mode: str = "noancilla",
    ) -> None:
        """Create a new logical OR circuit.

        Args:
            num_variable_qubits: The qubits of which the OR is computed. The result will be written
                into an additional result qubit.
            flags: A list of +1/0/-1 marking negations or omissions of qubits.
            mcx_mode: The mode to be used to implement the multi-controlled X gate.
        """
        self.num_variable_qubits = num_variable_qubits
        self.flags = flags

        # add registers
        qr_variable = QuantumRegister(num_variable_qubits, name="variable")
        qr_result = QuantumRegister(1, name="result")
        circuit = QuantumCircuit(qr_variable, qr_result, name="or")

        # determine the control qubits: all that have a nonzero flag
        flags = flags or [1] * num_variable_qubits
        control_qubits = [q for q, flag in zip(qr_variable, flags) if flag != 0]

        # determine the qubits that need to be flipped (if a flag is > 0)
        flip_qubits = [q for q, flag in zip(qr_variable, flags) if flag > 0]

        # determine the number of ancillas
        num_ancillas = MCXGate.get_num_ancilla_qubits(len(control_qubits), mode=mcx_mode)
        if num_ancillas > 0:
            qr_ancilla = AncillaRegister(num_ancillas, "ancilla")
            circuit.add_register(qr_ancilla)
        else:
            qr_ancilla = []

        circuit.x(qr_result)
        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)
        circuit.mcx(control_qubits, qr_result[:], qr_ancilla[:], mode=mcx_mode)
        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)

        super().__init__(*circuit.qregs, name="or")
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
