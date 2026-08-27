# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the sum of two equally sized qubit registers."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_func


class Adder(QuantumCircuit):
    r"""Compute the sum of two equally sized qubit registers.

    For two registers :math:`|a\rangle_n` and :math:`|b\rangle_n` with :math:`n` qubits each, an
    adder performs the following operation

    .. math::

        |a\rangle_n |b\rangle_n \mapsto |a\rangle_n |a + b\rangle_{n + 1}.

    The quantum register :math:`|a\rangle_n` (and analogously :math:`|b\rangle_n`)

    .. math::

        |a\rangle_n = |a_0\rangle \otimes \cdots \otimes |a_{n - 1}\rangle,

    for :math:`a_i \in \{0, 1\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \cdots + 2^{n - 1}a_{n - 1}.

    """

    @deprecate_func(
        since="2.1",
        additional_msg=(
            "Use the adder gates provided in qiskit.circuit.library.arithmetic instead. "
            "The gate type depends on the adder kind: fixed, half, full are represented by "
            "ModularAdderGate, HalfAdderGate, FullAdderGate, respectively. For different adder "
            "implementations, see https://quantum.cloud.ibm.com/docs/api/qiskit/synthesis.",
        ),
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(self, num_state_qubits: int, name: str = "Adder") -> None:
        """
        Args:
            num_state_qubits: The number of qubits in each of the registers.
            name: The name of the circuit.
        """
        super().__init__(name=name)
        self._num_state_qubits = num_state_qubits

    @property
    def num_state_qubits(self) -> int:
        """The number of state qubits, i.e. the number of bits in each input register.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits


class HalfAdderGate(Gate):
    r"""Compute the sum of two equally-sized qubit registers, including a carry-out bit.

    For two registers :math:`|a\rangle_n` and :math:`|b\rangle_n` with :math:`n` qubits each, an
    adder performs the following operation

    .. math::

        |a\rangle_n |b\rangle_n |0\rangle \mapsto |a\rangle_n |a + b\rangle_{n + 1}.

    The final input qubit is a clean qubit initialized to :math:`|0\rangle`.
    It stores the carry-out bit, which is why the output sum register has
    :math:`n + 1` qubits.

    The quantum register :math:`|a\rangle_n` (and analogously :math:`|b\rangle_n`)

    .. math::

        |a\rangle_n = |a_0\rangle \otimes \cdots \otimes |a_{n - 1}\rangle,

    for :math:`a_i \in \{0, 1\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \cdots + 2^{n - 1}a_{n - 1}.

    """

    def __init__(self, num_state_qubits: int, label: str | None = None) -> None:
        """
        Args:
            num_state_qubits: The number of qubits in each of the registers.
            label: An optional label for identifying the instruction.
        """
        if num_state_qubits < 1:
            raise ValueError("Need at least 1 state qubit.")

        super().__init__("HalfAdder", 2 * num_state_qubits + 1, [], label=label)
        self._num_state_qubits = num_state_qubits

    @property
    def num_state_qubits(self) -> int:
        """The number of state qubits, i.e. the number of bits in each input register.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    def _define(self):
        """Populates self.definition with some decomposition of this gate."""
        from qiskit.synthesis.arithmetic import adder_ripple_r25

        # This particular decomposition does not use any ancilla qubits.
        # Note that the transpiler may choose a different decomposition
        # based on the number of ancilla qubits available.
        self.definition = adder_ripple_r25(self.num_state_qubits)


class ModularAdderGate(Gate):
    r"""Compute the sum modulo an integer :math:`N` of two :math:`n`-sized qubit registers.

    For two registers :math:`|a\rangle_n` and :math:`|b\rangle_n` with :math:`n` qubits each, an
    adder performs the following operation

    .. math::

        |a\rangle_n |b\rangle_n \mapsto |a\rangle_n |a + b \text{ mod } N\rangle_n.

    The quantum register :math:`|a\rangle_n` (and analogously :math:`|b\rangle_n`)

    .. math::

        |a\rangle_n = |a_0\rangle \otimes \cdots \otimes |a_{n - 1}\rangle,

    for :math:`a_i \in \{0, 1\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \cdots + 2^{n - 1}a_{n - 1}.

    If ``modulus`` is not given (or equals :math:`2^n`), :math:`N = 2^n` and the operation is
    simply addition modulo the full range representable on :math:`n` qubits, as above. For a
    smaller modulus :math:`N < 2^n`, the map :math:`b \mapsto (a + b) \text{ mod } N` is only
    well-defined -- and therefore the action of this gate is only specified -- on inputs with
    :math:`a, b < N`. Applying the gate to a state with :math:`a \geq N` or :math:`b \geq N` is
    undefined behaviour and is left to the chosen synthesis method.

    """

    def __init__(
        self, num_state_qubits: int, label: str | None = None, modulus: int | None = None
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in each of the registers.
            label: An optional label for identifying the instruction.
            modulus: The modulus :math:`N` of the modular addition. Must satisfy
                :math:`1 \leq N \leq 2^n`, where :math:`n` is ``num_state_qubits``. Defaults to
                :math:`2^n`, in which case this gate is addition modulo the full range of
                ``num_state_qubits`` qubits, and is identical to the behaviour before this
                argument was introduced.

        Raises:
            ValueError: If ``num_state_qubits`` is smaller than 1, or if ``modulus`` is given
                and is not in the range :math:`[1, 2^n]`.
        """
        if num_state_qubits < 1:
            raise ValueError("Need at least 1 state qubit.")

        if modulus is not None and not 1 <= modulus <= 2**num_state_qubits:
            raise ValueError(
                "modulus must satisfy 1 <= modulus <= 2 ** num_state_qubits "
                f"(2 ** {num_state_qubits} = {2**num_state_qubits}), got {modulus}."
            )

        super().__init__("ModularAdder", 2 * num_state_qubits, [], label=label)
        self._num_state_qubits = num_state_qubits
        self.modulus = modulus

    @property
    def num_state_qubits(self) -> int:
        """The number of state qubits, i.e. the number of bits in each input register.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    def _define(self):
        """Populates self.definition with some decomposition of this gate."""
        from qiskit.synthesis.arithmetic import adder_modular_v17

        if self.modulus not in (None, 2**self.num_state_qubits):
            # No ancilla-free (or otherwise) synthesis method is registered yet for an
            # arbitrary modulus; see https://github.com/Qiskit/qiskit/issues/13608.
            raise CircuitError(
                "No default synthesis method is available for a ModularAdderGate with "
                f"modulus={self.modulus}. A synthesis method for this gate is only "
                "available when modulus is None or equal to 2 ** num_state_qubits. "
                "Use qiskit.transpiler.passes.HighLevelSynthesis with a plugin that "
                "supports this modulus, once one is registered for the 'ModularAdder' "
                "high-level-synthesis key."
            )

        # This particular decomposition does not use any ancilla qubits.
        # Note that the transpiler may choose a different decomposition
        # based on the number of ancilla qubits available.
        self.definition = adder_modular_v17(self.num_state_qubits)


class FullAdderGate(Gate):
    r"""Compute the sum of two :math:`n`-sized qubit registers, including carry-in and -out bits.

    For two registers :math:`|a\rangle_n` and :math:`|b\rangle_n` with :math:`n` qubits each, an
    adder performs the following operation

    .. math::

        |c_{\text{in}}\rangle_1 |a\rangle_n |b\rangle_n
        \mapsto |a\rangle_n |c_{\text{in}} + a + b \rangle_{n + 1}.

    The quantum register :math:`|a\rangle_n` (and analogously :math:`|b\rangle_n`)

    .. math::

        |a\rangle_n = |a_0\rangle \otimes \cdots \otimes |a_{n - 1}\rangle,

    for :math:`a_i \in \{0, 1\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \cdots + 2^{n - 1}a_{n - 1}.

    """

    def __init__(self, num_state_qubits: int, label: str | None = None) -> None:
        """
        Args:
            num_state_qubits: The number of qubits in each of the registers.
            label: An optional label for identifying the instruction.
        """
        if num_state_qubits < 1:
            raise ValueError("Need at least 1 state qubit.")

        super().__init__("FullAdder", 2 * num_state_qubits + 2, [], label=label)
        self._num_state_qubits = num_state_qubits

    @property
    def num_state_qubits(self) -> int:
        """The number of state qubits, i.e. the number of bits in each input register.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    def _define(self):
        """Populates self.definition with a decomposition of this gate."""
        from qiskit.synthesis.arithmetic import adder_ripple_c04

        # In the case of a full adder, this method does not use any ancilla qubits
        self.definition = adder_ripple_c04(self.num_state_qubits, kind="full")
