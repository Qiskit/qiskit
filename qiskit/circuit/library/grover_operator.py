# -*- coding: utf-8 -*-

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

"""The Grover operator."""

from typing import List, Optional, Union
import numpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from .standard_gates import MCXGate


class GroverOperator(QuantumCircuit):
    r"""The Grover operator.

    Grover's search algorithm [1, 2] consists of repeated applications of the so-called
    Grover operator used to amplify the amplitudes of the desired output states.
    This operator consists of the oracle, $\mathcal{S}_f$, that multiplies the amplitude of the
    good states by -1, a reflection $\mathcal{S}_0$ about the $\ket{0}^{\otimes n}$ state
    and an input state $\mathcal{A}$. For the textbook Grover search, $\mathcal{A} = H^{\otimes n}$,
    however for generic amplitude amplification the input state might differ [3].
    With these terms, the Grover operator can be written as

    .. math::

        \mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A} \mathcal{S_f}

    .. note::

        Sometimes the Grover operator is defined with a negative sign, in that case
        $\mathcal{S}_0$ multiplies all states *except* $\ket{0}^{\otimes n}$ with -1.
        In our formulation, $\mathcal{S}_0$ only multiplies $\ket{0}^{\otimes n}$ with -1.

    Examples:
        >>> from qiskit.circuit import QuantumCircuit
        >>> from qiskit.circuit.library import GroverOperator
        >>> oracle = QuantumCircuit(2)
        >>> oracle.z(0)  # good state = first qubit is |1>
        >>> grover_op = GroverOperator(oracle, insert_barriers=True)
        >>> grover_op.draw()
                 ┌───┐ ░ ┌───┐ ░ ┌───┐          ┌───┐      ░ ┌───┐
        state_0: ┤ Z ├─░─┤ H ├─░─┤ X ├───────■──┤ X ├──────░─┤ H ├
                 └───┘ ░ ├───┤ ░ ├───┤┌───┐┌─┴─┐├───┤┌───┐ ░ ├───┤
        state_1: ──────░─┤ H ├─░─┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├─░─┤ H ├
                       ░ └───┘ ░ └───┘└───┘└───┘└───┘└───┘ ░ └───┘

        >>> oracle = QuantumCircuit(1)
        >>> oracle.z(0)  # the qubit state |1> is the good state
        >>> state_in = QuantumCircuit(1)
        >>> state_in.ry(0.2, 0)  # non-uniform state preparation
        >>> grover_op = GroverOperator(oracle, state_in)
        >>> grover_op.draw()
                 ┌───┐┌──────────┐┌───┐┌───┐┌───┐┌─────────┐
        state_0: ┤ Z ├┤ RY(-0.2) ├┤ X ├┤ Z ├┤ X ├┤ RY(0.2) ├
                 └───┘└──────────┘└───┘└───┘└───┘└─────────┘

        >>> oracle = QuantumCircuit(4)
        >>> oracle.z(3)
        >>> reflection_qubits = [0]
        >>> state_in = QuantumCircuit(4)
        >>> state_in.cry(0.1, 0, 3)
        >>> state_in.ry(0.5, 3)
        >>> grover_op = GroverOperator(oracle, state_in, reflection_qubits=reflection_qubits)
        >>> grover_op.draw()
                                              ┌───┐          ┌───┐
        state_0: ──────────────────────■──────┤ X ├───────■──┤ X ├──────────■────────────────
                                       │      └───┘       │  └───┘          │
        state_1: ──────────────────────┼──────────────────┼─────────────────┼────────────────
                                       │                  │                 │
        state_2: ──────────────────────┼──────────────────┼─────────────────┼────────────────
                 ┌───┐┌──────────┐┌────┴─────┐┌───┐┌───┐┌─┴─┐┌───┐┌───┐┌────┴────┐┌─────────┐
        state_3: ┤ Z ├┤ RY(-0.5) ├┤ RY(-0.1) ├┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├┤ RY(0.1) ├┤ RY(0.5) ├
                 └───┘└──────────┘└──────────┘└───┘└───┘└───┘└───┘└───┘└─────────┘└─────────┘

        >>> mark_state = Statevector.from_label('011')
        >>> diffuse_operator = 2 * DensityMatrix.from_label('000') - Operator.from_label('III')
        >>> grover_op = GroverOperator(oracle=mark_state, zero_reflection=diffuse_operator)
        >>> grover_op.draw(fold=70)
                 ┌─────────────────┐      ┌───┐                          »
        state_0: ┤0                ├──────┤ H ├──────────────────────────»
                 │                 │┌─────┴───┴─────┐     ┌───┐          »
        state_1: ┤1 UCRZ(0,pi,0,0) ├┤0              ├─────┤ H ├──────────»
                 │                 ││  UCRZ(pi/2,0) │┌────┴───┴────┐┌───┐»
        state_2: ┤2                ├┤1              ├┤ UCRZ(-pi/4) ├┤ H ├»
                 └─────────────────┘└───────────────┘└─────────────┘└───┘»
        «         ┌─────────────────┐      ┌───┐
        «state_0: ┤0                ├──────┤ H ├─────────────────────────
        «         │                 │┌─────┴───┴─────┐    ┌───┐
        «state_1: ┤1 UCRZ(pi,0,0,0) ├┤0              ├────┤ H ├──────────
        «         │                 ││  UCRZ(pi/2,0) │┌───┴───┴────┐┌───┐
        «state_2: ┤2                ├┤1              ├┤ UCRZ(pi/4) ├┤ H ├
        «         └─────────────────┘└───────────────┘└────────────┘└───┘

    References:
        [1]: L. K. Grover (1996), A fast quantum mechanical algorithm for database search,
            `arXiv:quant-ph/9605043 <https://arxiv.org/abs/quant-ph/9605043>`_.
        [2]: I. Chuang & M. Nielsen, Quantum Computation and Quantum Information,
            Cambridge: Cambridge University Press, 2000. Chapter 6.1.2.
        [3]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
            Quantum Amplitude Amplification and Estimation.
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(self, oracle: Union[QuantumCircuit, Statevector],
                 state_in: Optional[QuantumCircuit] = None,
                 zero_reflection: Optional[Union[QuantumCircuit, DensityMatrix, Operator]] = None,
                 reflection_qubits: Optional[List[int]] = None,
                 insert_barriers: bool = False,
                 mcx: str = 'noancilla',
                 name: str = 'Q') -> None:
        """
        Args:
            oracle: The oracle implementing a reflection about the bad state.
            state_in: The operator preparing the good and bad state. For Grover's algorithm,
                this is a n-qubit Hadamard gate and for Amplitude Amplification or Estimation
                the operator A.
            zero_reflection: The reflection about the zero state.
            reflection_qubits: Qubits on which the the zero reflection act on.
            insert_barriers: Whether barriers should be inserted between the reflections and A.
            mcx: The mode to use for building the default zero reflection.
            name: The name of the circuit.
        """
        super().__init__(name=name)

        # store inputs
        if isinstance(oracle, Statevector):
            from qiskit.circuit.library import Diagonal  # pylint: disable=cyclic-import
            oracle = Diagonal((-1) ** oracle.data)
        self._oracle = oracle

        if isinstance(zero_reflection, (Operator, DensityMatrix)):
            from qiskit.circuit.library import Diagonal  # pylint: disable=cyclic-import
            zero_reflection = Diagonal(zero_reflection.data.diagonal())
        self._zero_reflection = zero_reflection

        self._reflection_qubits = reflection_qubits
        self._state_in = state_in
        self._insert_barriers = insert_barriers
        self._mcx = mcx

        # build circuit
        self._build()

    @property
    def reflection_qubits(self):
        """Reflection qubits, on which S0 is applied (if S0 is not user-specified)."""
        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas

        if self._reflection_qubits is not None:
            return list(set(self._reflection_qubits + [num_state_qubits - 1]))

        return list(range(num_state_qubits))

    @property
    def zero_reflection(self) -> QuantumCircuit:
        """The subcircuit implementing the reflection about 0."""
        if self._zero_reflection is not None:
            return self._zero_reflection

        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        qubits = self.reflection_qubits
        return _zero_reflection(num_state_qubits, qubits, self._mcx)

    @property
    def state_in(self) -> QuantumCircuit:
        """The subcircuit implementing the A operator or Hadamards."""
        if self._state_in is not None:
            return self._state_in

        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        hadamards = QuantumCircuit(num_state_qubits, name='H')
        # apply Hadamards only on reflection qubits, rest will cancel out
        hadamards.h(self.reflection_qubits)
        return hadamards

    @property
    def oracle(self):
        """The oracle implementing a reflection about the bad state."""
        return self._oracle

    def _build(self):
        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        self.add_register(QuantumRegister(num_state_qubits, name='state'))
        num_ancillas = numpy.max([self.oracle.num_ancillas,
                                  self.zero_reflection.num_ancillas,
                                  self.state_in.num_ancillas])
        if num_ancillas > 0:
            self.add_register(AncillaRegister(num_ancillas, name='ancilla'))

        self.compose(self.oracle, list(range(self.oracle.num_qubits)), inplace=True)
        if self._insert_barriers:
            self.barrier()
        self.compose(self.state_in.inverse(), list(range(self.state_in.num_qubits)), inplace=True)
        if self._insert_barriers:
            self.barrier()
        self.compose(self.zero_reflection, list(range(self.zero_reflection.num_qubits)),
                     inplace=True)
        if self._insert_barriers:
            self.barrier()
        self.compose(self.state_in, list(range(self.state_in.num_qubits)), inplace=True)


# TODO use the oracle compiler or the bit string oracle
def _zero_reflection(num_state_qubits: int, qubits: List[int], mcx: Optional[str] = None
                     ) -> QuantumCircuit:
    qr_state = QuantumRegister(num_state_qubits, 'state')
    reflection = QuantumCircuit(qr_state, name='S_0')

    num_ancillas = MCXGate.get_num_ancilla_qubits(num_state_qubits - 1, mcx)
    if num_ancillas > 0:
        qr_ancilla = AncillaRegister(num_ancillas, 'ancilla')
        reflection.add_register(qr_ancilla)
    else:
        qr_ancilla = []

    reflection.x(qubits)
    if len(qubits) == 1:
        reflection.z(0)  # MCX does not allow 0 control qubits, therefore this is separate
    else:
        reflection.h(qubits[-1])
        reflection.mcx(qubits[:-1], qubits[-1], qr_ancilla[:], mode=mcx)
        reflection.h(qubits[-1])
    reflection.x(qubits)

    return reflection
