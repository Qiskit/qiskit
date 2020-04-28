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

"""The RY 2-local circuit."""

from typing import Union, Optional, List, Tuple, Callable, Any
import numpy as np

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.library.standard_gates import RYGate, CZGate
from .two_local import TwoLocal


class RYAnsatz(TwoLocal):
    r"""The RY 2-local circuit.

    The RY trial wave function consistes of alternating layers of :math:`Y` rotations
    and entanglements. The entanglements are usually realized using CZ or CX gates.

    For example a RY circuit with 2 repetitions on 3 qubits with full CZ entanglement is

    .. parsed-literal::

         ┌──────────┐ ░           ░ ┌──────────┐ ░           ░ ┌──────────┐
         ┤ RY(θ[0]) ├─░──■──■─────░─┤ RY(θ[3]) ├─░──■──■─────░─┤ RY(θ[6]) ├
         ├──────────┤ ░  │  │     ░ ├──────────┤ ░  │  │     ░ ├──────────┤
         ┤ RY(θ[1]) ├─░──■──┼──■──░─┤ RY(θ[4]) ├─░──■──┼──■──░─┤ RY(θ[7]) ├
         ├──────────┤ ░     │  │  ░ ├──────────┤ ░     │  │  ░ ├──────────┤
         ┤ RY(θ[2]) ├─░─────■──■──░─┤ RY(θ[5]) ├─░─────■──■──░─┤ RY(θ[8]) ├
         └──────────┘ ░           ░ └──────────┘ ░           ░ └──────────┘


    The qubits can be entangled using different structures. This can be set using the
    ``entanglement`` keyword as string or a list of index-pairs, see the documentation of
    :class:`~qiskit.circuit.library.TwoLocal` and :class:`~qiskit.circuit.NLocal` for more detail.

    Additional options that can be set include skipping rotation gates on qubits that are not
    entangled, leaving out the final rotation layer and inserting barriers in between the
    rotation and entanglement layers.

    When all qubits are entangled and the entanglement gates themselves have no additional
    parameters, the number of parameters of this circuit is  :math:`q \times (r + 1)`,
    where :math:`q` is the total number of qubits and :math:`r` is the number of repetitions,
    set using `reps`. It is sensible to remove the rotation gates on the unentangled qubits since
    a sequence of RY gates can be reduced to a single RY with summed rotation angles and more
    parameters introduce overhead in optimization routines.

    If the form uses entanglement gates with parameters (such as ``'crx'``) the number of parameters
    increases by the number of entanglements. For instance with `'linear'` or `'sca'` entanglement
    the total number of parameters is :math:`2q \times (d + 1/2)`. For `'full'` entanglement an
    additional :math:`q \times (q - 1)/2 \times d` parameters, hence a total of
    :math:`d \times q \times (q + 1) / 2 + q`.

    Examples:

         >>> ry = RYAnsatz(3, reps=2)  # create the variational form on 3 qubits
         >>> print(ry)  # show the circuit
              ┌──────────┐      ┌──────────┐                  ┌──────────┐
         q_0: ┤ RY(θ[0]) ├─■──■─┤ RY(θ[3]) ├─────────────■──■─┤ RY(θ[6]) ├────────────
              ├──────────┤ │  │ └──────────┘┌──────────┐ │  │ └──────────┘┌──────────┐
         q_1: ┤ RY(θ[1]) ├─■──┼──────■──────┤ RY(θ[4]) ├─■──┼──────■──────┤ RY(θ[7]) ├
              ├──────────┤    │      │      ├──────────┤    │      │      ├──────────┤
         q_2: ┤ RY(θ[2]) ├────■──────■──────┤ RY(θ[5]) ├────■──────■──────┤ RY(θ[8]) ├
              └──────────┘                  └──────────┘                  └──────────┘

         >>> ry = RYAnsatz(3, entanglement='linear', reps=2, insert_barriers=True)
         >>> qc = QuantumCircuit(3)  # create a circuit and append the RY variational form
         >>> qc += ry.to_circuit()
         >>> qc.decompose().draw()
              ┌──────────┐ ░        ░ ┌──────────┐ ░        ░ ┌──────────┐
         q_0: ┤ RY(θ[0]) ├─░──■─────░─┤ RY(θ[3]) ├─░──■─────░─┤ RY(θ[6]) ├
              ├──────────┤ ░  │     ░ ├──────────┤ ░  │     ░ ├──────────┤
         q_1: ┤ RY(θ[1]) ├─░──■──■──░─┤ RY(θ[4]) ├─░──■──■──░─┤ RY(θ[7]) ├
              ├──────────┤ ░     │  ░ ├──────────┤ ░     │  ░ ├──────────┤
         q_2: ┤ RY(θ[2]) ├─░─────■──░─┤ RY(θ[5]) ├─░─────■──░─┤ RY(θ[8]) ├
              └──────────┘ ░        ░ └──────────┘ ░        ░ └──────────┘

         >>> ry = RYAnsatz(4, 'crx', entanglement='circular', reps=2, insert_barriers=True)
         >>> print(ry)
              ┌──────────┐ ░ ┌──────────┐                                     ░  ┌──────────┐
         q_0: ┤ RY(θ[0]) ├─░─┤ Rx(θ[4]) ├─────■───────────────────────────────░──┤ RY(θ[8]) ├
              ├──────────┤ ░ └────┬─────┘┌────┴─────┐                         ░  ├──────────┤
         q_1: ┤ RY(θ[1]) ├─░──────┼──────┤ Rx(θ[5]) ├─────■───────────────────░──┤ RY(θ[9]) ├
              ├──────────┤ ░      │      └──────────┘┌────┴─────┐             ░ ┌┴──────────┤
         q_2: ┤ RY(θ[2]) ├─░──────┼──────────────────┤ Rx(θ[6]) ├─────■───────░─┤ RY(θ[10]) ├
              ├──────────┤ ░      │                  └──────────┘┌────┴─────┐ ░ ├───────────┤
         q_3: ┤ RY(θ[3]) ├─░──────■──────────────────────────────┤ Rx(θ[7]) ├─░─┤ RY(θ[11]) ├
              └──────────┘ ░                                     └──────────┘ ░ └───────────┘

         >>> entanglement = [[0, 1], [0, 2]]
         >>> ry = RYAnsatz(3, 'cx', entanglement, reps=2)
         >>> print(ry)
              ┌──────────┐                      ┌──────────┐                      ┌──────────┐
         q_0: ┤ RY(θ[0]) ├──■────────────────■──┤ RY(θ[3]) ├──■────────────────■──┤ RY(θ[6]) ├
              ├──────────┤┌─┴─┐┌──────────┐  │  └──────────┘┌─┴─┐┌──────────┐  │  └──────────┘
         q_1: ┤ RY(θ[1]) ├┤ X ├┤ RY(θ[4]) ├──┼──────────────┤ X ├┤ RY(θ[7]) ├──┼──────────────
              ├──────────┤└───┘└──────────┘┌─┴─┐┌──────────┐└───┘└──────────┘┌─┴─┐┌──────────┐
         q_2: ┤ RY(θ[2]) ├─────────────────┤ X ├┤ RY(θ[5]) ├─────────────────┤ X ├┤ RY(θ[8]) ├
              └──────────┘                 └───┘└──────────┘                 └───┘└──────────┘
    """

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 entanglement_blocks: Union[
                     str, type, Instruction, QuantumCircuit,
                     List[Union[str, type, Instruction, QuantumCircuit]]
                 ] = CZGate,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 reps: int = 3,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 initial_state: Optional[Any] = None,
                 ) -> None:
        """Create a new RY 2-local circuit.

        Args:
          num_qubits: The number of qubits of the RY circuit.
          reps: Specifies how often the structure of a rotation layer followed by an entanglement
               layer is repeated.
          entanglement_blocks: The gates used in the entanglement layer. Can be specified via the
               name of a gate (e.g. ``'cx'``), the gate type itself (e.g. ``CXGate``) or a
               ``QuantumCircuit`` with two qubits.
               If only one gate is provided, the gate same gate is applied to each qubit.
               If a list of gates is provided, all gates are applied to each qubit in the provided
               order.
          entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
               or 'sca'), a list of integer-pairs specifying the indices of qubits
               entangled with one another, or a callable returning such a list provided with
               the index of the entanglement layer.
               See the Examples section of :class:`~qiskit.circuit.library.TwoLocal` for more
               detail.
          initial_state: An `InitialState` object to prepend to the circuit.
          skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
               that are entangled with another qubit. If False, the single qubit gates are applied
               to each qubit in the Ansatz. Defaults to False.
          skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
               that are entangled with another qubit. If False, the single qubit gates are applied
               to each qubit in the Ansatz. Defaults to False.
          skip_final_rotation_layer: If False, a rotation layer is added at the end of the
               ansatz. If True, no rotation layer is added.
          parameter_prefix: The parameterized gates require a parameter to be defined, for which
               we use :class:`~qiskit.circuit.ParameterVector`.
          insert_barriers: If True, barriers are inserted in between each layer. If False,
               no barriers are inserted.

        """
        super().__init__(num_qubits=num_qubits,
                         reps=reps,
                         rotation_blocks=RYGate,
                         entanglement_blocks=entanglement_blocks,
                         entanglement=entanglement,
                         initial_state=initial_state,
                         skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer,
                         parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers)

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
             The parameter bounds.
        """
        return self.num_parameters * [(-np.pi, np.pi)]
