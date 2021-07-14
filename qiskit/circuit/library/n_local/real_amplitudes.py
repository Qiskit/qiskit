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

"""The real-amplitudes 2-local circuit."""

from typing import Union, Optional, List, Tuple, Callable, Any
import numpy as np

from qiskit.circuit.library.standard_gates import RYGate, CXGate
from .two_local import TwoLocal


class RealAmplitudes(TwoLocal):
    r"""The real-amplitudes 2-local circuit.

    The ``RealAmplitudes`` circuit is a heuristic trial wave function used as Ansatz in chemistry
    applications or classification circuits in machine learning. The circuit consists of
    of alternating layers of :math:`Y` rotations and :math:`CX` entanglements. The entanglement
    pattern can be user-defined or selected from a predefined set.
    It is called ``RealAmplitudes`` since the prepared quantum states will only have
    real amplitudes, the complex part is always 0.

    For example a ``RealAmplitudes`` circuit with 2 repetitions on 3 qubits with ``'full'``
    entanglement is

    .. parsed-literal::

         ┌──────────┐ ░                 ░ ┌──────────┐ ░                 ░ ┌──────────┐
         ┤ RY(θ[0]) ├─░───■────■────────░─┤ RY(θ[3]) ├─░───■────■────────░─┤ RY(θ[6]) ├
         ├──────────┤ ░ ┌─┴─┐  │        ░ ├──────────┤ ░ ┌─┴─┐  │        ░ ├──────────┤
         ┤ RY(θ[1]) ├─░─┤ X ├──┼────■───░─┤ RY(θ[4]) ├─░─┤ X ├──┼────■───░─┤ RY(θ[7]) ├
         ├──────────┤ ░ └───┘┌─┴─┐┌─┴─┐ ░ ├──────────┤ ░ └───┘┌─┴─┐┌─┴─┐ ░ ├──────────┤
         ┤ RY(θ[2]) ├─░──────┤ X ├┤ X ├─░─┤ RY(θ[5]) ├─░──────┤ X ├┤ X ├─░─┤ RY(θ[8]) ├
         └──────────┘ ░      └───┘└───┘ ░ └──────────┘ ░      └───┘└───┘ ░ └──────────┘

    The entanglement can be set using the ``entanglement`` keyword as string or a list of
    index-pairs. See the documentation of :class:`~qiskit.circuit.library.TwoLocal` and
    :class:`~qiskit.circuit.NLocal` for more detail. Additional options that can be set include the
    number of repetitions, skipping rotation gates on qubits that are not entangled, leaving out
    the final rotation layer and inserting barriers in between the rotation and entanglement
    layers.

    If some qubits are not entangled with other qubits it makes sense to not apply rotation gates
    on these qubits, since a sequence of :math:`Y` rotations can be reduced to a single :math:`Y`
    rotation with summed rotation angles.

    Examples:

        >>> ansatz = RealAmplitudes(3, reps=2)  # create the circuit on 3 qubits
        >>> print(ansatz)
             ┌──────────┐          ┌──────────┐                      ┌──────────┐
        q_0: ┤ RY(θ[0]) ├──■────■──┤ RY(θ[3]) ├──────────────■────■──┤ RY(θ[6]) ├────────────
             ├──────────┤┌─┴─┐  │  └──────────┘┌──────────┐┌─┴─┐  │  └──────────┘┌──────────┐
        q_1: ┤ RY(θ[1]) ├┤ X ├──┼───────■──────┤ RY(θ[4]) ├┤ X ├──┼───────■──────┤ RY(θ[7]) ├
             ├──────────┤└───┘┌─┴─┐   ┌─┴─┐    ├──────────┤└───┘┌─┴─┐   ┌─┴─┐    ├──────────┤
        q_2: ┤ RY(θ[2]) ├─────┤ X ├───┤ X ├────┤ RY(θ[5]) ├─────┤ X ├───┤ X ├────┤ RY(θ[8]) ├
             └──────────┘     └───┘   └───┘    └──────────┘     └───┘   └───┘    └──────────┘

        >>> ansatz = RealAmplitudes(3, entanglement='linear', reps=2, insert_barriers=True)
        >>> qc = QuantumCircuit(3)  # create a circuit and append the RY variational form
        >>> qc.compose(ansatz, inplace=True)
        >>> qc.draw()
             ┌──────────┐ ░            ░ ┌──────────┐ ░            ░ ┌──────────┐
        q_0: ┤ RY(θ[0]) ├─░───■────────░─┤ RY(θ[3]) ├─░───■────────░─┤ RY(θ[6]) ├
             ├──────────┤ ░ ┌─┴─┐      ░ ├──────────┤ ░ ┌─┴─┐      ░ ├──────────┤
        q_1: ┤ RY(θ[1]) ├─░─┤ X ├──■───░─┤ RY(θ[4]) ├─░─┤ X ├──■───░─┤ RY(θ[7]) ├
             ├──────────┤ ░ └───┘┌─┴─┐ ░ ├──────────┤ ░ └───┘┌─┴─┐ ░ ├──────────┤
        q_2: ┤ RY(θ[2]) ├─░──────┤ X ├─░─┤ RY(θ[5]) ├─░──────┤ X ├─░─┤ RY(θ[8]) ├
             └──────────┘ ░      └───┘ ░ └──────────┘ ░      └───┘ ░ └──────────┘

        >>> ansatz = RealAmplitudes(4, reps=1, entanglement='circular', insert_barriers=True)
        >>> print(ansatz)
             ┌──────────┐ ░ ┌───┐                ░ ┌──────────┐
        q_0: ┤ RY(θ[0]) ├─░─┤ X ├──■─────────────░─┤ RY(θ[4]) ├
             ├──────────┤ ░ └─┬─┘┌─┴─┐           ░ ├──────────┤
        q_1: ┤ RY(θ[1]) ├─░───┼──┤ X ├──■────────░─┤ RY(θ[5]) ├
             ├──────────┤ ░   │  └───┘┌─┴─┐      ░ ├──────────┤
        q_2: ┤ RY(θ[2]) ├─░───┼───────┤ X ├──■───░─┤ RY(θ[6]) ├
             ├──────────┤ ░   │       └───┘┌─┴─┐ ░ ├──────────┤
        q_3: ┤ RY(θ[3]) ├─░───■────────────┤ X ├─░─┤ RY(θ[7]) ├
             └──────────┘ ░                └───┘ ░ └──────────┘

        >>> ansatz = RealAmplitudes(4, reps=2, entanglement=[[0,3], [0,2]],
        ... skip_unentangled_qubits=True)
        >>> print(ansatz)
             ┌──────────┐                 ┌──────────┐                 ┌──────────┐
        q_0: ┤ RY(θ[0]) ├──■───────■──────┤ RY(θ[3]) ├──■───────■──────┤ RY(θ[6]) ├
             └──────────┘  │       │      └──────────┘  │       │      └──────────┘
        q_1: ──────────────┼───────┼────────────────────┼───────┼──────────────────
             ┌──────────┐  │     ┌─┴─┐    ┌──────────┐  │     ┌─┴─┐    ┌──────────┐
        q_2: ┤ RY(θ[1]) ├──┼─────┤ X ├────┤ RY(θ[4]) ├──┼─────┤ X ├────┤ RY(θ[7]) ├
             ├──────────┤┌─┴─┐┌──┴───┴───┐└──────────┘┌─┴─┐┌──┴───┴───┐└──────────┘
        q_3: ┤ RY(θ[2]) ├┤ X ├┤ RY(θ[5]) ├────────────┤ X ├┤ RY(θ[8]) ├────────────
             └──────────┘└───┘└──────────┘            └───┘└──────────┘

    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "full",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: Optional[Any] = None,
        name: str = "RealAmplitudes",
    ) -> None:
        """Create a new RealAmplitudes 2-local circuit.

        Args:
            num_qubits: The number of qubits of the RealAmplitudes circuit.
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                See the Examples section of :class:`~qiskit.circuit.library.TwoLocal` for more
                detail.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
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
        super().__init__(
            num_qubits=num_qubits,
            reps=reps,
            rotation_blocks=RYGate,
            entanglement_blocks=CXGate,
            entanglement=entanglement,
            initial_state=initial_state,
            skip_unentangled_qubits=skip_unentangled_qubits,
            skip_final_rotation_layer=skip_final_rotation_layer,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
            The parameter bounds.
        """
        return self.num_parameters * [(-np.pi, np.pi)]
