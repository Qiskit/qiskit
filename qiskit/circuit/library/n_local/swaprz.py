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

"""The SwapRZ 2-local circuit."""

from typing import Union, Optional, List, Tuple, Callable, Any
from numpy import pi

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library.standard_gates import RZGate
from .two_local import TwoLocal


class SwapRZ(TwoLocal):
    r"""The SwapRZ ansatz.

    This trial wave function is layers of :math:`Z` rotations with entanglements using 2-qubit
    :math:`XX+YY` rotations. It was designed principally to be a particle-preserving wave function
    for :mod:`qiskit.chemistry`. Given an initial state as a set of 1's and 0's it will preserve
    the number of 1's - where for chemistry a 1 will indicate a particle.

    Note:

        In chemistry, to define the particles for SwapRZ, use a
        :class:`~qiskit.chemistry.components.initial_states.HartreeFock` initial state with
        the `Jordan-Wigner` qubit mapping

    See :class:`~qiskit.circuit.library.RY` for more detail on the possible arguments and options
    such as skipping unentanglement qubits, which apply here too.

    The rotations of the SwapRZ ansatz can be written as

    .. math::
        R_Z(\theta) = e^{-i \theta Z}

    and

    .. math::

        R_{XX+YY}(\theta) = e^{-i \theta / 2 (X \otimes X + Y \otimes Y)}
                          \approx e^{-i \theta / 2 X \otimes X} e^{-i \theta /2 Y \otimes Y }
                          = R_{XX}(\theta) R_{YY}(\theta)

    where the approximation used comes from the Trotter expansion of the sum in the exponential.

    Examples:

        >>> swaprz = SwapRZ(3, reps=1, insert_barriers=True, entanglement='linear')
        >>> print(swaprz)  # show the circuit
             ┌──────────┐ ░ ┌────────────┐┌────────────┐                             ░ ┌──────────┐
        q_0: ┤ RZ(θ[0]) ├─░─┤0           ├┤0           ├─────────────────────────────░─┤ RZ(θ[5]) ├
             ├──────────┤ ░ │  RXX(θ[3]) ││  RYY(θ[3]) │┌────────────┐┌────────────┐ ░ ├──────────┤
        q_1: ┤ RZ(θ[1]) ├─░─┤1           ├┤1           ├┤0           ├┤0           ├─░─┤ RZ(θ[6]) ├
             ├──────────┤ ░ └────────────┘└────────────┘│  RXX(θ[4]) ││  RYY(θ[4]) │ ░ ├──────────┤
        q_2: ┤ RZ(θ[2]) ├─░─────────────────────────────┤1           ├┤1           ├─░─┤ RZ(θ[7]) ├
             └──────────┘ ░                             └────────────┘└────────────┘ ░ └──────────┘

        >>> swaprz = SwapRZ(2, reps=1)
        >>> qc = QuantumCircuit(2)  # create a circuit and append the RY variational form
        >>> qc.cry(0.2, 0, 1)  # do some previous operation
        >>> qc.compose(swaprz, inplace=True)  # add the swaprz
        >>> qc.draw()
                        ┌──────────┐┌────────────┐┌────────────┐┌──────────┐
        q_0: ─────■─────┤ RZ(θ[0]) ├┤0           ├┤0           ├┤ RZ(θ[3]) ├
             ┌────┴────┐├──────────┤│  RXX(θ[2]) ││  RYY(θ[2]) │├──────────┤
        q_1: ┤ RY(0.2) ├┤ RZ(θ[1]) ├┤1           ├┤1           ├┤ RZ(θ[4]) ├
             └─────────┘└──────────┘└────────────┘└────────────┘└──────────┘

    """

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 reps: int = 3,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 initial_state: Optional[Any] = None,
                 ) -> None:
        """Create a new SwapRZ 2-local circuit.

        Args:
            num_qubits: The number of qubits of the SwapRZ circuit.
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
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
            skip_final_rotation_layer: If True, a rotation layer is added at the end of the
                ansatz. If False, no rotation layer is added. Defaults to True.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use :class:`~qiskit.circuit.ParameterVector`.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.

        """

        theta = Parameter('θ')
        rxxyy = QuantumCircuit(2, name='Rxx+yy')
        rxxyy.rxx(theta, 0, 1)
        rxxyy.ryy(theta, 0, 1)

        super().__init__(num_qubits=num_qubits,
                         rotation_blocks=RZGate,
                         entanglement_blocks=rxxyy,
                         entanglement=entanglement,
                         reps=reps,
                         skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer,
                         parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers,
                         initial_state=initial_state)

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
            The parameter bounds.
        """
        return self.num_parameters * [(-pi, pi)]
