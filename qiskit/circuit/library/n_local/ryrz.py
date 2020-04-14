# -*- coding: utf-8 -*-

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

"""The RYRZ variational form."""

from typing import Union, Optional, List, Tuple, Callable
from numpy import pi

from qiskit.extensions.standard import RYGate, RZGate, CZGate
from qiskit.util import deprecate_arguments
from qiskit.aqua.components.initial_states import InitialState
from .two_local import TwoLocal


class RYRZ(TwoLocal):
    r"""The RYRZ variational form.

    The RYRZ trial wave function is layers of :math:`y` plus :math:`z` rotations with entanglements.
    When none of qubits are unentangled to other qubits, the number of optimizer parameters this
    form creates and uses is given by :math:`q \times (d + 1) \times 2`, where :math:`q` is the
    total number of qubits and :math:`d` is the depth of the circuit.
    Nonetheless, in some cases, if an `entangler_map` does not include all qubits, that is, some
    qubits are not entangled by other qubits. The number of parameters is reduced by
    :math:`d \times q' \times 2` where :math:`q'` is the number of unentangled qubits.
    This is because adding more parameters to the unentangled qubits only introduce overhead
    without bringing any benefit; furthermore, theoretically, applying multiple Ry and Rz gates
    in a row can be reduced to a single Ry gate and one Rz gate with the summed rotation angles.

    See :class:`RY` for more detail on `entanglement` which applies here too.
    """

    @deprecate_arguments({'entangler_map': 'entanglement'})
    def __init__(self,
                 num_qubits: Optional[int] = None,
                 depth: int = 3,
                 entanglement_gates: Union[str, List[str], type, List[type]] = CZGate,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 initial_state: Optional[InitialState] = None,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 entangler_map: Optional[List[List[int]]] = None,  # pylint: disable=unused-argument
                 ) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
            depth: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            entanglement_gates: The gates used in the entanglement layer. Can be specified via the
                name of a gate (e.g. 'cx') or the gate type itself (e.g. CnotGate).
                If only one gate is provided, the gate same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
                See the Examples section for more detail.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
                See the Examples section for more detail.
            initial_state: An `InitialState` object to prepend to the Ansatz.
                TODO deprecate this feature in favor of prepend or overloading __add__ in
                the initial state class
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If True, a rotation layer is added at the end of the
                ansatz. If False, no rotation layer is added. Defaults to True.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use instances of `qiskit.circuit.Parameter`. The name of each parameter is the
                number of its occurrence with this specified prefix.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.
                Defaults to False.
            entangler_map: Deprecated, use `entanglement` instead. This argument now also supports
                entangler maps.

        Examples:
            >>> ryrz = RYRZ(3)  # create the variational form on 3 qubits
            >>> print(ryrz)  # show the circuit
            TODO: circuit diagram

            >>> ryrz = RYRZ(4, entanglement='full', reps=1)
            >>> qc = QuantumCircuit(3)  # create a circuit and append the RY variational form
            >>> qc += ryrz.to_circuit()
            >>> qc.draw()
            TODO: circuit diagram

            >>> ryrz_crx = RYRZ(2, entanglement_gate='crx', 'sca', reps=1, insert_barriers=True)
            >>> print(ryrz_crx)
            TODO: circuit diagram

            >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
            >>> ry = RYRZ(3, 'cx', entangler_map, reps=2)
            >>> print(ryrz)
            TODO: circuit diagram

            >>> ryrz = RYRZ(2, entanglement='linear', reps=1)
            >>> ry = RY(2, entanglement='full', reps=1)
            >>> my_varform = ryrz + ry
            >>> print(my_varform)
        """
        super().__init__(num_qubits=num_qubits,
                         depth=depth,
                         rotation_gates=[RYGate, RZGate],
                         entanglement_gates=entanglement_gates,
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
        return self.num_parameters * [(-pi, pi)]
