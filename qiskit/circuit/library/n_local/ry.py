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

"""The RY variational form."""

from typing import Union, Optional, List, Tuple, Callable

import numpy as np
from qiskit.extensions.standard import RYGate, CZGate
from qiskit.util import deprecate_arguments
from qiskit.aqua.components.initial_states import InitialState
from .two_local import TwoLocal


class RY(TwoLocal):
    r"""The RY Variational Form.

    The RY trial wave function is layers of :math:`y` rotations with entanglements.
    When none of qubits are unentangled to other qubits the number of parameters
    and the entanglement gates themselves have no additional parameters,
    the number of optimizer parameters this form creates and uses is given by
    :math:`q \times (d + 1)`, where :math:`q` is the total number of qubits and :math:`d` is
    the depth of the circuit.

    Nonetheless, in some cases, if an `entangler_map` does not include all qubits, that is, some
    qubits are not entangled by other qubits. The number of parameters is reduced by
    :math:`d \times q'` where :math:`q'` is the number of unentangled qubits.
    This is because adding more parameters to the unentangled qubits only introduce overhead
    without bringing any benefit; furthermore, theoretically, applying multiple RY gates in a row
    can be reduced to a single RY gate with the summed rotation angles.

    If the form uses entanglement gates with parameters (such as `'crx'`) the number of parameters
    increases by the number of entanglements. For instance with `'linear'` or `'sca'` entanglement
    the total number of parameters is :math:`2q \times (d + 1/2)`. For `'full'` entanglement an
    additional :math:`q \times (q - 1)/2 \times d` parameters, hence a total of
    :math:`d \times q \times (q + 1) / 2 + q`. It is possible to skip the final layer or :math:`y`
    rotations by setting the argument `skip_final_ry` to True.
    Then the number of parameters in above formulae decreases by :math:`q`.

    * 'full' entanglement is each qubit is entangled with all the others.
    * 'linear' entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
      for all :math:`i \in \{0, 1, ... , q - 2\}`, where :math:`q` is the total number of qubits.
    * 'sca' (shifted-circular-alternating) entanglement it is a generalized and modified version of
      the proposed circuit 14 in `Sim et al. <https://arxiv.org/abs/1905.10876>`__.
      It consists of circular entanglement where the 'long' entanglement connecting the first with
      the last qubit is shifted by one each block.  Furthermore the role of control and target
      qubits are swapped every block (therefore alternating).

    The `entanglement` parameter can be overridden by an `entangler_map` explicitly
    The entangler map is specified in the form of a list; where each element in the
    list is a list pair of a source qubit and a target qubit index. Indexes are integer values
    from :math:`0` to :math:`q-1`, where :math:`q` is the total number of qubits,
    as in the following example:

    >>> entangler_map = [[0, 1], [0, 2], [1, 3]]

    """

    @deprecate_arguments({'entangler_map': 'entanglement',
                          'skip_final_ry': 'skip_final_rotation_layer',
                          'entanglement_gate': 'entanglement_gates'})
    def __init__(self,
                 num_qubits: Optional[int] = None,
                 reps: int = 3,
                 entanglement_gates: Union[str, List[str], type, List[type]] = CZGate,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 initial_state: Optional[InitialState] = None,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 entangler_map: Optional[List[List[int]]] = None,  # pylint: disable=unused-argument
                 skip_final_ry: Optional[bool] = None,  # pylint: disable=unused-argument
                 entanglement_gate: Optional[str] = None,  # pylint: disable=unused-argument
                 ) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
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
            skip_final_ry: Deprecated, use `skip_final_rotation_layer` instead.
            entanglement_gate: Deprecated, use `entanglement_gates` instead.

        Examples:
            >>> ry = RY(3)  # create the variational form on 3 qubits
            >>> print(ry)  # show the circuit
            TODO: circuit diagram

            >>> ry = RY(3, entanglement='linear', reps=2, insert_barriers=True)
            >>> qc = QuantumCircuit(3)  # create a circuit and append the RY variational form
            >>> qc += ry.to_circuit()
            >>> qc.draw()
            TODO: circuit diagram

            >>> ry = RY(2, entanglement_gate='crx', 'sca', reps=2, insert_barriers=True)
            >>> print(ry)
            TODO: circuit diagram

            >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
            >>> ry = RY(3, 'cx', entangler_map, reps=2)
            >>> print(ry)
            TODO: circuit diagram

            >>> ry_linear = RY(2, entanglement='linear', reps=1)
            >>> ry_full = RY(2, entanglement='full', reps=1)
            >>> my_ry = ry_linear + ry_full
            >>> print(my_ry)
        """
        super().__init__(num_qubits=num_qubits,
                         reps=reps,
                         rotation_gates=RYGate,
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
        return self.num_parameters * [(-np.pi, np.pi)]
