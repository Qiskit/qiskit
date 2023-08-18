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

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018,
# which was supervised by Raban Iten and Prof. Renato Renner.

"""
(Abstract) base class for uniformly controlled (also called multiplexed) single-qubit rotations R_t.
This class provides a basis for the decomposition of uniformly controlled R_x,R_y and R_z gates
(i.e., for t=x,y,z). These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_t(a_i) is applied to the target qubit for a (real) angle a_i.
"""

from qiskit.circuit.library.generalized_gates.uc_pauli_rot import (
    UCPauliRotGate as NewUCPauliRotGate,
)
from qiskit.utils.deprecation import deprecate_func


class UCPauliRotGate(NewUCPauliRotGate):
    """
    Uniformly controlled rotations (also called multiplexed rotations).
    The decomposition is based on 'Synthesis of Quantum Logic Circuits'
    by Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf)

    Input:
    angle_list = list of (real) rotation angles [a_0,...,a_{2^k-1}]. Must have at least one entry.

    rot_axis = rotation axis for the single qubit rotations
               (currently, 'X', 'Y' and 'Z' are supported)
    """

    @deprecate_func(
        since="0.45.0", additional_msg="This object moved to qiskit.circuit.library.UCPauliRotGate."
    )
    def __init__(self, angle_list, rot_axis):
        super().__init__(angle_list, rot_axis)
