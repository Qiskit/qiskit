# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Initialize qubit registers to desired arbitrary state.
"""

from qiskit.circuit.library.data_preparation.initializer import Initialize as NewInitialize
from qiskit.utils.deprecation import deprecate_func

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Initialize(NewInitialize):
    """Complex amplitude initialization.

    Class that initializes some flexible collection of qubit registers, implemented by calling
    the :class:`qiskit.extensions.StatePreparation` Class.
    Note that Initialize is an Instruction and not a Gate since it contains a reset instruction,
    which is not unitary.
    """

    @deprecate_func(
        since="0.45.0", additional_msg="This object moved to qiskit.circuit.library.Initialize."
    )
    def __init__(self, params, num_qubits=None, normalize=False):
        r"""Create new initialize composite.

        Args:
            params (str, list, int or Statevector):
                * Statevector: Statevector to initialize to.
                * list: vector of complex amplitudes to initialize to.
                * string: labels of basis states of the Pauli eigenstates Z, X, Y. See
                  :meth:`.Statevector.from_label`.
                  Notice the order of the labels is reversed with respect to the qubit index to
                  be applied to. Example label '01' initializes the qubit zero to :math:`|1\rangle`
                  and the qubit one to :math:`|0\rangle`.
                * int: an integer that is used as a bitmap indicating which qubits to initialize
                  to :math:`|1\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit 2
                  to :math:`|1\rangle` and qubit 1 to :math:`|0\rangle`.

            num_qubits (int): This parameter is only used if params is an int. Indicates the total
                number of qubits in the `initialize` call. Example: `initialize` covers 5 qubits
                and params is 3. This allows qubits 0 and 1 to be initialized to :math:`|1\rangle`
                and the remaining 3 qubits to be initialized to :math:`|0\rangle`.
            normalize (bool): Whether to normalize an input array to a unit vector.
        """
        super().__init__(params, num_qubits, normalize)
