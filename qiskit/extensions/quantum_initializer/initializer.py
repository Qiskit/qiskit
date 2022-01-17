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
import math
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.reset import Reset
from .state_preparation import StatePreparation

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Initialize(Instruction):
    """Complex amplitude initialization.

    Class that implements the (complex amplitude) initialization of some
    flexible collection of qubit registers.
    Note that Initialize is an Instruction and not a Gate since it contains a reset instruction,
    which is not unitary.
    """

    def __init__(self, params, num_qubits=None):
        """Create new initialize composite.

        params (str, list, int or Statevector):
          * Statevector: Statevector to initialize to.
          * list: vector of complex amplitudes to initialize to.
          * string: labels of basis states of the Pauli eigenstates Z, X, Y. See
               :meth:`~qiskit.quantum_info.states.statevector.Statevector.from_label`.
               Notice the order of the labels is reversed with respect to the qubit index to
               be applied to. Example label '01' initializes the qubit zero to `|1>` and the
               qubit one to `|0>`.
          * int: an integer that is used as a bitmap indicating which qubits to initialize
               to `|1>`. Example: setting params to 5 would initialize qubit 0 and qubit 2
               to `|1>` and qubit 1 to `|0>`.
        num_qubits (int): This parameter is only used if params is an int. Indicates the total
            number of qubits in the `initialize` call. Example: `initialize` covers 5 qubits
            and params is 3. This allows qubits 0 and 1 to be initialized to `|1>` and the
            remaining 3 qubits to be initialized to `|0>`.
        """
        # self.params = params
        # self.num_qubits = num_qubits

        # pylint: disable=cyclic-import
        # from qiskit.quantum_info import Statevector

        # if isinstance(params, Statevector):
        #     params = params.data

        # if not isinstance(params, int) and num_qubits is not None:
        #     raise QiskitError(
        #         "The num_qubits parameter to Initialize should only be"
        #         " used when params is an integer"
        #     )
        # self._from_label = False
        # self._from_int = False

        # if isinstance(params, str):
        #     self._from_label = True
        #     num_qubits = len(params)
        # elif isinstance(params, int):
        #     self._from_int = True
        #     if num_qubits is None:
        #         num_qubits = int(math.log2(params)) + 1
        #     params = [params]
        # else:
        #     num_qubits = math.log2(len(params))

        #     # Check if param is a power of 2
        #     if num_qubits == 0 or not num_qubits.is_integer():
        #         raise QiskitError("Desired statevector length not a positive power of 2.")

        #     # Check if probabilities (amplitudes squared) sum to 1
        #     if not math.isclose(sum(np.absolute(params) ** 2), 1.0, abs_tol=_EPS):
        #         raise QiskitError("Sum of amplitudes-squared does not equal one.")

        #     num_qubits = int(num_qubits)

        # self._define(params, num_qubits)

        # self.reset(self.qubits)
        # self.append(StatePreparation(params, num_qubits), self.qubits)

        # replace with StatePreparation.get_num_qubits
        self._stateprep = StatePreparation(params, num_qubits)

        super().__init__("initialize",self._stateprep.num_qubits, 0, params)

    def _define(self): #??? only called with qc.decompose not qc.initialise?
        print('initialize _define')
        # print('num_qubits: ' + str(num_qubits))
        # print('params' + str(params))
        # call static method StatePreparation.get_num_qubits
        q = QuantumRegister(self._stateprep.num_qubits, "q")
        initialize_circuit = QuantumCircuit(q, name="init_def")
        initialize_circuit.reset(q)
        # initialize_circuit.append(StatePreparation(self.params, etc etc), q)
        self.definition = initialize_circuit

def initialize(self, params, qubits=None):
    print('initialize')
    r"""Initialize qubits in a specific state.

    Qubit initialization is done by first resetting the qubits to :math:`|0\rangle`
    followed by an state preparing unitary. Both these steps are included in the
    `Initialize` instruction.

    Args:
        params (str or list or int):
            * str: labels of basis states of the Pauli eigenstates Z, X, Y. See
                :meth:`~qiskit.quantum_info.states.statevector.Statevector.from_label`.
                Notice the order of the labels is reversed with respect to the qubit index to
                be applied to. Example label '01' initializes the qubit zero to `|1>` and the
                qubit one to `|0>`.
            * list: vector of complex amplitudes to initialize to.
            * int: an integer that is used as a bitmap indicating which qubits to initialize
               to `|1>`. Example: setting params to 5 would initialize qubit 0 and qubit 2
               to `|1>` and qubit 1 to `|0>`.
        qubits (QuantumRegister or int):
            * QuantumRegister: A list of qubits to be initialized [Default: None].
            * int: Index of qubit to initialized [Default: None].

    Returns:
        qiskit.circuit.Instruction: a handle to the instruction that was just initialized

    Examples:
        Prepare a qubit in the state :math:`(|0\rangle - |1\rangle) / \sqrt{2}`.

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.initialize([1/np.sqrt(2), -1/np.sqrt(2)], 0)
            circuit.draw()

        output:
             ┌──────────────────────────────┐
        q_0: ┤ initialize(0.70711,-0.70711) ├
             └──────────────────────────────┘


        Initialize from a string two qubits in the state `|10>`.
        The order of the labels is reversed with respect to qubit index.
        More information about labels for basis states are in
        :meth:`~qiskit.quantum_info.states.statevector.Statevector.from_label`.

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.initialize('01', circuit.qubits)
            circuit.draw()

        output:
             ┌──────────────────┐
        q_0: ┤0                 ├
             │  initialize(0,1) │
        q_1: ┤1                 ├
             └──────────────────┘


        Initialize two qubits from an array of complex amplitudes
        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.initialize([0, 1/np.sqrt(2), -1.j/np.sqrt(2), 0], circuit.qubits)
            circuit.draw()

        output:
             ┌────────────────────────────────────┐
        q_0: ┤0                                   ├
             │  initialize(0,0.70711,-0.70711j,0) │
        q_1: ┤1                                   ├
             └────────────────────────────────────┘
    """
    print('initialize')
    print('qubits:' + str(qubits))
    print('self.qubits:' + str(self.qubits))
    print('self.num_qubits:' + str(self.num_qubits))
    if qubits is None:
        qubits = self.qubits
    else:
        if isinstance(qubits, int):
            qubits = [qubits]
        qubits = self._bit_argument_conversion(qubits, self.qubits)

    num_qubits = None if not isinstance(params, int) else len(qubits)
    # num_qubits = len(qubits)

    return self.append(Initialize(params, num_qubits), qubits) #??? why does this method have access to .append, .reset etc.
    # self.reset(self.qubits)
    # return self.append(StatePreparation(params, num_qubits), qubits)


QuantumCircuit.initialize = initialize
