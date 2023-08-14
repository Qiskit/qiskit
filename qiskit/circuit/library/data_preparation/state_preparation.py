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
"""Prepare a quantum state from the state where all qubits are 0."""

from typing import Union, Optional

import math
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.x import CXGate, XGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.s import SGate, SdgGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Statevector

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class StatePreparation(Gate):
    """Complex amplitude state preparation.

    Class that implements the (complex amplitude) state preparation of some
    flexible collection of qubit registers.
    """

    def __init__(
        self,
        params: Union[str, list, int, Statevector],
        num_qubits: Optional[int] = None,
        inverse: bool = False,
        label: Optional[str] = None,
        normalize: bool = False,
    ):
        r"""
        Args:
            params:
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
            num_qubits: This parameter is only used if params is an int. Indicates the total
                number of qubits in the `initialize` call. Example: `initialize` covers 5 qubits
                and params is 3. This allows qubits 0 and 1 to be initialized to :math:`|1\rangle`
                and the remaining 3 qubits to be initialized to :math:`|0\rangle`.
            inverse: if True, the inverse state is constructed.
            label: An optional label for the gate
            normalize (bool): Whether to normalize an input array to a unit vector.

        Raises:
            QiskitError: ``num_qubits`` parameter used when ``params`` is not an integer

        When a Statevector argument is passed the state is prepared using a recursive
        initialization algorithm, including optimizations, from [1], as well
        as some additional optimizations including removing zero rotations and double cnots.

        **References:**
        [1] Shende, Bullock, Markov. Synthesis of Quantum Logic Circuits (2004)
        [`https://arxiv.org/abs/quant-ph/0406176v5`]

        """
        self._params_arg = params
        self._inverse = inverse
        self._name = "state_preparation_dg" if self._inverse else "state_preparation"

        if label is None:
            self._label = "State Preparation Dg" if self._inverse else "State Preparation"
        else:
            self._label = f"{label} Dg" if self._inverse else label

        if isinstance(params, Statevector):
            params = params.data

        if not isinstance(params, int) and num_qubits is not None:
            raise QiskitError(
                "The num_qubits parameter to StatePreparation should only be"
                " used when params is an integer"
            )
        self._from_label = isinstance(params, str)
        self._from_int = isinstance(params, int)

        # if initialized from a vector, check that the parameters are normalized
        if not self._from_label and not self._from_int:
            norm = np.linalg.norm(params)
            if normalize:
                params = np.array(params, dtype=np.complex128) / norm
            elif not math.isclose(norm, 1.0, abs_tol=_EPS):
                raise QiskitError(f"Sum of amplitudes-squared is not 1, but {norm}.")

        num_qubits = self._get_num_qubits(num_qubits, params)
        params = [params] if isinstance(params, int) else params

        super().__init__(self._name, num_qubits, params, label=self._label)

    def _define(self):
        if self._from_label:
            self.definition = self._define_from_label()
        elif self._from_int:
            self.definition = self._define_from_int()
        else:
            self.definition = self._define_synthesis()

    def _define_from_label(self):
        q = QuantumRegister(self.num_qubits, "q")
        initialize_circuit = QuantumCircuit(q, name="init_def")

        for qubit, param in enumerate(reversed(self.params)):
            if param == "1":
                initialize_circuit.append(XGate(), [q[qubit]])
            elif param == "+":
                initialize_circuit.append(HGate(), [q[qubit]])
            elif param == "-":
                initialize_circuit.append(XGate(), [q[qubit]])
                initialize_circuit.append(HGate(), [q[qubit]])
            elif param == "r":  # |+i>
                initialize_circuit.append(HGate(), [q[qubit]])
                initialize_circuit.append(SGate(), [q[qubit]])
            elif param == "l":  # |-i>
                initialize_circuit.append(HGate(), [q[qubit]])
                initialize_circuit.append(SdgGate(), [q[qubit]])

        if self._inverse:
            initialize_circuit = initialize_circuit.inverse()

        return initialize_circuit

    def _define_from_int(self):
        q = QuantumRegister(self.num_qubits, "q")
        initialize_circuit = QuantumCircuit(q, name="init_def")

        # Convert to int since QuantumCircuit converted to complex
        # and make a bit string and reverse it
        intstr = f"{int(np.real(self.params[0])):0{self.num_qubits}b}"[::-1]

        # Raise if number of bits is greater than num_qubits
        if len(intstr) > self.num_qubits:
            raise QiskitError(
                "StatePreparation integer has %s bits, but this exceeds the"
                " number of qubits in the circuit, %s." % (len(intstr), self.num_qubits)
            )

        for qubit, bit in enumerate(intstr):
            if bit == "1":
                initialize_circuit.append(XGate(), [q[qubit]])

        # note: X is it's own inverse, so even if self._inverse is True,
        # we don't need to invert anything
        return initialize_circuit

    def _define_synthesis(self):
        """Calculate a subcircuit that implements this initialization

        Implements a recursive initialization algorithm, including optimizations,
        from "Synthesis of Quantum Logic Circuits" Shende, Bullock, Markov
        https://arxiv.org/abs/quant-ph/0406176v5

        Additionally implements some extra optimizations: remove zero rotations and
        double cnots.
        """
        # call to generate the circuit that takes the desired vector to zero
        disentangling_circuit = self._gates_to_uncompute()

        # invert the circuit to create the desired vector from zero (assuming
        # the qubits are in the zero state)
        if self._inverse is False:
            initialize_instr = disentangling_circuit.to_instruction().inverse()
        else:
            initialize_instr = disentangling_circuit.to_instruction()

        q = QuantumRegister(self.num_qubits, "q")
        initialize_circuit = QuantumCircuit(q, name="init_def")
        initialize_circuit.append(initialize_instr, q[:])

        return initialize_circuit

    def _get_num_qubits(self, num_qubits, params):
        """Get number of qubits needed for state preparation"""
        if isinstance(params, str):
            num_qubits = len(params)
        elif isinstance(params, int):
            if num_qubits is None:
                num_qubits = int(math.log2(params)) + 1
        else:
            num_qubits = math.log2(len(params))

            # Check if param is a power of 2
            if num_qubits == 0 or not num_qubits.is_integer():
                raise QiskitError("Desired statevector length not a positive power of 2.")

            num_qubits = int(num_qubits)
        return num_qubits

    def inverse(self):
        """Return inverted StatePreparation"""

        label = (
            None if self._label in ("State Preparation", "State Preparation Dg") else self._label
        )

        return StatePreparation(self._params_arg, inverse=not self._inverse, label=label)

    def broadcast_arguments(self, qargs, cargs):
        flat_qargs = [qarg for sublist in qargs for qarg in sublist]

        if self.num_qubits != len(flat_qargs):
            raise QiskitError(
                "StatePreparation parameter vector has %d elements, therefore expects %s "
                "qubits. However, %s were provided."
                % (2**self.num_qubits, self.num_qubits, len(flat_qargs))
            )
        yield flat_qargs, []

    def validate_parameter(self, parameter):
        """StatePreparation instruction parameter can be str, int, float, and complex."""

        # StatePreparation instruction parameter can be str
        if isinstance(parameter, str):
            if parameter in ["0", "1", "+", "-", "l", "r"]:
                return parameter
            raise CircuitError(
                "invalid param label {} for instruction {}. Label should be "
                "0, 1, +, -, l, or r ".format(type(parameter), self.name)
            )

        # StatePreparation instruction parameter can be int, float, and complex.
        if isinstance(parameter, (int, float, complex)):
            return complex(parameter)
        elif isinstance(parameter, np.number):
            return complex(parameter.item())
        else:
            raise CircuitError(f"invalid param type {type(parameter)} for instruction  {self.name}")

    def _return_repeat(self, exponent: float) -> "Gate":
        return Gate(name=f"{self.name}*{exponent}", num_qubits=self.num_qubits, params=[])

    def _gates_to_uncompute(self):
        """Call to create a circuit with gates that take the desired vector to zero.

        Returns:
            QuantumCircuit: circuit to take self.params vector to :math:`|{00\\ldots0}\\rangle`
        """
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q, name="disentangler")

        # kick start the peeling loop, and disentangle one-by-one from LSB to MSB
        remaining_param = self.params

        for i in range(self.num_qubits):
            # work out which rotations must be done to disentangle the LSB
            # qubit (we peel away one qubit at a time)
            (remaining_param, thetas, phis) = StatePreparation._rotations_to_disentangle(
                remaining_param
            )

            # perform the required rotations to decouple the LSB qubit (so that
            # it can be "factored" out, leaving a shorter amplitude vector to peel away)

            add_last_cnot = True
            if np.linalg.norm(phis) != 0 and np.linalg.norm(thetas) != 0:
                add_last_cnot = False

            if np.linalg.norm(phis) != 0:
                rz_mult = self._multiplex(RZGate, phis, last_cnot=add_last_cnot)
                circuit.append(rz_mult.to_instruction(), q[i : self.num_qubits])

            if np.linalg.norm(thetas) != 0:
                ry_mult = self._multiplex(RYGate, thetas, last_cnot=add_last_cnot)
                circuit.append(ry_mult.to_instruction().reverse_ops(), q[i : self.num_qubits])
        circuit.global_phase -= np.angle(sum(remaining_param))
        return circuit

    @staticmethod
    def _rotations_to_disentangle(local_param):
        """
        Static internal method to work out Ry and Rz rotation angles used
        to disentangle the LSB qubit.
        These rotations make up the block diagonal matrix U (i.e. multiplexor)
        that disentangles the LSB.

        [[Ry(theta_1).Rz(phi_1)  0   .   .   0],
        [0         Ry(theta_2).Rz(phi_2) .  0],
                                    .
                                        .
        0         0           Ry(theta_2^n).Rz(phi_2^n)]]
        """
        remaining_vector = []
        thetas = []
        phis = []

        param_len = len(local_param)

        for i in range(param_len // 2):
            # Ry and Rz rotations to move bloch vector from 0 to "imaginary"
            # qubit
            # (imagine a qubit state signified by the amplitudes at index 2*i
            # and 2*(i+1), corresponding to the select qubits of the
            # multiplexor being in state |i>)
            (remains, add_theta, add_phi) = StatePreparation._bloch_angles(
                local_param[2 * i : 2 * (i + 1)]
            )

            remaining_vector.append(remains)

            # rotations for all imaginary qubits of the full vector
            # to move from where it is to zero, hence the negative sign
            thetas.append(-add_theta)
            phis.append(-add_phi)

        return remaining_vector, thetas, phis

    @staticmethod
    def _bloch_angles(pair_of_complex):
        """
        Static internal method to work out rotation to create the passed-in
        qubit from the zero vector.
        """
        [a_complex, b_complex] = pair_of_complex
        # Force a and b to be complex, as otherwise numpy.angle might fail.
        a_complex = complex(a_complex)
        b_complex = complex(b_complex)
        mag_a = abs(a_complex)
        final_r = np.sqrt(mag_a**2 + np.absolute(b_complex) ** 2)
        if final_r < _EPS:
            theta = 0
            phi = 0
            final_r = 0
            final_t = 0
        else:
            theta = 2 * np.arccos(mag_a / final_r)
            a_arg = np.angle(a_complex)
            b_arg = np.angle(b_complex)
            final_t = a_arg + b_arg
            phi = b_arg - a_arg

        return final_r * np.exp(1.0j * final_t / 2), theta, phi

    def _multiplex(self, target_gate, list_of_angles, last_cnot=True):
        """
        Return a recursive implementation of a multiplexor circuit,
        where each instruction itself has a decomposition based on
        smaller multiplexors.

        The LSB is the multiplexor "data" and the other bits are multiplexor "select".

        Args:
            target_gate (Gate): Ry or Rz gate to apply to target qubit, multiplexed
                over all other "select" qubits
            list_of_angles (list[float]): list of rotation angles to apply Ry and Rz
            last_cnot (bool): add the last cnot if last_cnot = True

        Returns:
            DAGCircuit: the circuit implementing the multiplexor's action
        """
        list_len = len(list_of_angles)
        local_num_qubits = int(math.log2(list_len)) + 1

        q = QuantumRegister(local_num_qubits)
        circuit = QuantumCircuit(q, name="multiplex" + str(local_num_qubits))

        lsb = q[0]
        msb = q[local_num_qubits - 1]

        # case of no multiplexing: base case for recursion
        if local_num_qubits == 1:
            circuit.append(target_gate(list_of_angles[0]), [q[0]])
            return circuit

        # calc angle weights, assuming recursion (that is the lower-level
        # requested angles have been correctly implemented by recursion
        angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]], np.identity(2 ** (local_num_qubits - 2)))

        # calc the combo angles
        list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

        # recursive step on half the angles fulfilling the above assumption
        multiplex_1 = self._multiplex(target_gate, list_of_angles[0 : (list_len // 2)], False)
        circuit.append(multiplex_1.to_instruction(), q[0:-1])

        # attach CNOT as follows, thereby flipping the LSB qubit
        circuit.append(CXGate(), [msb, lsb])

        # implement extra efficiency from the paper of cancelling adjacent
        # CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
        # second lower-level multiplex)
        multiplex_2 = self._multiplex(target_gate, list_of_angles[(list_len // 2) :], False)
        if list_len > 1:
            circuit.append(multiplex_2.to_instruction().reverse_ops(), q[0:-1])
        else:
            circuit.append(multiplex_2.to_instruction(), q[0:-1])

        # attach a final CNOT
        if last_cnot:
            circuit.append(CXGate(), [msb, lsb])

        return circuit


def prepare_state(self, state, qubits=None, label=None, normalize=False):
    r"""Prepare qubits in a specific state.

    This class implements a state preparing unitary. Unlike
    :class:`qiskit.extensions.Initialize` it does not reset the qubits first.

    Args:
        state (str or list or int or Statevector):
            * Statevector: Statevector to initialize to.
            * str: labels of basis states of the Pauli eigenstates Z, X, Y. See
              :meth:`.Statevector.from_label`. Notice the order of the labels is reversed with respect
              to the qubit index to be applied to. Example label '01' initializes the qubit zero to
              :math:`|1\rangle` and the qubit one to :math:`|0\rangle`.
            * list: vector of complex amplitudes to initialize to.
            * int: an integer that is used as a bitmap indicating which qubits to initialize
              to :math:`|1\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit 2
              to :math:`|1\rangle` and qubit 1 to :math:`|0\rangle`.

        qubits (QuantumRegister or Qubit or int):
            * QuantumRegister: A list of qubits to be initialized [Default: None].
            * Qubit: Single qubit to be initialized [Default: None].
            * int: Index of qubit to be initialized [Default: None].
            * list: Indexes of qubits to be initialized [Default: None].
        label (str): An optional label for the gate
        normalize (bool): Whether to normalize an input array to a unit vector.

    Returns:
        qiskit.circuit.Instruction: a handle to the instruction that was just initialized

    Examples:
        Prepare a qubit in the state :math:`(|0\rangle - |1\rangle) / \sqrt{2}`.

        .. code-block::

           import numpy as np
           from qiskit import QuantumCircuit

           circuit = QuantumCircuit(1)
           circuit.prepare_state([1/np.sqrt(2), -1/np.sqrt(2)], 0)
           circuit.draw()

        output:

        .. parsed-literal::

                 ┌─────────────────────────────────────┐
            q_0: ┤ State Preparation(0.70711,-0.70711) ├
                 └─────────────────────────────────────┘


        Prepare from a string two qubits in the state :math:`|10\rangle`.
        The order of the labels is reversed with respect to qubit index.
        More information about labels for basis states are in
        :meth:`.Statevector.from_label`.

        .. code-block::

            import numpy as np
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.prepare_state('01', circuit.qubits)
            circuit.draw()

        output:

        .. parsed-literal::

                 ┌─────────────────────────┐
            q_0: ┤0                        ├
                 │  State Preparation(0,1) │
            q_1: ┤1                        ├
                 └─────────────────────────┘


        Initialize two qubits from an array of complex amplitudes
        .. code-block::

            import numpy as np
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.prepare_state([0, 1/np.sqrt(2), -1.j/np.sqrt(2), 0], circuit.qubits)
            circuit.draw()

        output:

        .. parsed-literal::

                 ┌───────────────────────────────────────────┐
            q_0: ┤0                                          ├
                 │  State Preparation(0,0.70711,-0.70711j,0) │
            q_1: ┤1                                          ├
                 └───────────────────────────────────────────┘
    """

    if qubits is None:
        qubits = self.qubits
    elif isinstance(qubits, (int, np.integer, slice, Qubit)):
        qubits = [qubits]

    num_qubits = len(qubits) if isinstance(state, int) else None

    return self.append(
        StatePreparation(state, num_qubits, label=label, normalize=normalize), qubits
    )


QuantumCircuit.prepare_state = prepare_state
