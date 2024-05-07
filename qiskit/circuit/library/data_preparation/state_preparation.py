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
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.s import SGate, SdgGate
from qiskit.circuit.library.generalized_gates import Isometry
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.states.statevector import Statevector  # pylint: disable=cyclic-import

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

        When a Statevector argument is passed the state is prepared based on the
        :class:`~.library.Isometry` synthesis described in [1].

        References:
            1. Iten et al. Quantum circuits for isometries.
               `PhysRevA.93.032318 <`https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318`>`_

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
            self.definition = self._define_synthesis_isom()

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

    def _define_synthesis_isom(self):
        """Calculate a subcircuit that implements this initialization via isometry"""
        q = QuantumRegister(self.num_qubits, "q")
        initialize_circuit = QuantumCircuit(q, name="init_def")

        isom = Isometry(self._params_arg, 0, 0)
        initialize_circuit.append(isom, q[:])

        # invert the circuit to create the desired vector from zero (assuming
        # the qubits are in the zero state)
        if self._inverse is True:
            return initialize_circuit.inverse()

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

    def inverse(self, annotated: bool = False):
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
