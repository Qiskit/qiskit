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

"""
A generic opaque instruction.

OpaqueInstructions can be implementable on hardware or in simulator
(snapshot, noise, etc.).

OpaqueInstructions are identified by the following:

    name: A string to identify the type of instruction.
          Used to request a specific instruction on the backend, or in visualizing circuits.

    num_qubits, num_clbits: dimensions of the instruction.

    params: List of parameters to specialize a specific instruction instance.

OpaqueInstructions do not have any context about where they are in a circuit (which qubits/clbits).
The circuit itself keeps this context.
"""
from .instruction import Instruction


class OpaqueInstruction(Instruction):
    """Opaque instruction"""
    def validate_parameter(self, parameter):
        return parameter
