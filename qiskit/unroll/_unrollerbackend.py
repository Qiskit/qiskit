# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Base backend object for the unroller that raises BackendError.
"""
from ._backenderror import BackendError


class UnrollerBackend(object):
    """Backend for the unroller that raises BackendError.

    This backend also serves as a base class for other unroller backends.
    """
    # pylint: disable=unused-argument

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        if basis:
            basis = []

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit.

        basis is a list of operation name strings.
        """
        raise BackendError("Backend set_basis unimplemented")

    def version(self, version):
        """Print the version string.

        v is a version number.
        """
        raise BackendError("Backend version unimplemented")

    def new_qreg(self, name, size):
        """Create a new quantum register.

        name = name of the register
        sz = size of the register
        """
        raise BackendError("Backend new_qreg unimplemented")

    def new_creg(self, name, size):
        """Create a new classical register.

        name = name of the register
        sz = size of the register
        """
        raise BackendError("Backend new_creg unimplemented")

    def define_gate(self, name, gatedata):
        """Define a new quantum gate.

        name is a string.
        gatedata is the AST node for the gate.
        """
        raise BackendError("Backend define_gate unimplemented")

    def u(self, arg, qubit, nested_scope=None):
        """Fundamental single qubit gate.

        arg is 3-tuple of Node expression objects.
        qubit is (regname,idx) tuple.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        # pylint: disable=invalid-name
        raise BackendError("Backend u unimplemented")

    def cx(self, qubit0, qubit1):
        """Fundamental two qubit gate.

        qubit0 is (regname,idx) tuple for the control qubit.
        qubit1 is (regname,idx) tuple for the target qubit.
        """
        # pylint: disable=invalid-name
        raise BackendError("Backend cx unimplemented")

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        raise BackendError("Backend measure unimplemented")

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        raise BackendError("Backend barrier unimplemented")

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        raise BackendError("Backend reset unimplemented")

    def set_condition(self, creg, cval):
        """Attach a current condition.

        creg is a name string.
        cval is the integer value for the test.
        """
        raise BackendError("Backend set_condition unimplemented")

    def drop_condition(self):
        """Drop the current condition."""
        raise BackendError("Backend drop_condition unimplemented")

    def start_gate(self, name, args, qubits, nested_scope=None):
        """Begin a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        raise BackendError("Backend start_gate unimplemented")

    def end_gate(self, name, args, qubits, nested_scope=None):
        """End a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        raise BackendError("Backend end_gate unimplemented")

    def get_output(self):
        """Returns the output generated by the backend.
        Depending on the type of Backend, the output could have different types.
        It must be called once the Qasm parsing has finished
        """
        raise BackendError("Backend get_output unimplemented")
