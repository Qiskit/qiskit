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
QISKit extensions for local_qiskit_simulator gates
"""

# =============================================================================
# Add custom gate extensions to qiskit
# =============================================================================

try:
    from qiskit import QuantumCircuit
    from qiskit import Gate
    from qiskit import CompositeGate
    from qiskit._quantumregister import QuantumRegister
    from qiskit._instructionset import InstructionSet

    # Custom Gate Classes

    class U0Gate(Gate):
        """Wait gate."""

        def __init__(self, m, qubit, circ=None):
            """Create new u0 gate."""
            super(U0Gate, self).__init__("u0", [m], [qubit], circ)

        def qasm(self):
            """Return OPENQASM string."""
            qubit = self.arg[0]
            m = self.param[0]
            return self._qasmif("u0(%f) %s[%d];" % (m,
                                                    qubit[0].name,
                                                    qubit[1]))

        def inverse(self):
            """Invert this gate."""
            return self  # self-inverse

        def reapply(self, circ):
            """Reapply this gate to corresponding qubits in circ."""
            self._modifiers(circ.u0(self.param[0], self.arg[0]))

    class WaitGate(Gate):
        """Wait gate."""

        def __init__(self, t, qubit, circ=None):
            """Create new wait gate."""
            super(WaitGate, self).__init__("wait", [t], [qubit], circ)

        def qasm(self):
            """Return OPENQASM string."""
            qubit = self.arg[0]
            t = self.param[0]
            return self._qasmif("wait(%f) %s[%d];" % (t,
                                                      qubit[0].name,
                                                      qubit[1]))

        def inverse(self):
            """Invert this gate."""
            return self  # self-inverse

        def reapply(self, circ):
            """Reapply this gate to corresponding qubits in circ."""
            self._modifiers(circ.wait(self.param[0], self.arg[0]))

    class SaveGate(Gate):
        """Simulator save operation."""

        def __init__(self, m, qubit, circ=None):
            """Create new save gate."""
            super(SaveGate, self).__init__("save", [m], [qubit], circ)

        def qasm(self):
            """Return OPENQASM string."""
            qubit = self.arg[0]
            m = self.param[0]
            return self._qasmif("save(%d) %s[%d];" % (m,
                                                      qubit[0].name,
                                                      qubit[1]))

        def inverse(self):
            """Invert this gate."""
            return self  # self-inverse

        def reapply(self, circ):
            """Reapply this gate to corresponding qubits in circ."""
            self._modifiers(circ.save(self.param[0], self.arg[0]))

    class LoadGate(Gate):
        """Simulator load operation."""

        def __init__(self, m, qubit, circ=None):
            """Create new load gate."""
            super(LoadGate, self).__init__("load", [m], [qubit], circ)

        def qasm(self):
            """Return OPENQASM string."""
            qubit = self.arg[0]
            m = self.param[0]
            return self._qasmif("load(%d) %s[%d];" % (m, qubit[0].name,
                                                      qubit[1]))

        def inverse(self):
            """Invert this gate."""
            return self  # self-inverse

        def reapply(self, circ):
            """Reapply this gate to corresponding qubits in circ."""
            self._modifiers(circ.load(self.param[0], self.arg[0]))

    class NoiseGate(Gate):
        """Simulator save operation."""

        def __init__(self, m, qubit, circ=None):
            """Create new save gate."""
            super(NoiseGate, self).__init__("noise", [m], [qubit], circ)

        def qasm(self):
            """Return OPENQASM string."""
            qubit = self.arg[0]
            m = self.param[0]
            return self._qasmif("noise(%d) %s[%d];" % (m,
                                                       qubit[0].name,
                                                       qubit[1]))

        def inverse(self):
            """Invert this gate."""
            return self  # self-inverse

        def reapply(self, circ):
            """Reapply this gate to corresponding qubits in circ."""
            self._modifiers(circ.noise(self.param[0], self.arg[0]))

    class UZZGate(Gate):
        """controlled-Z gate."""

        def __init__(self, theta, ctl, tgt, circ=None):
            """Create new uzz gate."""
            super(UZZGate, self).__init__("uzz", [theta], [ctl, tgt], circ)

        def qasm(self):
            """Return OPENQASM string."""
            ctl = self.arg[0]
            tgt = self.arg[1]
            theta = self.param[0]
            return self._qasmif("izz %s[%d],%s[%d];" % (theta,
                                                        ctl[0].name, ctl[1],
                                                        tgt[0].name, tgt[1]))

        def inverse(self):
            """Invert this gate."""
            self.param[0] = -self.param[0]
            return self

        def reapply(self, circ):
            """Reapply this gate to corresponding qubits in circ."""
            self._modifiers(circ.uzz(self.param[0], self.arg[0], self.arg[1]))

    # Custom gate functions

    def u0(self, m, q):
        """Apply u0 with length m to q."""
        if isinstance(q, QuantumRegister):
            gs = InstructionSet()
            for j in range(q.size):
                gs.add(self.u0(m, (q, j)))
            return gs
        else:
            self._check_qubit(q)
            return self._attach(U0Gate(m, q, self))

    def wait(self, t, q):
        """Apply wait for time t to q."""
        if isinstance(q, QuantumRegister):
            gs = InstructionSet()
            for j in range(q.size):
                gs.add(self.wait(t, (q, j)))
            return gs
        else:
            self._check_qubit(q)
            return self._attach(WaitGate(t, q, self))

    def save(self, m, q):
        """Cache the quantum state of local_qiskit_simulator."""
        if isinstance(q, QuantumRegister):
            gs = InstructionSet()
            for j in range(q.size):
                gs.add(self.save(m, (q, j)))
            return gs
        else:
            self._check_qubit(q)
            return self._attach(SaveGate(m, q, self))

    def load(self, m, q):
        """Load cached quantum state of local_qiskit_simulator."""
        if isinstance(q, QuantumRegister):
            gs = InstructionSet()
            for j in range(q.size):
                gs.add(self.load(m, (q, j)))
            return gs
        else:
            self._check_qubit(q)
            return self._attach(LoadGate(m, q, self))

    def noise(self, m, q):
        """Cache the quantum state of locla_qiskit_simulator."""
        if isinstance(q, QuantumRegister):
            gs = InstructionSet()
            for j in range(q.size):
                gs.add(self.noise(m, (q, j)))
            return gs
        else:
            self._check_qubit(q)
            return self._attach(NoiseGate(m, q, self))

    def uzz(self, theta, ctl, tgt):
        """Apply CZ to circuit."""
        if isinstance(ctl, QuantumRegister) and \
                isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
            # apply cx to qubits between two registers
            instructions = InstructionSet()
            for i in range(ctl.size):
                instructions.add(self.uzz(theta, (ctl, i), (tgt, i)))
            return instructions
        else:
            self._check_qubit(ctl)
            self._check_qubit(tgt)
            self._check_dups([ctl, tgt])
            return self._attach(UZZGate(theta, ctl, tgt, self))

    def attach_local_qiskit_simulator_gates():
        """Add custom local_qiskit_simulator gates to qiskit."""

        # Add to QuantumCircuit class
        QuantumCircuit.u0 = u0
        QuantumCircuit.wait = wait
        QuantumCircuit.save = save
        QuantumCircuit.load = load
        QuantumCircuit.noise = noise
        QuantumCircuit.uzz = uzz

        # Add to CompositeGate class
        CompositeGate.u0 = u0
        CompositeGate.wait = wait
        CompositeGate.save = save
        CompositeGate.load = load
        CompositeGate.noise = noise
        CompositeGate.uzz = uzz

        # Add to QASM header for parsing
        QuantumCircuit.header += \
            "\n// local_qiskit_simulator gates:" + \
            "\ngate wait(t) a {}  // idle for time t" + \
            "\ngate save(m) a {}  // cache quantum state" + \
            "\ngate load(m) a {}  // load cached quantum state" + \
            "\ngate noise(m) a {} // switch noise off (0) or on (1)" + \
            "\ngate uzz(theta) a, b {}  // Uzz rotation by angle theta"

except:
    pass
