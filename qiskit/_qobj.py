# -*- coding: utf-8 -*-
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
"""Models for QObj and its related components."""


class Qobj(object):
    """ Representation of a Quantum Object.

    Attributes:
        id_ (str): Qobj identifier.
        config (QobjConfig): config settings for the Qobj.
        circuits (list(QobjCircuit)): list of circuits.
    """
    def __init__(self, id, config, circuits=None):
        # pylint: disable=redefined-builtin,invalid-name
        self.id = id
        self.config = config
        self.circuits = circuits or []

    def as_dict(self):
        """
        Returns:
            dict: a dictionary representation of the Qobj.
        """
        return {
            'id': self.id,
            'config': self.config.as_dict(),
            'circuits': [circuit.as_dict() for circuit in self.circuits]
        }


class QobjConfig(object):
    """Configuration for a Qobj.

    Attributes:
        max_credits (int): maximum number of credits allowed (online backends
            only).
        shots (int): number of shots.
        backend (str): name of the backend.
    """
    def __init__(self, max_credits, shots, backend):
        self.max_credits = max_credits
        self.shots = shots
        self.backend = backend

    def __eq__(self, other):
        attrs = ['max_credits', 'shots', 'backend']
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in attrs)

    def as_dict(self):
        """
        Returns:
            dict: a dictionary representation of the QobjConfig.
        """
        return {
            'max_credits': self.max_credits,
            'shots': self.shots,
            'backend': self.backend
        }


class QobjCircuit(object):
    """Quantum circuit represented inside a Qobj.

    Attributes:
        name (str): circuit name.
        circuit (str): uncompiled quantum circuit.
        compiled_circuit (str): compiled quantum circuit (JSON format).
        compiled_circuit_qasm (str): compiled quantum circuit (QASM format).
        config (QobjCircuitConfig): config settings for the circuit.
    """
    def __init__(self, name, config, compiled_circuit, circuit=None,
                 compiled_circuit_qasm=None):
        self.name = name
        self.config = config
        self.compiled_circuit = compiled_circuit

        self.circuit = circuit
        self.compiled_circuit_qasm = compiled_circuit_qasm

    def as_dict(self):
        """
        Returns:
            dict: a dictionary representation of the QobjCircuit.
        """
        return {
            'name': self.name,
            'compiled_circuit': self.compiled_circuit,
            'circuit': self.circuit,
            'compiled_circuit_qasm': self.compiled_circuit_qasm,
            'config': self.config.as_dict()
        }


class QobjCircuitConfig(object):
    """Configuration for a QobjCircuit.

    Attributes:
        coupling_map (dict): adjacency list.
        basis_gates (str): comma-separated gate names.
        layout (dict): layout computed by mapper.
        seed (int): initial seed for the simulator.

    Note:
        Please note that the backends and custom applications can append
        additional attributes to the configuration via the **kwargs parameter
        of the constructor.
    """
    def __init__(self, coupling_map=None, basis_gates='u1,u2,u3,cx,id',
                 layout=None, seed=None, **kwargs):
        self.coupling_map = coupling_map
        self.basis_gates = basis_gates
        self.layout = layout
        self.seed = seed

        for key, value in kwargs.items():
            setattr(self, key, value)

    def as_dict(self):
        """
        Returns:
            dict: a dictionary representation of the QobjCircuitConfig.
        """
        # Construct the list of attribute names dynamically, as additional
        # attribute might have been set by the backends.
        attributes = [i for i in dir(self) if not i.startswith('__') and
                      not callable(getattr(self, i))]

        return {attribute: getattr(self, attribute)
                for attribute in attributes}
