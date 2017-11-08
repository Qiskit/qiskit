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
    def __init__(self, id_, config, circuits=None):
        """
        Args:
            id_ (string): job id
            config (QobjConfig): config settings for the Qobj
            circuits (list(QobjCircuit)): list of circuits
        """
        self.id_ = id_
        self.config = config
        self.circuits = circuits or []


class QobjConfig(object):
    def __init__(self, max_credits, shots, backend):
        self.max_credits = max_credits
        self.shots = shots
        self.backend = backend

    def __eq__(self, other):
        attrs = ['max_credits', 'shots', 'backend']
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in attrs)


class QobjCircuit(object):
    def __init__(self, name, config, compiled_circuit, circuit=None,
                 compiled_circuit_qasm=None):
        self.name = name
        self.config = config
        self.compiled_circuit = compiled_circuit

        self.circuit = circuit
        self.compiled_circuit_qasm = compiled_circuit_qasm


class QobjCircuitConfig(object):
    def __init__(self, coupling_map=None, basis_gates='u1,u2,u3,cx,id',
                 layout=None, seed=None, **kwargs):
        self.coupling_map = coupling_map
        self.basis_gates = basis_gates
        self.layout = layout
        self.seed = seed

        for key, value in kwargs.items():
            setattr(self, key, value)
