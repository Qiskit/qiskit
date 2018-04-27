# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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

"""Models for QObj and its related components."""

from types import SimpleNamespace

from ._utils import QObjType


class QObjStructure(SimpleNamespace):
    """
    General QObj structure.
    """
    def as_dict(self):
        """Return a dictionary representation of the QObjStructure, recursively
        converting its public attributes.

        Returns:
            dict: a dictionary.
        """
        def expand_item(obj):
            """
            Return a valid representation of `obj` depending on its type.
            """
            if isinstance(obj, list):
                return [expand_item(item) for item in obj]
            if isinstance(obj, QObjStructure):
                return obj.as_dict()
            return obj

        return {key: expand_item(value) for key, value
                in self.__dict__.items() if not key.startswith('_')}


class QObj(QObjStructure):
    """Representation of a QObj.

    Attributes:
        id (str): QObj identifier.
        config (QObjConfig): config settings for the QObj.
        experiments (list[QObjExperiment]): list of experiments.
        headers (QObjStructure): headers.
        type (str): experiment type (QASM/PULSE).
    """
    def __init__(self, id, config, experiments, headers, **kwargs):
        # pylint: disable=redefined-builtin,invalid-name
        self.id = id
        self.config = config
        self.experiments = experiments
        self.headers = headers
        self.type = QObjType.QASM.value

        super().__init__(**kwargs)


class QObjConfig(QObjStructure):
    """Configuration for a QObj.

    Attributes:
        shots (int): number of shots.
        register_slots (int): number of classical register slots.
    """
    def __init__(self, shots, register_slots, **kwargs):
        self.shots = shots
        self.register_slots = register_slots

        super().__init__(**kwargs)


class QObjExperiment(QObjStructure):
    """Quantum experiment represented inside a QObj.

    Attributes:
        instructions(list[QObjInstruction]): list of instructions
    """
    def __init__(self, instructions, **kwargs):
        self.instructions = instructions

        super().__init__(**kwargs)


class QObjInstruction(QObjStructure):
    """Quantum Instruction.

    Attributes:
        name(str): name of the gate.
        qubits(list): list of qubits to apply to the gate.
        params(list): list of parameters for the gate.
    """
    def __init__(self, name, qubits, params, **kwargs):
        self.name = name
        self.qubits = qubits
        self.params = params

        super().__init__(**kwargs)
