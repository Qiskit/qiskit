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

from ._utils import QObjType


class QObjStructure(object):
    """
    General QObj structure.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


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
        self.type = QObjType.PULSE.value

        super().__init__(**kwargs)

    def as_dict(self):
        """
        Returns:
            dict: a dictionary representation of the QObj.
        """
        return {
            'id': self.id,
            'config': self.config.as_dict(),
            'experiments': [experiment.as_dict() for experiment
                            in self.experiments],
            'headers': self.headers.as_dict()
        }


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

    def __eq__(self, other):
        attrs = ['max_credits', 'shots', 'backend']
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in attrs)


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
        instructions(list[QObjInstruction]): list of instructions
    """
    def __init__(self, instructions, **kwargs):
        self.instructions = instructions

        super().__init__(**kwargs)
