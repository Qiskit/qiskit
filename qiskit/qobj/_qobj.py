# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for Qobj and its related components."""

from types import SimpleNamespace

import numpy
import sympy

from ._validation import QobjValidationError
from ._utils import QobjType

# Current version of the Qobj schema.
QOBJ_VERSION = '1.0.0'
# Qobj schema versions:
# * 1.0.0: Qiskit 0.6
# * 0.0.1: Qiskit 0.5.x format (pre-schemas).


class QobjItem(SimpleNamespace):
    """Generic Qobj structure.

    Single item of a Qobj structure, acting as a superclass of the rest of the
    more specific elements.
    """
    REQUIRED_ARGS = ()

    def as_dict(self):
        """
        Return a dictionary representation of the QobjItem, recursively
        converting its public attributes.
        Returns:
            dict: a dictionary.
        """
        return {key: self._expand_item(value) for key, value
                in self.__dict__.items() if not key.startswith('_')}

    @classmethod
    def _expand_item(cls, obj):
        # pylint: disable=too-many-return-statements
        """
        Return a valid representation of `obj` depending on its type.
        """
        if isinstance(obj, (list, tuple)):
            return [cls._expand_item(item) for item in obj]
        if isinstance(obj, dict):
            return {key: cls._expand_item(value) for key, value in obj.items()}
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.float):
            return float(obj)
        if isinstance(obj, sympy.Symbol):
            return str(obj)
        if isinstance(obj, sympy.Basic):
            return float(obj.evalf())
        if isinstance(obj, numpy.ndarray):
            return cls._expand_item(obj.tolist())
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        if hasattr(obj, 'as_dict'):
            return obj.as_dict()
        return obj

    @classmethod
    def from_dict(cls, obj):
        """
        Return a QobjItem from a dictionary recursively, checking for the
        required attributes.

        Returns:
            QobjItem: a new QobjItem.

        Raises:
            QobjValidationError: if the dictionary does not contain the
                required attributes for that class.
        """
        if not all(key in obj.keys() for key in cls.REQUIRED_ARGS):
            raise QobjValidationError(
                'The dict does not contain all required keys: missing "%s"' %
                [key for key in cls.REQUIRED_ARGS if key not in obj.keys()])

        return cls(**{key: cls._qobjectify_item(value)
                      for key, value in obj.items()})

    @classmethod
    def _qobjectify_item(cls, obj):
        """
        Return a valid value for a QobjItem from a object.
        """
        if isinstance(obj, dict):
            # TODO: should use the subclasses for finer control over the
            # required arguments.
            return QobjItem.from_dict(obj)
        elif isinstance(obj, list):
            return [cls._qobjectify_item(item) for item in obj]
        return obj

    def __reduce__(self):
        """
        Customize the reduction in order to allow serialization, as the Qobjs
        are automatically serialized due to the use of futures.
        """
        init_args = tuple(getattr(self, key) for key in self.REQUIRED_ARGS)
        extra_args = {key: value for key, value in self.__dict__.items()
                      if key not in self.REQUIRED_ARGS}
        return self.__class__, init_args, extra_args


class Qobj(QobjItem):
    """Representation of a Qobj.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (QobjConfig): config settings for the Qobj.
        experiments (list[QobjExperiment]): list of experiments.
        header (QobjHeader): headers.
        type (str): experiment type (QASM/PULSE).
        schema_version (str): Qobj version.
    """

    REQUIRED_ARGS = ['qobj_id', 'config', 'experiments', 'header']

    def __init__(self, qobj_id, config, experiments, header, **kwargs):
        # pylint: disable=redefined-builtin,invalid-name
        self.qobj_id = qobj_id
        self.config = config
        self.experiments = experiments
        self.header = header

        self.type = QobjType.QASM.value
        self.schema_version = QOBJ_VERSION

        super().__init__(**kwargs)


class QobjConfig(QobjItem):
    """Configuration for a Qobj.

    Attributes:
        shots (int): number of shots.
        memory_slots (int): number of measurements slots in the classical
            memory on the backend.

    Attributes defined in the schema but not required:
        max_credits (int): number of credits.
        seed (int): random seed.
    """
    REQUIRED_ARGS = ['shots', 'memory_slots']

    def __init__(self, shots, memory_slots, **kwargs):
        self.shots = shots
        self.memory_slots = memory_slots

        super().__init__(**kwargs)


class QobjHeader(QobjItem):
    """Header for a Qobj.

    Attributes defined in the schema but not required:
        backend_name (str): name of the backend
        backend_version (str): the backend version this set of experiments was generated for.
        qubit_labels (list): map physical qubits to qregs (for QASM).
        clbit_labels (list): map classical clbits to memory_slots (for QASM).
    """
    pass


class QobjExperiment(QobjItem):
    """Quantum experiment represented inside a Qobj.

        instructions (list[QobjInstruction)): list of instructions.

    Attributes defined in the schema but not required:
        header (QobjExperimentHeader): header.
        config (QobjItem): config settings for the Experiment.
    """
    REQUIRED_ARGS = ['instructions']

    def __init__(self, instructions, **kwargs):
        self.instructions = instructions

        super().__init__(**kwargs)


class QobjExperimentHeader(QobjItem):
    """Header for a Qobj.

    Attributes defined in the schema but not required:
        name (str): experiment name.
    """
    pass


class QobjInstruction(QobjItem):
    """Quantum Instruction.

    Attributes:
        name(str): name of the gate.
        qubits(list): list of qubits to apply to the gate.
    """
    REQUIRED_ARGS = ['name']

    def __init__(self, name, **kwargs):
        self.name = name

        super().__init__(**kwargs)
