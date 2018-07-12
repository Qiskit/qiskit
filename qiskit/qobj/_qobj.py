# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for Qobj and its related components."""

from types import SimpleNamespace

from . import _utils

# Current version of the Qobj schema.
QOBJ_VERSION = '0.0.2'
# Previous Qobj schema versions:
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
        """
        Return a valid representation of `obj` depending on its type.
        """
        if isinstance(obj, list):
            return [cls._expand_item(item) for item in obj]
        if isinstance(obj, QobjItem):
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
            raise _utils.QobjValidationError(
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
        id (str): Qobj identifier.
        config (QobjConfig): config settings for the Qobj.
        experiments (list[QobjExperiment]): list of experiments.
        header (QobjHeader): headers.
        type (str): experiment type (QASM/PULSE).
        _version (str): Qobj version.
    """

    REQUIRED_ARGS = ['id', 'config', 'experiments', 'header']

    def __init__(self, id, config, experiments, header, **kwargs):
        # pylint: disable=redefined-builtin,invalid-name
        self.id = id
        self.config = config
        self.experiments = experiments
        self.header = header

        self.type = _utils.QobjType.QASM.value
        self._version = QOBJ_VERSION

        super().__init__(**kwargs)


class QobjConfig(QobjItem):
    """Configuration for a Qobj.

    Attributes:
        shots (int): number of shots.
        register_slots (int): number of classical register slots.

    Attributes defined in the schema but not required:
        max_credits (int): number of credits.
        seed (int):
    """
    REQUIRED_ARGS = ['shots', 'register_slots']

    def __init__(self, shots, register_slots, **kwargs):
        self.shots = shots
        self.register_slots = register_slots

        super().__init__(**kwargs)


class QobjHeader(QobjItem):
    """Header for a Qobj.

    Attributes defined in the schema but not required:
        backend_name (str): name of the backend
        backend_version (str):
        qubit_labels (list):
        clbit_labels (list):
    """
    pass


class QobjExperiment(QobjItem):
    """Quantum experiment represented inside a Qobj.

        instructions (list[QobjInstruction)): list of instructions.

    Attributes defined in the schema but not required:
        header (QobjExperimentHeader):
        config (QobjItem):
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
