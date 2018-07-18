# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for Qobj and its related components."""

from types import SimpleNamespace

from ._utils import QobjType, QobjValidationError

# Current version of the Qobj schema.
QOBJ_VERSION = '0.0.1'


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
        id (str): Qobj identifier.
        config (QobjConfig): config settings for the Qobj.
        circuits (list[QobjExperiment]): list of experiments.
        type (str): experiment type (QASM/PULSE).
        _version (str): Qobj version.
    """

    REQUIRED_ARGS = ['id', 'config', 'circuits']

    def __init__(self, id, config, circuits, **kwargs):
        # pylint: disable=redefined-builtin,invalid-name
        self.id = id
        self.config = config
        self.circuits = circuits

        self.type = QobjType.QASM.value
        self._version = QOBJ_VERSION

        super().__init__(**kwargs)


class QobjConfig(QobjItem):
    """Configuration for a Qobj.

    Attributes:
        max_credits (int): number of credits.
        shots (int): number of shots.
        backend_name (str): name of the backend.
    """
    REQUIRED_ARGS = ['max_credits', 'shots', 'backend_name']

    def __init__(self, max_credits, shots, backend_name, **kwargs):
        self.max_credits = max_credits
        self.shots = shots
        self.backend_name = backend_name

        super().__init__(**kwargs)


class QobjExperiment(QobjItem):
    """Quantum experiment represented inside a Qobj.

    Attributes:
        name (str): name of the experiment.
        config (QobjExperimentConfig): config settings for the experiment.
        compiled_circuit (QobjCompiledCircuit): list of instructions
        compiled_circuit_qasm (str)
    """
    REQUIRED_ARGS = ['name', 'config', 'compiled_circuit',
                     'compiled_circuit_qasm']

    def __init__(self, name, config, compiled_circuit, compiled_circuit_qasm,
                 **kwargs):
        self.name = name
        self.config = config
        self.compiled_circuit = compiled_circuit
        self.compiled_circuit_qasm = compiled_circuit_qasm

        super().__init__(**kwargs)


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


# TODO: Remove when new schema is in place?
class QobjExperimentConfig(QobjItem):
    """Configuration for a experiment.

    Attributes:
        seed (int): seed.
        basis_gates (str): basis gates
        coupling_map (list): coupling map
        layout (list): layout
    """
    REQUIRED_ARGS = ['seed', 'basis_gates', 'coupling_map', 'layout']

    def __init__(self, seed, basis_gates, coupling_map, layout, **kwargs):
        self.seed = seed
        self.basis_gates = basis_gates
        self.coupling_map = coupling_map
        self.layout = layout

        super().__init__(**kwargs)


# TODO: Remove when new schema is in place?
class QobjCompiledCircuit(QobjItem):
    """Compiled circuit.

    Attributes:
        header (QobjItem): header.
        operations (list[QobjInstruction]): list of instructions.
    """
    REQUIRED_ARGS = ['header', 'operations']

    def __init__(self, header, operations, **kwargs):
        self.header = header
        self.operations = operations

        super().__init__(**kwargs)
