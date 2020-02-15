# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Module providing definitions of QASM Qobj classes."""

import os

import json
import fastjsonschema


path_part = 'schemas/qobj_schema.json'
path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    path_part)
with open(path, 'r') as fd:
    json_schema = json.loads(fd.read())
validator = fastjsonschema.compile(json_schema)


class QasmQobjInstruction:
    """A class representing a single instruction in an QasmQobj Experiment."""

    def __init__(self, name, params=None, qubits=None, register=None,
                 memory=None, condition=None, conditional=None, label=None,
                 mask=None, relation=None, val=None):
        """Instatiate a new QasmQobjInstruction object.

        Args:
            name (str): The name of the instruction
            params (list): The list of parameters for the gate
            qubits (list): A list of `int`s representing the qubits the
                instruction operates on
            register (list): If a ``measure`` instruction this is a list
                of `int`s containing the list of register slots in which to
                store the measurement results (must be the same length as
                qubits). If a ``bfunc`` instruction this is a single `int`
                of the register slot in which to store the result.
            memory (list): If a ``measure`` instruction this is a list
                of `int`s containing the list of memory slots to store the
                measurement results in (must be the same length as qubits).
                If a ``bfunc`` instruction this is a single `int` of the
                memory slot to store the boolean function result in.
            condition (tuple): A tuple of the form ``(int, int)`` where the
                first `int` is the control register and the second `int` is
                the control value if the gate has a condition.
            conditional (int):  The register index of the condition
            label (str): An optional label assigned to the instruction
            mask (int): For a ``bfunc`` instruction the hex value which is
                applied as an ``AND`` to the register bits.
            relation (str): Relational  operator  for  comparing  the  masked
                register to the `val` kwarg. Can be either ``==`` (equals) or
                ``!=`` (not equals).
            val (int): Value to which to compare the masked register. In other
                words, the output of the function is (``register AND mask)
                relation val.
        """
        super(QasmQobjInstruction, self).__init__()
        self.name = name
        if params is not None:
            self.params = params
        if qubits is not None:
            self.qubits = qubits
        if register is not None:
            self.register = register
        if memory is not None:
            self.memory = memory
        if condition is not None:
            self._condition = condition
        if conditional is not None:
            self.conditional = conditional
        if label is not None:
            self.label = label
        if mask is not None:
            self.mask = mask
        if relation is not None:
            self.relation = relation
        if val is not None:
            self.val = val

    def to_dict(self):
        """Return a dictionary format representation of the Instruction.

        Returns:
            dict: The dictionary form of the QasmQobjInstruction.
        """
        out_dict = {'name': self.name}
        for attr in ['params', 'qubits', 'register', 'memory', '_condition',
                     'conditional', 'label', 'mask', 'relation', 'val']:
            if hasattr(self, attr):
                out_dict[attr] = getattr(self, attr)

        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjInstruction object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            QasmQobjInstruction: The object from the input dictionary.
        """
        name = data.pop('name')
        return cls(name, **data)

    def __eq__(self, other):
        if isinstance(other, QasmQobjInstruction):
            if self.to_dict() == other.to_dict():
                return True
        return False


class QasmQobjExperiment:
    """A QASM Qobj Experiment.

    Each instance of this class is used to represent a QASM experiment as
    part of a larger QASM qobj.
    """

    def __init__(self, config=None, header=None, instructions=None):
        """Instatiate a QasmQobjExperiment.

        Args:
            config (QasmQobjExperimentConfig): A config object for the experiment
            header (QasmQobjExperimentHeader): A header object for the experiment
            instructions (list): A list of :class:`QasmQobjInstruction` objects
        """
        self.config = config or QasmQobjExperimentConfig()
        self.header = header or QasmQobjExperimentHeader()
        self.instructions = instructions or []

    def to_dict(self):
        """Return a dictionary format representation of the Experiment.

        Returns:
            dict: The dictionary form of the QasmQObjExperiment.
        """
        out_dict = {
            'config': self.config.to_dict(),
            'header': self.header.to_dict(),
            'instructions': [x.to_dict() for x in self.instructions]
        }
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjExperiment object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            QasmQobjExperiment: The object from the input dictionary.
        """
        config = None
        if 'config' in data:
            config = QasmQobjExperimentConfig.from_dict(data.pop('config'))
        header = None
        if 'header' in data:
            header = QasmQobjExperimentHeader.from_dict(data.pop('header'))
        instructions = None
        if 'instructions' in data:
            instructions = [
                QasmQobjInstruction.from_dict(
                    inst) for inst in data.pop('instructions')]
        return cls(config, header, instructions)

    def __eq__(self, other):
        if isinstance(other, QasmQobjExperiment):
            if self.to_dict() == other.to_dict():
                return True
        return False


class QasmQobjConfig:
    """A configuration for a QASM Qobj."""

    _data = {}

    def __init__(self, shots=None, max_credits=None, seed_simulator=None,
                 memory=None, parameter_binds=None, memory_slots=None,
                 n_qubits=None, **kwargs):
        """Model for RunConfig.

        Args:
            shots (int): the number of shots.
            max_credits (int): the max_credits to use on the IBMQ public devices.
            seed_simulator (int): the seed to use in the simulator
            memory (bool): whether to request memory from backend (per-shot readouts)
            parameter_binds (list[dict]): List of parameter bindings
            memory_slots (int): The number of memory slots on the device
            n_qubits (int): The number of qubits on the device
            kwargs: Additional free form key value fields to add to the
                configuration.
        """
        self._data = {}
        if shots is not None:
            self._data['shots'] = int(shots)

        if max_credits is not None:
            self._data['max_credits'] = int(max_credits)

        if seed_simulator is not None:
            self._data['seed_simulator'] = int(seed_simulator)

        if memory is not None:
            self._data['memory'] = bool(memory)

        if parameter_binds is not None:
            self._data['parameter_binds'] = parameter_binds

        if memory_slots is not None:
            self._data['memory_slots'] = memory_slots

        if n_qubits is not None:
            self._data['n_qubits'] = n_qubits

        if kwargs:
            self._data.update(kwargs)

    @property
    def shots(self):
        """The number of shots to run for each experiment."""
        return self._data.get('shots')

    @shots.setter
    def shots(self, value):
        self._data['shots'] = value

    @property
    def max_credits(self):
        """The max number of credits to use on the IBMQ public devices."""
        return self._data.get('max_credits')

    @max_credits.setter
    def max_credits(self, value):
        self._data['max_credits'] = value

    @property
    def memory(self):
        """Whether to request memory from backend (per-shot readouts)."""
        return self._data.get('memory')

    @memory.setter
    def memory(self, value):
        self._data['memory'] = value

    @property
    def parameter_binds(self):
        """List of parameter bindings."""
        return self._data.get('parameter_binds')

    @parameter_binds.setter
    def parameter_binds(self, value):
        self._data['parameter_binds'] = value

    @property
    def memory_slots(self):
        """The number of memory slots on the device."""
        self._data.get('memory_slots')

    @memory_slots.setter
    def memory_slots(self, value):
        self._data['memory_slots'] = value

    @property
    def n_qubits(self):
        """The number of qubits on the device."""
        nqubits = self._data.get('n_qubits')
        if nqubits is None:
            raise AttributeError('Attribute n_qubits is not defined')
        return nqubits

    @n_qubits.setter
    def n_qubits(self, value):
        self._data['n_qubits'] = value

    def to_dict(self):
        """Return a dictionary format representation of the QASM Qobj config.

        Returns:
            dict: The dictionary form of the QasmQobjConfig.
        """
        return self._data

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjConfig object from a dictionary.

        Args:
            data (dict): A dictionary for the config

        Returns:
            QasmQobjConfig: The object from the input dictionary.
        """
        return cls(**data)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError('Attribute %s is not defined' % name)

    def __eq__(self, other):
        if isinstance(other, QasmQobjConfig):
            if self.to_dict() == other.to_dict():
                return True
        return False


class QobjDictField:
    """A class used to represent a dictionary field in Qobj

    Exists as a backwards compatibility shim around a dictionary for Qobjs
    previously constructed using marshmallow.
    """

    _data = {}

    def __init__(self, **kwargs):
        """Instantiate a new Qobj dict field object.

        Args:
            kwargs: arbitrary keyword arguments that can be accessed as
                attributes of the object.
        """
        self._data = kwargs

    def __getstate__(self):
        return self._data

    def __setstate__(self, state):
        self._data = state

    def __getattr__(self, attr):
        try:
            return self._data[attr]
        except KeyError:
            raise AttributeError('Attribute %s is not defined' % attr)

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            self._data[name] = value
        else:
            super().__setattr__(name, value)

    def to_dict(self):
        """Return a dictionary format representation of the QASM Qobj.

        Returns:
            dict: The dictionary form of the QobjHeader.

        """
        return self._data

    @classmethod
    def from_dict(cls, data):
        """Create a new QobjHeader object from a dictionary.

        Args:
            data (dict): A dictionary representing the QobjHeader to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            QobjHeader: The QobjHeader from the input dictionary.
        """

        return cls(**data)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.to_dict() == other.to_dict():
                return True
        return False


class QasmQobjExperimentHeader(QobjDictField):
    """A header for a single QASM experiment in the qobj."""
    pass


class QasmQobjExperimentConfig(QobjDictField):
    """Configuration for a single QASM experiment in the qobj."""
    pass


class QobjHeader(QobjDictField):
    """A class used to represent a dictionary header in Qobj objects."""
    pass


class QobjExperimentHeader(QobjHeader):
    """A class representing a header dictionary for a Qobj Experiment."""
    pass


class QasmQobj:
    """A QASM Qobj."""

    def __init__(self, qobj_id=None, config=None, experiments=None,
                 header=None):
        """Instatiate a new QASM Qobj Object.

        Each QASM Qobj object is used to represent a single payload that will
        be passed to a Qiskit provider. It mirrors the Qobj the published
        `Qobj specification <https://arxiv.org/abs/1809.03452>`_ for OpenQASM
        experiments.

        Args:
            qobj_id (str): An identifier for the qobj
            config (QasmQobjRunConfig): A config for the entire run
            header (QobjHeader): A header for the entire run
            experiments (list): A list of lists of :class:`QasmQobjExperiment`
                objects representing an experiment
        """
        self.header = header or QobjHeader()
        self.config = config or QasmQobjConfig()
        self.experiments = experiments or []
        self.qobj_id = qobj_id

    def to_dict(self, validate=False):
        """Return a dictionary format representation of the QASM Qobj.

        Note this dict is not in the json wire format expected by IBMQ and qobj
        specification because complex numbers are still of type complex. Also
        this may contain native numpy arrays. When serializing this output
        for use with IBMQ you can leverage a json encoder that converts these
        as expected. For example:

        .. code-block::

            import json
            import numpy

            class QobjEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, numpy.ndarray):
                        return obj.tolist()
                    if isinstance(obj, complex):
                        return (obj.real, obj.imag)
                    return json.JSONEncoder.default(self, obj)

            json.dumps(qobj.to_dict(), cls=QobjEncoder)


        Args:
            validate (bool): When set to true validate the output dictionary
                against the jsonschema for qobj spec.

        Returns:
            dict: A dictionary representation of the QasmQobj object
        """
        out_dict = {
            'qobj_id': self.qobj_id,
            'header': self.header.to_dict(),
            'config': self.config.to_dict(),
            'schema_version': '1.1.0',
            'type': 'QASM',
            'experiments': [x.to_dict() for x in self.experiments]
        }
        if validate:
            validator(out_dict)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QASMQobj object from a dictionary.

        Args:
            data (dict): A dictionary representing the QasmQobj to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            QasmQobj: The QasmQobj from the input dictionary.
        """
        config = None
        if 'config' in data:
            config = QasmQobjConfig.from_dict(data['config'])
        experiments = None
        if 'experiments' in data:
            experiments = [
                QasmQobjExperiment.from_dict(
                    exp) for exp in data['experiments']]
        header = None
        if 'header' in data:
            header = QobjHeader.from_dict(data['header'])

        return cls(qobj_id=data.get('qobj_id'), config=config,
                   experiments=experiments, header=header)

    def __eq__(self, other):
        if isinstance(other, QasmQobj):
            if self.to_dict() == other.to_dict():
                return True
        return False
