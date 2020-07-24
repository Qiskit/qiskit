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
import pprint
from types import SimpleNamespace

import json
import fastjsonschema

from qiskit.circuit.parameterexpression import ParameterExpression


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
                 mask=None, relation=None, val=None, snapshot_type=None):
        """Instatiate a new QasmQobjInstruction object.

        Args:
            name (str): The name of the instruction
            params (list): The list of parameters for the gate
            qubits (list): A list of ``int`` representing the qubits the
                instruction operates on
            register (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of register slots in which to
                store the measurement results (must be the same length as
                qubits). If a ``bfunc`` instruction this is a single ``int``
                of the register slot in which to store the result.
            memory (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of memory slots to store the
                measurement results in (must be the same length as qubits).
                If a ``bfunc`` instruction this is a single ``int`` of the
                memory slot to store the boolean function result in.
            condition (tuple): A tuple of the form ``(int, int)`` where the
                first ``int`` is the control register and the second ``int`` is
                the control value if the gate has a condition.
            conditional (int):  The register index of the condition
            label (str): An optional label assigned to the instruction
            mask (int): For a ``bfunc`` instruction the hex value which is
                applied as an ``AND`` to the register bits.
            relation (str): Relational  operator  for  comparing  the  masked
                register to the ``val`` kwarg. Can be either ``==`` (equals) or
                ``!=`` (not equals).
            val (int): Value to which to compare the masked register. In other
                words, the output of the function is ``(register AND mask)``
            snapshot_type (str): For snapshot instructions the type of snapshot
                to use
        """
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
        if snapshot_type is not None:
            self.snapshot_type = snapshot_type

    def to_dict(self):
        """Return a dictionary format representation of the Instruction.

        Returns:
            dict: The dictionary form of the QasmQobjInstruction.
        """
        out_dict = {'name': self.name}
        for attr in ['params', 'qubits', 'register', 'memory', '_condition',
                     'conditional', 'label', 'mask', 'relation', 'val',
                     'snapshot_type']:
            if hasattr(self, attr):
                # TODO: Remove the param type conversion when Aer understands
                # ParameterExpression type
                if attr == 'params':
                    params = []
                    for param in list(getattr(self, attr)):
                        if isinstance(param, ParameterExpression):
                            params.append(float(param))
                        else:
                            params.append(param)
                    out_dict[attr] = params
                else:
                    out_dict[attr] = getattr(self, attr)

        return out_dict

    def __repr__(self):
        out = "QasmQobjInstruction(name='%s'" % self.name
        for attr in ['params', 'qubits', 'register', 'memory', '_condition',
                     'conditional', 'label', 'mask', 'relation', 'val',
                     'snapshot_type']:
            attr_val = getattr(self, attr, None)
            if attr_val is not None:
                if isinstance(attr_val, str):
                    out += ', %s="%s"' % (attr, attr_val)
                else:
                    out += ", %s=%s" % (attr, attr_val)
        out += ')'
        return out

    def __str__(self):
        out = "Instruction: %s\n" % self.name
        for attr in ['params', 'qubits', 'register', 'memory', '_condition',
                     'conditional', 'label', 'mask', 'relation', 'val',
                     'snapshot_type']:
            if hasattr(self, attr):
                out += '\t\t%s: %s\n' % (attr, getattr(self, attr))
        return out

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

    def __repr__(self):
        instructions_str = [repr(x) for x in self.instructions]
        instructions_repr = '[' + ', '.join(instructions_str) + ']'
        out = "QasmQobjExperiment(config=%s, header=%s, instructions=%s)" % (
            repr(self.config), repr(self.header), instructions_repr)
        return out

    def __str__(self):
        out = '\nQASM Experiment:\n'
        config = pprint.pformat(self.config.to_dict())
        header = pprint.pformat(self.header.to_dict())
        out += 'Header:\n%s\n' % header
        out += 'Config:\n%s\n\n' % config
        for instruction in self.instructions:
            out += '\t%s\n' % instruction
        return out

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


class QasmQobjConfig(SimpleNamespace):
    """A configuration for a QASM Qobj."""

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
        if shots is not None:
            self.shots = int(shots)

        if max_credits is not None:
            self.max_credits = int(max_credits)

        if seed_simulator is not None:
            self.seed_simulator = int(seed_simulator)

        if memory is not None:
            self.memory = bool(memory)

        if parameter_binds is not None:
            self.parameter_binds = parameter_binds

        if memory_slots is not None:
            self.memory_slots = memory_slots

        if n_qubits is not None:
            self.n_qubits = n_qubits

        if kwargs:
            self.__dict__.update(kwargs)

    def to_dict(self):
        """Return a dictionary format representation of the QASM Qobj config.

        Returns:
            dict: The dictionary form of the QasmQobjConfig.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjConfig object from a dictionary.

        Args:
            data (dict): A dictionary for the config

        Returns:
            QasmQobjConfig: The object from the input dictionary.
        """
        return cls(**data)

    def __eq__(self, other):
        if isinstance(other, QasmQobjConfig):
            if self.to_dict() == other.to_dict():
                return True
        return False


class QobjDictField(SimpleNamespace):
    """A class used to represent a dictionary field in Qobj

    Exists as a backwards compatibility shim around a dictionary for Qobjs
    previously constructed using marshmallow.
    """

    def __init__(self, **kwargs):
        """Instantiate a new Qobj dict field object.

        Args:
            kwargs: arbitrary keyword arguments that can be accessed as
                attributes of the object.
        """
        self.__dict__.update(kwargs)

    def to_dict(self):
        """Return a dictionary format representation of the QASM Qobj.

        Returns:
            dict: The dictionary form of the QobjHeader.

        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """Create a new QobjHeader object from a dictionary.

        Args:
            data (dict): A dictionary representing the QobjHeader to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            QobjDictFieldr: The QobjDictField from the input dictionary.
        """

        return cls(**data)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.__dict__ == other.__dict__:
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
        self.type = 'QASM'
        self.schema_version = '1.2.0'

    def __repr__(self):
        experiments_str = [repr(x) for x in self.experiments]
        experiments_repr = '[' + ', '.join(experiments_str) + ']'
        out = "QasmQobj(qobj_id='%s', config=%s, experiments=%s, header=%s)" % (
            self.qobj_id, repr(self.config), experiments_repr,
            repr(self.header))
        return out

    def __str__(self):
        out = "QASM Qobj: %s:\n" % self.qobj_id
        config = pprint.pformat(self.config.to_dict())
        out += "Config: %s\n" % str(config)
        header = pprint.pformat(self.header.to_dict())
        out += "Header: %s\n" % str(header)
        out += "Experiments:\n"
        for experiment in self.experiments:
            out += "%s" % str(experiment)
        return out

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
            'schema_version': '1.2.0',
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
