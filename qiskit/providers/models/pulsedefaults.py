# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Model and schema for pulse defaults."""
import warnings

from collections import defaultdict
from marshmallow.validate import Length, Range

from qiskit.util import _to_tuple
from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.base import ObjSchema
from qiskit.validation.fields import (Integer, List, Nested, Number, String)
from qiskit.qobj import PulseLibraryItemSchema, PulseQobjInstructionSchema
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.pulse.schedule import Schedule, ParameterizedSchedule
from qiskit.pulse.exceptions import PulseError


class MeasurementKernelSchema(BaseSchema):
    """Schema for MeasurementKernel."""

    # Optional properties.
    name = String()
    params = Nested(ObjSchema)


class DiscriminatorSchema(BaseSchema):
    """Schema for Discriminator."""

    # Optional properties.
    name = String()
    params = Nested(ObjSchema)


class CommandSchema(BaseSchema):
    """Schema for Command."""

    # Required properties.
    name = String(required=True)

    # Optional properties.
    qubits = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    sequence = Nested(PulseQobjInstructionSchema, many=True)


class PulseDefaultsSchema(BaseSchema):
    """Schema for PulseDefaults."""

    # Required properties.
    qubit_freq_est = List(Number(), required=True, validate=Length(min=1))
    meas_freq_est = List(Number(), required=True, validate=Length(min=1))
    buffer = Integer(required=True, validate=Range(min=0))
    pulse_library = Nested(PulseLibraryItemSchema, required=True, many=True)
    cmd_def = Nested(CommandSchema, many=True, required=True)

    # Optional properties.
    meas_kernel = Nested(MeasurementKernelSchema)
    discriminator = Nested(DiscriminatorSchema)


@bind_schema(MeasurementKernelSchema)
class MeasurementKernel(BaseModel):
    """Model for MeasurementKernel.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``MeasurementKernelSchema``.
    """
    pass


@bind_schema(DiscriminatorSchema)
class Discriminator(BaseModel):
    """Model for Discriminator.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``DiscriminatorSchema``.
    """
    pass


@bind_schema(CommandSchema)
class Command(BaseModel):
    """Model for Command.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``CommandSchema``.

    Attributes:
        name (str): Pulse command name.
    """
    def __init__(self, name, **kwargs):
        self.name = name

        super().__init__(**kwargs)


@bind_schema(PulseDefaultsSchema)
class PulseDefaults(BaseModel):
    """Description of default settings for Pulse systems. These are operations or settings that
    may be good starting points for the Pulse user. The user may modify these defaults through
    the provided methods to build a reference to custom operations, which may in turn be used
    for building Schedules or converting circuits to Schedules.
    """

    def __init__(self, qubit_freq_est, meas_freq_est, buffer,
                 pulse_library, cmd_def, **kwargs):
        """
        Validate and reformat transport layer inputs to initialize this.

        Args:
            qubit_freq_est (list[number]): Estimated qubit frequencies in GHz.
            meas_freq_est (list[number]): Estimated measurement cavity frequencies
                in GHz.
            buffer (int): Default buffer time (in units of dt) between pulses.
            pulse_library (list[PulseLibraryItem]): Pulse name and sample definitions.
            cmd_def (list[Command]): Operation name and definition in terms of Commands.
        """
        
        super().__init__(**kwargs)
        
        self.buffer = buffer
        self._qubit_freq_est = qubit_freq_est
        self._meas_freq_est = meas_freq_est
        self._pulse_library = pulse_library
        self._cmd_def = cmd_def

        # The processed and reformatted circuit operation definitions
        self._ops_def = defaultdict(dict)
        # A backwards mapping from qubit to supported operation
        self._qubit_ops = defaultdict(list)
        # Build the above dictionaries from pulse_library and cmd_def
        converter = QobjToInstructionConverter(pulse_library, buffer)
        for op in cmd_def:
            qubits = _to_tuple(op.qubits)
            self._qubit_ops[qubits].append(op.name)
            sched = ParameterizedSchedule(*[converter(inst) for inst in op.sequence],
                                          name=op.name)
            self._ops_def[op.name][qubits] = sched

    @property
    def pulse_library(self):
        warnings.warn("Direct access to the pulse_library is being deprecated. Please use "
                      "the `replace_pulse` method to modify pulse specifications.",
                      DeprecationWarning)
        return self._pulse_library

    @property
    def cmd_def(self):
        warnings.warn("Direct access to cmd_def is being deprecated. Please use the various "
                      "operation methods (such as ops, get, and add) to modify or extract "
                      "circuit operation definitions.",
                      DeprecationWarning)
        return self._cmd_def

    def qubit_freq_est(self, qubit):
        """
        Return the estimated resonant frequency for the given qubit in Hz.

        Args:
            qubit (int): Index of the qubit of interest.
        Raises:
            PulseError: If the frequency is not available.
        Returns:
            float: The frequency of the qubit resonance in Hz.
        """
        warnings.warn("The qubit frequency estimation was previously returned in GHz, and "
                      "now is returned in Hz.")
        try:
            return self._qubit_freq_est[qubit] * 1e9
        except IndexError:
            raise PulseError("Cannot get the qubit frequency for qubit {qub}, this system only "
                             "has {num} qubits.".format(qub=qubit, num=self.n_qubits))

    def meas_freq_est(self, qubit):
        """
        Return the estimated measurement stimulus frequency to readout from the given qubit.

        Args:
            qubit (int): Index of the qubit of interest.
        Raises:
            PulseError: If the frequency is not available.
        Returns:
            float: The measurement stimulus frequency in Hz.
        """
        warnings.warn("The measurement frequency estimation was previously returned in GHz, and "
                      "now is returned in Hz.")
        try:
            return self._meas_freq_est[qubit] * 1e9
        except IndexError:
            raise PulseError("Cannot get the measurement frequency for qubit {qub}, this system "
                             "only has {num} qubits.".format(qub=qubit, num=self.n_qubits))


    def replace(self, pulse_name, samples):
        """
        Replace the named pulse with the given samples.
        Note: This will update existing operation definitions which are dependent on the
              modified pulse!

        Args:
            pulse_name (str):
            samples (list(complex)):
        Returns:
            None
        Raises:
            PulseError:
        """
        # TODO: need to lazy build schedules so we can have the following function, OR rebuild when this executes
        if not hasattr(self, '__pulse_library'):
            # Is this the right time to do this? probably?
            self.__pulse_library = {}
            for pulse in self.pulse_library:
                self.__pulse_library[pulse.name] = pulse.samples
        if pulse_name not in self.__pulse_library:
            raise PulseError("Tried to replace pulse '{}' but it is not present in the pulse "
                             "library.".format(pulse_name))
        self.__pulse_library[pulse_name] = samples
        # Need to look into either making the following faster, or modifying when scheds are built
        # or, modify get, get_parameters, and pop
        # Probably, get happens many more times, so `replace` should be the slower function.
        converter = QobjToInstructionConverter(self.__pulse_library, buffer)
        for op in cmd_def:
            # want to only do this for ops with the replaced pulse!!!!!!!!!!!
            sched = ParameterizedSchedule(*[converter(inst) for inst in op.sequence],
                                          name=op.name)
            self._ops_def[op.name][_to_tuple(op.qubits)] = sched

    @property
    def ops(self):
        """
        Return all operations which are defined by default. (This is essentially the basis gates
        along with measure and reset.)

        Returns:
            list: The names of all the circuit operations which have Schedule definitions in this.
        """
        return list(self._ops_def.keys())

    def op_qubits(self, operation):
        """
        Return a list of the qubits for which the given operation is defined. Single qubit
        operations return a flat list, and multiqubit operations return a list of tuples.

        Args:
            operation (str): The name of the circuit operation.
        Returns:
            list[Union[int, Tuple[int]]]: Qubit indices which have the given operation defined.
                This is a list of tuples if the operation has an arity greater than 1, or a flat
                list of ints otherwise.
        """
        return [qs[0] if len(qs) == 1 else qs
                for qs in sorted(self._ops_def[operation].keys())]

    def qubit_ops(self, qubits):
        """
        Return a list of the operation names that are defined by the backend for the given qubit
        or qubits.

        Args:
            qubits (Union[int, Iterable[int]]): A qubit index, or a list or tuple of indices.
        Returns:
            list[str]: All the operations which are defined on the qubits. For 1 qubit, all the 1Q
                operations defined. For multiple qubits, all the operations which apply to that
                whole set of qubits (e.g. qubits=[0, 1] may return ['cx']).
        """
        return self._qubit_ops[_to_tuple(qubits)]

    def has(self, operation, qubits):
        """
        Is the operation defined for the given qubits?

        Args:
            operation (str): The operation for which to look.
            qubits (list[Union[int, Tuple[int]]]): The specific qubits for the operation.
        Returns:
            bool: True iff the operation is defined.
        """
        return operation in self._ops_def and \
            _to_tuple(qubits) in self._ops_def[operation]

    def assert_has(self, operation, qubits):
        """
        Convenience method to check that the given operation is defined, and error if it is not.

        Args:
            operation (str): The operation for which to look.
            qubits (list[Union[int, Tuple[int]]]): The specific qubits for the operation.
        Returns:
            None
        Raises:
            PulseError: If the operation is not defined on the qubits.
        """
        if not self.has(operation, _to_tuple(qubits)):
            raise PulseError("Operation {op} for qubits {qubits} is not defined for this "
                             "system.".format(op=operation, qubits=qubits))

    def get(self,
            operation,
            qubits,
            *params,
            **kwparams):
        """
        Return the defined Schedule for the given operation on the given qubits.

        Args:
            operation (str): Name of the operation.
            qubits (list[Union[int, Tuple[int]]]): The qubits for the operation.
            *params (list[Union[int, float, complex]]): Command parameters for generating the
                                                        output schedule.
            **kwparams (Dict[str, Union[int, float, complex]]): Keyworded command parameters
                                                                for generating the schedule.
        Returns:
            Schedule: The Schedule defined for the input.

        Raises:
            PulseError: If the operation is not defined on the qubits.
        """
        self.assert_has(operation, qubits)
        sched = self._ops_def[operation].get(_to_tuple(qubits))
        if isinstance(sched, ParameterizedSchedule):
            sched = sched.bind_parameters(*params, **kwparams)
        return sched

    def get_parameters(self, operation, qubits):
        """
        Return the list of parameters taken by the given operation on the given qubits.

        Args:
            operation (str): Name of the operation.
            qubits (list[Union[int, Tuple[int]]]): The qubits for the operation.
        Returns:
            Tuple[str]: The parameters required by the operation.

        Raises:
            PulseError: If the operation is not defined on the qubits.
        """
        self.assert_has(operation, qubits)
        return self._ops_def[operation][_to_tuple(qubits)].parameters

    def add(self, operation, qubits, schedule):
        """
        Add a new known operation.

        Args:
            operation (str): The name of the operation to add.
            qubits (list[Union[int, Tuple[int]]]): The qubits which the operation applies to.
            schedule (Schedule): The Schedule that implements the given operation.
        Returns:
            None
        Raises:
            PulseError: If the qubits are provided as an empty iterable.
        """
        qubits = _to_tuple(qubits)
        if qubits == ():
            raise PulseError("Cannot add definition {} with no target qubits.".format(operation))
        if not (isinstance(schedule, Schedule) or isinstance(schedule, ParameterizedSchedule)):
            raise PulseError("Attemping to add an invalid schedule type.")
        self._ops_def[operation][qubits] = schedule

    def remove(self, operation, qubits):
        """Remove the given operation from the defined operations.

        Args:
            operation (str): The name of the operation to add.
            qubits (list[Union[int, Tuple[int]]]): The qubits which the operation applies to.
        Returns:
            None
        Raises:
            PulseError: If the operation is not present.
        """
        self.assert_has(operation, qubits)
        self._ops_def[operation].pop(_to_tuple(qubits))

    def pop(self,
            operation,
            qubits,
            *params,
            **kwparams):
        """
        Remove and return the defined Schedule for the given operation on the given qubits.

        Args:
            operation (str): Name of the operation.
            qubits (list[Union[int, Tuple[int]]]): The qubits for the operation.
            *params (list[Union[int, float, complex]]): Command parameters for generating the
                                                        output schedule.
            **kwparams (Dict[str, Union[int, float, complex]]): Keyworded command parameters
                                                                for generating the schedule.
        Returns:
            Schedule: The Schedule defined for the input.

        Raises:
            PulseError: If command for qubits is not available
        """
        self.assert_has(operation, qubits)
        sched = self._ops_def[operation].pop(_to_tuple(qubits))
        if isinstance(schedule, ParameterizedSchedule):
            return sched.bind_parameters(*params, **kwparams)
        return sched

    def __repr__(self):
        single_qops = "1Q operations:\n"
        multi_qops = "Multi qubit operations:\n"
        for qubits, ops in self._qubit_ops.items():
            if len(qubits) == 1:
                single_qops += "  q{qubit}: {ops}\n".format(qubit=qubits[0], ops=ops)
            else:
                multi_qops += "  {qubits}: {ops}\n".format(qubits=qubits, ops=ops)
        ops = single_qops + multi_qops
        qfreq = "Qubit Frequencies [GHz]\n{freqs}".format(freqs=self._qubit_freq_est)
        mfreq = "Measurement Frequencies [GHz]\n{freqs} ".format(freqs=self._meas_freq_est)
        return ("<{name}({ops}{delim}\n{qfreq}\n{mfreq})\n>"
                "".format(name=self.__class__.__name__,
                          ops=ops,
                          delim="_" * 80,
                          qfreq=qfreq,
                          mfreq=mfreq))
