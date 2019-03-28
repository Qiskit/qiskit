# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""The pulse qobj models."""

from marshmallow.validate import Range, Regexp, Length, OneOf

from qiskit.qobj.utils import MeasReturnType
from qiskit.validation import bind_schema, BaseSchema, BaseModel
from qiskit.validation.fields import (Integer, String, Number, Complex, Boolean,
                                      List, Nested, DictParameters)
from .base import (QobjInstructionSchema, QobjExperimentConfigSchema, QobjExperimentSchema,
                   QobjConfigSchema, QobjInstruction, QobjExperimentConfig,
                   QobjExperiment, QobjConfig)


class QobjMeasurementOptionSchema(BaseSchema):
    """Schema for QobjMeasOptiton."""

    # Required properties.
    name = String(required=True)
    params = DictParameters(valid_value_types=(int, float, str, bool, type(None)),
                            required=True)


class QobjPulseLibrarySchema(BaseSchema):
    """Schema for QobjPulseLibrary."""

    # Required properties.
    name = String(required=True)
    samples = List(Complex(), required=True, validate=Length(min=1))


class QobjHamiltonianSchema(BaseSchema):
    """Schema for QobjHamiltonian."""
    # pylint: disable=invalid-name

    # Required properties.
    h_latex = String(required=True)
    h_str = List(String(), required=True, validate=Length(min=1))
    vars = DictParameters(valid_value_types=(int, float, complex),
                          required=True)
    osc = DictParameters(valid_value_types=(int, ),
                         required=True)


class QobjOscillatorNoiseSchema(BaseSchema):
    """Schema for QobjOscillatorNoise."""

    # Required properties.
    n_th = DictParameters(valid_value_types=(int, float),
                          required=True)
    coupling = DictParameters(valid_value_types=(int, float),
                              required=True)


class QobjNoiseSchema(BaseSchema):
    """Schema for QobjNoise."""

    # Required properties.
    qubit = DictParameters(valid_value_types=(int, ),
                           required=True)
    oscillator = Nested(QobjOscillatorNoiseSchema,
                        required=True)


class QobjOdeOptionsSchema(BaseSchema):
    """Schema for OdeOptions."""

    # Optional properties.
    atol = Number()
    rtol = Number()
    nsteps = Integer(validate=Range(min=0))
    max_step = Integer(validate=Range(min=0))
    num_cpus = Integer(validate=Range(min=0))
    norm_tol = Number()
    norm_steps = Integer(validate=Range(min=0))
    rhs_reuse = Boolean()
    rhs_filename = String()


class PulseQobjInstructionSchema(QobjInstructionSchema):
    """Schema for PulseQobjInstruction."""
    # pylint: disable=invalid-name

    # Required properties
    t0 = Integer(required=True, validate=Range(min=0))

    # Optional properties.
    ch = String(validate=Regexp('[dum]([0-9])+'))
    conditional = Integer(validate=Range(min=0))
    phase = Number()
    val = Complex()
    duration = Integer(validate=Range(min=1))
    qubits = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    memory_slot = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    register_slot = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    kernels = Nested(QobjMeasurementOptionSchema, many=True)
    discriminators = Nested(QobjMeasurementOptionSchema, many=True)
    label = String()
    type = String()


class PulseQobjExperimentConfigSchema(QobjExperimentConfigSchema):
    """Schema for PulseQobjExperimentConfig."""

    # Optional properties.
    qubit_lo_freq = List(Number())
    meas_lo_freq = List(Number())


class PulseQobjExperimentSchema(QobjExperimentSchema):
    """Schema for PulseQobjExperiment."""

    # Required properties.
    instructions = Nested(PulseQobjInstructionSchema, required=True, many=True,
                          validate=Length(min=1))

    # Optional properties.
    config = Nested(PulseQobjExperimentConfigSchema)


class PulseQobjDevConfigSchema(QobjConfigSchema):
    """Schema for PulseQobjDevConfig of device backend."""

    # Required properties.
    meas_level = Integer(required=True, validate=Range(min=0, max=2))
    memory_slot_size = Integer(required=True)
    pulse_library = Nested(QobjPulseLibrarySchema, many=True, required=True)
    qubit_lo_freq = List(Number(), required=True)
    meas_lo_freq = List(Number(), required=True)
    rep_time = Integer(required=True)
    meas_return = String(validate=OneOf(choices=(MeasReturnType.AVERAGE,
                                                 MeasReturnType.SINGLE)))


class PulseQobjSimConfigSchema(BaseSchema):
    """Schema for PulseQobjDevConfig of simulator."""
    # pylint: disable=invalid-name

    # Requested properties.
    pulse_library = Nested(QobjPulseLibrarySchema, many=True, required=True)
    qubit_lo_freq = List(Number(), required=True)
    meas_lo_freq = List(Number(), required=True)
    hamiltonian = Nested(QobjHamiltonianSchema, required=True)
    dt = Number(required=True)

    # Optional properties.
    noise = Nested(QobjNoiseSchema)
    ode_options = Nested(QobjOdeOptionsSchema)


@bind_schema(QobjMeasurementOptionSchema)
class QobjMeasurementOption(BaseModel):
    """Model for QobjMeasurementOption.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjMeasurementOptionSchema``.

    Attributes:
        name (str): name of option specified in the backend
        params (dict): measurement parameter
    """
    def __init__(self, name, params, **kwargs):
        self.name = name
        self.params = params

        super().__init__(**kwargs)


@bind_schema(QobjPulseLibrarySchema)
class QobjPulseLibrary(BaseModel):
    """Model for QobjPulseLibrary.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjPulseLibrarySchema``.

    Attributes:
        name (str): name of pulse
        samples (list[complex]]): list of complex values defining pulse shape
    """

    def __init__(self, name, samples, **kwargs):
        self.name = name
        self.samples = samples

        super().__init__(**kwargs)


@bind_schema(PulseQobjInstructionSchema)
class PulseQobjInstruction(QobjInstruction):
    """Model for PulseQobjInstruction inherit from QobjInstruction.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjInstructionSchema``.

    Attributes:
        name (str): name of the instruction
        t0 (int): timing of executing the instruction
    """
    def __init__(self, name, t0, **kwargs):
        # pylint: disable=invalid-name
        self.t0 = t0

        super().__init__(name=name,
                         t0=t0,
                         **kwargs)


@bind_schema(PulseQobjExperimentConfigSchema)
class PulseQobjExperimentConfig(QobjExperimentConfig):
    """Model for PulseQobjExperimentConfig inherit from QobjExperimentConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjExperimentConfigSchema``.
    """
    pass


@bind_schema(PulseQobjExperimentSchema)
class PulseQobjExperiment(QobjExperiment):
    """Model for PulseQobjExperiment inherit from QobjExperiment.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjExperimentSchema``.

    Attributes:
        instructions (list[PulseQobjInstruction]): list of instructions.
    """
    def __init__(self, instructions, **kwargs):

        super().__init__(instructions=instructions,
                         **kwargs)


@bind_schema(PulseQobjDevConfigSchema)
class PulseQobjDevConfig(QobjConfig):
    """Model for PulseQobjDevConfig inherit from QobjConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjDevConfigSchema``.

    Attributes:
        meas_level (int): a value represents the level of measurement.
        memory_slot_size (int): size of memory slot
        meas_return (str): a level of measurement information.
        pulse_library (list[QobjPulseLibrary]): a pulse library.
        qubit_lo_freq (list): the list of frequencies for qubit drive LO's in GHz.
        meas_lo_freq (list): the list of frequencies for measurement drive LO's in GHz.
        rep_time (int): the value of repetition time of experiment in us.
    """
    def __init__(self, meas_level, memory_slot_size, meas_return,
                 pulse_library, qubit_lo_freq, meas_lo_freq, rep_time,
                 **kwargs):
        self.meas_level = meas_level
        self.memory_slot_size = memory_slot_size
        self.meas_return = meas_return
        self.pulse_library = pulse_library
        self.qubit_lo_freq = qubit_lo_freq
        self.meas_lo_freq = meas_lo_freq
        self.rep_time = rep_time

        super().__init__(meas_level=meas_level,
                         memory_slot_size=memory_slot_size,
                         meas_return=meas_return,
                         pulse_library=pulse_library,
                         qubit_lo_freq=qubit_lo_freq,
                         meas_lo_freq=meas_lo_freq,
                         rep_time=rep_time,
                         **kwargs)


@bind_schema(PulseQobjSimConfigSchema)
class PulseQobjSimConfig(QobjConfig):
    """Model for PulseQobjSimConfig inherit from QobjConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjSimConfigSchema``.

    Attributes:
        pulse_library (list[QobjPulseLibrary]): a pulse library.
        qubit_lo_freq (list): the list of frequencies for qubit drive LO's in GHz.
        meas_lo_freq (list): the list of frequencies for measurement drive LO's in GHz.
        hamiltonian (dict): the dictionary defining system Hamiltonian.
        dt (float): time interval of pulse envelope.
    """
    def __init__(self, pulse_library, qubit_lo_freq, meas_lo_freq,
                 hamiltonian, dt, **kwargs):
        # pylint: disable=invalid-name
        self.pulse_library = pulse_library
        self.qubit_lo_freq = qubit_lo_freq
        self.meas_lo_freq = meas_lo_freq
        self.hamiltonian = hamiltonian
        self.dt = dt

        super().__init__(pulse_library=pulse_library,
                         qubit_lo_freq=qubit_lo_freq,
                         meas_lo_freq=meas_lo_freq,
                         hamiltonian=hamiltonian,
                         dt=dt,
                         **kwargs)
