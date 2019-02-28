# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for RunConfig and its related components."""
from marshmallow.validate import Range, OneOf

from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Boolean, Integer, List, String


class RunConfigSchema(BaseSchema):
    """Schema for RunConfig."""

    # Required properties.
    # None

    # Optional properties.
    shots = Integer(validate=Range(min=1))
    max_credits = Integer(validate=Range(min=3, max=10))  # TODO: can we check the range
    seed = Integer()
    memory = Boolean()  # set default to be False

    # Optional properties for PULSE
    meas_level = Integer(validate=Range(min=0, max=2))
    pulse_library = List()
    memory_slot_size = Integer()
    meas_return = String(validate=OneOf(choices=('avg', 'single')))
    qubit_lo_freq = List()
    meas_lo_freq = List()
    rep_time = Integer()


@bind_schema(RunConfigSchema)
class RunConfig(BaseModel):
    """Model for RunConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``RunConfigSchema``.

    Attributes:
        shots (int): the number of shots.
        max_credits (int): the max_credits to use on the IBMQ public devices.
        seed (int): the seed to use in the simulator for the first experiment.
        memory (bool): to use memory.
        meas_level (int): set the appropriate level of the measurement output.
        pulse_library (list): list of sample pulses used in the experiments.
        memory_slot_size (int): size of each memory slot if the output is Level 0.
        meas_return (str): indicates the level of measurement information to return.
        qubit_lo_freq (list): list of qubit driving frequency.
        meas_lo_freq (list): list of measurement frequency.
        rep_time (int): repetition time of the experiment in μs.
    """
