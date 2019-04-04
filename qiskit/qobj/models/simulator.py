# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""The simulator qobj models."""

from marshmallow.validate import Range, Length, Regexp

from qiskit.validation import bind_schema, BaseSchema, BaseModel
from qiskit.validation.fields import (Integer, String, Number, Complex, List, Nested, Boolean,
                                      Dict, DictParameters, InstructionParameter)
from qiskit.validation.validate import PatternProperties


class PulseNoiseModelSchema(BaseSchema):
    """Schema for PulseNoiseModel."""

    # Required properties.
    qubit = List(
        Dict(validate=PatternProperties({
            Regexp('^([A-Za-z])+$'): Number()
        })),
        required=True)
    oscillator = List(
        List(Number, validate=Length(equal=2)),
        required=True
    )

    # Optional properties.


class PulseOdeOptionSchema(BaseSchema):
    """Schema for PulseOdeOption."""

    # Required properties.

    # Optional properties.
    atol = Number()
    rtol = Number()
    nsteps = Integer(validate=Range(min=0))
    max_steps = Integer(validate=Range(min=0))
    num_cpus = Integer(validate=Range(min=0))
    norm_tol = Number()
    norm_steps = Integer(validate=Range(min=0))
    rhs_reuse = Boolean()
    rhs_filename = String(validate=Length(min=1))


class PulseSimulatorSpecSchema(BaseSchema):
    """Schema for PulseSimulatorSpec."""
    # pylint: disable=invalid-name

    # Required properties.
    hamiltonian = List(String(), validate=Length(min=1), required=True)
    dim_osc = List(Integer(validate=Range(min=1)), required=True)
    dim_qub = List(Integer(validate=Range(min=2)), required=True)
    vars = Dict(validate=PatternProperties({
        Regexp('^([a-z0-9])+$'): InstructionParameter()
    }), required=True)
    dt = Number(required=True)

    # Optional properties.
    noise_model = Nested(PulseNoiseModelSchema)
    ode_options = Nested(PulseOdeOptionSchema)


@bind_schema(PulseNoiseModelSchema)
class PulseNoiseModel(BaseModel):
    """Model for PulseNoiseModel.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseNoiseModelSchema``.

    Attributes:
        qubit (list[dict[str, int]]): noise model for qubit
        oscillator (list[list[float]]): noise model for oscillator (n_th, coupling)
    """
    def __init__(self, qubit, oscillator, **kwargs):
        self.qubit = qubit
        self.oscillator = oscillator

        super().__init__(**kwargs)


@bind_schema(PulseOdeOptionSchema)
class PulseOdeOption(BaseModel):
    """Model for PulseOdeOption.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseOdeOptionSchema``.
    """
    pass


@bind_schema(PulseSimulatorSpecSchema)
class PulseSimulatorSpec(BaseModel):
    """Model for PulseSimulatorSpec.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseSimulatorSpecSchema``.

    Attributes:
        hamiltonian (list[str]): system Hamiltonian in machine readable format
        dim_osc (list[int]): size of Fock state of each oscillator subspace
        dim_qub (list[int]): size of Fock state of each qubit subspace
        vars (dict[str, float]): dictionary of variables in Hamiltonian
        dt (float): time interval of input pulse sampling points
    """
    def __init__(self, hamiltonian, dim_osc, dim_qub,
                 vars, dt, **kwargs):
        # pylint: disable=invalid-name

        self.hamiltonian = hamiltonian
        self.dim_osc = dim_osc
        self.dim_qub = dim_qub
        self.vars = vars
        self.dt = dt

        super().__init__(**kwargs)
