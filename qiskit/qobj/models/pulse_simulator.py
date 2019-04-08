# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""The simulator qobj models."""

from marshmallow.validate import Range, Length, Regexp

from qiskit.validation import bind_schema, BaseSchema, BaseModel
from qiskit.validation.fields import (Integer, String, Number, List, Nested, Boolean,
                                      Dict, InstructionParameter)
from qiskit.validation.validate import PatternProperties


class PulseHamiltonianSchema(BaseSchema):
    """Schema for PulseHamiltonian."""
    # pylint: disable=redefined-builtin

    # Required properties.
    ham_string = List(String(), validate=Length(min=1), required=True)
    dim_osc = List(Integer(validate=Range(min=1)), required=True)
    dim_qub = List(Integer(validate=Range(min=2)), required=True)
    vars = Dict(validate=PatternProperties({
        Regexp('^([a-z0-9])+$'): InstructionParameter()
    }), required=True)

    # Optional properties.


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
    # pylint: invalid-name

    # Required properties.
    hamiltonian = Nested(PulseHamiltonianSchema)
    dt = Number(required=True)

    # Optional properties.
    noise_model = Nested(PulseNoiseModelSchema)
    ode_options = Nested(PulseOdeOptionSchema)


@bind_schema(PulseHamiltonianSchema)
class PulseHamiltonian(BaseModel):
    """Model for PulseHamiltonian.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseHamiltonianSchema``.

    Attributes:

    """
    def __init__(self, ham_string, dim_osc, dim_qub, vars, **kwargs):
        # pylint: disable=redefined-builtin

        self.ham_string = ham_string
        self.dim_osc = dim_osc
        self.dim_qub = dim_qub
        self.vars = vars

        super().__init__(**kwargs)


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
        hamiltonian (PulseHamiltonian): system Hamiltonian configuration
        dt (float): time interval of input pulse sampling points
    """
    def __init__(self, hamiltonian, dt, **kwargs):
        # pylint: disable=redefined-builtin,invalid-name

        self.hamiltonian = hamiltonian
        self.dt = dt

        super().__init__(**kwargs)
