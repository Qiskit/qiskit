# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Model and schema for backend configuration."""

from marshmallow.validate import Length, OneOf, Range, Regexp

from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Boolean, DateTime, Integer, List, Nested, String


class GateConfigSchema(BaseSchema):
    """Schema for GateConfig."""

    # Required properties.
    name = String(required=True)
    parameters = List(String(), required=True)
    qasm_def = String(required=True)

    # Optional properties.
    coupling_map = List(List(Integer(),
                             validate=Length(min=1)),
                        validate=Length(min=1))
    latency_map = List(List(Integer(validate=OneOf([0, 1])),
                            validate=Length(min=1)),
                       validate=Length(min=1))
    conditional = Boolean()
    description = String()


class BackendConfigurationSchema(BaseSchema):
    """Schema for BackendConfiguration."""

    # Required properties.
    backend_name = String(required=True)
    backend_version = String(required=True,
                             validate=Regexp("[0-9]+.[0-9]+.[0-9]+$"))
    n_qubits = Integer(required=True, validate=Range(min=1))
    basis_gates = List(String(), required=True,
                       validate=Length(min=1))
    gates = Nested(GateConfigSchema, required=True, many=True,
                   validate=Length(min=1))
    local = Boolean(required=True)
    simulator = Boolean(required=True)
    conditional = Boolean(required=True)
    open_pulse = Boolean(required=True)
    memory = Boolean(required=True)
    max_shots = Integer(required=True, validate=Range(min=1))

    # Optional properties.
    max_experiments = Integer(validate=Range(min=1))
    sample_name = String()
    coupling_map = List(List(Integer(),
                             validate=Length(min=1)),
                        validate=Length(min=1), allow_none=True)
    n_registers = Integer(validate=Range(min=1))
    register_map = List(List(Integer(validate=OneOf([0, 1])),
                             validate=Length(min=1)),
                        validate=Length(min=1))
    configurable = Boolean()
    credits_required = Boolean()
    online_date = DateTime()
    display_name = String()
    description = String()
    tags = List(String())


@bind_schema(GateConfigSchema)
class GateConfig(BaseModel):
    """Model for GateConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``GateConfigSchema``.

    Attributes:
        name (str): the gate name as it will be referred to in QASM.
        parameters (list[str]): variable names for the gate parameters (if any).
        qasm_def (str): definition of this gate in terms of QASM primitives U
            and CX.
    """

    def __init__(self, name, parameters, qasm_def, **kwargs):
        self.name = name
        self.parameters = parameters
        self.qasm_def = qasm_def

        super().__init__(**kwargs)


@bind_schema(BackendConfigurationSchema)
class BackendConfiguration(BaseModel):
    """Model for BackendConfiguration.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``BackendConfigurationSchema``.
    Attributes:
        backend_name (str): backend name.
        backend_version (str): backend version in the form X.Y.Z.
        n_qubits (int): number of qubits.
        basis_gates (list[str]): list of basis gates names on the backend.
        gates (GateConfig): list of basis gates on the backend.
        local (bool): backend is local or remote.
        simulator (bool): backend is a simulator.
        conditional (bool): backend supports conditional operations.
        open_pulse (bool): backend supports open pulse.
        memory (bool): backend supports memory.
        max_shots (int): maximum number of shots supported.
    """

    def __init__(self, backend_name, backend_version, n_qubits, basis_gates,
                 gates, local, simulator, conditional, open_pulse, memory,
                 max_shots, **kwargs):
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.n_qubits = n_qubits
        self.basis_gates = basis_gates
        self.gates = gates
        self.local = local
        self.simulator = simulator
        self.conditional = conditional
        self.open_pulse = open_pulse
        self.memory = memory
        self.max_shots = max_shots

        super().__init__(**kwargs)
