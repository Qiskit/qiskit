# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Model and schema for backend configuration."""
import datetime
from typing import Any, Iterable, Tuple, Union, List

from marshmallow.validate import Length, Regexp

from qiskit.validation import fields
from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.providers.exceptions import BackendPropertyError


class NduvSchema(BaseSchema):
    """Schema for name-date-unit-value."""

    # Required properties.
    date = fields.DateTime(required=True)
    name = fields.String(required=True)
    unit = fields.String(required=True)
    value = fields.Number(required=True)


class GateSchema(BaseSchema):
    """Schema for Gate."""

    # Required properties.
    qubits = fields.List(fields.Integer(), required=True,
                         validate=Length(min=1))
    gate = fields.String(required=True)
    parameters = fields.Nested(NduvSchema, required=True, many=True,
                               validate=Length(min=1))


class BackendPropertiesSchema(BaseSchema):
    """Schema for BackendProperties."""

    # Required properties.
    backend_name = fields.String(required=True)
    backend_version = fields.String(required=True,
                                    validate=Regexp("[0-9]+.[0-9]+.[0-9]+$"))
    last_update_date = fields.DateTime(required=True)
    qubits = fields.List(fields.Nested(NduvSchema, many=True,
                                       validate=Length(min=1)), required=True,
                         validate=Length(min=1))
    gates = fields.Nested(GateSchema, required=True, many=True,
                          validate=Length(min=1))
    general = fields.Nested(NduvSchema, required=True, many=True)


@bind_schema(NduvSchema)
class Nduv(BaseModel):
    """Model for name-date-unit-value.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``NduvSchema``.

    Attributes:
        date: date.
        name: name.
        unit: unit.
        value: value.
    """

    def __init__(self,
                 date: datetime.datetime,
                 name: str,
                 unit: str,
                 value: float,
                 **kwargs):
        self.date = date
        self.name = name
        self.unit = unit
        self.value = value

        super().__init__(**kwargs)


@bind_schema(GateSchema)
class Gate(BaseModel):
    """Model for Gate.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``GateSchema``.

    Attributes:
        qubits: qubits.
        gate: gate.
        parameters: parameters.
    """

    def __init__(self, qubits: List[int], gate: str, parameters: Nduv, **kwargs):
        self.qubits = qubits
        self.gate = gate
        self.parameters = parameters

        super().__init__(**kwargs)


@bind_schema(BackendPropertiesSchema)
class BackendProperties(BaseModel):
    """Model for BackendProperties.

    Container class which holds backend properties which have been measured by the provider.
    All properties which are provided are provided optionally. These properties may describe
    qubits, gates, or other general properties of the backend.
    """

    def __init__(self,
                 backend_name: str,
                 backend_version: str,
                 last_update_date: datetime.datetime,
                 qubits: List[List[Nduv]],
                 gates: List[Gate],
                 general: List[Nduv],
                 **kwargs):  # pylint: disable=missing-param-doc
        """Initialize a BackendProperties instance.

        Args:
            backend_name: Backend name.
            backend_version: Backend version in the form X.Y.Z.
            last_update_date: Last date/time that a property was updated.
            qubits: System qubit parameters.
            gates: System gate parameters.
            general: General parameters.
        """
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.last_update_date = last_update_date
        self.general = general
        self.qubits = qubits
        self.gates = gates

        self._qubits = {}
        for qubit, props in enumerate(qubits):
            formatted_props = {}
            for prop in props:
                value = self._apply_prefix(prop.value, prop.unit)
                formatted_props[prop.name] = (value, prop.date)
                self._qubits[qubit] = formatted_props

        self._gates = {}
        for gate in gates:
            if gate.gate not in self._gates:
                self._gates[gate.gate] = {}
            formatted_props = {}
            for param in gate.parameters:
                value = self._apply_prefix(param.value, param.unit)
                formatted_props[param.name] = (value, param.date)
            self._gates[gate.gate][tuple(gate.qubits)] = formatted_props

        super().__init__(**kwargs)

    def gate_property(self,
                      gate: str,
                      qubits: Union[int, Iterable[int]] = None,
                      name: str = None) -> Tuple[Any, datetime.datetime]:
        """
        Return the property of the given gate.

        Args:
            gate: Name of the gate.
            qubits: The qubit to find the property for.
            name: Optionally used to specify which gate property to return.

        Returns:
            Gate property as a tuple of the value and the time it was measured.

        Raises:
            BackendPropertyError: If the property is not found or name is
                                  specified but qubit is not.
        """
        try:
            result = self._gates[gate]
            if qubits is not None:
                if isinstance(qubits, int):
                    qubits = tuple([qubits])
                result = result[tuple(qubits)]
                if name:
                    result = result[name]
            elif name:
                raise BackendPropertyError("Provide qubits to get {n} of {g}".format(n=name,
                                                                                     g=gate))
        except KeyError:
            raise BackendPropertyError("Could not find the desired property for {g}".format(g=gate))
        return result

    def gate_error(self, gate: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return gate error estimates from backend properties.

        Args:
            gate: The gate for which to get the error.
            qubits: The specific qubits for the gate.

        Returns:
            Gate error of the given gate and qubit(s).
        """
        return self.gate_property(gate, qubits,
                                  'gate_error')[0]  # Throw away datetime at index 1

    def gate_length(self, gate: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return the duration of the gate in units of seconds.

        Args:
            gate: The gate for which to get the duration.
            qubits: The specific qubits for the gate.

        Returns:
            Gate length of the given gate and qubit(s).
        """
        return self.gate_property(gate, qubits,
                                  'gate_length')[0]  # Throw away datetime at index 1

    def qubit_property(self,
                       qubit: int,
                       name: str = None) -> Tuple[Any, datetime.datetime]:
        """
        Return the property of the given qubit.

        Args:
            qubit: The property to look for.
            name: Optionally used to specify within the hierarchy which property to return.

        Returns:
            Qubit property as a tuple of the value and the time it was measured.

        Raises:
            BackendPropertyError: If the property is not found.
        """
        try:
            result = self._qubits[qubit]
            if name is not None:
                result = result[name]
        except KeyError:
            raise BackendPropertyError("Couldn't find the propert{name} for qubit "
                                       "{qubit}.".format(name="y '" + name + "'" if name else "ies",
                                                         qubit=qubit))
        return result

    def t1(self, qubit: int) -> float:  # pylint: disable=invalid-name
        """
        Return the T1 time of the given qubit.

        Args:
            qubit: Qubit for which to return the T1 time of.

        Returns:
            T1 time of the given qubit.
        """
        return self.qubit_property(qubit, 'T1')[0]  # Throw away datetime at index 1

    def t2(self, qubit: int) -> float:  # pylint: disable=invalid-name
        """
        Return the T2 time of the given qubit.

        Args:
            qubit: Qubit for which to return the T2 time of.

        Returns:
            T2 time of the given qubit.
        """
        return self.qubit_property(qubit, 'T2')[0]  # Throw away datetime at index 1

    def frequency(self, qubit: int) -> float:
        """
        Return the frequency of the given qubit.

        Args:
            qubit: Qubit for which to return frequency of.

        Returns:
            Frequency of the given qubit.
        """
        return self.qubit_property(qubit, 'frequency')[0]  # Throw away datetime at index 1

    def readout_error(self, qubit: int) -> float:
        """
        Return the readout error of the given qubit.

        Args:
            qubit: Qubit for which to return the readout error of.

        Return:
            Readout error of the given qubit,
        """
        return self.qubit_property(qubit, 'readout_error')[0]  # Throw away datetime at index 1

    def _apply_prefix(self, value: float, unit: str) -> float:
        """
        Given a SI unit prefix and value, apply the prefix to convert to standard SI unit.

        Args:
            value: The number to apply prefix to.
            unit: String prefix.

        Returns:
            Converted value.

        Raises:
            BackendPropertyError: If the units aren't recognized.
        """
        prefixes = {
            'p': 1e-12,
            'n': 1e-9,
            'u': 1e-6,
            'µ': 1e-6,
            'm': 1e-3,
            'k': 1e3,
            'M': 1e6,
            'G': 1e9
        }
        if not unit:
            return value
        try:
            return value * prefixes[unit[0]]
        except KeyError:
            raise BackendPropertyError("Could not understand units: {u}".format(u=unit))
