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
from typing import Any, Iterable, Tuple, Union

from marshmallow.validate import Length, Regexp
from qiskit.util import _to_tuple
from qiskit.validation.fields import DateTime, List, Nested, Number, String, Integer
from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.pulse.exceptions import PulseError


class NduvSchema(BaseSchema):
    """Schema for name-date-unit-value."""

    # Required properties.
    date = DateTime(required=True)
    name = String(required=True)
    unit = String(required=True)
    value = Number(required=True)


class GateSchema(BaseSchema):
    """Schema for Gate."""

    # Required properties.
    qubits = List(Integer(), required=True,
                  validate=Length(min=1))
    gate = String(required=True)
    parameters = Nested(NduvSchema, required=True, many=True,
                        validate=Length(min=1))


class BackendPropertiesSchema(BaseSchema):
    """Schema for BackendProperties."""

    # Required properties.
    backend_name = String(required=True)
    backend_version = String(required=True,
                             validate=Regexp("[0-9]+.[0-9]+.[0-9]+$"))
    last_update_date = DateTime(required=True)
    qubits = List(Nested(NduvSchema, many=True,
                         validate=Length(min=1)), required=True,
                  validate=Length(min=1))
    gates = Nested(GateSchema, required=True, many=True,
                   validate=Length(min=1))
    general = Nested(NduvSchema, required=True, many=True)


@bind_schema(NduvSchema)
class Nduv(BaseModel):
    """Model for name-date-unit-value.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``NduvSchema``.

    Attributes:
        date (datetime): date.
        name (str): name.
        unit (str): unit.
        value (Number): value.
    """

    def __init__(self, date, name, unit, value, **kwargs):
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
        qubits (list[Number]): qubits.
        gate (str): gate.
        parameters (Nduv): parameters.
    """

    def __init__(self, qubits, gate, parameters, **kwargs):
        self.qubits = qubits
        self.gate = gate
        self.parameters = parameters

        super().__init__(**kwargs)


@bind_schema(BackendPropertiesSchema)
class BackendProperties(BaseModel):
    """Model for BackendProperties.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``BackendPropertiesSchema``.

    Attributes:
        backend_name (str): backend name.
        backend_version (str): backend version in the form X.Y.Z.
        last_update_date (datetime): last date/time that a property was updated.
        qubits (list[list[Nduv]]): system qubit parameters.
        gates (list[Gate]): system gate parameters.
        general (list[Nduv]): general parameters.
    """

    def __init__(self, backend_name, backend_version, last_update_date,
                 qubits, gates, general, **kwargs):
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
            self._gates[gate.gate][_to_tuple(gate.qubits)] = formatted_props

        super().__init__(**kwargs)

    def gate_property(self,
                      operation: str,
                      qubits: Union[int, Iterable[int]] = None,
                      name: str = None) -> Tuple[Any, datetime.datetime]:
        """
        Return the gate property of the given operation.

        Args:
            operation: Name of the gate.
            qubits: The qubit to find the property for.
            name: Optionally used to specify within the heirarchy which
                  property to return.

        Return:
            Gate property.

        Raises:
	        PulseError: If the property is not found or name is specified but qubit is not.
        """
        try:
            result = self._gates[operation]
            if qubits is not None:
                result = result[_to_tuple(qubits)]
                if name:
                    result = result[name]
            elif name:
                raise PulseError("Provide qubits to get {n} of '{o}'.".format(n=name, o=operation))
        except KeyError:
            raise PulseError("Could not find the desired property for {o}.".format(o=operation))
        return result

    def gate_error(self, operation: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return gate error estimates from backend properties.

        Args:
            operation: The operation for which to get the error.
            qubits: The specific qubits for the operation.

        Return:
            Gate error of the given operation and qubit(s).
        """
        return self.gate_property(operation, qubits,
                                  'gate_error')[0]  # Throw away datetime at index 1

    def gate_length(self, operation: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return the duration of the gate in units of seconds.

        Args:
            operation: The operation for which to get the duration.
            qubits: The specific qubits for the operation.

        Return:
            Gate length of the given operation and qubit(s).
        """
        return self.gate_property(operation, qubits,
                                  'gate_length')[0]  # Throw away datetime at index 1

    def qubit_property(self,
                       qubit: int,
                       name: str = None) -> Tuple[Any, datetime.datetime]:
        """
        Return the property of the given qubit.

        Args:
            qubit: The property to look for.
            name: Optionally used to specify within the heirarchy which property to return.

        Return:
            Qubit property.

        Raises:
	        PulseError: If the property is not found.
        """
        try:
            result = self._qubits[qubit]
            if name is not None:
                result = result[name]
        except KeyError:
            raise PulseError("Couldn't find the desired property for {q}.".format(q=qubit))
        return result

    def t1(self, qubit: int) -> Tuple[Any, datetime.datetime]:
        """
        Return the T1 time of the given qubit.

        Args:
            qubit: Qubit for which to return the T1 time of.

        Return:
            T1 time of the given qubit.
        """
        return self.qubit_property(qubit, 'T1')

    def _apply_prefix(self, value: float, unit: str) -> float:
        """
        Given a SI unit prefix and value, apply the prefix to convert to standard SI unit.

        Args:
            value: The number to apply prefix to.
            unit: String prefix.

        Return:
            Converted value.

        Raises:
            PulseError: If the units aren't recognized.
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
            raise PulseError("Could not understand units: {}".format(unit))
