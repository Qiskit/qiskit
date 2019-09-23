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
from collections import defaultdict
from marshmallow.validate import Length, Regexp
from typing import Any, Iterable, List, Tuple, Union

from qiskit.pulse.exceptions import PulseError
from qiskit.util import _to_tuple
from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import DateTime, List, Nested, Number, String, Integer


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

        self._qubits = defaultdict(dict)
        for qubit, props in enumerate(qubits):
            formatted_props = {}
            for prop in props:
                value = self._apply_prefix(prop.value, prop.unit)
                formatted_props[prop.name] = (value, prop.date)
                self._qubits[qubit] = formatted_props

        self._gates = defaultdict(dict)
        for gate in gates:
            qubits = _to_tuple(gate.qubits)
            formatted_props = {}
            formatted_props['name'] = gate.name
            for param in gate.parameters:
                value = self._apply_prefix(param.value, param.unit)
                formatted_props[param.name] = (value, param.date)
            self._gates[gate.gate][qubits] = formatted_props

        super().__init__(**kwargs)

    def gate_error(self, operation: str, qubits: Union[int, Iterable[int]]):
        """
        Return gate error estimates from backend properties.

        Args:
            operation (str): The operation for which to get the error.
            qubits (Union[int, Iterable[int]]): The specific qubits for the operation.

        Raises:
	        PulseError: If error is True and the property is not found.
        """
        # Throw away datetime at index 1
        return self.get_gate_property(operation, qubits, 'gate_error')[0]

    def gate_length(self, operation: str, qubits: Union[int, Iterable[int]]):
        """
        Return the duration of the gate in units of seconds.

        Args:
            operation (str): The operation for which to get the duration.
            qubits (Union[int, Iterable[int]]): The specific qubits for the operation.

        Raises:
	        PulseError: If error is True and the property is not found.
        """
        # Throw away datetime at index 1
        return self.get_gate_property(operation, qubits, 'gate_length')[0]

    def get_gate_property(self,
                          gate: str = None,
                          qubits: Union[int, Iterable[int]] = None,
                          gate_property: str = None) -> Tuple[Any, datetime.datetime]:
        """
        Return the gate properties of the given qubit and property.

        Args:
            gate (str) : Name of the gate
            qubits (Union[int, Iterable[int]]): The property to look for.
            gate_property (str): Optionally used to specify within the heirarchy which
            property to return.

        Return:
            Gate properties of the given qubit and property, if it was given by the
            backend, otherwise, return `None`.

        Raises:
	        PulseError: If error is True and the property is not found.
        """
        result = self._gates
        try:
            if gate is not None:
                result = result[gate]
                if qubits is not None:
                    result = result[_to_tuple(qubits)]
                    if gate_property is not None:
                        result = result[gate_property]
        except (KeyError, TypeError):
            raise PulseError("Could not find the desired property.")
        return result

    def get_qubit_property(self, qubit: int = None,
                           name: str = None) -> Tuple[Any, datetime.datetime]:
        """
        Return the qubit properties of the given qubit and name.

        Args:
            qubit (int): The property to look for.
            name (str): Optionally used to specify within the heirarchy which property to return.

        Return:
            Qubit properties of the given qubit and name, if it was given by
            the backend, otherwise, return `None`.

        Raises:
	        PulseError: If error is True and the property is not found.
        """
        result = self._qubits
        try:
            if qubit is not None:
                result = result[qubit]
                if name is not None:
                    result = result[name]
        except (KeyError, TypeError):
            raise PulseError("Could not find the desired property.")
        return result

    def t1(self, qubit: int):
        """
        Return the properties of T1 of the given qubit.

        Args:
            qubit (int): The property to look for.

        Return:
            Properties of T1 of the given qubit, if it was given by the backend, otherwise,
            return `None`.

        Raises:
	        PulseError: If error is True and the property is not found.
        """
        return self.get_qubit_property(qubit, 'T1')

    def _apply_prefix(self, value, unit):
        prefixes = {
           'p': 1e-12, 'n': 1e-9,
           'u': 1e-6,  'µ': 1e-6,
           'm': 1e-3,  'k': 1e3,
           'M': 1e6,   'G': 1e9,
        }
        if not unit:
            return value
        try:
            return value * prefixes[unit[0]]
        except KeyError:
            raise PulseError("Could not understand units: {}".format(unit))
