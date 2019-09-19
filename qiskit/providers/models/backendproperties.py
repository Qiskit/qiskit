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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from marshmallow.validate import Length, Regexp

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
        for qubit, props in enumerate(_qubits):
            formatted_props = {}
            for prop in props:
                value = self._apply_prefix(prop.value, prop.unit)
                formatted_props[prop.name] = (value, prop.date)
                self._qubits[qubit] = formatted_props

        self._gates = defaultdict(dict)
        for gate in _gates:
            qubits = _to_tuple(gate.qubits)
            formatted_props = {}
            formatted_props['name'] = gate.name
            for param in gate.parameters:
                value = self._apply_prefix(param.value, param.unit)
                formatted_props[param.name] = (value, param.date)
            self._gates[gate.gate][qubits] = formatted_props

        super().__init__(**kwargs)

    def gate_error(self, operation, qubits):
        """
        Return gate error estimates from backend properties.
        Args:
            operation: The operation for which to get the error.
            qubits: The specific qubits for the operation.
        """
        try:
            result = self._gates.get(operation, {}).get(qubits, {}).get('gate_error')[0]
            if result is None:
                raise error
            # return self.gates[operation][_to_tuple(qubits)]['gate_error'][0]
        except KeyError:
            #TODO - add a better and clear error message
            if error:
                raise PulseError("Could not find the desired property.")
            raise PulseError("Could not find the desired property.")
        return result

    def gate_length(self, operation, qubits):
        """
        Return the duration of the gate in units of seconds.
        Args:
            operation: The operation for which to get the duration.
            qubits: The specific qubits for the operation.
        """
        try:
            # Throw away datetime at index 1
            result = self._gates.get(operation).get(_to_tuple(qubits)).get('gate_length')[0]
            if result is None:
                raise error
            # return self.gates[operation][_to_tuple(qubits)]['gate_length'][0]
        except KeyError:
            #TODO - add a better and clear error message
            if error:
                raise PulseError("Could bit find the desired property")
            raise PulseError("Could bit find the desired property")
        return result

    def get_gate_property(self,
                          gate: str = None,
                          qubits: Union[int, Iterable[int]] = None,
                          gate_property: str = None) -> Tuple[Any, datetime.datetime]:
        """
        Return the gate properties of the given qubit and property, if it was given by the backend, otherwise,
        return `None` or raise an error.

        Args:
            gate (str) : Name of the gate
            qubit (Union[int, Iterable[int]]): The property to look for.
            gate_property (str): Optionally used to specify within the heirarchy which property to return.

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
            else:
                raise error
        except (KeyError, TypeError):
            if error:
                raise PulseError("Could not find the desired property as your gate is ", gate)
            raise PulseError("Could not find the desired property.")
        return result

    def get_qubit_property(self, qubit: int = None, name: str = None) -> Tuple[Any, datetime.datetime]:
        """
        Return the qubit properties of the given qubit and name, if it was given by the backend, otherwise,
        return `None` or raise an error.

        Args:
            qubit (int): The property to look for.
            name (str): Optionally used to specify within the heirarchy which property to return.

        Raises:
	        PulseError: If error is True and the property is not found.
        """
        result = self._qubits
        try:
            if qubit is not None:
                result = result[qubit]
                if name is not None:
                    result = result[name]
            else:
                raise error
        except (KeyError, TypeError):
            if error:
                raise PulseError("Could not find the desired property as your qubit is ", qubit)
            raise PulseError("Could not find the desired property.")
        return result

    def t1(self, qubit: int):
        """
        Return the properties of T1 of the given qubit, if it was given by the backend, otherwise,
        return `None` or raise an error.

        Args:
            qubit (int): The property to look for.

        Raises:
	        PulseError: If error is True and the property is not found.
        """
        #TODO investigate the possibilites of errors
        try:
            ret = get_qubit_property(qubit, 'T1')
        except:
            raise PulseError("Could not find the desired property")
        return ret

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
