# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Backend Properties classes."""

import copy
import datetime
from typing import Any, Iterable, Tuple, Union
import dateutil.parser

from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix


class Nduv:
    """Class representing name-date-unit-value

    Attributes:
        date: date.
        name: name.
        unit: unit.
        value: value.
    """

    def __init__(self, date, name, unit, value):
        """Initialize a new name-date-unit-value object

        Args:
            date (datetime): Date field
            name (str): Name field
            unit (str): Nduv unit
            value (float): The value of the Nduv
        """
        self.date = date
        self.name = name
        self.unit = unit
        self.value = value

    @classmethod
    def from_dict(cls, data):
        """Create a new Nduv object from a dictionary.

        Args:
            data (dict): A dictionary representing the Nduv to create.
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            Nduv: The Nduv from the input dictionary.
        """
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the object.

        Returns:
            dict: The dictionary form of the Nduv.
        """
        out_dict = {
            "date": self.date,
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
        }
        return out_dict

    def __eq__(self, other):
        if isinstance(other, Nduv):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __repr__(self):
        return f"Nduv({repr(self.date)}, {self.name}, {self.unit}, {self.value})"


class Gate:
    """Class representing a gate's properties

    Attributes:
    qubits: qubits.
    gate: gate.
    parameters: parameters.
    """

    _data = {}

    def __init__(self, qubits, gate, parameters, **kwargs):
        """Initialize a new Gate object

        Args:
            qubits (list): A list of integers representing qubits
            gate (str): The gates name
            parameters (list): List of :class:`Nduv` objects for the
                name-date-unit-value for the gate
            kwargs: Optional additional fields
        """
        self._data = {}
        self.qubits = qubits
        self.gate = gate
        self.parameters = parameters
        self._data.update(kwargs)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f"Attribute {name} is not defined") from ex

    @classmethod
    def from_dict(cls, data):
        """Create a new Gate object from a dictionary.

        Args:
            data (dict): A dictionary representing the Gate to create.
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            Gate: The Nduv from the input dictionary.
        """
        in_data = copy.copy(data)
        nduvs = []
        for nduv in in_data.pop("parameters"):
            nduvs.append(Nduv.from_dict(nduv))
        in_data["parameters"] = nduvs
        return cls(**in_data)

    def to_dict(self):
        """Return a dictionary format representation of the BackendStatus.

        Returns:
            dict: The dictionary form of the Gate.
        """
        out_dict = {}
        out_dict["qubits"] = self.qubits
        out_dict["gate"] = self.gate
        out_dict["parameters"] = [x.to_dict() for x in self.parameters]
        out_dict.update(self._data)
        return out_dict

    def __eq__(self, other):
        if isinstance(other, Gate):
            if self.to_dict() == other.to_dict():
                return True
        return False


class BackendProperties:
    """Class representing backend properties

    This holds backend properties measured by the provider. All properties
    which are provided optionally. These properties may describe qubits, gates,
    or other general properties of the backend.
    """

    _data = {}

    def __init__(
        self, backend_name, backend_version, last_update_date, qubits, gates, general, **kwargs
    ):
        """Initialize a BackendProperties instance.

        Args:
            backend_name (str): Backend name.
            backend_version (str): Backend version in the form X.Y.Z.
            last_update_date (datetime or str): Last date/time that a property was
                updated. If specified as a ``str``, it must be in ISO format.
            qubits (list): System qubit parameters as a list of lists of
                           :class:`Nduv` objects
            gates (list): System gate parameters as a list of :class:`Gate`
                          objects
            general (list): General parameters as a list of :class:`Nduv`
                            objects
            kwargs: optional additional fields
        """
        self._data = {}
        self.backend_name = backend_name
        self.backend_version = backend_version
        if isinstance(last_update_date, str):
            last_update_date = dateutil.parser.isoparse(last_update_date)
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
        self._data.update(kwargs)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f"Attribute {name} is not defined") from ex

    @classmethod
    def from_dict(cls, data):
        """Create a new Gate object from a dictionary.

        Args:
            data (dict): A dictionary representing the Gate to create.
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            BackendProperties: The BackendProperties from the input
                               dictionary.
        """
        in_data = copy.copy(data)
        backend_name = in_data.pop("backend_name")
        backend_version = in_data.pop("backend_version")
        last_update_date = in_data.pop("last_update_date")
        qubits = []
        for qubit in in_data.pop("qubits"):
            nduvs = []
            for nduv in qubit:
                nduvs.append(Nduv.from_dict(nduv))
            qubits.append(nduvs)
        gates = [Gate.from_dict(x) for x in in_data.pop("gates")]
        general = [Nduv.from_dict(x) for x in in_data.pop("general")]
        return cls(
            backend_name, backend_version, last_update_date, qubits, gates, general, **in_data
        )

    def to_dict(self):
        """Return a dictionary format representation of the BackendProperties.

        Returns:
            dict: The dictionary form of the BackendProperties.
        """
        out_dict = {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "last_update_date": self.last_update_date,
        }
        out_dict["qubits"] = []
        for qubit in self.qubits:
            qubit_props = []
            for item in qubit:
                qubit_props.append(item.to_dict())
            out_dict["qubits"].append(qubit_props)
        out_dict["gates"] = [x.to_dict() for x in self.gates]
        out_dict["general"] = [x.to_dict() for x in self.general]
        out_dict.update(self._data)
        return out_dict

    def __eq__(self, other):
        if isinstance(other, BackendProperties):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def gate_property(
        self, gate: str, qubits: Union[int, Iterable[int]] = None, name: str = None
    ) -> Tuple[Any, datetime.datetime]:
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
                raise BackendPropertyError(f"Provide qubits to get {name} of {gate}")
        except KeyError as ex:
            raise BackendPropertyError(f"Could not find the desired property for {gate}") from ex
        return result

    def faulty_qubits(self):
        """Return a list of faulty qubits."""
        faulty = []
        for qubit in self._qubits:
            if not self.is_qubit_operational(qubit):
                faulty.append(qubit)
        return faulty

    def faulty_gates(self):
        """Return a list of faulty gates."""
        faulty = []
        for gate in self.gates:
            if not self.is_gate_operational(gate.gate, gate.qubits):
                faulty.append(gate)
        return faulty

    def is_gate_operational(self, gate: str, qubits: Union[int, Iterable[int]] = None) -> bool:
        """
        Return the operational status of the given gate.

        Args:
            gate: Name of the gate.
            qubits: The qubit to find the operational status for.

        Returns:
            bool: Operational status of the given gate. True if the gate is operational,
            False otherwise.
        """
        properties = self.gate_property(gate, qubits)
        if "operational" in properties:
            return bool(properties["operational"][0])
        return True  # if property operational not existent, then True.

    def gate_error(self, gate: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return gate error estimates from backend properties.

        Args:
            gate: The gate for which to get the error.
            qubits: The specific qubits for the gate.

        Returns:
            Gate error of the given gate and qubit(s).
        """
        return self.gate_property(gate, qubits, "gate_error")[0]  # Throw away datetime at index 1

    def gate_length(self, gate: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return the duration of the gate in units of seconds.

        Args:
            gate: The gate for which to get the duration.
            qubits: The specific qubits for the gate.

        Returns:
            Gate length of the given gate and qubit(s).
        """
        return self.gate_property(gate, qubits, "gate_length")[0]  # Throw away datetime at index 1

    def qubit_property(self, qubit: int, name: str = None) -> Tuple[Any, datetime.datetime]:
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
        except KeyError as ex:
            raise BackendPropertyError(
                "Couldn't find the propert{name} for qubit "
                "{qubit}.".format(name="y '" + name + "'" if name else "ies", qubit=qubit)
            ) from ex
        return result

    def t1(self, qubit: int) -> float:  # pylint: disable=invalid-name
        """
        Return the T1 time of the given qubit.

        Args:
            qubit: Qubit for which to return the T1 time of.

        Returns:
            T1 time of the given qubit.
        """
        return self.qubit_property(qubit, "T1")[0]  # Throw away datetime at index 1

    def t2(self, qubit: int) -> float:  # pylint: disable=invalid-name
        """
        Return the T2 time of the given qubit.

        Args:
            qubit: Qubit for which to return the T2 time of.

        Returns:
            T2 time of the given qubit.
        """
        return self.qubit_property(qubit, "T2")[0]  # Throw away datetime at index 1

    def frequency(self, qubit: int) -> float:
        """
        Return the frequency of the given qubit.

        Args:
            qubit: Qubit for which to return frequency of.

        Returns:
            Frequency of the given qubit.
        """
        return self.qubit_property(qubit, "frequency")[0]  # Throw away datetime at index 1

    def readout_error(self, qubit: int) -> float:
        """
        Return the readout error of the given qubit.

        Args:
            qubit: Qubit for which to return the readout error of.

        Return:
            Readout error of the given qubit.
        """
        return self.qubit_property(qubit, "readout_error")[0]  # Throw away datetime at index 1

    def readout_length(self, qubit: int) -> float:
        """
        Return the readout length [sec] of the given qubit.

        Args:
            qubit: Qubit for which to return the readout length of.

        Return:
            Readout length of the given qubit.
        """
        return self.qubit_property(qubit, "readout_length")[0]  # Throw away datetime at index 1

    def is_qubit_operational(self, qubit: int) -> bool:
        """
        Return the operational status of the given qubit.

        Args:
            qubit: Qubit for which to return operational status of.

        Returns:
            Operational status of the given qubit.
        """
        properties = self.qubit_property(qubit)
        if "operational" in properties:
            return bool(properties["operational"][0])
        return True  # if property operational not existent, then True.

    def _apply_prefix(self, value: float, unit: str) -> float:
        """
        Given a SI unit prefix and value, apply the prefix to convert to
        standard SI unit.

        Args:
            value: The number to apply prefix to.
            unit: String prefix.

        Returns:
            Converted value.

        Raises:
            BackendPropertyError: If the units aren't recognized.
        """
        try:
            return apply_prefix(value, unit)
        except Exception as ex:
            raise BackendPropertyError(f"Could not understand units: {unit}") from ex
