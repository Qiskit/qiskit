# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Blueprint circuit object."""

from typing import Optional
from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametertable import ParameterTable


class BlueprintCircuit(QuantumCircuit, ABC):
    """Blueprint circuit object.

    In many applications it is necessary to pass around the structure a circuit will have without
    explicitly knowing e.g. its number of qubits, or other missing information. This can be solved
    by having a circuit that knows how to construct itself, once all information is available.

    This class provides an interface for such circuits. Before internal data of the circuit is
    accessed, the ``_build`` method is called. There the configuration of the circuit is checked.
    """

    def __init__(self, *regs, name: Optional[str] = None) -> None:
        """Create a new blueprint circuit.

        The ``_data`` argument storing the internal circuit data is set to ``None`` to indicate
        that the circuit has not been built yet.
        """
        super().__init__(*regs, name=name)
        self._data = None
        self._qregs = []
        self._cregs = []
        self._qubits = []

    @abstractmethod
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration allows the circuit to be built.

        Args:
            raise_on_failure: If True, raise if the configuration is invalid. If False, return
                False if the configuration is invalid.

        Returns:
            True, if the configuration is valid. Otherwise, depending on the value of
            ``raise_on_failure`` an error is raised or False is returned.
        """
        raise NotImplementedError

    @abstractmethod
    def _build(self) -> None:
        """Build the circuit."""
        # do not build the circuit if _data is already populated
        if self._data:
            return

        self._data = []

        # check whether the configuration is valid
        self._check_configuration()

    def _invalidate(self) -> None:
        """Invalidate the current circuit build."""
        self._data = None
        self._parameter_table = ParameterTable()

    @property
    def qregs(self):
        """A list of the quantum registers associated with the circuit."""
        return self._qregs

    @qregs.setter
    def qregs(self, qregs):
        """Set the quantum registers associated with the circuit."""
        self._qregs = qregs
        self._qubits = [qbit for qreg in qregs for qbit in qreg]
        self._invalidate()

    @property
    def data(self):
        if self._data is None:
            self._build()
        return super().data

    def qasm(self, formatted=False, filename=None):
        if self._data is None:
            self._build()
        return super().qasm(formatted, filename)

    def append(self, instruction, qargs=None, cargs=None):
        if self._data is None:
            self._build()
        return super().append(instruction, qargs, cargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def size(self):
        if self._data is None:
            self._build()
        return super().size()

    def depth(self):
        if self._data is None:
            self._build()
        return super().depth()

    def count_ops(self):
        if self._data is None:
            self._build()
        return super().count_ops()

    def num_nonlocal_gates(self):
        if self._data is None:
            self._build()
        return super().num_nonlocal_gates()

    def num_connected_components(self, unitary_only=False):
        if self._data is None:
            self._build()
        return super().num_connected_components(unitary_only=unitary_only)

    def copy(self, name=None):
        if self._data is None:
            self._build()
        return super().copy(name=name)
