# -*- coding: utf-8 -*-

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

"""Gate equivalence library."""

from collections import namedtuple

from .exceptions import CircuitError
from .parameterexpression import ParameterExpression

Key = namedtuple('Key', ['name',
                         'num_qubits'])

Entry = namedtuple('Entry', ['search_base',
                             'equivalences'])

Equivalence = namedtuple('Equivalence', ['params',  # Ordered to match Gate.params
                                         'circuit'])


class EquivalenceLibrary():
    """A library providing a one-way mapping of Gates to their equivalent
    implementations as QuantumCircuits."""

    def __init__(self, *, base=None):
        """Create a new equivalence library.

        Args:
            base (Optional[EquivalenceLibrary]):  Base equivalence library to
                will be referenced if an entry is not found in this library.
        """
        self._base = base

        self._map = {}

    def add_equivalence(self, gate, equivalent_circuit):
        """Add a new equivalence to the library. Future queries for the Gate
        will include the given circuit, in addition to all existing equivalences
        (including those from base).

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            equivalent_circuit (QuantumCircuit): A circuit equivalently
                implementing the given Gate.
        """

        _raise_if_shape_mismatch(gate, equivalent_circuit)
        _raise_if_param_mismatch(gate.params, equivalent_circuit.parameters)

        key = Key(name=gate.name,
                  num_qubits=gate.num_qubits)

        equiv = Equivalence(params=gate.params.copy(),
                            circuit=equivalent_circuit.copy())

        if key not in self._map:
            self._map[key] = Entry(search_base=True, equivalences=[])

        self._map[key].equivalences.append(equiv)

    def set_entry(self, gate, entry):
        """Set the equivalence record for a Gate. Future queries for the Gate
        will return only the circuits provided.

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            entry (List['QuantumCircuit']) : A list of QuantumCircuits, each
                equivalently implementing the given Gate.
        """

        for equiv in entry:
            _raise_if_shape_mismatch(gate, equiv)
            _raise_if_param_mismatch(gate.params, equiv.parameters)

        key = Key(name=gate.name,
                  num_qubits=gate.num_qubits)

        equivs = [Equivalence(params=gate.params.copy(),
                              circuit=equiv.copy())
                  for equiv in entry]

        self._map[key] = Entry(search_base=False,
                               equivalences=equivs)

    def get_entry(self, gate):
        """Gets the set of QuantumCircuits circuits from the library which
        equivalently implement the given Gate.

        Parameterized circuits will have their parameters replaced with the
        corresponding entries from Gate.params.

        Args:
            gate (Gate) - Gate: A Gate instance.

        Returns:
            List[QuantumCircuit]: A list of equivalent QuantumCircuits. If empty,
                library contains no known decompositions of Gate.

                Returned circuits will be ordered according to their insertion in
                the library, from earliest to latest, from top to base. The
                ordering of the StandardEquivalenceLibrary will not generally be
                consistent across Qiskit versions.
        """

        key = Key(name=gate.name,
                  num_qubits=gate.num_qubits)

        query_params = gate.params

        if key in self._map:
            entry = self._map[key]
            search_base, equivs = entry

            rtn = [_rebind_equiv(equiv, query_params) for equiv in equivs]

            if search_base and self._base is not None:
                return rtn + self._base.get_entry(gate)
            return rtn

        if self._base is None:
            return []

        return self._base.get_entry(gate)


def _raise_if_param_mismatch(gate_params, circuit_parameters):
    gate_parameters = [p for p in gate_params
                       if isinstance(p, ParameterExpression)]

    if set(gate_parameters) != circuit_parameters:
        raise CircuitError('Cannot add equivalence between circuit and gate '
                           'of different parameters. Gate params: {}. '
                           'Circuit params: {}.'.format(
                               gate_parameters,
                               circuit_parameters))


def _raise_if_shape_mismatch(gate, circuit):
    if (gate.num_qubits != circuit.n_qubits
            or gate.num_clbits != circuit.n_clbits):
        raise CircuitError('Cannot add equivalence between circuit and gate '
                           'of different shapes. Gate: {} qubits and {} clbits. '
                           'Circuit: {} qubits and {} clbits.'.format(
                               gate.num_qubits, gate.num_clbits,
                               circuit.n_qubits, circuit.n_clbits))


def _rebind_equiv(equiv, query_params):
    equiv_params, equiv_circuit = equiv

    param_map = dict(zip(equiv_params, query_params))

    symbolic_param_map, numeric_param_map = _partition_dict(
        param_map,
        lambda k, v: isinstance(v, ParameterExpression))

    equiv = equiv_circuit.bind_parameters(numeric_param_map)
    equiv._substitute_parameters(symbolic_param_map)

    return equiv


def _partition_dict(dict_, predicate):
    return ({k: v for k, v in dict_.items() if predicate(k, v)},
            {k: v for k, v in dict_.items() if not predicate(k, v)})
