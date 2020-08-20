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

"""Oracle class"""

import ast

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.exceptions import QiskitError
from qiskit.transpiler.oracle_synthesis.tweedledum import Tweedledum
from .oracle_visitor import OracleVisitor


class Oracle(Gate):
    """An oracle object represents an oracle function and its logic network."""

    def __init__(self, source, synthesizer=None):
        """Creates a ``Oracle`` from Python source code in ``source``. The code should be
        a single function with types.

        Args:
            source (str): Python code with type hints.
            synthesizer(Synthesizer): TODO default: Tweedledum
        Raises:
            ImportError: If tweedledum is not installed.
            QiskitError: If source is not a string.
        """
        if not isinstance(source, str):
            raise QiskitError('Oracle needs a source code as a string.')
        _oracle_visitor = OracleVisitor()
        self.ast = _oracle_visitor.visit(ast.parse(source))
        self.synthesizer = synthesizer if synthesizer else Tweedledum
        self._synth_instance = None
        self.scopes = _oracle_visitor.scopes
        self.args = _oracle_visitor.args
        self.name = _oracle_visitor.name
        super().__init__(self.name, num_qubits=sum([qreg.size for qreg in self.qregs]), params=[])

    @property
    def synth_instance(self):
        if self._synth_instance is None:
            self._synth_instance = self.synthesizer(self.ast)
        return self._synth_instance

    @property
    def types(self):
        """Dumps a list of scopes with their variables and types.
        Returns:
            list(dict): A list of scopes as dicts, where key is the variable name and
            value is its type.
        """
        ret = []
        for scope in self.scopes:
            ret.append({k: v[0] for k, v in scope.items()})
        return ret

    def simulate(self):
        """Runs ``tweedledum.simulate`` on the logic network."""
        return self.synth_instance.simulate()

    def synth(self, arg_regs=False) -> QuantumCircuit:
        """Synthesis the logic network into a ``QuantumCircuit``.

        Args:
            arg_regs (bool): Default ``False``. If ``True`` uses the parameter names to create
            registers with those names. Otherwise, creates a circuit with a flat quantum register.

        Returns:
            QuantumCircuit: A circuit implementing the logic network.
        """
        if arg_regs:
            qregs = self.qregs
        else:
            qregs = None
        return self.synth_instance.synth(name=self.name, qregs=qregs)

    def _define(self):
        """The definition of the oracle is its synthesis"""
        self.definition = self.synth()

    @property
    def qregs(self):
        """The list of qregs used by the oracle"""
        qregs = [QuantumRegister(1, name=arg) for arg in self.args if self.types[0][arg] == 'Int1']
        qregs.reverse()
        if self.types[0]['return'] == 'Int1':
            qregs.append(QuantumRegister(1, name='return'))
        return qregs
