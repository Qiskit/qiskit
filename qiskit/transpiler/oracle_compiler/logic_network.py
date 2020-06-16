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

"""LogicNetwork and the related exceptions"""

import ast
try:
    from tweedledum import synthesize_xag, simulate  # pylint: disable=no-name-in-module
    HAS_TWEEDLEDUM = True
except Exception:  # pylint: disable=broad-except
    HAS_TWEEDLEDUM = False
from qiskit import QuantumCircuit, QuantumRegister
from .oracle_visitor import OracleVisitor
from .utils import tweedledum2qiskit


class LogicNetwork:
    """A logical network represents an oracle function."""

    def __init__(self, source):
        """Creates a LogicNetwork from Python source code in ``source``. The code should be
        a single function with type hints.

        Args:
            source (str): Python code with type hints.

        Raises:
            ImportError: If tweedledum is not installed.
        """
        if not HAS_TWEEDLEDUM:
            raise ImportError("To use the oracle compiler, tweedledum "
                              "must be installed. To install tweedledum run "
                              '"pip install tweedledum".')
        _oracle_visitor = OracleVisitor()
        _oracle_visitor.visit(ast.parse(source))
        self._network = _oracle_visitor._network
        self.scopes = _oracle_visitor.scopes
        self.args = _oracle_visitor.args
        self.name = _oracle_visitor.name
        super().__init__()

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
        return simulate(self._network)

    def synth(self, arg_regs=False) -> QuantumCircuit:
        """Synthesis the logic network into a ``QuantumCircuit``.

        Args:
            arg_regs (bool): Default ``False``. If ``True`` uses the parameter names to create
            registers with those names. Otherwise, creates a circuit with a flat quantum register.

        Returns:
            QuantumCircuit: A circuit implementing the logic network.
        """
        if arg_regs:
            qregs = [QuantumRegister(1, name=arg) for arg in self.args
                     if self.types[0][arg] == 'Bit']
            qregs.reverse()
            if self.types[0]['return'] == 'Bit':
                qregs.append(QuantumRegister(1, name='return'))
        else:
            qregs = None
        return tweedledum2qiskit(synthesize_xag(self._network), name=self.name, qregs=qregs)
