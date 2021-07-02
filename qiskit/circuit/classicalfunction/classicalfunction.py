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

"""ClassicalFunction class"""

import ast
from typing import Callable, Optional

from tweedledum.classical import simulate
from tweedledum.synthesis import pkrm_synth

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from .classical_element import ClassicalElement
from .classical_function_visitor import ClassicalFunctionVisitor
from .utils import tweedledum2qiskit


class ClassicalFunction(ClassicalElement):
    """Represent a classical function function and its logic network."""

    def __init__(self, source, name=None):
        """Creates a ``ClassicalFunction`` from Python source code in ``source``.

        The code should be a single function with types.

        Args:
            source (str): Python code with type hints.
            name (str): Optional. Default: "*classicalfunction*". ClassicalFunction name.

        Raises:
            QiskitError: If source is not a string.
        """
        if not isinstance(source, str):
            raise QiskitError("ClassicalFunction needs a source code as a string.")
        self._ast = ast.parse(source)
        self._network = None
        self._scopes = None
        self._args = None
        self._truth_table = None
        super().__init__(
            name or "*classicalfunction*",
            num_qubits=sum(qreg.size for qreg in self.qregs),
            params=[],
        )

    def compile(self):
        """Parses and creates the logical circuit"""
        _classical_function_visitor = ClassicalFunctionVisitor()
        _classical_function_visitor.visit(self._ast)
        self._network = _classical_function_visitor._network
        self._scopes = _classical_function_visitor.scopes
        self._args = _classical_function_visitor.args
        self.name = _classical_function_visitor.name

    @property
    def network(self):
        """Returns the logical network"""
        if self._network is None:
            self.compile()
        return self._network

    @property
    def scopes(self):
        """Returns the scope dict"""
        if self._scopes is None:
            self.compile()
        return self._scopes

    @property
    def args(self):
        """Returns the classicalfunction arguments"""
        if self._args is None:
            self.compile()
        return self._args

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

    def simulate(self, bitstring: str) -> bool:
        """Evaluate the expression on a bitstring.

        This evaluation is done classically.

        Args:
            bitstring: The bitstring for which to evaluate.

        Returns:
            bool: result of the evaluation.
        """
        return simulate(self._network, bitstring)

    def simulate_all(self):
        """
        Returns a truth table.

        Returns:
            str: a bitstring with a truth table
        """
        result = list()
        for position in range(2 ** self._network.num_pis()):
            sim_result = "".join([str(int(tt[position])) for tt in self.truth_table])
            result.append(sim_result)

        return "".join(reversed(result))

    @property
    def truth_table(self):
        """Returns (and computes) the truth table"""
        if self._truth_table is None:
            self._truth_table = simulate(self._network)
        return self._truth_table

    def synth(
        self,
        registerless: bool = True,
        synthesizer: Optional[Callable[[ClassicalElement], QuantumCircuit]] = None,
    ) -> QuantumCircuit:
        """Synthesis the logic network into a :class:`~qiskit.circuit.QuantumCircuit`.

        Args:
            registerless: Default ``True``. If ``False`` uses the parameter names to create
            registers with those names. Otherwise, creates a circuit with a flat quantum register.
            synthesizer: Optional. If None tweedledum's pkrm_synth is used.

        Returns:
            QuantumCircuit: A circuit implementing the logic network.
        """
        if registerless:
            qregs = None
        else:
            qregs = self.qregs

        if synthesizer:
            return synthesizer(self)

        return tweedledum2qiskit(pkrm_synth(self.truth_table[0]), name=self.name, qregs=qregs)

    def _define(self):
        """The definition of the classical function is its synthesis"""
        self.definition = self.synth()

    @property
    def qregs(self):
        """The list of qregs used by the classicalfunction"""
        qregs = [QuantumRegister(1, name=arg) for arg in self.args if self.types[0][arg] == "Int1"]
        qregs.reverse()
        if self.types[0]["return"] == "Int1":
            qregs.append(QuantumRegister(1, name="return"))
        return qregs
