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
try:
    from tweedledum.passes import xag_synth  # pylint: disable=no-name-in-module
    HAS_TWEEDLEDUM = True
except Exception:  # pylint: disable=broad-except
    HAS_TWEEDLEDUM = False
from qiskit.circuit import quantumregister
from qiskit.circuit import gate
from qiskit.exceptions import QiskitError
from .utils import tweedledum2qiskit
from .classical_function_visitor import ClassicalFunctionVisitor


class ClassicalFunction(gate.Gate):
    """Represent a classical function function and its logic network."""

    def __init__(self, source, name=None):
        """Creates a ``ClassicalFunction`` from Python source code in ``source``.

        The code should be a single function with types.

        Args:
            source (str): Python code with type hints.
            name (str): Optional. Default: "*classicalfunction*". ClassicalFunction name.
        Raises:
            ImportError: If tweedledum is not installed.
            QiskitError: If source is not a string.
        """
        if not isinstance(source, str):
            raise QiskitError('ClassicalFunction needs a source code as a string.')
        if not HAS_TWEEDLEDUM:
            raise ImportError("To use the classicalfunction compiler, tweedledum "
                              "must be installed. To install tweedledum run "
                              '"pip install tweedledum".')
        self._ast = ast.parse(source)
        self._network = None
        self._scopes = None
        self._args = None
        super().__init__(name or '*classicalfunction*',
                         num_qubits=sum([qreg.size for qreg in self.qregs]),
                         params=[])

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
        from tweedledum.classical import simulate  # pylint: disable=no-name-in-module
        return simulate(self._network, bitstring)

    def simulate_all(self):
        from tweedledum.classical import simulate  # pylint: disable=no-name-in-module

        _truth_table = simulate(self._network)

        result = list()
        for position in range(2 ** self._network.num_pis()):
            sim_result = ''.join([str(int(tt[position])) for tt in _truth_table])
            result.append(sim_result)

        return ''.join(reversed(result))


    def synth(self, registerless=True):
        """Synthesis the logic network into a :class:`~qiskit.circuit.QuantumCircuit`.

        Args:
            registerless (bool): Default ``True``. If ``False`` uses the parameter names to create
            registers with those names. Otherwise, creates a circuit with a flat quantum register.

        Returns:
            QuantumCircuit: A circuit implementing the logic network.
        """
        from tweedledum.passes import pkrm_synth  # pylint: disable=no-name-in-module

        if registerless:
            qregs = None
        else:
            qregs = self.qregs

        return tweedledum2qiskit(pkrm_synth(self._network), name=self.name, qregs=qregs)

    def _define(self):
        """The definition of the classical function is its synthesis"""
        self.definition = self.synth()

    @property
    def qregs(self):
        """The list of qregs used by the classicalfunction"""
        qregs = [
            quantumregister.QuantumRegister(
                1, name=arg) for arg in self.args if self.types[0][arg] == 'Int1']
        qregs.reverse()
        if self.types[0]['return'] == 'Int1':
            qregs.append(quantumregister.QuantumRegister(1, name='return'))
        return qregs
