# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Phase Oracle object."""

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.classicalfunction.boolean_expression import BooleanExpression


class PhaseOracle(QuantumCircuit):
    r"""Phase Oracle.

    The Phase Oracle object constructs circuits for any arbitrary
    input logical expressions. A logical expression is composed of logical operators
    `&` (`AND`), `|` (`OR`), `~` (`NOT`), and `^` (`XOR`).
    as well as symbols for literals (variables).
    For example, `'a & b'`, and `(v0 | ~v1) & (~v2 & v3)`
    are both valid string representation of boolean logical expressions.

    For convenience, this oracle, in addition to parsing arbitrary logical expressions,
    also supports input strings in the `DIMACS CNF format
    <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
    which is the standard format for specifying SATisfiability (SAT) problem instances in
    `Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__,
    which is a conjunction of one or more clauses, where a clause is a disjunction of one
    or more literals. See :meth:`qiskit.circuit.library.phase_oracle.PhaseOracle.from_dimacs_file`.
    """

    def __init__(self, expression: str) -> None:  # pylint: disable=super-init-not-called
        self.boolean_expression = BooleanExpression(expression)
        # input qubits for the oracle
        self.state_qubits = range(self.boolean_expression.num_qubits - 1)

        self.compose(self._build_from_boolean_expression(), inplace=True)

    def _build_from_boolean_expression(self):
        # initialize the quantumcircuit
        qr_state = QuantumRegister(len(self.state_qubits), 'state')

        super().__init__(qr_state, name='Phase Oracle')

        from tweedledum.passes import pkrm_synth  # pylint: disable=no-name-in-module

        return self.boolean_expression.synth(
            synthesizer=lambda logic_network: pkrm_synth(logic_network,
                                                         {"pkrm_synth": {"phase_esop": True}}))

    def evaluate_bitstring(self, bitstring: str) -> bool:
        """Evaluate the oracle on a bitstring.
        This evaluation is done classically without any quantum circuit.
        Args:
            bitstring: The bitstring for which to evaluate.
        Returns:
            True if the bitstring is a good state, False otherwise.
        """
        return self.boolean_expression.simulate(bitstring)

    @classmethod
    def from_dimacs_file(cls, filename: str):
        r"""Create a PhaseOracle from the string in the DIMACS format.

        It is possible to build a PhaseOracle from a file in `DIMACS CNF format
        <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
        which is the standard format for specifying SATisfiability (SAT) problem instances in
        `Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__,
        which is a conjunction of one or more clauses, where a clause is a disjunction of one
        or more literals.

        The following is an example of a CNF expressed in the DIMACS format:

        .. code:: text

          c DIMACS CNF file with 3 satisfying assignments: 1 -2 3, -1 -2 -3, 1 2 -3.
          p cnf 3 5
          -1 -2 -3 0
          1 -2 3 0
          1 2 -3 0
          1 -2 -3 0
          -1 2 3 0

        The first line, following the `c` character, is a comment. The second line specifies that
        the CNF is over three boolean variables --- let us call them  :math:`x_1, x_2, x_3`, and
        contains five clauses.  The five clauses, listed afterwards, are implicitly joined by the
        logical `AND` operator, :math:`\land`, while the variables in each clause, represented by
        their indices, are implicitly disjoined by the logical `OR` operator, :math:`lor`. The
        :math:`-` symbol preceding a boolean variable index corresponds to the logical `NOT`
        operator, :math:`lnot`. Character `0` (zero) marks the end of each clause.  Essentially,
        the code above corresponds to the following CNF:

        :math:`(\lnot x_1 \lor \lnot x_2 \lor \lnot x_3)
        \land (x_1 \lor \lnot x_2 \lor x_3)
        \land (x_1 \lor x_2 \lor \lnot x_3)
        \land (x_1 \lor \lnot x_2 \lor \lnot x_3)
        \land (\lnot x_1 \lor x_2 \lor x_3)`.


        Args:
            filename (str): A file in DIMACS format.

        Returns:
            PhaseOracle: A quantum circuit with a phase oracle.
        """
        phase_oracle = cls.__new__(cls)
        phase_oracle.boolean_expression = BooleanExpression.from_dimacs_file(filename)

        # input qubits for the oracle
        phase_oracle.state_qubits = range(phase_oracle.boolean_expression.num_qubits - 1)

        phase_oracle.compose(phase_oracle._build_from_boolean_expression(), inplace=True)

        return phase_oracle
