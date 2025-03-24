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

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, Gate

from qiskit.synthesis.boolean.boolean_expression import BooleanExpression


class PhaseOracle(QuantumCircuit):
    r"""Phase Oracle.

    The Phase Oracle object constructs circuits for any arbitrary
    input logical expressions. A logical expression is composed of logical operators
    `&` (logical `AND`), `|` (logical  `OR`),
    `~` (logical  `NOT`), and `^` (logical  `XOR`).
    as well as symbols for literals (variables).
    For example, `'a & b'`, and `(v0 | ~v1) & (~v2 & v3)`
    are both valid string representation of boolean logical expressions.

    A phase oracle for a boolean function `f(x)` performs the following
    quantum operation:

    .. math::

            |x\rangle \mapsto (-1)^{f(x)}|x\rangle

    For convenience, this oracle, in addition to parsing arbitrary logical expressions,
    also supports input strings in the `DIMACS CNF format
    <https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
    which is the standard format for specifying SATisfiability (SAT) problem instances in
    `Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__,
    which is a conjunction of one or more clauses, where a clause is a disjunction of one
    or more literals. See :meth:`qiskit.circuit.library.phase_oracle.PhaseOracle.from_dimacs_file`.

    From 16 variables on, possible performance issues should be expected when using the
    default synthesizer.
    """

    def __init__(
        self,
        expression: str,
        var_order: list[str] | None = None,
    ) -> None:
        """
        Args:
            expression: A Python-like boolean expression.
            var_order: A list with the order in which variables will be created.
               (default: by appearance)
        """
        self.boolean_expression = BooleanExpression(expression, var_order=var_order)
        oracle = self.boolean_expression.synth(circuit_type="phase")

        super().__init__(oracle.num_qubits, name="Phase Oracle")

        self.compose(oracle, inplace=True, copy=False)

    def evaluate_bitstring(self, bitstring: str) -> bool:
        """Evaluate the oracle on a bitstring.
        This evaluation is done classically without any quantum circuit.

        Args:
            bitstring: The bitstring for which to evaluate. The input bitstring is expected to be
                in little-endian order.

        Returns:
            True if the bitstring is a good state, False otherwise.
        """
        return self.boolean_expression.simulate(bitstring[::-1])

    @classmethod
    def from_dimacs_file(cls, filename: str):
        r"""Create a PhaseOracle from the string in the DIMACS format.

        It is possible to build a PhaseOracle from a file in `DIMACS CNF format
        <https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
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
            filename: A file in DIMACS format.

        Returns:
            PhaseOracle: A quantum circuit with a phase oracle.
        """
        expr = BooleanExpression.from_dimacs_file(filename)
        return cls(expr)


class PhaseOracleGate(Gate):
    r"""Implements a phase oracle.

    The Phase Oracle Gate object constructs circuits for any arbitrary
    input logical expressions. A logical expression is composed of logical operators
    `&` (logical `AND`), `|` (logical  `OR`),
    `~` (logical  `NOT`), and `^` (logical  `XOR`).
    as well as symbols for literals (variables).
    For example, `'a & b'`, and `(v0 | ~v1) & (~v2 & v3)`
    are both valid string representation of boolean logical expressions.

    A phase oracle for a boolean function `f(x)` performs the following
    quantum operation:

    .. math::

            |x\rangle \mapsto (-1)^{f(x)}|x\rangle

    For convenience, this oracle, in addition to parsing arbitrary logical expressions,
    also supports input strings in the `DIMACS CNF format
    <https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
    which is the standard format for specifying SATisfiability (SAT) problem instances in
    `Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__,
    which is a conjunction of one or more clauses, where a clause is a disjunction of one
    or more literals. See :meth:`qiskit.circuit.library.phase_oracle.PhaseOracleGate.from_dimacs_file`.

    From 16 variables on, possible performance issues should be expected when using the
    default synthesizer.
    """

    def __init__(
        self,
        expression: str,
        var_order: list[str] | None = None,
        label: str | None = None,
    ) -> None:
        """
        Args:
            expression: A Python-like boolean expression.
            var_order: A list with the order in which variables will be created.
               (default: by appearance)
            label: A label for the gate to display in visualizations. Per default, the label is
                set to display the textual represntation of the boolean expression (truncated if needed)
        """
        self.boolean_expression = BooleanExpression(expression, var_order=var_order)

        if label is None:
            short_expr_for_name = (expression[:15] + "...") if len(expression) > 15 else expression
            label = short_expr_for_name

        super().__init__(
            name="Phase Oracle",
            num_qubits=self.boolean_expression.num_bits,
            params=[],
            label=label,
        )

    def _define(self):
        """Defined by the synthesized phase oracle"""
        self.definition = self.boolean_expression.synth(circuit_type="phase")

    @classmethod
    def from_dimacs_file(cls, filename: str) -> PhaseOracleGate:
        r"""Create a PhaseOracle from the string in the DIMACS format.

        It is possible to build a PhaseOracle from a file in `DIMACS CNF format
        <https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
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
        their indices, are implicitly disjoined by the logical `OR` operator, :math:`\lor`. The
        :math:`-` symbol preceding a boolean variable index corresponds to the logical `NOT`
        operator, :math:`\lnot`. Character `0` (zero) marks the end of each clause.  Essentially,
        the code above corresponds to the following CNF:

        :math:`(\lnot x_1 \lor \lnot x_2 \lor \lnot x_3)
        \land (x_1 \lor \lnot x_2 \lor x_3)
        \land (x_1 \lor x_2 \lor \lnot x_3)
        \land (x_1 \lor \lnot x_2 \lor \lnot x_3)
        \land (\lnot x_1 \lor x_2 \lor x_3)`.


        Args:
            filename: A file in DIMACS format.

        Returns:
            PhaseOracleGate: A quantum circuit with a phase oracle.
        """
        expr = BooleanExpression.from_dimacs_file(filename)
        return cls(expr)
