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

"""A class for parsingand  synthesizing boolean expressions"""

import ast
import itertools
import re
from os.path import basename, isfile

from .boolean_expression_visitor import (
    BooleanExpressionEvalVisitor,
    BooleanExpressionArgsCollectorVisitor,
)


class BooleanExpression:
    """A Boolean Expression"""

    def __init__(self, expression: str, name: str = None, var_order: list = None) -> None:
        """
        Args:
            expression (str): The logical expression string.
            name (str): Optional. Instruction gate name. Otherwise part of the expression is
               going to be used.
            var_order(list): A list with the order in which variables will be created.
               (default: by appearance)
        """
        self.expression = expression
        self.expression_ast = ast.parse(expression)
        args_collector = BooleanExpressionArgsCollectorVisitor()
        args_collector.visit(self.expression_ast)
        self.args = args_collector.get_sorted_args()
        if var_order is not None:
            missing_args = set(self.args) - set(var_order)
            if len(missing_args) > 0:
                raise ValueError(f"var_order missing the variable(s) {', '.join(missing_args)}")
            self.args.sort(key=var_order.index)

    def simulate(self, bitstring: str) -> bool:
        """Evaluate the expression on a bitstring.

        This evaluation is done classically.

        Args:
            bitstring: The bitstring for which to evaluate.

        Returns:
            bool: result of the evaluation.
        """
        eval_visitor = BooleanExpressionEvalVisitor()
        if len(self.args) != len(bitstring):
            raise ValueError(
                f"bitstring length differs from the number of variables "
                f"({len(bitstring)} != {len(self.args)})"
            )
        for arg, bit in zip(self.args, bitstring):
            if not bit in ["0", "1"]:
                raise ValueError("bitstring must be composed of 0 and 1 only")
            eval_visitor.arg_values[arg] = bit == "1"
        return eval_visitor.visit(self.expression_ast)

    def truth_table(self) -> dict:
        """Generates the full truth table for the expression
        Returns:
            dict: A dictionary mapping boolean assignments to the boolean result
        """
        return {
            assignment: self.simulate("".join("1" if val else "0" for val in assignment))
            for assignment in itertools.product([False, True], repeat=len(self.args))
        }

    def synth(self, circuit_type: str = "bit"):
        r"""Synthesize the logic network into a :class:`~qiskit.circuit.QuantumCircuit`.
        There are two common types of circuits for a boolean function :math:`f(x)`:

        1. **Bit-flip oracles** which compute:

         .. math::

            |x\rangle|y\rangle |-> |x\rangle|f(x)\oplusy\rangle

        2. **Phase-flip** oracles which compute:

         .. math::

            |x\rangle |-> (-1)^{f(x)}|x\rangle

        By default the bit-flip oracle is generated.

        Args:
            circuit_type: which type of oracle to create, 'bit' or 'phase' flip oracle.
        Returns:
            QuantumCircuit: A circuit implementing the logic network.
        Raises:
            ValueError: If ``circuit_type`` is not either 'bit' or 'phase'.
        """
        # pylint: disable=cyclic-import
        from .boolean_expression_synth import (
            synth_bit_oracle_from_esop,
            synth_phase_oracle_from_esop,
            EsopGenerator,
        )  # import here to avoid cyclic import

        # generating the esop currntly requires generating the full truth table
        # there are many optimizations that can be done to improve this step
        esop = EsopGenerator(self.truth_table()).esop
        if circuit_type == "bit":
            return synth_bit_oracle_from_esop(esop, len(self.args) + 1)
        if circuit_type == "phase":
            return synth_phase_oracle_from_esop(esop, len(self.args))
        raise ValueError("'circuit_type' must be either 'bit' or 'phase'")

    def _define(self):
        """The definition of the boolean expression is its synthesis"""
        self.definition = self.synth()

    @staticmethod
    def from_dimacs(dimacs: str, name: str = None):
        """Create a BooleanExpression from a string in the DIMACS format.
        Args:
            dimacs : A string in DIMACS format.
            name: an optional name for the BooleanExpression

        Returns:
            BooleanExpression: A gate for the input string

        Raises:
            ValueError: If the string is not formatted according to DIMACS rules
        """
        header_regex = re.compile(r"p\s+cnf\s+(\d+)\s+(\d+)")
        clause_regex = re.compile(r"(-?\d+)")
        lines = [
            line for line in dimacs.split("\n") if not line.startswith("c") and line != ""
        ]  # DIMACS comment line start with c
        header_match = header_regex.match(lines[0])
        if not header_match:
            raise ValueError("First line must start with 'p cnf'")
        num_vars, _ = map(int, header_match.groups())
        variables = [f"x{i+1}" for i in range(num_vars)]
        clauses = []
        for line in lines[1:]:
            literals = clause_regex.findall(line)
            if len(literals) == 0 or literals[-1] != "0":
                continue
            clauses.append([int(c) for c in literals[:-1]])
        clause_strings = [
            " | ".join([f'{"~" if lit < 0 else ""}{variables[abs(lit)-1]}' for lit in clause])
            for clause in clauses
        ]
        expr = " & ".join([f"({c})" for c in clause_strings])
        return BooleanExpression(expr, name=name, var_order=variables)

    @staticmethod
    def from_dimacs_file(filename: str):
        """Create a BooleanExpression from a file in the DIMACS format.
        Args:
            filename: A file in DIMACS format.

        Returns:
            BooleanExpression: A gate for the input string

        Raises:
            FileNotFoundError: If filename is not found.
        """
        if not isfile(filename):
            raise FileNotFoundError(f"The file {filename} does not exists.")
        with open(filename, "r") as dimacs_file:
            dimacs = dimacs_file.read()
        return BooleanExpression.from_dimacs(dimacs, name=basename(filename))
