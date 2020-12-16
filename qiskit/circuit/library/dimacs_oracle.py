# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A quantum oracle constructed from a DIMACS format logical expression."""

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.classicalfunction import ClassicalFunction


class DIMACSOracle(QuantumCircuit):
    r"""The Logical Expression Quantum Oracle.

    The Logical Expression Oracle, as its name suggests, constructs circuits for any arbitrary
    input logical expressions. A logical expression is composed of logical operators
    `&` (`AND`), `|` (`OR`), `~` (`NOT`), and `^` (`XOR`),
    as well as symbols for literals (variables).
    For example, `'a & b'`, and `(v0 | ~v1) ^ (~v2 & v3)`
    are both valid string representation of boolean logical expressions.

    For convenience, this oracle, in addition to parsing arbitrary logical expressions,
    also supports input strings in the `DIMACS CNF format
    <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
    which is the standard format for specifying SATisfiability (SAT) problem instances in
    `Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__,
    which is a conjunction of one or more clauses, where a clause is a disjunction of one
    or more literals.

    The following is an example of a CNF expressed in DIMACS format:

    .. code:: text

      c This is an example DIMACS CNF file with 3 satisfying assignments: 1 -2 3, -1 -2 -3, 1 2 -3.
      p cnf 3 5
      -1 -2 -3 0
      1 -2 3 0
      1 2 -3 0
      1 -2 -3 0
      -1 2 3 0

    The first line, following the `c` character, is a comment. The second line specifies that the
    CNF is over three boolean variables --- let us call them  :math:`x_1, x_2, x_3`, and contains
    five clauses.  The five clauses, listed afterwards, are implicitly joined by the logical `AND`
    operator, :math:`\land`, while the variables in each clause, represented by their indices,
    are implicitly disjoined by the logical `OR` operator, :math:`lor`. The :math:`-` symbol
    preceding a boolean variable index corresponds to the logical `NOT` operator, :math:`lnot`.
    Character `0` (zero) marks the end of each clause.  Essentially, the code above corresponds
    to the following CNF:

    :math:`(\lnot x_1 \lor \lnot x_2 \lor \lnot x_3)
    \land (x_1 \lor \lnot x_2 \lor x_3)
    \land (x_1 \lor x_2 \lor \lnot x_3)
    \land (x_1 \lor \lnot x_2 \lor \lnot x_3)
    \land (\lnot x_1 \lor x_2 \lor x_3)`.

    This is an example showing how to search for a satisfying assignment to an SAT problem encoded
    in DIMACS using the `Logical Expression oracle with the Grover algorithm.
    <https://github.com/Qiskit/qiskit-tutorials-community/blob/master/optimization/grover.ipynb>`__

    Logic expressions, regardless of the input formats, are parsed and stored as Abstract Syntax
    Tree (AST) tuples, from which the corresponding circuits are constructed. The oracle circuits
    can then be used with any oracle-oriented algorithms when appropriate. For example, an oracle
    built from a DIMACS input can be used with the Grover's algorithm to search for a satisfying
    assignment to the encoded SAT instance.

    By default, the Logical Expression oracle will not try to apply any optimization when building
    the circuits. For any DIMACS input, the constructed circuit truthfully recreates each inner
    disjunctive clauses as well as the outermost conjunction; For other arbitrary input expression,
    It only tries to convert it to a CNF or DNF (Disjunctive Normal Form, similar to CNF, but with
    inner conjunctions and a outer disjunction) before constructing its circuit. This, for example,
    could be good for educational purposes, where a user would like to compare a built circuit
    against their input expression to examine and analyze details. However, this often leads
    to relatively deep circuits that possibly also involve many ancillary qubits. The oracle
    therefore, provides the option to try to optimize the input logical expression before
    building its circuit.
    """

    def __init__(self, dimacs: str) -> None:
        """
        Args:
            dimacs: The string in the DIMACS format.
        """
        # convert the dimacs string to a AST source and create a classicalfunction object
        source = self._dimacs_to_source(dimacs)
        classicalfunction = ClassicalFunction(source)

        # store the classically simulated results for the ``evaluate_bitstring`` method
        self._result_lookup = classicalfunction.simulate()

        # build the circuit
        oracle = classicalfunction.synth()

        # initialize the quantumcircuit
        qr_state = QuantumRegister(oracle.num_qubits - 1, 'q')
        qr_flag = QuantumRegister(1, 'state')
        super().__init__(qr_state, qr_flag, name='DIMACS Oracle')

        # to convert from the bitflip oracle provided from classicalfunction we
        # additionally apply a hadamard and X gates
        self.x(qr_flag)
        self.h(qr_flag)
        self.compose(oracle, inplace=True)
        self.h(qr_flag)
        self.x(qr_flag)

    def _dimacs_to_source(self, dimacs, name='f'):
        # sanitize input and convert the dimacs string to a list
        lines = []
        for line in dimacs.strip().split('\n'):
            if line[0] == 'c':
                continue  # skip comments
            if len(line) == 0:
                continue  # skip empty lines
            lines.append(line.strip().lower())

        # check the header is actually in DIMACS CNF format
        if not lines[0][:6] == 'p cnf ':
            raise ValueError('Unrecognized DIMACS CNF header {}.'.format(lines[0]))

        # the number of variables is a value in the header
        num_var = int(lines[0].split()[2])

        # create the variables and header for the AST
        argnames = [f'x{i+1}' for i in range(num_var)]
        parsed = f'def {name}('
        for arg in argnames[:-1]:
            parsed += f'{arg}: Int1, '
        parsed += f'{argnames[-1]}: Int1) -> Int1:\n'
        indent = 4 * ' '
        clauses = [f'c{i}' for i, _ in enumerate(lines[1:])]

        # add one clause variable for each line
        for i, line in enumerate(lines[1:]):
            toks = line.split()
            print(toks)
            if not toks[-1] == '0':
                raise ValueError('Unrecognized dimacs line {}.'.format(line))

            parsed += f'{indent}{clauses[i]} = '
            parsed += ' or '.join(['not ' + argnames[int(t[1:])-1]
                                   if t[0] == '-' else argnames[int(t)-1] for t in toks[:-1]])
            parsed += '\n'

        # combine all clauses
        parsed += f'{indent}return ' + ' and '.join(clauses)

        return parsed

    def evaluate_bitstring(self, bitstring: str) -> bool:
        """Evaluate the oracle on a bitstring.

        This evaluation is done classically without any quantum circuit.

        Args:
            bitstring: The bitstring for which to evaluate.

        Returns:
            True if the bitstring is a good state, False otherwise.
        """
        index = int(bitstring, 2)
        return self._result_lookup[index] == '0'
