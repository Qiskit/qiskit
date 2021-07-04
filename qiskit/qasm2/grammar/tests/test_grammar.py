import io
import os
from contextlib import redirect_stderr
import pytest

import yaml

from antlr4 import *
from antlr4.tree.Trees import Trees

# add lexer, parser to Python path and import
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qasm2Lexer import qasm2Lexer
from qasm2Parser import qasm2Parser


def get_pretty_tree(
    tree: "ParseTree", rule_names: list = None, parser: Parser = None, level: int = 0
) -> str:
    """Take antlr ``ParseTree`` and return indented tree format for test comparison.

    Adapted from ``antrl4.tree.Trees.toStringTree()`` method.

    Args:
        tree: The antlr parse tree.
        rule_names: Names of parser rules.
        parser: The parser used to generated the tree.
        level: Level of tree (used for indentation).

    Returns:
        Pretty tree format (indents of one space at each level).
    """
    indent_value = "  "  # indent using two spaces to match ``yaml`` reference files

    if parser is not None:
        rule_names = parser.ruleNames

    node_text = Trees.getNodeText(tree, rule_names)
    pretty_tree = level * indent_value + node_text + "\n"

    if tree.getChildCount() > 0:
        for i in range(0, tree.getChildCount()):
            pretty_tree += get_pretty_tree(tree.getChild(i), rule_names=rule_names, level=level + 1)

    return pretty_tree


def build_parse_tree(input_str: str, using_file: bool = False) -> str:
    """Build indented parse tree in string format.

    Args:
        input_str: Input program or file path.
        using_file: Whether input string is source program or file path.

    Raises:
        Exception: If build fails (at any stage: lexing or parsing).

    Returns:
        Parse tree string in indented format.
    """
    input = FileStream(input_str, encoding="utf-8") if using_file else InputStream(input_str)
    pretty_tree = ""
    # antlr errors (lexing and parsing) sent to stdout -> redirect to variable err
    with io.StringIO() as err, redirect_stderr(err):
        lexer = qasm2Lexer(input)
        stream = CommonTokenStream(lexer)
        parser = qasm2Parser(stream)
        tree = parser.program()

        pretty_tree = get_pretty_tree(tree, None, parser)

        error = err.getvalue()
        if error:
            raise Exception("Parse tree build failed. Error:\n" + error)

    return pretty_tree


class TestGrammar:
    """Test the ANTLR grammar w/ pytest."""

    @pytest.fixture(scope="function", autouse=True)
    def setup(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))  # tests/ dir
        root_dir = os.path.dirname(test_dir)  # project root dir
        self.examples_path = os.path.join(root_dir, "examples/")
        self.test_path = os.path.join(test_dir, "outputs")

    def load_and_compare_yaml(self, test_str):
        """Process test yaml files. Yaml is expected to contain OpenQasm3.0 source code, which is
        parsed. The resulting parse tree is compared to a reference output.

        The yaml keys are ``source`` and ``reference``, respectively.

        Args:
            test_str (str): Relative path of test yaml file, ie ``add.yaml``.
        """
        if not "yaml" in test_str:
            raise ValueError("Test file should be in YAML format.")

        test_path = os.path.join(self.test_path, test_str)
        with open(test_path) as test_file:
            test_dict = yaml.load(test_file, Loader=yaml.FullLoader)

        if sorted(list(test_dict.keys())) != ["reference", "source"]:
            raise KeyError("Reference YAML file contain only ``source`` and ``reference`` keys.")

        qasm_source = test_dict["source"]
        parse_tree = build_parse_tree(qasm_source)

        reference = test_dict["reference"]
        assert parse_tree == reference

    def test_header(self):
        """Test header."""
        self.load_and_compare_yaml("header.yaml")

    def test_global_statement(self):
        """Test global statements."""
        self.load_and_compare_yaml("subroutine.yaml")
        self.load_and_compare_yaml("kernel.yaml")
        self.load_and_compare_yaml("quantum_gate.yaml")
        self.load_and_compare_yaml("empty_gate.yaml")
        # TODO: Add calibration test when pulse grammar is filled in

    def test_declaration(self):
        """Test classical and quantum declaration."""
        self.load_and_compare_yaml("declaration.yaml")

    def test_pragma(self):
        """Test pragma directive."""
        self.load_and_compare_yaml("pragma.yaml")

    def test_expression(self):
        """Test expressions."""
        self.load_and_compare_yaml("binary_expr.yaml")
        self.load_and_compare_yaml("order_of_ops.yaml")
        self.load_and_compare_yaml("unary_expr.yaml")
        self.load_and_compare_yaml("built_in_call.yaml")
        self.load_and_compare_yaml("sub_and_kern_call.yaml")

    def test_assignment(self):
        """Test assignment statements."""
        self.load_and_compare_yaml("assignment.yaml")

    def test_branching(self):
        """Test branching statements."""
        self.load_and_compare_yaml("branching.yaml")
        self.load_and_compare_yaml("branch_binop.yaml")

    def test_loop_and_control_directive(self):
        """Test loop and control directive statements."""
        self.load_and_compare_yaml("loop.yaml")

    def test_aliasing(self):
        """Test alias statements."""
        self.load_and_compare_yaml("alias.yaml")

    def test_examples(self):
        """Loop through all example files, parse and verify that no errors are raised.

        Examples located at: ``qiskit/qasm2/grammar/examples``.
        """
        examples = os.listdir(self.examples_path)
        for e in examples:
            example_file = os.path.join(self.examples_path, e)
            if os.path.isfile(example_file):
                tree = build_parse_tree(example_file, using_file=True)
