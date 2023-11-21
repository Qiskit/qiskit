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
"""Pygments tools for Qasm.
"""

from pygments.lexer import RegexLexer
from pygments.token import Comment, String, Keyword, Name, Number, Text
from pygments.style import Style


class QasmTerminalStyle(Style):
    """A style for OpenQasm in a Terminal env (e.g. Jupyter print)."""

    styles = {
        String: "ansibrightred",
        Number: "ansibrightcyan",
        Keyword.Reserved: "ansibrightgreen",
        Keyword.Declaration: "ansibrightgreen",
        Keyword.Type: "ansibrightmagenta",
        Name.Builtin: "ansibrightblue",
        Name.Function: "ansibrightyellow",
    }


class QasmHTMLStyle(Style):
    """A style for OpenQasm in a HTML env (e.g. Jupyter widget)."""

    styles = {
        String: "ansired",
        Number: "ansicyan",
        Keyword.Reserved: "ansigreen",
        Keyword.Declaration: "ansigreen",
        Keyword.Type: "ansimagenta",
        Name.Builtin: "ansiblue",
        Name.Function: "ansiyellow",
    }


class OpenQASMLexer(RegexLexer):
    """A pygments lexer for OpenQasm."""

    name = "OpenQASM"
    aliases = ["qasm"]
    filenames = ["*.qasm"]

    gates = [
        "id",
        "cx",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "h",
        "t",
        "tdg",
        "ccx",
        "c3x",
        "c4x",
        "c3sqrtx",
        "rx",
        "ry",
        "rz",
        "cz",
        "cy",
        "ch",
        "swap",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cu3",
        "rxx",
        "rzz",
        "rccx",
        "rc3x",
        "u1",
        "u2",
        "u3",
    ]

    tokens = {
        "root": [
            (r"\n", Text),
            (r"[^\S\n]+", Text),
            (r"//\n", Comment),
            (r"//.*?$", Comment.Single),
            # Keywords
            (r"(OPENQASM|include)\b", Keyword.Reserved, "keywords"),
            (r"(qreg|creg)\b", Keyword.Declaration),
            # Treat 'if' special
            (r"(if)\b", Keyword.Reserved, "if_keywords"),
            # Constants
            (r"(pi)\b", Name.Constant),
            # Special
            (r"(barrier|measure|reset)\b", Name.Builtin, "params"),
            # Gates (Types)
            ("(" + "|".join(gates) + r")\b", Keyword.Type, "params"),
            (r"[unitary\d+]", Keyword.Type),
            # Functions
            (r"(gate)\b", Name.Function, "gate"),
            # Generic text
            (r"[a-zA-Z_][a-zA-Z0-9_]*", Text, "index"),
        ],
        "keywords": [
            (r'\s*("([^"]|"")*")', String, "#push"),
            (r"\d+", Number, "#push"),
            (r".*\(", Text, "params"),
        ],
        "if_keywords": [
            (r"[a-zA-Z0-9_]*", String, "#pop"),
            (r"\d+", Number, "#push"),
            (r".*\(", Text, "params"),
        ],
        "params": [
            (r"[a-zA-Z_][a-zA-Z0-9_]*", Text, "#push"),
            (r"\d+", Number, "#push"),
            (r"(\d+\.\d*|\d*\.\d+)([eEf][+-]?[0-9]+)?", Number, "#push"),
            (r"\)", Text),
        ],
        "gate": [(r"[unitary\d+]", Keyword.Type, "#push"), (r"p\d+", Text, "#push")],
        "index": [(r"\d+", Number, "#pop")],
    }
