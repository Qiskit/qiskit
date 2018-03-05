# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
OPENQASM Lexer.

This is a wrapper around the PLY lexer to support the "include" statement
by creating a stack of lexers.
"""

import os

import ply.lex as lex
from sympy import Number

from . import _node as node
from ._qasmerror import QasmError

CORE_LIBS_PATH = os.path.join(os.path.dirname(__file__), 'libs')
CORE_LIBS = os.listdir(CORE_LIBS_PATH)


class QasmLexer(object):
    """OPENQASM Lexer.

    This is a wrapper around the PLY lexer to support the "include" statement
    by creating a stack of lexers.
    """
    # pylint: disable=invalid-name,missing-docstring,unused-argument
    # pylint: disable=attribute-defined-outside-init

    def __mklexer__(self, filename):
        """Create a PLY lexer."""
        self.lexer = lex.lex(module=self, debug=False)
        self.filename = filename
        self.lineno = 1

        if filename:
            with open(filename, 'r') as ifile:
                self.data = ifile.read()
            self.lexer.input(self.data)

    def __init__(self, filename):
        """Create the OPENQASM lexer."""
        self.__mklexer__(filename)
        self.stack = []

    def input(self, data):
        """Set the input text data."""
        self.data = data
        self.lexer.input(data)

    def token(self):
        """Return the next token."""
        ret = self.lexer.token()
        return ret

    def pop(self):
        """Pop a PLY lexer off the stack."""
        self.lexer = self.stack.pop()
        self.filename = self.lexer.qasm_file
        self.lineno = self.lexer.qasm_line

    def push(self, filename):
        """Push a PLY lexer on the stack to parse filename."""
        self.lexer.qasm_file = self.filename
        self.lexer.qasm_line = self.lineno
        self.stack.append(self.lexer)
        self.__mklexer__(filename)

    # ---- Beginning of the PLY lexer ----
    literals = r'=()[]{};<>,.+-/*^"'
    reserved = {
        'barrier': 'BARRIER',
        'creg': 'CREG',
        'gate': 'GATE',
        'if': 'IF',
        'measure': 'MEASURE',
        'opaque': 'OPAQUE',
        'qreg': 'QREG',
        'pi': 'PI',
        'reset': 'RESET',
    }
    tokens = [
        'NNINTEGER',
        'REAL',
        'CX',
        'U',
        'FORMAT',
        'ASSIGN',
        'MATCHES',
        'ID',
        'STRING',
    ] + list(reserved.values())

    def t_REAL(self, t):
        r'(([0-9]+|([0-9]+)?\.[0-9]+|[0-9]+\.)[eE][+-]?[0-9]+)|(([0-9]+)?\.[0-9]+|[0-9]+\.)'
        t.value = Number(t.value)
        # tad nasty, see mkfloat.py to see how this is derived from python spec
        return t

    def t_NNINTEGER(self, t):
        r'[1-9]+[0-9]*|0'
        t.value = int(t.value)
        return t

    def t_ASSIGN(self, t):
        '->'
        return t

    def t_MATCHES(self, t):
        '=='
        return t

    def t_STRING(self, t):
        r'\"([^\\\"]|\\.)*\"'
        return t

    def t_INCLUDE(self, t):
        'include'
        #
        # Now eat up the next two tokens which must be
        # 1 - the name of the include file, and
        # 2 - a terminating semicolon
        #
        # Then push the current lexer onto the stack, create a new one from
        # the include file, and push it onto the stack.
        #
        # When we hit eof (the t_eof) rule, we pop.
        next_token = self.lexer.token()
        lineno = next_token.lineno
        # print('NEXT', next, "next.value", next.value, type(next))
        if isinstance(next_token.value, str):
            incfile = next_token.value.strip('"')
        else:
            raise QasmError("Invalid include: must be a quoted string.")

        if incfile in CORE_LIBS:
            incfile = os.path.join(CORE_LIBS_PATH, incfile)

        next_token = self.lexer.token()
        if next_token is None or next_token.value != ';':
            raise QasmError('Invalid syntax, missing ";" at line', str(lineno))

        if not os.path.exists(incfile):
            raise QasmError(
                'Include file %s cannot be found, line %s, file %s' %
                (incfile, str(next_token.lineno), self.filename))
        self.push(incfile)
        return self.lexer.token()

    def t_FORMAT(self, t):
        r'OPENQASM\s+(\d+)\.(\d+)'
        return t

    def t_COMMENT(self, t):
        r'//.*'
        pass

    def t_CX(self, t):
        'CX'
        return t

    def t_U(self, t):
        'U'
        return t

    def t_ID(self, t):
        r'[a-z][a-zA-Z0-9_]*'

        t.type = self.reserved.get(t.value, 'ID')
        if t.type == 'ID':
            t.value = node.Id(t.value, self.lineno, self.filename)
        return t

    def t_newline(self, t):
        r'\n+'
        self.lineno += len(t.value)
        t.lexer.lineno = self.lineno

    def t_eof(self, t):
        if self.stack:
            self.pop()
            return self.lexer.token()
        return None

    t_ignore = ' \t\r'

    def t_error(self, t):
        print("Unable to match any token rule, got -->%s<--" % t.value[0])
        print("Check your OPENQASM source and any include statements.")
        # t.lexer.skip(1)
