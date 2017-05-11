"""
OPENQASM Lexer.

This is a wrapper around the PLY lexer to support the "include" statement
by creating a stack of lexers.

Author: Jim Challenger
"""
import os
import ply.lex as lex
from ._qasmexception import QasmException
from . import _node as node

CORE_LIBS_PATH = os.path.join(os.path.dirname(__file__), 'libs')
# TODO: Get dinamically from the folder "qasm/lib"
CORE_LIBS = ['qelib1.inc']


class QasmLexer(object):
    """OPENQASM Lexer.

    This is a wrapper around the PLY lexer to support the "include" statement
    by creating a stack of lexers.
    """

    def __mklexer__(self, filename):
        """Create a PLY lexer."""
        self.lexer = lex.lex(module=self, debug=False)
        self.filename = filename
        self.lineno = 1

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
        self.data = open(filename).read()
        self.lexer.input(self.data)

    # ---- Beginning of the PLY lexer ----
    literals = r'=()[]{};<>,.+-/*"'
    tokens = (
        'NNINTEGER',
        'BARRIER',
        'OPAQUE',
        'RESET',
        'IF',
        'REAL',
        'QREG',
        'CREG',
        'GATE',
        'PI',
        'CX',
        'U',
        'MEASURE',
        'MAGIC',
        'ASSIGN',
        'MATCHES',
        'ID',
        'STRING',
    )

    def t_REAL(self, t):
        r'(([0-9]+|([0-9]+)?\.[0-9]+|[0-9]+\.)[eE][+-]?[0-9]+)|(([0-9]+)?\.[0-9]+|[0-9]+\.)'
        t.value = float(t.value)
        # tad nasty, see mkfloat.py to see how this is derived from python spec
        return t

    def t_NNINTEGER(self, t):
        r'[1-9]+[0-9]*|0'
        t.value = int(t.value)
        return t

    def t_QREG(self, t):
        'qreg'
        return t

    def t_CREG(self, t):
        'creg'
        return t

    def t_GATE(self, t):
        'gate'
        return t

    def t_MEASURE(self, t):
        'measure'
        return t

    def t_IF(self, t):
        'if'
        return t

    def t_RESET(self, t):
        'reset'
        return t

    def t_ASSIGN(self, t):
        '->'
        return t

    def t_MATCHES(self, t):
        '=='
        return t

    def t_BARRIER(self, t):
        'barrier'
        return t

    def t_OPAQUE(self, t):
        'opaque'
        return t

    def t_STRING(self, t):
        r'\"([^\\\"]|\\.)*\"'
        return t

    def t_INCLUDE(self, t):
        'include'

        '''
        Now eat up the next two tokens which must be
        1 - the name of the include file, and
        2 - a terminating semicolon

        Then push the current lexer onto the stack, create a new one from
        the include file, and push it onto the stack.

        When we hit eof (the t_eof) rule, we pop.
        '''
        next = self.lexer.token()
        lineno = next.lineno
        # print('NEXT', next, "next.value", next.value, type(next))
        if isinstance(next.value, str):
            incfile = next.value.strip('"')
        else:
            raise QasmException("Invalid include: must be a quoted string.")

        if incfile in CORE_LIBS:
            incfile = os.path.join(CORE_LIBS_PATH, incfile)

        next = self.lexer.token()
        if next is None or next.value != ';':
            raise QasmException('Invalid syntax, missing ";" at line',
                                str(lineno))

        if not os.path.exists(incfile):
            raise QasmException('Include file', incfile,
                                'cannot be found, line', str(next.lineno),
                                ', file', self.filename)
        self.push(incfile)
        return self.lexer.token()

    def t_MAGIC(self, t):
        'OPENQASM'
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

        if t.value == 'pi':
            t.type = 'PI'
            return t

        t.value = node.Id(t.value, self.lineno, self.filename)
        return t

    def t_newline(self, t):
        r'\n+'
        self.lineno += len(t.value)
        t.lexer.lineno = self.lineno

    def t_eof(self, t):
        if len(self.stack) > 0:
            self.pop()
            return self.lexer.token()
        else:
            return None

    t_ignore = ' \t\r'

    def t_error(self, t):
        print("Unable to match any token rule, got -->%s<--" % t.value[0])
        print("Check your OPENQASM source and any include statements.")
        # t.lexer.skip(1)
