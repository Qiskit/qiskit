#!/usr/bin/env python
# Author: Jim Challenger, Andrew Cross
import os
import sys
import math
import traceback
import ply.lex as lex
import ply.yacc as yacc

class QasmException(Exception):
    def __init__(self, *msg):
        self.msg = ' '.join(msg)

    def __str__(self):
        return repr(self.msg)

class Node(object):
    def __init__(self, type, children=None, root=None):
        self.type = type
        if ( children ):
            self.children = children
        else:
            self.children = []
        self.root = root
        self.slist = []
        self.expression = False

    def is_expression(self):
        return self.expression

    def add_child(self, n):
        self.children.append(n)

    def to_string(self, indent):
        ind = indent * ' '

        if ( self.root ):
            print(ind, self.type, '---', self.root)
        else:
            print(ind, self.type)

        indent = indent + 3
        ind = indent * ' '

        for c in self.children:
            if ( c == None ):
                print("OOPS! type of parent is", type(self))
                print(self.children)

            if ( type(c) is str ):
                print(ind, c)
            elif ( type(c) is int ):
                print(ind, str(c))
            elif ( type(c) is float ):
                print(ind, str(c))
            else:
                c.to_string(indent)

class Program(Node):
    def __init__(self, children):
        Node.__init__(self, 'program', children, None)
    def qasm(self):
        s = ""
        for c in self.children:
            s += c.qasm() + "\n"
        return s


class Qreg(Node):
    def __init__(self, children):
        Node.__init__(self, 'qreg', children, None)

        self.id = children[0]         # this is the indexed id, the full id[n] object
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        self.index = self.id.index

    def to_string(self, indent):
        ind = indent * ' '
        print(ind, 'qreg')
        self.children[0].to_string(indent + 3)

    def qasm(self):
        return "qreg " + self.id.qasm() + ";"

class Creg(Node):
    def __init__(self, children):
        Node.__init__(self, 'creg', children, None)

        self.id = children[0]         # this is the indexed id, the full id[n] object
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        self.index = self.id.index

    def to_string(self, indent):
        ind = indent * ' '
        print(ind, 'creg')
        self.children[0].to_string(indent + 3)

    def qasm(self):
        return "creg " + self.id.qasm() + ";"

class Id(Node):
    def __init__(self, id, line, file):
        Node.__init__(self, "id", None, None);

        self.name = id
        self.line = line
        self.file = file

        self.is_bit = False            # to help with scoping rules, so we know the id is a bit

        # print('Define id', id, 'at line', line)

    def to_string(self, indent):
        ind = indent * ' '
        # too noisy but valid
        # print(ind, 'id', self.id, 'line', self.line, 'file', self.file)
        print(ind, 'id', self.name)

    def qasm(self):
        return self.name

class Int(Node):
    def __init__(self, id):
        Node.__init__(self, "int", None, None);

        self.value = id

    def to_string(self, indent):
        ind = indent * ' '
        print(ind, 'int', self.value)

    def qasm(self):
        return "%d"%self.value

class Real(Node):
    def __init__(self, id):
        Node.__init__(self, "real", None, None);

        self.value = id

    def to_string(self, indent):
        ind = indent * ' '
        print(ind, 'real', self.value)

    def qasm(self):
        # TODO: control this
        return "%0.15f"%self.value

class IndexedId(Node):
    def __init__(self, children):
        Node.__init__(self, 'indexed_id', children, None)

        self.id = children[0]
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        self.index = children[1]

    def to_string(self, indent):
        ind = indent * ' '
        # too noisy but shows where to get stuff
        #print(ind, 'indexed_id', self.id.id, self.index, 'line', self.id.line, 'file', self.id.file)
        print(ind, 'indexed_id', self.name, self.index)

    def qasm(self):
        return self.name + "[%d]"%self.index

class IdList(Node):
    def __init__(self, children):
        Node.__init__(self, 'id_list', children, None)

    def size(self):
        return len(self.children)

    def qasm(self):
        return ",".join([self.children[j].qasm() for j in range(self.size())])

class PrimaryList(Node):
    def __init__(self, children):
        Node.__init__(self, 'primary_list', children, None)

    def size(self):
        return len(self.children)

    def qasm(self):
        return ",".join([self.children[j].qasm() for j in range(self.size())])

class GateBody(Node):
    def __init__(self, children):
        Node.__init__(self, 'gate_body', children, None)
    def qasm(self):
        s = ""
        for c in self.children:
            s += "  " + c.qasm() + "\n"
        return s
    def calls(self):
        l = []
        for c in self.children:
          if c.type == "custom_unitary":
            l.append(c.name)
        return l

class Gate(Node):
    def __init__(self, children):
        Node.__init__(self, 'gate', children, None)

        self.id = children[0]
        self.name = self.id.name             # these are required by the symtab
        self.line = self.id.line
        self.file = self.id.file

        if  len(children) == 3:
            self.arguments = None
            self.bitlist = children[1]
            self.body = children[2]
        else:
            self.arguments = children[1]
            self.bitlist = children[2]
            self.body = children[3]

    def n_args(self):
        if ( self.arguments ):
            return self.arguments.size()
        return 0

    def n_bits(self):
        return self.bitlist.size()

    def qasm(self):
        s = "gate " + self.name
        if self.arguments is not None:
            s += "(" + self.arguments.qasm() + ")"
        s += " " + self.bitlist.qasm() + "\n"
        s += "{\n"  + self.body.qasm() + "}"
        return s

class CustomUnitary(Node):
    def __init__(self, children):
        Node.__init__(self, 'custom_unitary', children, None)

        self.id = children[0]
        self.name = self.id.name
        if ( len(children) == 3 ):
            self.arguments = children[1]
            self.bitlist = children[2]
        else:
            self.arguments = None
            self.bitlist = children[1]
    def qasm(self):
        s = self.name
        if self.arguments is not None:
            s += "(" + self.arguments.qasm() + ")"
        s += " " + self.bitlist.qasm() + ";"
        return s


class UniversalUnitary(Node):
    def __init__(self, children):
        Node.__init__(self, 'universal_unitary', children, None)
    def qasm(self):
        return "U(" + self.children[0].qasm() + ") " + \
               self.children[1].qasm() + ";"

class Cnot(Node):
    def __init__(self, children):
        Node.__init__(self, 'cnot', children, None)
    def qasm(self):
        return "CX " + self.children[0].qasm() + "," + \
               self.children[1].qasm() + ";"

class ExpressionList(Node):
    def __init__(self, children):
        Node.__init__(self, 'expression_list', children, None)

    def size(self):
        return len(self.children)

    def qasm(self):
        return ",".join([self.children[j].qasm() for j in range(self.size())])

class BinaryOp(Node):
    def __init__(self, children):
        Node.__init__(self, 'binop', children, None)
    def qasm(self):
        return "(" + self.children[1].qasm() + self.children[0] + \
               self.children[2].qasm() + ")"

class Prefix(Node):
    def __init__(self, children):
        Node.__init__(self, 'prefix', children, None)
    def qasm(self):
        return self.children[0] + "(" + self.children[1].qasm() + ")"

class Measure(Node):
    def __init__(self, children):
        Node.__init__(self, 'measure', children, None)
    def qasm(self):
        return "measure " + self.children[0].qasm() + " -> " + \
               self.children[1].qasm() + ";"

class Magic(Node):
    def __init__(self, children):
        Node.__init__(self, 'magic', children, None)
    def qasm(self):
        return "OPENQASM %.1f;"%self.children[0]

class Barrier(Node):
    def __init__(self, children):
        Node.__init__(self, 'barrier', children, None)
    def qasm(self):
        return "barrier " + self.children[0].qasm() + ";"

class Reset(Node):
    def __init__(self, children):
        Node.__init__(self, 'reset', children, None)
    def qasm(self):
        return "reset " + self.children[0].qasm() + ";"

class If(Node):
    def __init__(self, children):
        Node.__init__(self, 'if', children, None)
    def qasm(self):
        return "if(" + self.children[0].qasm() + "==" + \
          self.children[1].qasm() + ") " + self.children[2].qasm()

class Opaque(Node):
    def __init__(self, children):
        Node.__init__(self, 'opaque', children, None)

        self.id = children[0]
        self.name = self.id.name             # these are required by the symtab
        self.line = self.id.line
        self.file = self.id.file

        if ( len(children) == 3 ):
            self.arguments = children[1]
            self.bitlist = children[2]
        else:
            self.arguments = None
            self.bitlist = children[1]

    def n_args(self):
        if ( self.arguments ):
            return self.arguments.size()
        return 0

    def n_bits(self):
        return self.bitlist.size()

    def qasm(self):
        s = "opaque %s"%self.name
        if self.arguments is not None:
          s += "(" + self.arguments.qasm() + ")"
        s += self.bitlist.qasm() + ";"
        return s

class External(Node):
    def __init__(self, children):
        Node.__init__(self, 'external', children, None)

class QasmLexer(object):
    '''
       This is a wrapper around the ply lexer to support the "include" statement
       by creating a stack of lexers.
    '''

    def __mklexer__(self, filename):
        self.lexer = lex.lex(module=self, debug=False)
        self.filename = filename
        self.lineno = 1

    def __init__(self, filename):
        self.__mklexer__(filename)
        self.stack = []


    def input(self, data):
        self.data = data
        self.lexer.input(data)

    def token(self):
        ret = self.lexer.token()
        # print("RETURNING TOKEN from", self.filename, "last line", self.lineno, '-->', ret)
        return ret

    def pop(self):
        self.lexer = self.stack.pop()
        self.filename = self.lexer.qasm_file
        self.lineno = self.lexer.qasm_line

    def push(self, filename):

        self.lexer.qasm_file = self.filename
        self.lexer.qasm_line = self.lineno
        self.stack.append(self.lexer)
        self.__mklexer__(filename)
        self.data = open(filename).read()
        self.lexer.input(self.data)

    # --------------------------------------------------------------------------------
    #                   L E X E R
    # --------------------------------------------------------------------------------

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
        r'\d+'
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
        2 - a terminating semocolon

        Then push the current lexer onto the stack, create a new one from the include file,
        and push it onto the stack.

        When we hit eof (the t_eof) rule, we pop.
        '''
        next = self.lexer.token()
        lineno = next.lineno
        # print('NEXT', next, "next.value", next.value, type(next))
        if ( type(next.value) == str ):
            incfile = next.value.strip('"')
        else:
            raise QasmException("Invalid include: must be a quoted string.")

        next = self.lexer.token()
        if ( next == None or next.value != ';' ):
            raise QasmException('Invalid syntax, missing ";" at line', str(lineno))

        if not os.path.exists(incfile):
            raise QasmException('Include file', incfile, 'cannot be found, line', str(next.lineno), ', file', self.filename)


        self.push(incfile)

        return self.lexer.token()


    def t_MAGIC(self, t):
        'OPENQASM'
        return t

    def t_COMMENT(self, t):
        r'//.*'
        pass

    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'

        if t.value == 'U':
            t.type = 'U'
            return t

        if t.value == 'CX':
            t.type = 'CX'
            return t

        if t.value == 'pi':
            t.type = 'PI'
            return t

        t.value = Id(t.value, self.lineno, self.filename)
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
        print("Junk in the line -->%s<--" % t.value[0])
        # t.lexer.skip(1)


class QasmParser(object):

    def __init__(self, filename):
        self.lexer = QasmLexer(filename)
        self.tokens = self.lexer.tokens
        self.parser = yacc.yacc(module=self, debug=True)
        self.qasm = None
        self.parseDeb = False

        self.global_symtab = {}                          # global symtab
        self.current_symtab = self.global_symtab         # top of symbol stack
        self.symbols = []                                # symbol stack

    def update_symtab(self, obj):
        #
        # Everything in the symtab must be a node with these attributes:
        #   name - the string name of the object
        #   type - the string type of the object
        #   line - the source line where the type was first found
        #   file - the source file where the type was first found
        #
        if ( obj.name in self.current_symtab ):
            prev = self.current_symtab[obj.name]
            raise QasmException("Duplicate declaretion for", obj.type + " '" + obj.name + "' at line", str(obj.line) + ', file',
                                obj.file + '.\nPrevious occurance at line', str(prev.line) + ', file', prev.file)
        self.current_symtab[obj.name] = obj

    def verify_declared_bit(self, obj):
        #
        # We are verifying gate args against the formal parameters of a gate prototype.
        #
        if ( not (obj.name in self.current_symtab) ):
            raise QasmException("Cannot find symbol '" + obj.name + "' in argument list for gate, line", str(obj.line), 'file', obj.file)

        #
        # This insures the thing is from the bitlist and not from the argument list
        #
        sym = self.current_symtab[obj.name]
        if ( not ( sym.type == 'id' and sym.is_bit) ):
            raise QasmException("Bit", obj.name, 'is not declared as a bit in the argument list.')

    def verify_bit_list(self, obj):
        #
        # We expect the object to be a bitlist or an idlist, we don't care.  We will
        # iterate it and insure everything in it is declared as a bit, and throw if not
        #
        for b in obj.children:
            self.verify_declared_bit(b)

    def verify_exp_list(self, obj):
        #
        # A tad harder.  This is a list of expressions each of which could the the head of a tree
        # We need to recursively walk each of these and insure that any Id elements resolve to the
        # current stack.
        #
        # I believe we only have to look at the current symtab.

        if obj.children != None:
            for child in obj.children:
                if ( isinstance(child, Id) ):
                    if child.name in ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt']:
                        continue

                    if ( not child.name in self.current_symtab ):
                        raise QasmException("Argument '" + child.name + "' in expression cannot be found, line", str(child.line), "file", child.file)
                else:
                    if hasattr(child, "children"):
                        self.verify_exp_list(child)


    def verify_as_gate(self, obj, bitlist, arglist=None):
        if ( not (obj.name in self.global_symtab) ):
            raise QasmException("Cannot find gate definition for '" + obj.name + "', line", str(obj.line), 'file', obj.file)
        g = self.global_symtab[obj.name]
        if ( not (( g.type == 'gate') or ( g.type == 'opaque')) ):
            raise QasmException("'" + obj.name + "' is used as a gate or opaque call but the symbol is neither it is a '" + g.type + "' line",
                                str(obj.line), 'file', obj.file)

        if ( g.n_bits() != bitlist.size() ):
            raise QasmException("Gate or opaque call to '" + obj.name + "' uses", str(bitlist.size()), "qubits but is declared for",
                                str(g.n_bits()), "qubits", "line", str(obj.line), 'file', obj.file)

        if ( arglist ):
            if ( g.n_args() != arglist.size() ):
                raise QasmException("Gate or opaque call to '" + obj.name + "' uses", str(arglist.size()),
                                    "qubits but is declared for",
                                    str(g.n_args()), "qubits", "line", str(obj.line), 'file', obj.file)
        else:
            if ( g.n_args() > 0 ):
                raise QasmException("Gate or opaque call to '" + obj.name + "' has no arguments but is declared for",
                                    str(g.n_args()), "qubits", "line", str(obj.line), 'file', obj.file)


    def verify_reg(self, obj, typ):
        # how to verify:
        #    types must match
        #    indexes must be checked
        if ( not obj.name in self.global_symtab ):
            raise QasmException('Cannot find definition for', typ, "'" + obj.name + "'", 'at line', str(obj.line), 'file', obj.file)

        sym = self.global_symtab[obj.name]

        if ( sym.type != typ ):
            raise QasmException("Type for '" + sym.name + "' should be '" + typ + "' but was found to be '" + sym.type + "'",
                                "line", str(obj.line), "file", obj.file)

        if ( obj.type == 'indexed_id' ):
            bound = sym.index
            ndx = obj.index
            if ( (ndx < 0) or ( ndx >= bound ) ):
                raise QasmException("Register index for '" + sym.name + "' out of bounds. Index is", str(ndx),
                                    "bound is 0 <= index <", str(bound),
                                    "at line", str(obj.line), "file", obj.file);

    def verify_reg_list(self, obj, typ):
        #
        # We expect the object to be a bitlist or an idlist, we don't care.  We will
        # iterate it and insure everything in it is declared as a bit, and throw if not
        #
        for b in obj.children:
            self.verify_reg(b, typ)

    #def verify_declared_id(self, obj):
    #    #
    #    # We are verifying gate args against the formal parameters of a gate prototype.
    #    #
    #    print(self.current_symtab)
    #    print('Verifying obj', obj, 'obj.name', obj.name)
    #    if ( not (obj.name in self.current_symtab) ):
    #        raise QasmException("Cannot find symbol '" + obj.name + "' in argument list for gate, line", str(obj.line), 'file', obj.file)

    def pop_scope(self):
        self.current_symtab = self.symbols.pop()

    def push_scope(self):
        self.symbols.append(self.current_symtab)
        self.current_symtab = {}


    # --------------------------------------------------------------------------------
    #                   P A R S E R
    # --------------------------------------------------------------------------------
    start = 'main'


    def p_main(self, p):
        '''
            main : program
        '''
        #print("------- make a main")
        self.qasm = p[1]


    # ----------------------------------------
    #  program : statement
    #          | program statement
    # ----------------------------------------
    def p_program_0(self, p):
        '''
           program : statement
        '''
        #print("------- make a program 0")

        p[0] = Program( [p[1]] )

    def p_program_1(self, p):
        '''
           program : program statement
        '''
        #print("------- make a program 1")
        p[0] = p[1]
        p[0].add_child(p[2])


    # ----------------------------------------
    #  statement : decl
    #            | quantum_op ';'
    #            | magic ';'
    # ----------------------------------------
    def p_statement(self, p):
        '''
           statement : decl
                     | quantum_op ';'
                     | magic ';'
                     | ignore
                     | quantum_op error
                     | magic error
        '''
        #print("------- make a statement")
        if ( len(p) > 2 ):
            if ( p[2] != ';' ):
                raise QasmException("Missing ';' at end of statement; received", str(p[2].value))

        p[0] = p[1]

    def p_magic(self, p):
        '''
           magic : MAGIC REAL
        '''
        p[0] = Magic( [p[2]] )

    def p_magic_0(self, p):
        '''
           magic : MAGIC error
        '''
        magic = "2.0;"
        raise QasmException("Invalid magic string. Expected '" + magic + "'.  Is the semocolon missing?")

    # ----------------------------------------
    #  id : ID
    # ----------------------------------------
    def p_id(self, p):
        '''
           id : ID
        '''
        #print('------ make an id, type is', type(p[1]))
        p[0] = p[1]

    def p_id_e(self, p):
        '''
           id : error
        '''
        raise QasmException("Expected an ID, received '" + str(p[1].value) + "'")

    # ----------------------------------------
    #  indexed_id : ID [ int ]
    # ----------------------------------------
    def p_indexed_id(self, p):
        '''
           indexed_id : id '[' NNINTEGER ']'
                      | id '[' NNINTEGER error
                      | id '[' error
        '''
        # print("------- make a indexed id from", p[1], type(p[1]))
        if ( len(p) == 4 ):
            raise QasmException("Expecting an integer index; received", str(p[3].value))
        if ( p[4] != ']' ):
            raise QasmException("Missing ']' in indexed ID; received", str(p[4].value))
        p[0] = IndexedId( [ p[1], p[3] ] )


    # ----------------------------------------
    #  primary : id
    #          | indexed_id
    # ----------------------------------------
    def p_primary(self, p):
        '''
           primary : id
                   | indexed_id
        '''
        p[0] = p[1]

    # ----------------------------------------
    #  id_list : id
    #          | id_list ',' id
    # ----------------------------------------
    def p_id_list_0(self, p):
        '''
           id_list : id
        '''
        #print("------- make a id_list_0 with", p[1].name)
        p[0] = IdList( [p[1]] )

    def p_id_list_1(self, p):
        '''
           id_list : id_list ',' id
        '''
        #print("------- make a id_list_1 with", p[3].name)
        p[0] = p[1]
        p[0].add_child(p[3])

    # ----------------------------------------
    #  gate_id_list : id
    #               | gate_id_list ',' id
    # ----------------------------------------
    def p_gate_id_list_0(self, p):
        '''
           gate_id_list : id
        '''
        #print("------- make a gate_id_list_0 with", p[1].name)
        p[0] = IdList( [p[1]] )
        self.update_symtab(p[1])

    def p_gate_id_list_1(self, p):
        '''
           gate_id_list : gate_id_list ',' id
        '''
        #print("------- make a gate_id_list_1 with", p[3].name)
        p[0] = p[1]
        p[0].add_child(p[3])
        self.update_symtab(p[3])


    # ----------------------------------------
    #  bit_list : bit
    #           | bit_list ',' bit
    # ----------------------------------------
    def p_bit_list_0(self, p):
        '''
           bit_list : id
        '''
        #print("------- make a bit_list_0 with", p[1].name)
        p[0] = IdList( [p[1]] )
        p[1].is_bit = True
        self.update_symtab(p[1])

    def p_bit_list_1(self, p):
        '''
           bit_list : bit_list ',' id
        '''
        #print("------- make a bit_list_1 with", p[3].name)
        p[0] = p[1]
        p[0].add_child(p[3])
        p[3].is_bit = True
        self.update_symtab(p[3])

    # ----------------------------------------
    #  primary_list : primary
    #               | primary_list ',' primary
    # ----------------------------------------
    def p_primary_list_0(self, p):
        '''
           primary_list : primary
        '''
        #print("------- make a primary_list_0");
        p[0] = PrimaryList( [p[1]] )

    def p_primary_list_1(self, p):
        '''
           primary_list : primary_list ',' primary
        '''
        #print("------- make a primary_list_0");
        p[0] = p[1]
        p[1].add_child(p[3])


    # ----------------------------------------
    #  decl : qreg_decl
    #       | creg_decl
    #       | gate_decl
    # ----------------------------------------
    def p_decl(self, p):
        '''
           decl : qreg_decl ';'
                | creg_decl ';'
                | qreg_decl error
                | creg_decl error
                | gate_decl
        '''
        if ( len(p) > 2 ):
            if ( p[2] != ';' ):
                raise QasmException("Missing ';' in qreg or creg declaraton. Instead received '" + p[2].value + "'")

        #print("------- make a decl")
        p[0] = p[1]

    # ----------------------------------------
    #  qreg_decl : QREG indexed_id
    # ----------------------------------------
    def p_qreg_decl(self, p):
        '''
           qreg_decl : QREG indexed_id
        '''
        # print("------- make a qreg from", p[2])
        p[0] = Qreg( [p[2]] )
        self.update_symtab(p[0])

    def p_qreg_decl_e(self, p):
        '''
           qreg_decl : QREG error
        '''
        raise QasmException("Expecting indexed id (ID[int]) in QREG declaration; received", p[2].value)

    # ----------------------------------------
    #  creg_decl : QREG indexed_id
    # ----------------------------------------
    def p_creg_decl(self, p):
        '''
           creg_decl : CREG indexed_id
        '''
        #print("------- make a creg")
        p[0] = Creg( [p[2]] )
        self.update_symtab(p[0])


    def p_creg_decl_e(self, p):
        '''
           creg_decl : CREG error
        '''
        raise QasmException("Expecting indexed id (ID[int]) in CREG declaration; received", p[2].value)

    # Gate_body will throw if there are erros so we don't need to cover that here.
    # Same with the id_lists - if there are not legal we die before we get here
    #
    # ----------------------------------------
    #  gate_decl : GATE id gate_scope                      bit_list gate_body
    #            | GATE id gate_scope '(' ')'              bit_list gate_body
    #            | GATE id gate_scope '(' gate_id_list ')' bit_list gate_body
    #
    # ----------------------------------------
    def p_gate_decl_0(self, p):
        '''
           gate_decl : GATE id gate_scope bit_list gate_body
        '''
        #print("------- make a gate_decl_0")
        p[0] = Gate( [ p[2], p[4], p[5] ] )
        self.pop_scope()
        self.update_symtab(p[0])


    def p_gate_decl_1(self, p):
        '''
           gate_decl : GATE id gate_scope '(' ')' bit_list gate_body
        '''
        #print("------- make a gate_decl_1")
        p[0] = Gate( [ p[2], p[6], p[7] ] )
        self.pop_scope()
        self.update_symtab(p[0])


    def p_gate_decl_2(self, p):
        '''
           gate_decl : GATE id gate_scope '(' gate_id_list ')' bit_list gate_body
        '''
        #print("------- make a gate_decl_2")
        p[0] = Gate ( [ p[2], p[5], p[7], p[8] ] )
        self.pop_scope()
        self.update_symtab(p[0])

    def p_gate_scope(self, p):
        '''
           gate_scope :
        '''
        self.push_scope()

    #def p_unset_flag(self, p):
    #    '''
    #       unset_flag :
    #    '''
    #    # kludgy but not clear how to do better because
    #    self.defining_gate = False

    # ----------------------------------------
    #  gate_body : '{' gate_op_list '}'
    #            | '{' '}'
    #
    #            | '{' gate_op_list error
    #            | '{' error
    #
    # Error handling: gete_op will throw if there's a problem so we won't
    #                 get here with in the gate_op_list
    # ----------------------------------------
    def p_gate_body_0(self, p):
        '''
           gate_body : '{' '}'
        '''
        if ( p[2] != '}' ):
            raise QasmException("Missing '}' in gate definition; received'" + str(p[2].value) + "'")

        #print("------- make a gate_body_0")
        p[0] = GateBody(None)

    def p_gate_body_1(self, p):
        '''
           gate_body : '{' gate_op_list '}'
        '''

        #print("------- make a gate_body_1")
        p[0] = GateBody( p[2] )



    # ----------------------------------------
    #  gate_op_list : gate_op
    #               | gate_op_ist gate_op
    #
    # Error handling: gete_op will throw if there's a problem so we won't
    #                 get here with errors
    # ----------------------------------------
    def p_gate_op_list_0(self, p):
        '''
            gate_op_list : gate_op
        '''
        p[0] = [ p[1] ]

    def p_gate_op_list_1(self, p):
        '''
            gate_op_list : gate_op_list gate_op
        '''
        p[0] = p[1]
        p[0].append(p[2])

    # ----------------------------------------
    # These are for use outside of gate_bodies and allow
    # indexed ids everywhere.
    #
    # unitary_op : U '(' exp_list ')'  primary
    #            | CX                  primary ',' primary
    #            | id                  pirmary_list
    #            | id '(' ')'          primary_list
    #            | id '(' exp_list ')' primary_list
    #
    # Note that it might not be unitary - this is the mechanism that is also used
    # to invoke calls to 'opaque'
    # ----------------------------------------
    def p_unitary_op_0(self, p):
       '''
          unitary_op : U '(' exp_list ')' primary
       '''
       #print("------- make a unitary_op_0 (universal)")
       p[0] = UniversalUnitary( [ p[3], p[5] ] )
       self.verify_reg(p[5], 'qreg')
       self.verify_exp_list(p[3])


    def p_unitary_op_1(self, p):
       '''
       unitary_op : CX primary ',' primary
       '''
       #print("------- make a unitary_op_1 (CNOT)")
       p[0] = Cnot( [ p[2], p[4] ] )
       self.verify_reg(p[2], 'qreg')
       self.verify_reg(p[4], 'qreg')


    def p_unitary_op_2(self, p):
       '''
          unitary_op : id primary_list
       '''
       #print("------- make a unitary_op_2")
       p[0] = CustomUnitary( [ p[1], p[2] ] )
       self.verify_as_gate(p[1], p[2])
       self.verify_reg_list(p[2], 'qreg')

    def p_unitary_op_3(self, p):
       '''
          unitary_op : id '(' ')' primary_list
       '''
       #print("------- make a unitary_op_3")
       p[0] = CustomUnitary( [ p[1], p[4] ] )
       self.verify_as_gate(p[1], p[4])
       self.verify_reg_list(p[4], 'qreg')


    def p_unitary_op_4(self, p):
       '''
          unitary_op : id '(' exp_list ')' primary_list
       '''
       #print("------- make a unitary_op_4")
       p[0] = CustomUnitary( [p[1], p[3], p[5]] )
       self.verify_as_gate(p[1], p[5], arglist=p[3])
       self.verify_reg_list(p[5], 'qreg')
       self.verify_exp_list(p[3])


    # ----------------------------------------
    # This is a restricted set of "quantum_op" which also
    # prohobits indexed ids, for use in a gate_body
    #
    # gate_op : U '(' exp_list ')'  id         ';'
    #         | CX                  id ',' id  ';'
    #         | id                  id_list    ';'
    #         | id '(' ')'          id_list    ';'
    #         | id '(' exp_list ')' id_list    ';'
    #         | BARRIER id_list                ';'
    # ----------------------------------------
    def p_gate_op_0(self, p):
       '''
          gate_op : U '(' exp_list ')' id ';'
       '''
       #print("------- make a gate_op_0 (universal) from", p[3], p[5])
       p[0] = UniversalUnitary( [ p[3], p[5] ] )
       self.verify_declared_bit(p[5])
       self.verify_exp_list(p[3])

    def p_gate_op_0e1(self, p):
       '''
          gate_op : U '(' exp_list ')' error
       '''
       raise QasmException("Invalid U inside gate definition.  Missing bit id or ';'");

    def p_gate_op_0e2(self, p):
       '''
          gate_op : U '(' exp_list error
       '''
       raise QasmException("Missing ')' in U invocation in gate definition.");

    def p_gate_op_1(self, p):
       '''
       gate_op : CX id ',' id ';'
       '''
       #print("------- make a gate_op_1 (CNOT)")
       p[0] = Cnot( [ p[2], p[4] ] )
       self.verify_declared_bit(p[2])
       self.verify_declared_bit(p[4])

    def p_gate_op_1e1(self, p):
       '''
       gate_op : CX error
       '''
       raise QasmException("Invalid CX inside gate definition.  Expectedn an ID or '.', received '" + str(p[2].value) + "'");


    def p_gate_op_1e2(self, p):
       '''
       gate_op : CX id ',' error
       '''
       raise QasmException("Invalid CX inside gate definition.  Expectedn an ID or ';', received '" + str(p[4].value) + "'");


    def p_gate_op_2(self, p):
       '''
          gate_op : id id_list ';'
       '''
       #print("------- make a gate_op_2")
       p[0] = CustomUnitary( [p[1], p[2]] )
       # TO verify:
       # 1 id is declared as a gate in global scope
       # 2 everything in the id_list is declared as a bit in local scope
       self.verify_as_gate(p[1], p[2])
       self.verify_bit_list(p[2])

    def p_gate_op_2e(self, p):
       '''
          gate_op : id  id_list error
       '''
       raise QasmException("Invalid gate invocation inside gate definition.");

    def p_gate_op_3(self, p):
       '''
          gate_op : id '(' ')' id_list ';'
       '''
       #print("------- make a gate_op_3")
       p[0] = CustomUnitary( [p[1], p[4]] )
       self.verify_as_gate(p[1], p[4])
       self.verify_bit_list(p[4])

    def p_gate_op_4(self, p):
       '''
          gate_op : id '(' exp_list ')' id_list ';'
       '''
       #print("------- make a gate_op_4 from", p[1], p[3], p[5])
       p[0] = CustomUnitary( [p[1], p[3], p[5]] )
       self.verify_as_gate(p[1], p[5], arglist=p[3])
       self.verify_bit_list(p[5])
       self.verify_exp_list(p[3])

    def p_gate_op_4e0(self, p):
       '''
          gate_op : id '(' ')'  error
       '''
       raise QasmException("Invalid bit list inside gate definition or missing ';'");

    def p_gate_op_4e1(self, p):
       '''
          gate_op : id '('   error
        '''
       raise QasmException("Unmatched () for gate invocation inside gate invocation.");

    def p_gate_op_5(self, p):
       '''
           gate_op : BARRIER id_list ';'
       '''
       #print("------- make a gate_op_4")

       p[0] = Barrier( [p[2]] )
       self.verify_bit_list(p[2])


    def p_gate_op_5e(self, p):
       '''
           gate_op : BARRIER error
       '''
       raise QasmException("Invalid barrier inside gate definition.")

    # ----------------------------------------
    # opaque : OPAQUE id gate_scope                      bit_list
    #        | OPAQUE id gate_scope '(' ')'              bit_list
    #        | OPAQUE id gate_scope '(' gate_id_list ')' bit_list
    #
    # These are like gate declaratons only wihtout a body.
    # ----------------------------------------
    def p_opaque_0(self, p):
        '''
           opaque : OPAQUE id gate_scope bit_list
        '''
        #print("------- make opaque_0")
        p[0] = Opaque( [ p[2], p[4] ] )
        self.pop_scope()
        self.update_symtab(p[0])

    def p_opaque_1(self, p):
        '''
           opaque : OPAQUE id gate_scope '(' ')' bit_list
        '''
        #print("------- make opaque_1")
        p[0] = Opaque( [ p[2], p[6] ] )
        self.pop_scope()
        self.update_symtab(p[0])


    def p_opaque_2(self, p):
        '''
           opaque : OPAQUE id gate_scope '(' gate_id_list ')' bit_list
        '''
        #print("------- make opaque_2")
        p[0] = Opaque( [ p[2], p[5], p[7] ] )
        self.pop_scope()
        self.update_symtab(p[0])

    def p_opaque_1e(self, p):
        '''
           opaque : OPAQUE id gate_scope '(' error
        '''
        #print("------- make opaque_1")
        raise QasmException("Poorly formed OPAQUE statement.")

    # ----------------------------------------
    # measure : MEASURE primary ASSIGN primary
    # ----------------------------------------
    def p_measure(self, p):
        '''
           measure : MEASURE primary ASSIGN primary
        '''
        #print("------- make a measure")
        p[0] = Measure( [ p[2], p[4] ] )
        self.verify_reg(p[2], 'qreg')
        self.verify_reg(p[4], 'creg')

    def p_measure_e(self, p):

        '''
           measure : MEASURE primary error
        '''
        raise QasmException("Illegal measure statement." + str(p[3].value))

    # ----------------------------------------
    # barrier : BARRIER primary_list
    #
    # Errors are covered by handling erros in primary_list
    # ----------------------------------------
    def p_barrier(self, p):
       '''
           barrier : BARRIER primary_list
       '''
       #print("------- make a barrier")
       p[0] = Barrier( [p[2]] )
       self.verify_reg_list(p[2], 'qreg')

    # ----------------------------------------
    # reset : RESET primary
    # ----------------------------------------
    def p_reset(self, p):
       '''
           reset : RESET primary
       '''
       #print("------- make a reset")
       p[0] = Reset( [p[2]] )
       self.verify_reg(p[2], 'qreg')


    # ----------------------------------------
    # IF '(' ID MATCHES NNINTEGER ')' quantum_op
    # ----------------------------------------
    def p_if(self, p):
       '''
          if : IF '(' id MATCHES NNINTEGER ')' quantum_op
          if : IF '(' id error
          if : IF '(' id MATCHES error
          if : IF '(' id MATCHES NNINTEGER error
          if : IF error

       '''
       if ( len(p) == 3 ):
           raise QasmException("Ill-formed IF statement. Perhaps a missing '('?")
       if ( len(p) == 5 ):
           raise QasmException("Ill-formed IF statement.  Expected '==', received '" + str(p[4].value))
       if ( len(p) == 6 ):
           raise QasmException("Ill-formed IF statement.  Expected a number, received '" + str(p[5].value))
       if ( len(p) == 7 ):
           raise QasmException("Ill-formed IF statement, unmatched '('")

       #print("------- make an if")
       p[0] = If( [ p[3], p[5], p[7] ] )

    # ----------------------------------------
    # These are all the things you can have outside of a gate declaration
    #        quantum_op : unitary_op
    #                   | opaque
    #                   | measure
    #                   | reset
    #                   | barrier
    #                   | if
    #
    # ----------------------------------------
    def p_quantum_op(self, p):
        '''
            quantum_op : unitary_op
                       | opaque
                       | measure
                       | barrier
                       | reset
                       | if
        '''
        p[0] = p[1]

    # ----------------------------------------
    # unary : NNINTEGER
    #       | REAL
    #       | PI
    #       | ID
    #       | '(' expression ')'
    #       | id '(' expression ')'
    #
    # We will trust 'expression' to throw before we have to handle it here
    # ----------------------------------------
    def p_unary_0(self, p):
        '''
           unary : NNINTEGER
        '''
        #print("------- make a unary_0 NNINTEGER")
        p[0] = Int(p[1])

    def p_unary_1(self, p):
        '''
           unary : REAL
        '''
        #print("------- make a unary_1 REAL")
        p[0] = Real(p[1])

    def p_unary_2(self, p):
        '''
           unary : PI
        '''
        #print("------- make a unary_2 PI")
        p[0] = Real(math.pi)

    def p_unary_3(self, p):
        '''
           unary : id
        '''
        #print("------- make a unary_3 ID")
        p[0] = p[1]

    def p_unary_4(self, p):
        '''
           unary : '(' expression ')'
        '''
        #print("------- make a unary_4 (exp)")
        p[0] = p[2]

    def p_unary_6(self, p):
        '''
           unary : id '(' expression ')'
        '''
        # print("------- make a unary_5 external")
        # note this is a semantic check, not syntactic
        if ( not ( p[1].name in ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt']) ):
            raise QasmException("Illegal external function call: ", str(p[1].name))

        p[0] = External( [p[1], p[3] ] )


    # ----------------------------------------
    # Prefix
    # ----------------------------------------
    def p_prefix_expression_0(self, p):
        '''
           prefix_expression : unary
        '''
        #print("------- make a prefix_0")
        p[0] = p[1]

    def p_prefix_expression_1(self, p):
        '''
           prefix_expression : '+' prefix_expression
                             | '-' prefix_expression
        '''
        #print("------- make a prefix_1")
        p[0] = Prefix( [p[1], p[2] ])

    def p_additive_expression_0(self, p):
        '''
            additive_expression : prefix_expression
        '''
        #print("------- make a additive_0")
        p[0] = p[1]

    def p_additive_expression_1(self, p):
        '''
            additive_expression : additive_expression '+' prefix_expression
                                | additive_expression '-' prefix_expression
        '''
        #print("------- make a additive_1")
        p[0]= BinaryOp( [p[2], p[1], p[3] ])


    def p_multiplicative_expression_0(self, p):
        '''
            multiplicative_expression : additive_expression
        '''
        #print("------- make a multiplicative_0")
        p[0] = p[1]


    def p_multiplicative_expression_1(self, p):
        '''
            multiplicative_expression : multiplicative_expression '*' additive_expression
                                      | multiplicative_expression '/' additive_expression
        '''
        #print("------- make a multiplicative_1")
        p[0] = BinaryOp( [p[2], p[1], p[3] ])


    def p_expression_0(self, p):
        '''
            expression : multiplicative_expression
        '''
        #print("------- make a expression_0")
        p[0] = p[1]

    def p_expression_1(self, p):
        '''
            expression : expression '^' multiplicative_expression
        '''
        #print("------- make a expression_1")
        p[0] = BinaryOp( [p[2], p[1], p[3] ])

    # ----------------------------------------
    # exp_list : exp
    #          | exp_list ',' exp
    # ----------------------------------------
    def p_exp_list_0(self, p):
        '''
           exp_list : expression
        '''
        #print("------- make a expression_list_0")
        p[0] = ExpressionList( [p[1]] )

    def p_exp_list_1(self, p):
        '''
           exp_list : exp_list ',' expression
        '''
        #print("------- make a expression_list_1")

        p[0] = p[1]
        p[0].add_child(p[3])


    def p_ignore(self, p):
        '''
           ignore : STRING
        '''
        # this should never hit but it keeps the unsupressable warnings at bay
        pass

    def p_error(self, p):
        # EOF is a special case becase the stupid error token isn't placed on the stack
        if ( not p ):
            raise QasmException("Error at end of file.  Perhaps there is a missing ';'");

        col = self.find_column(self.lexer.data, p)
        print("Error near line", str(self.lexer.lineno),  'Column', col)

    # Compute column.
    #     input is the input text string
    #     token is a token instance
    def find_column(self, input, token):
        if ( token == None ):
            return 0
        last_cr = input.rfind('\n',0,token.lexpos)
        if last_cr < 0:
            last_cr = 0
        column = (token.lexpos - last_cr) + 1
        return column

    def print_tokens(self):
        '''
           Test method to verify tokenizer
        '''

        try:
            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                # todo this isn't really the column, it's the character position.  Need to do backtrack to the nearest \n to get
                # the actual column
                print('TOKEN:' + str(tok) + ":ENDTOKEN", 'at line', tok.lineno, 'column', tok.lexpos, 'file', self.lexer.filename)
        except QasmException as e:
            print('C--------------------------------------------------------------------------------')
            print('Exception tokenizing qasm file:', e.msg)
            print('C--------------------------------------------------------------------------------')
        except:
            print('C--------------------------------------------------------------------------------')
            print(sys.exc_info()[0], 'Exception tokenizing qasm file')
            traceback.print_exc()
            print('C--------------------------------------------------------------------------------')


    def parse_debug(self, val):
        if ( val == True ):
            self.parseDeb = True
        elif ( val == False ):
            self.parseDeb = False
        else:
            raise QasmException("Illegal debug value '" +str(val) + "' must be True or False.")

    def parse(self, data):
        self.parser.parse(data, lexer=self.lexer, debug=self.parseDeb)
        if ( self.qasm == None ):
            raise QasmException("Uncaught exception in parser; see previous messages for details.")
        return self.qasm

    def print_tree(self):
        if ( self.qasm != None ):
            self.qasm.to_string(0)
        else:
            print("No parsed qasm to print")


    # ----------------------------------------
    #  Parser runner, to use this module stand-alone
    # ----------------------------------------
    def run(self, data):

        ast = self.parser.parse(data, debug=True)
        self.parser.parse(data, debug=True)
        ast.to_string(0)


if __name__ == '__main__':
    sqp = QasmParser()
    print(sys.argv[1:])
    sqp.main(sys.argv[1:])
