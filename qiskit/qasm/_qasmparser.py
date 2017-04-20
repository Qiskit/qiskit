"""
OPENQASM parser.

Author: Jim Challenger
"""
import math
from ._qasmlexer import QasmLexer
import ply.yacc as yacc
from ._qasmexception import QasmException
from . import _node as node


class QasmParser(object):
    """OPENQASM Parser."""

    def __init__(self, filename):
        """Create the parser."""
        if filename is None:
            filename = ""
        self.lexer = QasmLexer(filename)
        self.tokens = self.lexer.tokens
        # For yacc, also, write_tables = Bool and optimize = Bool
        self.parser = yacc.yacc(module=self, debug=False)
        self.qasm = None
        self.parseDeb = False
        self.global_symtab = {}                          # global symtab
        self.current_symtab = self.global_symtab         # top of symbol stack
        self.symbols = []                                # symbol stack
        self.external_functions = ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt']

    def update_symtab(self, obj):
        """Update a node in the symbol table.

        Everything in the symtab must be a node with these attributes:
        name - the string name of the object
        type - the string type of the object
        line - the source line where the type was first found
        file - the source file where the type was first found
        """
        if obj.name in self.current_symtab:
            prev = self.current_symtab[obj.name]
            raise QasmException("Duplicate declaration for", obj.type + " '"
                                + obj.name + "' at line", str(obj.line)
                                + ', file', obj.file
                                + '.\nPrevious occurence at line',
                                str(prev.line) + ', file', prev.file)
        self.current_symtab[obj.name] = obj

    def verify_declared_bit(self, obj):
        """Verify a qubit id against the gate prototype."""
        # We are verifying gate args against the formal parameters of a
        # gate prototype.
        if obj.name not in self.current_symtab:
            raise QasmException("Cannot find symbol '" + obj.name
                                + "' in argument list for gate, line",
                                str(obj.line), 'file', obj.file)

        # This insures the thing is from the bitlist and not from the
        # argument list.
        sym = self.current_symtab[obj.name]
        if not (sym.type == 'id' and sym.is_bit):
            raise QasmException("Bit", obj.name,
                                'is not declared as a bit in the gate.')

    def verify_bit_list(self, obj):
        """Verify each qubit in a list of ids."""
        # We expect the object to be a bitlist or an idlist, we don't care.
        # We will iterate it and insure everything in it is declared as a bit,
        # and throw if not.
        for b in obj.children:
            self.verify_declared_bit(b)

    def verify_exp_list(self, obj):
        """Verify each expression in a list."""
        # A tad harder.  This is a list of expressions each of which could be
        # the head of a tree. We need to recursively walk each of these and
        # ensure that any Id elements resolve to the current stack.
        #
        # I believe we only have to look at the current symtab.
        if obj.children is not None:
            for child in obj.children:
                if isinstance(child, node.Id):
                    if child.name in self.external_functions:
                        continue

                    if child.name not in self.current_symtab:
                        raise QasmException("Argument '" + child.name
                                            + "' in expression cannot be "
                                            + "found, line", str(child.line),
                                            "file", child.file)
                else:
                    if hasattr(child, "children"):
                        self.verify_exp_list(child)

    def verify_as_gate(self, obj, bitlist, arglist=None):
        """Verify a user defined gate call."""
        if obj.name not in self.global_symtab:
            raise QasmException("Cannot find gate definition for '" + obj.name
                                + "', line", str(obj.line), 'file', obj.file)
        g = self.global_symtab[obj.name]
        if not (g.type == 'gate' or g.type == 'opaque'):
            raise QasmException("'" + obj.name + "' is used as a gate "
                                + "or opaque call but the symbol is neither;"
                                + " it is a '" + g.type + "' line",
                                str(obj.line), 'file', obj.file)

        if g.n_bits() != bitlist.size():
            raise QasmException("Gate or opaque call to '" + obj.name
                                + "' uses", str(bitlist.size()),
                                "qubits but is declared for",
                                str(g.n_bits()), "qubits", "line",
                                str(obj.line), 'file', obj.file)

        if arglist:
            if g.n_args() != arglist.size():
                raise QasmException("Gate or opaque call to '" + obj.name
                                    + "' uses", str(arglist.size()),
                                    "qubits but is declared for",
                                    str(g.n_args()), "qubits", "line",
                                    str(obj.line), 'file', obj.file)
        else:
            if g.n_args() > 0:
                raise QasmException("Gate or opaque call to '" + obj.name
                                    + "' has no arguments but is declared for",
                                    str(g.n_args()), "qubits", "line",
                                    str(obj.line), 'file', obj.file)

    def verify_reg(self, obj, typ):
        """Verify a register."""
        # How to verify:
        #    types must match
        #    indexes must be checked
        if obj.name not in self.global_symtab:
            raise QasmException('Cannot find definition for', typ, "'"
                                + obj.name + "'", 'at line', str(obj.line),
                                'file', obj.file)

        sym = self.global_symtab[obj.name]

        if sym.type != typ:
            raise QasmException("Type for '" + sym.name + "' should be '"
                                + typ + "' but was found to be '" + sym.type
                                + "'", "line", str(obj.line), "file", obj.file)

        if obj.type == 'indexed_id':
            bound = sym.index
            ndx = obj.index
            if ndx < 0 or ndx >= bound:
                raise QasmException("Register index for '" + sym.name
                                    + "' out of bounds. Index is", str(ndx),
                                    "bound is 0 <= index <", str(bound),
                                    "at line", str(obj.line), "file", obj.file)

    def verify_reg_list(self, obj, typ):
        """Verify a list of registers."""
        # We expect the object to be a bitlist or an idlist, we don't care.
        # We will iterate it and ensure everything in it is declared as a bit,
        # and throw if not.
        for b in obj.children:
            self.verify_reg(b, typ)

    def pop_scope(self):
        """Return to the previous scope."""
        self.current_symtab = self.symbols.pop()

    def push_scope(self):
        """Enter a new scope."""
        self.symbols.append(self.current_symtab)
        self.current_symtab = {}

    # ---- Begin the PLY parser ----
    start = 'main'

    def p_main(self, p):
        '''
            main : program
        '''
        self.qasm = p[1]

    # ----------------------------------------
    #  program : statement
    #          | program statement
    # ----------------------------------------
    def p_program_0(self, p):
        '''
           program : statement
        '''
        p[0] = node.Program([p[1]])

    def p_program_1(self, p):
        '''
           program : program statement
        '''
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
        if len(p) > 2:
            if p[2] != ';':
                raise QasmException("Missing ';' at end of statement; "
                                    + "received", str(p[2].value))
        p[0] = p[1]

    def p_magic(self, p):
        '''
           magic : MAGIC REAL
        '''
        p[0] = node.Magic([p[2]])

    def p_magic_0(self, p):
        '''
           magic : MAGIC error
        '''
        magic = "2.0;"
        raise QasmException("Invalid magic string. Expected '" + magic
                            + "'.  Is the semicolon missing?")

    # ----------------------------------------
    #  id : ID
    # ----------------------------------------
    def p_id(self, p):
        '''
           id : ID
        '''
        p[0] = p[1]

    def p_id_e(self, p):
        '''
           id : error
        '''
        raise QasmException("Expected an ID, received '"
                            + str(p[1].value) + "'")

    # ----------------------------------------
    #  indexed_id : ID [ int ]
    # ----------------------------------------
    def p_indexed_id(self, p):
        '''
           indexed_id : id '[' NNINTEGER ']'
                      | id '[' NNINTEGER error
                      | id '[' error
        '''
        if len(p) == 4:
            raise QasmException("Expecting an integer index; received",
                                str(p[3].value))
        if p[4] != ']':
            raise QasmException("Missing ']' in indexed ID; received",
                                str(p[4].value))
        p[0] = node.IndexedId([p[1], p[3]])

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
        p[0] = node.IdList([p[1]])

    def p_id_list_1(self, p):
        '''
           id_list : id_list ',' id
        '''
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
        p[0] = node.IdList([p[1]])
        self.update_symtab(p[1])

    def p_gate_id_list_1(self, p):
        '''
           gate_id_list : gate_id_list ',' id
        '''
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
        p[0] = node.IdList([p[1]])
        p[1].is_bit = True
        self.update_symtab(p[1])

    def p_bit_list_1(self, p):
        '''
           bit_list : bit_list ',' id
        '''
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
        p[0] = node.PrimaryList([p[1]])

    def p_primary_list_1(self, p):
        '''
           primary_list : primary_list ',' primary
        '''
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
        if len(p) > 2:
            if p[2] != ';':
                raise QasmException("Missing ';' in qreg or creg declaraton."
                                    + " Instead received '" + p[2].value + "'")
        p[0] = p[1]

    # ----------------------------------------
    #  qreg_decl : QREG indexed_id
    # ----------------------------------------
    def p_qreg_decl(self, p):
        '''
           qreg_decl : QREG indexed_id
        '''
        p[0] = node.Qreg([p[2]])
        if p[2].name in self.external_functions:
            raise QasmException("QREG names cannot be reserved words. "
                                + "Received '" + p[2].name + "'")
        if p[2].index == 0:
            raise QasmException("QREG size must be positive")
        self.update_symtab(p[0])

    def p_qreg_decl_e(self, p):
        '''
           qreg_decl : QREG error
        '''
        raise QasmException("Expecting indexed id (ID[int]) in QREG"
                            + " declaration; received", p[2].value)

    # ----------------------------------------
    #  creg_decl : QREG indexed_id
    # ----------------------------------------
    def p_creg_decl(self, p):
        '''
           creg_decl : CREG indexed_id
        '''
        p[0] = node.Creg([p[2]])
        if p[2].name in self.external_functions:
            raise QasmException("CREG names cannot be reserved words. "
                                + "Received '" + p[2].name + "'")
        if p[2].index == 0:
            raise QasmException("CREG size must be positive")
        self.update_symtab(p[0])

    def p_creg_decl_e(self, p):
        '''
           creg_decl : CREG error
        '''
        raise QasmException("Expecting indexed id (ID[int]) in CREG"
                            + " declaration; received", p[2].value)

    # Gate_body will throw if there are errors, so we don't need to cover
    # that here. Same with the id_lists - if they are not legal, we die
    # before we get here
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
        p[0] = node.Gate([p[2], p[4], p[5]])
        if p[2].name in self.external_functions:
            raise QasmException("GATE names cannot be reserved words. "
                                + "Received '" + p[2].name + "'")
        self.pop_scope()
        self.update_symtab(p[0])

    def p_gate_decl_1(self, p):
        '''
           gate_decl : GATE id gate_scope '(' ')' bit_list gate_body
        '''
        p[0] = node.Gate([p[2], p[6], p[7]])
        if p[2].name in self.external_functions:
            raise QasmException("GATE names cannot be reserved words. "
                                + "Received '" + p[2].name + "'")
        self.pop_scope()
        self.update_symtab(p[0])

    def p_gate_decl_2(self, p):
        '''
        gate_decl : GATE id gate_scope '(' gate_id_list ')' bit_list gate_body
        '''
        p[0] = node.Gate([p[2], p[5], p[7], p[8]])
        if p[2].name in self.external_functions:
            raise QasmException("GATE names cannot be reserved words. "
                                + "Received '" + p[2].name + "'")
        self.pop_scope()
        self.update_symtab(p[0])

    def p_gate_scope(self, p):
        '''
           gate_scope :
        '''
        self.push_scope()

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
        if p[2] != '}':
            raise QasmException("Missing '}' in gate definition; received'"
                                + str(p[2].value) + "'")
        p[0] = node.GateBody(None)

    def p_gate_body_1(self, p):
        '''
           gate_body : '{' gate_op_list '}'
        '''
        p[0] = node.GateBody(p[2])

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
        p[0] = [p[1]]

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
    # Note that it might not be unitary - this is the mechanism that
    # is also used to invoke calls to 'opaque'
    # ----------------------------------------
    def p_unitary_op_0(self, p):
        '''
          unitary_op : U '(' exp_list ')' primary
        '''
        p[0] = node.UniversalUnitary([p[3], p[5]])
        self.verify_reg(p[5], 'qreg')
        self.verify_exp_list(p[3])

    def p_unitary_op_1(self, p):
        '''
        unitary_op : CX primary ',' primary
        '''
        p[0] = node.Cnot([p[2], p[4]])
        self.verify_reg(p[2], 'qreg')
        self.verify_reg(p[4], 'qreg')
        # TODO: check that p[2] and p[4] are distinct

    def p_unitary_op_2(self, p):
        '''
        unitary_op : id primary_list
        '''
        p[0] = node.CustomUnitary([p[1], p[2]])
        self.verify_as_gate(p[1], p[2])
        self.verify_reg_list(p[2], 'qreg')
        # TODO: check that primary_list elements are distinct

    def p_unitary_op_3(self, p):
        '''
        unitary_op : id '(' ')' primary_list
        '''
        p[0] = node.CustomUnitary([p[1], p[4]])
        self.verify_as_gate(p[1], p[4])
        self.verify_reg_list(p[4], 'qreg')
        # TODO: check that primary_list elements are distinct

    def p_unitary_op_4(self, p):
        '''
        unitary_op : id '(' exp_list ')' primary_list
        '''
        p[0] = node.CustomUnitary([p[1], p[3], p[5]])
        self.verify_as_gate(p[1], p[5], arglist=p[3])
        self.verify_reg_list(p[5], 'qreg')
        self.verify_exp_list(p[3])
        # TODO: check that primary_list elements are distinct

    # ----------------------------------------
    # This is a restricted set of "quantum_op" which also
    # prohibits indexed ids, for use in a gate_body
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
        p[0] = node.UniversalUnitary([p[3], p[5]])
        self.verify_declared_bit(p[5])
        self.verify_exp_list(p[3])

    def p_gate_op_0e1(self, p):
        '''
        gate_op : U '(' exp_list ')' error
        '''
        raise QasmException("Invalid U inside gate definition. "
                            + "Missing bit id or ';'")

    def p_gate_op_0e2(self, p):
        '''
        gate_op : U '(' exp_list error
        '''
        raise QasmException("Missing ')' in U invocation in gate definition.")

    def p_gate_op_1(self, p):
        '''
        gate_op : CX id ',' id ';'
        '''
        p[0] = node.Cnot([p[2], p[4]])
        self.verify_declared_bit(p[2])
        self.verify_declared_bit(p[4])
        # TODO: check that p[2] and p[4] are distinct

    def p_gate_op_1e1(self, p):
        '''
        gate_op : CX error
        '''
        raise QasmException("Invalid CX inside gate definition. "
                            + "Expected an ID or '.', received '"
                            + str(p[2].value) + "'")

    def p_gate_op_1e2(self, p):
        '''
        gate_op : CX id ',' error
        '''
        raise QasmException("Invalid CX inside gate definition. "
                            + "Expected an ID or ';', received '"
                            + str(p[4].value) + "'")

    def p_gate_op_2(self, p):
        '''
        gate_op : id id_list ';'
        '''
        p[0] = node.CustomUnitary([p[1], p[2]])
        # To verify:
        # 1. id is declared as a gate in global scope
        # 2. everything in the id_list is declared as a bit in local scope
        self.verify_as_gate(p[1], p[2])
        self.verify_bit_list(p[2])
        # TODO: check that elements of id_list are distinct

    def p_gate_op_2e(self, p):
        '''
        gate_op : id  id_list error
        '''
        raise QasmException("Invalid gate invocation inside gate definition.")

    def p_gate_op_3(self, p):
        '''
        gate_op : id '(' ')' id_list ';'
        '''
        p[0] = node.CustomUnitary([p[1], p[4]])
        self.verify_as_gate(p[1], p[4])
        self.verify_bit_list(p[4])
        # TODO: check that elements of id_list are distinct

    def p_gate_op_4(self, p):
        '''
        gate_op : id '(' exp_list ')' id_list ';'
        '''
        p[0] = node.CustomUnitary([p[1], p[3], p[5]])
        self.verify_as_gate(p[1], p[5], arglist=p[3])
        self.verify_bit_list(p[5])
        self.verify_exp_list(p[3])
        # TODO: check that elements of id_list are distinct

    def p_gate_op_4e0(self, p):
        '''
        gate_op : id '(' ')'  error
        '''
        raise QasmException("Invalid bit list inside gate definition or"
                            + " missing ';'")

    def p_gate_op_4e1(self, p):
        '''
        gate_op : id '('   error
        '''
        raise QasmException("Unmatched () for gate invocation inside gate"
                            + " invocation.")

    def p_gate_op_5(self, p):
        '''
        gate_op : BARRIER id_list ';'
        '''
        p[0] = node.Barrier([p[2]])
        self.verify_bit_list(p[2])
        # TODO: check that elements of id_list are distinct

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
        p[0] = node.Opaque([p[2], p[4]])
        if p[2].name in self.external_functions:
            raise QasmException("OPAQUE names cannot be reserved words. "
                                + "Received '" + p[2].name + "'")
        self.pop_scope()
        self.update_symtab(p[0])

    def p_opaque_1(self, p):
        '''
           opaque : OPAQUE id gate_scope '(' ')' bit_list
        '''
        p[0] = node.Opaque([p[2], p[6]])
        self.pop_scope()
        self.update_symtab(p[0])

    def p_opaque_2(self, p):
        '''
           opaque : OPAQUE id gate_scope '(' gate_id_list ')' bit_list
        '''
        p[0] = node.Opaque([p[2], p[5], p[7]])
        if p[2].name in self.external_functions:
            raise QasmException("OPAQUE names cannot be reserved words. "
                                + "Received '" + p[2].name + "'")
        self.pop_scope()
        self.update_symtab(p[0])

    def p_opaque_1e(self, p):
        '''
           opaque : OPAQUE id gate_scope '(' error
        '''
        raise QasmException("Poorly formed OPAQUE statement.")

    # ----------------------------------------
    # measure : MEASURE primary ASSIGN primary
    # ----------------------------------------
    def p_measure(self, p):
        '''
           measure : MEASURE primary ASSIGN primary
        '''
        p[0] = node.Measure([p[2], p[4]])
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
        p[0] = node.Barrier([p[2]])
        self.verify_reg_list(p[2], 'qreg')
        # TODO: check that elements of primary_list are distinct

    # ----------------------------------------
    # reset : RESET primary
    # ----------------------------------------
    def p_reset(self, p):
        '''
        reset : RESET primary
        '''
        p[0] = node.Reset([p[2]])
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
        if len(p) == 3:
            raise QasmException("Ill-formed IF statement. Perhaps a"
                                + " missing '('?")
        if len(p) == 5:
            raise QasmException("Ill-formed IF statement.  Expected '==', "
                                + "received '" + str(p[4].value))
        if len(p) == 6:
            raise QasmException("Ill-formed IF statement.  Expected a number, "
                                + "received '" + str(p[5].value))
        if len(p) == 7:
            raise QasmException("Ill-formed IF statement, unmatched '('")

        if p[7].type == 'if':
            raise QasmException("Nested IF statements not allowed")
        if p[7].type == 'barrier':
            raise QasmException("barrier not permitted in IF statement")

        p[0] = node.If([p[3], p[5], p[7]])

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
        p[0] = node.Int(p[1])

    def p_unary_1(self, p):
        '''
           unary : REAL
        '''
        p[0] = node.Real(p[1])

    def p_unary_2(self, p):
        '''
           unary : PI
        '''
        p[0] = node.Real(math.pi)

    def p_unary_3(self, p):
        '''
           unary : id
        '''
        p[0] = p[1]

    def p_unary_4(self, p):
        '''
           unary : '(' expression ')'
        '''
        p[0] = p[2]

    def p_unary_6(self, p):
        '''
           unary : id '(' expression ')'
        '''
        # note this is a semantic check, not syntactic
        if p[1].name not in self.external_functions:
            raise QasmException("Illegal external function call: ",
                                str(p[1].name))
        p[0] = node.External([p[1], p[3]])

    # ----------------------------------------
    # Prefix
    # ----------------------------------------
    def p_prefix_expression_0(self, p):
        '''
           prefix_expression : unary
        '''
        p[0] = p[1]

    def p_prefix_expression_1(self, p):
        '''
           prefix_expression : '+' prefix_expression
                             | '-' prefix_expression
        '''
        p[0] = node.Prefix([p[1], p[2]])

    def p_additive_expression_0(self, p):
        '''
            additive_expression : prefix_expression
        '''
        p[0] = p[1]

    def p_additive_expression_1(self, p):
        '''
            additive_expression : additive_expression '+' prefix_expression
                                | additive_expression '-' prefix_expression
        '''
        p[0] = node.BinaryOp([p[2], p[1], p[3]])

    def p_multiplicative_expression_0(self, p):
        '''
            multiplicative_expression : additive_expression
        '''
        p[0] = p[1]

    def p_multiplicative_expression_1(self, p):
        '''
        multiplicative_expression : multiplicative_expression '*' additive_expression
                                  | multiplicative_expression '/' additive_expression
        '''
        p[0] = node.BinaryOp([p[2], p[1], p[3]])

    def p_expression_0(self, p):
        '''
            expression : multiplicative_expression
        '''
        p[0] = p[1]

    def p_expression_1(self, p):
        '''
            expression : expression '^' multiplicative_expression
        '''
        p[0] = node.BinaryOp([p[2], p[1], p[3]])

    # ----------------------------------------
    # exp_list : exp
    #          | exp_list ',' exp
    # ----------------------------------------
    def p_exp_list_0(self, p):
        '''
           exp_list : expression
        '''
        p[0] = node.ExpressionList([p[1]])

    def p_exp_list_1(self, p):
        '''
           exp_list : exp_list ',' expression
        '''
        p[0] = p[1]
        p[0].add_child(p[3])

    def p_ignore(self, p):
        '''
           ignore : STRING
        '''
        # this should never hit but it keeps the unsupressable warnings at bay
        pass

    def p_error(self, p):
        # EOF is a special case because the stupid error token isn't placed
        # on the stack
        if not p:
            raise QasmException("Error at end of file. "
                                + "Perhaps there is a missing ';'")

        col = self.find_column(self.lexer.data, p)
        print("Error near line", str(self.lexer.lineno),  'Column', col)

    def find_column(self, input, token):
        """Compute the column.

        Input is the input text string.
        token is a token instance.
        """
        if token is None:
            return 0
        last_cr = input.rfind('\n', 0, token.lexpos)
        if last_cr < 0:
            last_cr = 0
        column = (token.lexpos - last_cr) + 1
        return column

    def print_tokens(self):
        '''Test method to verify tokenizer.'''
        try:
            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                # TODO: This isn't really the column, it's the character
                # position.  Need to do backtrack to the nearest \n to get
                # the actual column.
                print('TOKEN:' + str(tok) + ":ENDTOKEN", 'at line',
                      tok.lineno, 'column', tok.lexpos, 'file',
                      self.lexer.filename)
        except QasmException as e:
            print('Exception tokenizing qasm file:', e.msg)

    def parse_debug(self, val):
        """Set the parseDeb field."""
        if val is True:
            self.parseDeb = True
        elif val is False:
            self.parseDeb = False
        else:
            raise QasmException("Illegal debug value '" + str(val)
                                + "' must be True or False.")

    def parse(self, data):
        """Parse some data."""
        self.parser.parse(data, lexer=self.lexer, debug=self.parseDeb)
        if self.qasm is None:
            raise QasmException("Uncaught exception in parser; "
                                + "see previous messages for details.")
        return self.qasm

    def print_tree(self):
        """Print parsed OPENQASM."""
        if self.qasm is not None:
            self.qasm.to_string(0)
        else:
            print("No parsed qasm to print")

    def run(self, data):
        """Parser runner.

        To use this module stand-alone.
        """
        ast = self.parser.parse(data, debug=True)
        self.parser.parse(data, debug=True)
        ast.to_string(0)
