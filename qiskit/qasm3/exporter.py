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

"""QASM3 Exporter"""

import collections
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union

from qiskit.circuit import (
    Barrier,
    Clbit,
    Gate,
    Instruction,
    Measure,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    Reset,
)
from qiskit.circuit.bit import Bit
from qiskit.circuit.controlflow import (
    IfElseOp,
    ForLoopOp,
    WhileLoopOp,
    ControlFlowOp,
    BreakLoopOp,
    ContinueLoopOp,
)
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check

from . import ast
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter


class Exporter:
    """QASM3 expoter main class."""

    def __init__(
        self,
        includes: Sequence[str] = ("stdgates.inc",),
        basis_gates: Sequence[str] = ("U",),
        disable_constants: bool = False,
        alias_classical_registers: bool = False,
        indent: str = "  ",
    ):
        """
        Args:
            includes: the filenames that should be emitted as includes.  These files will be parsed
                for gates, and any objects dumped from this exporter will use those definitions
                where possible.
            basis_gates: the basic defined gate set of the backend.
            disable_constants: if ``True``, always emit floating-point constants for numeric
                parameter values.  If ``False`` (the default), then values close to multiples of
                QASM 3 constants (``pi``, ``euler``, and ``tau``) will be emitted in terms of those
                constants instead, potentially improving accuracy in the output.
            alias_classical_registers: If ``True``, then classical bit and classical register
                declarations will look similar to quantum declarations, where the whole set of bits
                will be declared in a flat array, and the registers will just be aliases to
                collections of these bits.  This is inefficient for running OpenQASM 3 programs,
                however, and may not be well supported on backends.  Instead, the default behaviour
                of ``False`` means that individual classical registers will gain their own
                ``bit[size] register;`` declarations, and loose :obj:`.Clbit`\\ s will go onto their
                own declaration.  In this form, each :obj:`.Clbit` must be in either zero or one
                :obj:`.ClassicalRegister`\\ s.
            indent: the indentation string to use for each level within an indented block.  Can be
                set to the empty string to disable indentation.
        """
        self.basis_gates = basis_gates
        self.disable_constants = disable_constants
        self.alias_classical_registers = alias_classical_registers
        self.includes = list(includes)
        self.indent = indent

    def dumps(self, circuit):
        """Convert the circuit to QASM 3, returning the result as a string."""
        with io.StringIO() as stream:
            self.dump(circuit, stream)
            return stream.getvalue()

    def dump(self, circuit, stream):
        """Convert the circuit to QASM 3, dumping the result to a file or text stream."""
        builder = QASM3Builder(
            circuit,
            includeslist=self.includes,
            basis_gates=self.basis_gates,
            disable_constants=self.disable_constants,
            alias_classical_registers=self.alias_classical_registers,
        )
        BasicPrinter(stream, indent=self.indent).visit(builder.build_program())


class GlobalNamespace:
    """Global namespace dict-like."""

    qiskit_gates = {
        "p": standard_gates.PhaseGate,
        "x": standard_gates.XGate,
        "y": standard_gates.YGate,
        "z": standard_gates.ZGate,
        "h": standard_gates.HGate,
        "s": standard_gates.SGate,
        "sdg": standard_gates.SdgGate,
        "t": standard_gates.TGate,
        "tdg": standard_gates.TdgGate,
        "sx": standard_gates.SXGate,
        "rx": standard_gates.RXGate,
        "ry": standard_gates.RYGate,
        "rz": standard_gates.RZGate,
        "cx": standard_gates.CXGate,
        "cy": standard_gates.CYGate,
        "cz": standard_gates.CZGate,
        "cp": standard_gates.CPhaseGate,
        "crx": standard_gates.CRXGate,
        "cry": standard_gates.CRYGate,
        "crz": standard_gates.CRZGate,
        "ch": standard_gates.CHGate,
        "swap": standard_gates.SwapGate,
        "ccx": standard_gates.CCXGate,
        "cswap": standard_gates.CSwapGate,
        "cu": standard_gates.CUGate,
        "CX": standard_gates.CXGate,
        "phase": standard_gates.PhaseGate,
        "cphase": standard_gates.CPhaseGate,
        "id": standard_gates.IGate,
        "u1": standard_gates.U1Gate,
        "u2": standard_gates.U2Gate,
        "u3": standard_gates.U3Gate,
    }
    include_paths = [abspath(join(dirname(__file__), "..", "qasm", "libs"))]

    def __init__(self, includelist, basis_gates=()):
        self._data = {gate: None for gate in basis_gates}

        for includefile in includelist:
            if includefile == "stdgates.inc":
                self._data.update(self.qiskit_gates)
            else:
                # TODO What do if an inc file is not standard?
                # Should it be parsed?
                pass

    def __setitem__(self, name_str, instruction):
        self._data[name_str] = type(instruction)
        self._data[id(instruction)] = name_str

    def __getitem__(self, key):
        if isinstance(key, Instruction):
            return self._data.get(id(key), key.name)
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, instruction):
        if isinstance(instruction, standard_gates.UGate):
            return True
        if id(instruction) in self._data:
            return True
        if type(instruction) in [Gate, Instruction]:  # user-defined instructions/gate
            return self._data.get(instruction.name, None) == instruction
        if instruction.name in self._data:
            if self._data.get(instruction.name) is None:  # it is a basis gate:
                return True
            if isinstance(instruction, self._data.get(instruction.name)):
                return True
        return False

    def register(self, instruction):
        """Register an instruction in the namespace"""
        if instruction.name in self._data:
            self[f"{instruction.name}_{id(instruction)}"] = instruction
        else:
            self[instruction.name] = instruction


# A _Scope is the structure used in the builder to store the contexts and re-mappings of bits from
# the top-level scope where the bits were actually defined.  In the class, 'circuit' is an instance
# of QuantumCircuit that defines this level, and 'bit_map' is a mapping of 'Bit: Bit', where the
# keys are bits in the circuit in this scope, and the values are the Bit in the top-level scope in
# this context that this bit actually represents.  This is a cheap hack around actually implementing
# a proper symbol table.
_Scope = collections.namedtuple("_Scope", ("circuit", "bit_map"))


class QASM3Builder:
    """QASM3 builder constructs an AST from a QuantumCircuit."""

    builtins = (Barrier, Measure, Reset, BreakLoopOp, ContinueLoopOp)
    gate_parameter_prefix = "_gate_p"
    gate_qubit_prefix = "_gate_q"

    def __init__(
        self,
        quantumcircuit,
        includeslist,
        basis_gates,
        disable_constants,
        alias_classical_registers,
    ):
        # This is a stack of stacks; the outer stack is a list of "outer" look-up contexts, and the
        # inner stack is for scopes within these.  A "outer" look-up context in this sense means
        # the main program body or a gate/subroutine definition, whereas the scopes are for things
        # like the body of a ``for`` loop construct.
        self._circuit_ctx = []
        self.push_context(quantumcircuit)
        self.includeslist = includeslist
        self._gate_to_declare = {}
        self._subroutine_to_declare = {}
        self._opaque_to_declare = {}
        self._flat_reg = False
        self._physical_qubit = False
        self._loose_clbit_index_lookup = {}
        self.disable_constants = disable_constants
        self.alias_classical_registers = alias_classical_registers
        self.global_namespace = GlobalNamespace(includeslist, basis_gates)

    def _register_gate(self, gate):
        self.global_namespace.register(gate)
        self._gate_to_declare[id(gate)] = gate

    def _register_subroutine(self, instruction):
        self.global_namespace.register(instruction)
        self._subroutine_to_declare[id(instruction)] = instruction

    def _register_opaque(self, instruction):
        if instruction not in self.global_namespace:
            self.global_namespace.register(instruction)
            self._opaque_to_declare[id(instruction)] = instruction

    def build_header(self):
        """Builds a Header"""
        version = ast.Version("3")
        includes = self.build_includes()
        return ast.Header(version, includes)

    def build_program(self):
        """Builds a Program"""
        self.hoist_declarations(self.global_scope(assert_=True).circuit.data)
        return ast.Program(self.build_header(), self.build_global_statements())

    def hoist_declarations(self, instructions):
        """Walks the definitions in gates/instructions to make a list of gates to declare."""
        for instruction in instructions:
            if isinstance(instruction[0], ControlFlowOp):
                for block in instruction[0].blocks:
                    self.hoist_declarations(block.data)
                continue
            if instruction[0] in self.global_namespace or isinstance(instruction[0], self.builtins):
                continue

            if instruction[0].definition is None:
                self._register_opaque(instruction[0])
            else:
                self.hoist_declarations(instruction[0].definition.data)
                if isinstance(instruction[0], Gate):
                    self._register_gate(instruction[0])
                else:
                    self._register_subroutine(instruction[0])

    def global_scope(self, assert_=False):
        """Return the global circuit scope that is used as the basis of the full program.  If
        ``assert_=True``, then this raises :obj:`.QASM3ExporterError` if the current context is not
        the global one."""
        if assert_ and len(self._circuit_ctx) != 1 and len(self._circuit_ctx[0]) != 1:
            # Defensive code to help catch logic errors.
            raise QASM3ExporterError(  # pragma: no cover
                f"Not currently in the global context. Current contexts are: {self._circuit_ctx}"
            )
        return self._circuit_ctx[0][0]

    def current_outermost_scope(self):
        """Return the outermost scope for this context.  If building the main program, then this is
        the :obj:`.QuantumCircuit` instance that the full program is being built from.  If building
        a gate or subroutine definition, this is the body that defines the gate or subroutine."""
        return self._circuit_ctx[-1][0]

    def current_scope(self):
        """Return the current circuit scope."""
        return self._circuit_ctx[-1][-1]

    def current_context(self):
        """Return the current context (list of scopes)."""
        return self._circuit_ctx[-1]

    def push_scope(self, circuit: QuantumCircuit, qubits: Iterable[Qubit], clbits: Iterable[Clbit]):
        """Push a new scope (like a ``for`` or ``while`` loop body) onto the current context
        stack."""
        current_map = self.current_scope().bit_map
        qubits = tuple(current_map[qubit] for qubit in qubits)
        clbits = tuple(current_map[clbit] for clbit in clbits)
        if circuit.num_qubits != len(qubits):
            raise QASM3ExporterError(  # pragma: no cover
                f"Tried to push a scope whose circuit needs {circuit.num_qubits} qubits, but only"
                f" provided {len(qubits)} qubits to create the mapping."
            )
        if circuit.num_clbits != len(clbits):
            raise QASM3ExporterError(  # pragma: no cover
                f"Tried to push a scope whose circuit needs {circuit.num_clbits} clbits, but only"
                f" provided {len(clbits)} clbits to create the mapping."
            )
        mapping = dict(itertools.chain(zip(circuit.qubits, qubits), zip(circuit.clbits, clbits)))
        self._circuit_ctx[-1].append(_Scope(circuit, mapping))

    def pop_scope(self) -> _Scope:
        """Pop the current scope (like a ``for`` or ``while`` loop body) off the current context
        stack."""
        if len(self._circuit_ctx[-1]) <= 1:
            raise QASM3ExporterError(  # pragma: no cover
                "Tried to pop a scope from the current context, but there are no current scopes."
            )
        return self._circuit_ctx[-1].pop()

    def push_context(self, outer_context: QuantumCircuit):
        """Push a new context (like for a ``gate`` or ``def`` body) onto the stack."""
        mapping = {bit: bit for bit in itertools.chain(outer_context.qubits, outer_context.clbits)}
        self._circuit_ctx.append([_Scope(outer_context, mapping)])

    def pop_context(self):
        """Pop the current context (like for a ``gate`` or ``def`` body) onto the stack."""
        if len(self._circuit_ctx) == 1:
            raise QASM3ExporterError(  # pragma: no cover
                "Tried to pop the current context, but that is the global context."
            )
        if len(self._circuit_ctx[-1]) != 1:
            raise QASM3ExporterError(  # pragma: no cover
                "Tried to pop the current context while there are still"
                f" {len(self._circuit_ctx[-1]) - 1} unclosed scopes."
            )
        self._circuit_ctx.pop()

    def build_includes(self):
        """Builds a list of included files."""
        return [ast.Include(filename) for filename in self.includeslist]

    def build_global_statements(self) -> List[ast.Statement]:
        """
        globalStatement
            : subroutineDefinition
            | kernelDeclaration
            | quantumGateDefinition
            | calibration
            | quantumDeclarationStatement  # build_quantumdeclaration
            | pragma
            ;

        statement
            : expressionStatement
            | assignmentStatement
            | classicalDeclarationStatement
            | branchingStatement
            | loopStatement
            | endStatement
            | aliasStatement
            | quantumStatement  # build_quantuminstruction
            ;
        """
        definitions = self.build_definitions()
        inputs, outputs, variables = self.build_variable_declarations()
        bit_declarations = self.build_classical_declarations()
        context = self.global_scope(assert_=True).circuit
        if getattr(context, "_layout", None) is not None:
            self._physical_qubit = True
            quantum_declarations = []
        else:
            quantum_declarations = self.build_quantum_declarations()
        quantum_instructions = self.build_quantum_instructions(context.data)
        self._physical_qubit = False

        return [
            statement
            for source in (
                definitions,
                inputs,
                outputs,
                variables,
                bit_declarations,
                quantum_declarations,
                quantum_instructions,
            )
            for statement in source
        ]

    def build_definitions(self):
        """Builds all the definition."""
        ret = []
        for instruction in self._opaque_to_declare.values():
            ret.append(self.build_definition(instruction, self.build_opaque_definition))
        for instruction in self._subroutine_to_declare.values():
            ret.append(self.build_definition(instruction, self.build_subroutine_definition))
        for instruction in self._gate_to_declare.values():
            ret.append(self.build_definition(instruction, self.build_gate_definition))
        return ret

    def build_definition(self, instruction, builder):
        """Using a given definition builder, builds that definition."""
        try:
            return instruction._define_qasm3()
        except AttributeError:
            pass
        self._flat_reg = True
        definition = builder(instruction)
        self._flat_reg = False
        return definition

    def build_opaque_definition(self, instruction):
        """Builds an Opaque gate definition as a CalibrationDefinition"""
        name = self.global_namespace[instruction]
        quantum_arguments = [
            ast.Identifier(f"{self.gate_qubit_prefix}_{n}") for n in range(instruction.num_qubits)
        ]
        return ast.CalibrationDefinition(ast.Identifier(name), quantum_arguments)

    def build_subroutine_definition(self, instruction):
        """Builds a SubroutineDefinition"""
        name = self.global_namespace[instruction]
        self.push_context(instruction.definition)
        quantum_arguments = [
            ast.QuantumArgument(ast.Identifier(f"{self.gate_qubit_prefix}_{n_qubit}"))
            for n_qubit in range(len(instruction.definition.qubits))
        ]
        subroutine_body = ast.SubroutineBlock(
            self.build_quantum_instructions(instruction.definition.data),
        )
        self.pop_context()
        return ast.SubroutineDefinition(ast.Identifier(name), subroutine_body, quantum_arguments)

    def build_gate_definition(self, gate):
        """Builds a QuantumGateDefinition"""
        signature = self.build_gate_signature(gate)
        self.push_context(gate.definition)
        body = ast.QuantumBlock(self.build_quantum_instructions(gate.definition.data))
        self.pop_context()
        return ast.QuantumGateDefinition(signature, body)

    def build_gate_signature(self, gate):
        """Builds a QuantumGateSignature"""
        name = self.global_namespace[gate]
        params = []
        definition = gate.definition
        # Dummy parameters
        for num in range(len(gate.params) - len(definition.parameters)):
            param_name = f"{self.gate_parameter_prefix}_{num}"
            params.append(ast.Identifier(param_name))
        params += [ast.Identifier(param.name) for param in definition.parameters]
        quantum_arguments = [
            ast.Identifier(f"{self.gate_qubit_prefix}_{n_qubit}")
            for n_qubit in range(len(definition.qubits))
        ]
        return ast.QuantumGateSignature(ast.Identifier(name), quantum_arguments, params or None)

    def build_variable_declarations(self):
        """Builds lists of the input, output and standard variables used in this program."""
        inputs, outputs, variables = [], [], []
        global_scope = self.global_scope(assert_=True).circuit
        for parameter in global_scope.parameters:
            declaration = _infer_variable_declaration(global_scope, parameter)
            if declaration is None:
                continue
            if isinstance(declaration, ast.IODeclaration):
                if declaration.modifier is ast.IOModifier.INPUT:
                    inputs.append(declaration)
                else:
                    outputs.append(declaration)
            else:
                variables.append(declaration)
        return inputs, outputs, variables

    @property
    def base_classical_register_name(self):
        """The base register name"""
        name = "_all_clbits" if self.alias_classical_registers else "_loose_clbits"
        if name in self.global_namespace._data:
            raise NotImplementedError  # TODO choose a different name if there is a name collision
        return name

    @property
    def base_quantum_register_name(self):
        """The base register name"""
        name = "_all_qubits"
        if name in self.global_namespace._data:
            raise NotImplementedError  # TODO choose a different name if there is a name collision
        return name

    def build_classical_declarations(self):
        """Return a list of AST nodes declaring all the classical bits and registers.

        The behaviour of this function depends on the setting ``alias_classical_registers``. If this
        is ``True``, then the output will be in the same form as the output of
        :meth:`.build_classical_declarations`, with the registers being aliases.  If ``False``, it
        will instead return a :obj:`.ast.ClassicalDeclaration` for each classical register, and one
        for the loose :obj:`.Clbit` instances, and will raise :obj:`QASM3ExporterError` if any
        registers overlap.

        This function populates the lookup table ``self._loose_clbit_index_lookup``.
        """
        circuit = self.current_scope().circuit
        if self.alias_classical_registers:
            self._loose_clbit_index_lookup = {
                bit: index for index, bit in enumerate(circuit.clbits)
            }
            flat_declaration = self.build_clbit_declaration(
                len(circuit.clbits), self.base_classical_register_name
            )
            return [flat_declaration] + self.build_aliases(circuit.cregs)
        loose_register_size = 0
        for index, bit in enumerate(circuit.clbits):
            found_bit = circuit.find_bit(bit)
            if len(found_bit.registers) > 1:
                raise QASM3ExporterError(
                    f"Clbit {index} is in multiple registers, but 'alias_classical_registers' is"
                    f" False. Registers and indices: {found_bit.registers}."
                )
            if not found_bit.registers:
                self._loose_clbit_index_lookup[bit] = loose_register_size
                loose_register_size += 1
        if loose_register_size > 0:
            loose = [
                self.build_clbit_declaration(loose_register_size, self.base_classical_register_name)
            ]
        else:
            loose = []
        return loose + [
            self.build_clbit_declaration(len(register), register.name) for register in circuit.cregs
        ]

    def build_clbit_declaration(self, n_clbits: int, name: str) -> ast.ClassicalDeclaration:
        """Return a declaration of the :obj:`.Clbit`\\ s as a ``bit[n]``."""
        return ast.ClassicalDeclaration(ast.BitArrayType(n_clbits), ast.Identifier(name))

    def build_quantum_declarations(self):
        """Return a list of AST nodes declaring all the qubits in the current scope, and all the
        alias declarations for these qubits."""
        return [self.build_qubit_declarations()] + self.build_aliases(
            self.current_scope().circuit.qregs
        )

    def build_qubit_declarations(self):
        """Return a declaration of all the :obj:`.Qubit`\\ s in the current scope."""
        # Base register
        return ast.QuantumDeclaration(
            ast.Identifier(self.base_quantum_register_name),
            ast.Designator(self.build_integer(self.current_scope().circuit.num_qubits)),
        )

    def build_aliases(self, registers: Iterable[Register]) -> List[ast.AliasStatement]:
        """Return a list of alias declarations for the given registers.  The registers can be either
        classical or quantum."""
        out = []
        for register in registers:
            elements = []
            # Greedily consolidate runs of bits into ranges.  We don't bother trying to handle
            # steps; there's no need in generated code.  Even single bits are referenced as ranges
            # because the concatenation in an alias statement can only concatenate arraylike values.
            start_index, prev_index = None, None
            register_identifier = (
                ast.Identifier(self.base_quantum_register_name)
                if isinstance(register, QuantumRegister)
                else ast.Identifier(self.base_classical_register_name)
            )
            for bit in register:
                cur_index = self.find_bit(bit).index
                if start_index is None:
                    start_index = cur_index
                elif cur_index != prev_index + 1:
                    elements.append(
                        ast.SubscriptedIdentifier(
                            register_identifier,
                            ast.Range(
                                start=self.build_integer(start_index),
                                end=self.build_integer(prev_index),
                            ),
                        )
                    )
                    start_index = prev_index = cur_index
                prev_index = cur_index
            # After the loop, if there were any bits at all, there's always one unemitted range.
            if len(register) != 0:
                elements.append(
                    ast.SubscriptedIdentifier(
                        register_identifier,
                        ast.Range(
                            start=self.build_integer(start_index),
                            end=self.build_integer(prev_index),
                        ),
                    )
                )
            out.append(ast.AliasStatement(ast.Identifier(register.name), elements))
        return out

    def build_quantum_instructions(self, instructions):
        """Builds a list of call statements"""
        ret = []
        for instruction in instructions:
            if isinstance(instruction[0], Gate):
                if instruction[0].condition:
                    eqcondition = self.build_eqcondition(instruction[0].condition)
                    instruction_without_condition = instruction[0].copy()
                    instruction_without_condition.condition = None
                    true_body = self.build_program_block(
                        [(instruction_without_condition, instruction[1], instruction[2])]
                    )
                    ret.append(ast.BranchingStatement(eqcondition, true_body))
                else:
                    ret.append(self.build_gate_call(instruction))
            elif isinstance(instruction[0], Barrier):
                operands = [self.build_single_bit_reference(operand) for operand in instruction[1]]
                ret.append(ast.QuantumBarrier(operands))
            elif isinstance(instruction[0], Measure):
                measurement = ast.QuantumMeasurement(
                    [self.build_single_bit_reference(operand) for operand in instruction[1]]
                )
                qubit = self.build_single_bit_reference(instruction[2][0])
                ret.append(ast.QuantumMeasurementAssignment(qubit, measurement))
            elif isinstance(instruction[0], Reset):
                for operand in instruction[1]:
                    ret.append(ast.QuantumReset(self.build_single_bit_reference(operand)))
            elif isinstance(instruction[0], ForLoopOp):
                ret.append(self.build_for_loop(*instruction))
            elif isinstance(instruction[0], WhileLoopOp):
                ret.append(self.build_while_loop(*instruction))
            elif isinstance(instruction[0], IfElseOp):
                ret.append(self.build_if_statement(*instruction))
            elif isinstance(instruction[0], BreakLoopOp):
                ret.append(ast.BreakStatement())
            elif isinstance(instruction[0], ContinueLoopOp):
                ret.append(ast.ContinueStatement())
            else:
                ret.append(self.build_subroutine_call(instruction))
        return ret

    def build_if_statement(
        self, instruction: IfElseOp, qubits: Iterable[Qubit], clbits: Iterable[Clbit]
    ) -> ast.BranchingStatement:
        """Build an :obj:`.IfElseOp` into a :obj:`.ast.BranchingStatement`."""
        condition = self.build_eqcondition(instruction.condition)

        true_circuit = instruction.blocks[0]
        self.push_scope(true_circuit, qubits, clbits)
        true_body = self.build_program_block(true_circuit.data)
        self.pop_scope()
        if len(instruction.blocks) == 1:
            return ast.BranchingStatement(condition, true_body, None)

        false_circuit = instruction.blocks[1]
        self.push_scope(false_circuit, qubits, clbits)
        false_body = self.build_program_block(false_circuit.data)
        self.pop_scope()
        return ast.BranchingStatement(condition, true_body, false_body)

    def build_while_loop(
        self, instruction: WhileLoopOp, qubits: Iterable[Qubit], clbits: Iterable[Clbit]
    ) -> ast.WhileLoopStatement:
        """Build a :obj:`.WhileLoopOp` into a :obj:`.ast.WhileLoopStatement`."""
        condition = self.build_eqcondition(instruction.condition)
        loop_circuit = instruction.blocks[0]
        self.push_scope(loop_circuit, qubits, clbits)
        loop_body = self.build_program_block(loop_circuit.data)
        self.pop_scope()
        return ast.WhileLoopStatement(condition, loop_body)

    def build_for_loop(
        self, instruction: ForLoopOp, qubits: Iterable[Qubit], clbits: Iterable[Clbit]
    ) -> ast.ForLoopStatement:
        """Build a :obj:`.ForLoopOp` into a :obj:`.ast.ForLoopStatement`."""
        loop_parameter, indexset, loop_circuit = instruction.params
        if loop_parameter is None:
            # The loop parameter is implicitly declared by the ``for`` loop (see also
            # _infer_parameter_declaration), so it doesn't matter that we haven't declared this,
            # except for the concerns about symbol collision which can only be alleviated by
            # introducing a proper symbol table to this exporter.
            loop_parameter_ast = ast.Identifier("_")
        else:
            loop_parameter_ast = ast.Identifier(loop_parameter.name)
        if isinstance(indexset, range):
            # QASM 3 uses inclusive ranges on both ends, unlike Python.
            indexset_ast = ast.Range(
                start=self.build_integer(indexset.start),
                end=self.build_integer(indexset.stop - 1),
                step=self.build_integer(indexset.step) if indexset.step != 1 else None,
            )
        else:
            try:
                indexset_ast = ast.IndexSet([self.build_integer(value) for value in indexset])
            except QASM3ExporterError:
                raise QASM3ExporterError(
                    "The values in QASM 3 'for' loops must all be integers, but received"
                    f" '{indexset}'."
                ) from None
        self.push_scope(loop_circuit, qubits, clbits)
        body_ast = self.build_program_block(loop_circuit)
        self.pop_scope()
        return ast.ForLoopStatement(loop_parameter_ast, indexset_ast, body_ast)

    def build_integer(self, value) -> ast.Integer:
        """Build an integer literal, raising a :obj:`.QASM3ExporterError` if the input is not
        actually an
        integer."""
        if not isinstance(value, numbers.Integral):
            # This is meant to be purely defensive, in case a non-integer slips into the logic
            # somewhere, but no valid Terra object should trigger this.
            raise QASM3ExporterError(f"'{value}' is not an integer")  # pragma: no cover
        return ast.Integer(int(value))

    def build_program_block(self, instructions):
        """Builds a ProgramBlock"""
        return ast.ProgramBlock(self.build_quantum_instructions(instructions))

    def build_eqcondition(self, condition):
        """Classical Conditional condition from a instruction.condition"""
        if isinstance(condition[0], Clbit):
            condition_on = self.build_single_bit_reference(condition[0])
        else:
            condition_on = ast.Identifier(condition[0].name)
        return ast.ComparisonExpression(
            condition_on, ast.EqualsOperator(), self.build_integer(condition[1])
        )

    def build_gate_call(self, instruction):
        """Builds a QuantumGateCall"""
        if isinstance(instruction[0], standard_gates.UGate):
            gate_name = ast.Identifier("U")
        else:
            gate_name = ast.Identifier(self.global_namespace[instruction[0]])
        qubits = [self.build_single_bit_reference(qubit) for qubit in instruction[1]]
        if self.disable_constants:
            parameters = [ast.Expression(param) for param in instruction[0].params]
        else:
            parameters = [
                ast.Expression(pi_check(param, output="qasm")) for param in instruction[0].params
            ]

        return ast.QuantumGateCall(gate_name, qubits, parameters=parameters)

    def build_subroutine_call(self, instruction):
        """Builds a SubroutineCall"""
        identifier = ast.Identifier(self.global_namespace[instruction[0]])
        expressions = [ast.Expression(param) for param in instruction[0].params]
        # TODO: qubits should go inside the brackets of subroutine calls, but neither Terra nor the
        # AST here really support the calls, so there's no sensible way of writing it yet.
        bits = [self.build_single_bit_reference(bit) for bit in instruction[1]]
        return ast.SubroutineCall(identifier, bits, expressions)

    def build_single_bit_reference(self, bit: Bit) -> ast.Identifier:
        """Get an identifier node that refers to one particular bit."""
        found_bit = self.find_bit(bit)
        if self._physical_qubit and isinstance(bit, Qubit):
            return ast.PhysicalQubitIdentifier(ast.Identifier(str(found_bit.index)))
        if self._flat_reg:
            return ast.Identifier(f"{self.gate_qubit_prefix}_{found_bit.index}")
        if found_bit.registers:
            # We preferentially return a reference via a register in the hope that this is what the
            # user is used to seeing as well.
            register, index = found_bit.registers[0]
            return ast.SubscriptedIdentifier(
                ast.Identifier(register.name), self.build_integer(index)
            )
        # Otherwise reference via the list of all qubits, or the list of loose clbits.
        if isinstance(bit, Qubit):
            return ast.SubscriptedIdentifier(
                ast.Identifier(self.base_quantum_register_name), self.build_integer(found_bit.index)
            )
        return ast.SubscriptedIdentifier(
            ast.Identifier(self.base_classical_register_name),
            self.build_integer(self._loose_clbit_index_lookup[bit]),
        )

    def find_bit(self, bit: Bit):
        """Look up the bit using :meth:`.QuantumCircuit.find_bit` in the current outermost scope."""
        # This is a hacky work-around for now. Really this should be a proper symbol-table lookup,
        # but with us expecting to put in a whole new AST for Terra 0.20, this should be sufficient
        # for the use-cases we support.  (Jake, 2021-11-22.)
        if len(self.current_context()) > 1:
            ancestor_bit = self.current_scope().bit_map[bit]
            return self.current_outermost_scope().circuit.find_bit(ancestor_bit)
        return self.current_scope().circuit.find_bit(bit)


def _infer_variable_declaration(
    circuit: QuantumCircuit, parameter: Parameter
) -> Union[ast.ClassicalDeclaration, None]:
    """Attempt to infer what type a parameter should be declared as to work with a circuit.

    This is very simplistic; it assumes all parameters are real numbers that need to be input to the
    program, unless one is used as a loop variable, in which case it shouldn't be declared at all,
    because the ``for`` loop declares it implicitly (per the Qiskit/QSS reading of the OpenQASM
    spec at Qiskit/openqasm@8ee55ec).

    .. note::

        This is a hack around not having a proper type system implemented in Terra, and really this
        whole function should be removed in favour of proper symbol-table building and lookups.
        This function is purely to try and hack the parameters for ``for`` loops into the exporter
        for now.

    Args:
        circuit: The global-scope circuit, which is the base of the exported program.
        parameter: The parameter to infer the type of.

    Returns:
        A suitable :obj:`.ast.ClassicalDeclaration` node, or, if the parameter should *not* be
        declared, then ``None``.
    """

    def is_loop_variable(circuit, parameter):
        """Recurse into the instructions a parameter is used in, checking at every level if it is
        used as the loop variable of a ``for`` loop."""
        # This private access is hacky, and shouldn't need to happen; the type of a parameter
        # _should_ be an intrinsic part of the parameter, or somewhere publicly accessible, but
        # Terra doesn't have those concepts yet.  We can only try and guess at the type by looking
        # at all the places it's used in the circuit.
        for instruction, index in circuit._parameter_table[parameter]:
            if isinstance(instruction, ForLoopOp):
                # The parameters of ForLoopOp are (loop_parameter, indexset, body).
                if index == 0:
                    return True
            if isinstance(instruction, ControlFlowOp):
                if is_loop_variable(instruction.params[index], parameter):
                    return True
        return False

    if is_loop_variable(circuit, parameter):
        return None
    # Arbitrary choice of double-precision float for all other parameters, but it's what we actually
    # expect people to be binding to their Parameters right now.
    return ast.IODeclaration(
        ast.IOModifier.INPUT, ast.FloatType.DOUBLE, ast.Identifier(parameter.name)
    )
