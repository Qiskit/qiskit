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

# pylint: disable=invalid-name

"""QASM3 Exporter"""

import io
from os.path import dirname, join, abspath, exists
from typing import Sequence

from qiskit.circuit.tools import pi_check
from qiskit.circuit import Gate, Barrier, Measure, QuantumRegister, Instruction
from qiskit.circuit.library.standard_gates import (
    UGate,
    PhaseGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    SXGate,
    RXGate,
    RYGate,
    RZGate,
    CXGate,
    CYGate,
    CZGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CHGate,
    SwapGate,
    CCXGate,
    CSwapGate,
    CUGate,
    # CXGate Again
    # PhaseGate Again,
    # CPhaseGate Again,
    IGate,  # Is this id?
    U1Gate,
    U2Gate,
    U3Gate,
)
from qiskit.circuit.bit import Bit
from qiskit.circuit import Qubit, Clbit
from .ast import (
    Program,
    Version,
    Include,
    Header,
    Identifier,
    IndexIdentifier2,
    QuantumBlock,
    QuantumBarrier,
    Designator,
    Statement,
    SubroutineCall,
    SubroutineDefinition,
    SubroutineBlock,
    BranchingStatement,
    QuantumGateCall,
    QuantumDeclaration,
    QuantumGateSignature,
    QuantumGateDefinition,
    QuantumMeasurement,
    QuantumMeasurementAssignment,
    Integer,
    ProgramBlock,
    ComparisonExpression,
    BitDeclaration,
    EqualsOperator,
    QuantumArgument,
    Expression,
    CalibrationDefinition,
    IOModifier,
    IO,
    PhysicalQubitIdentifier,
    AliasStatement,
)
from .printer import BasicPrinter


class Exporter:
    """QASM3 expoter main class."""

    def __init__(
        self,
        includes: Sequence[str] = ("stdgates.inc",),
        basis_gates: Sequence[str] = ("U",),
        disable_constants: bool = False,
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
            indent: the indentation string to use for each level within an indented block.  Can be
                set to the empty string to disable indentation.
        """
        self.basis_gates = basis_gates
        self.disable_constants = disable_constants
        self.includes = list(includes)
        self.indent = indent

    def dumps(self, circuit):
        """Convert the circuit to QASM 3, returning the result as a string."""
        with io.StringIO() as stream:
            self.dump(circuit, stream)
            return stream.getvalue()

    def dump(self, circuit, stream):
        """Convert the circuit to QASM 3, dumping the result to a file or text stream."""
        builder = Qasm3Builder(circuit, self.includes, self.basis_gates, self.disable_constants)
        BasicPrinter(stream, indent=self.indent).visit(builder.build_program())


class GlobalNamespace:
    """Global namespace dict-like."""

    qiskit_gates = {
        "p": PhaseGate,
        "x": XGate,
        "y": YGate,
        "z": ZGate,
        "h": HGate,
        "s": SGate,
        "sdg": SdgGate,
        "t": TGate,
        "tdg": TdgGate,
        "sx": SXGate,
        "rx": RXGate,
        "ry": RYGate,
        "rz": RZGate,
        "cx": CXGate,
        "cy": CYGate,
        "cz": CZGate,
        "cp": CPhaseGate,
        "crx": CRXGate,
        "cry": CRYGate,
        "crz": CRZGate,
        "ch": CHGate,
        "swap": SwapGate,
        "ccx": CCXGate,
        "cswap": CSwapGate,
        "cu": CUGate,
        "CX": CXGate,
        "phase": PhaseGate,
        "cphase": CPhaseGate,
        "id": IGate,
        "u1": U1Gate,
        "u2": U2Gate,
        "u3": U3Gate,
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

    def _extract_gates_from_file(self, filename):
        gates = set()
        for filepath in self._find_lib(filename):
            with open(filepath) as fp:
                for line in fp.readlines():
                    if line.startswith("gate "):
                        gates.add(line[5:].split("(")[0].split()[0])
        return gates

    def _find_lib(self, filename):
        for include_path in self.include_paths:
            full_path = join(include_path, filename)
            if exists(full_path):
                yield full_path

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
        if isinstance(instruction, UGate):
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


class Qasm3Builder:
    """QASM3 builder constructs an AST from a QuantumCircuit."""

    builtins = (Barrier, Measure)

    def __init__(self, quantumcircuit, includeslist, basis_gates, disable_constants):
        self.circuit_ctx = [quantumcircuit]
        self.includeslist = includeslist
        self._gate_to_declare = {}
        self._subroutine_to_declare = {}
        self._opaque_to_declare = {}
        self._flat_reg = False
        self._physical_qubit = False
        self.disable_constants = disable_constants
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
        version = Version("3")
        includes = self.build_includes()
        return Header(version, includes)

    def build_program(self):
        """Builds a Program"""
        self.hoist_declarations(self.circuit_ctx[-1].data)
        return Program(self.build_header(), self.build_globalstatements())

    def hoist_declarations(self, instructions):
        """Walks the definitions in gates/instructions to make a list of gates to declare."""
        for instruction in instructions:
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

    def build_includes(self):
        """Builds a list of included files."""
        return [Include(filename) for filename in self.includeslist]

    def build_globalstatements(self) -> [Statement]:
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
        inputs = self.build_inputs()
        bitdeclarations = self.build_bitdeclarations()
        quantumdeclarations = None
        if hasattr(self.circuit_ctx[-1], "_layout") and self.circuit_ctx[-1]._layout is not None:
            self._physical_qubit = True
        else:
            quantumdeclarations = self.build_quantumdeclarations()
        quantuminstructions = self.build_quantuminstructions(self.circuit_ctx[-1].data)
        self._physical_qubit = False

        ret = []
        if definitions:
            ret += definitions
        if inputs:
            ret += inputs
        if bitdeclarations:
            ret += bitdeclarations
        if quantumdeclarations:
            ret += quantumdeclarations
        if quantuminstructions:
            ret += quantuminstructions
        return ret

    def build_definitions(self):
        """Builds all the definition."""
        ret = []
        for instruction in self._opaque_to_declare.values():
            ret.append(self.build_definition(instruction, self.build_opaquedefinition))
        for instruction in self._subroutine_to_declare.values():
            ret.append(self.build_definition(instruction, self.build_subroutinedefinition))
        for instruction in self._gate_to_declare.values():
            ret.append(self.build_definition(instruction, self.build_quantumgatedefinition))
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

    def build_opaquedefinition(self, instruction):
        """Builds an Opaque gate definition as a CalibrationDefinition"""
        name = self.global_namespace[instruction]
        quantumArgumentList = [Identifier(f"q_{n}") for n in range(instruction.num_qubits)]
        return CalibrationDefinition(Identifier(name), quantumArgumentList)

    def build_subroutinedefinition(self, instruction):
        """Builds a SubroutineDefinition"""
        name = self.global_namespace[instruction]
        self.circuit_ctx.append(instruction.definition)
        quantumArgumentList = self.build_quantumArgumentList(
            instruction.definition.qregs, instruction.definition
        )
        subroutineBlock = SubroutineBlock(
            self.build_quantuminstructions(instruction.definition.data),
        )
        self.circuit_ctx.pop()
        return SubroutineDefinition(Identifier(name), subroutineBlock, quantumArgumentList)

    def build_quantumgatedefinition(self, gate):
        """Builds a QuantumGateDefinition"""
        quantumGateSignature = self.build_quantumGateSignature(gate)

        self.circuit_ctx.append(gate.definition)
        quantumBlock = QuantumBlock(self.build_quantuminstructions(gate.definition.data))
        self.circuit_ctx.pop()
        return QuantumGateDefinition(quantumGateSignature, quantumBlock)

    def build_quantumGateSignature(self, gate):
        """Builds a QuantumGateSignature"""
        name = self.global_namespace[gate]
        params = []
        # Dummy parameters
        for num in range(len(gate.params) - len(gate.definition.parameters)):
            param_name = f"param_{num}"
            params.append(Identifier(param_name))
        params += [Identifier(param.name) for param in gate.definition.parameters]

        self.circuit_ctx.append(gate.definition)
        qargList = []
        for qreg in gate.definition.qregs:
            for qubit in qreg:
                qreg, idx = self.find_bit(qubit)
                qubit_name = f"{qreg.name}_{idx}"
                qargList.append(Identifier(qubit_name))
        self.circuit_ctx.pop()

        return QuantumGateSignature(Identifier(name), qargList, params or None)

    def build_inputs(self):
        """Builds a list of Inputs"""
        ret = []
        for param in self.circuit_ctx[-1].parameters:
            ret.append(IO(IOModifier.input, Identifier("float[32]"), Identifier(param.name)))
        return ret

    def build_bitdeclarations(self):
        """Builds a list of BitDeclarations"""
        ret = []
        for creg in self.circuit_ctx[-1].cregs:
            ret.append(BitDeclaration(Identifier(creg.name), Designator(Integer(creg.size))))
        return ret

    @property
    def base_register_name(self):
        """The base register name"""
        name = "_q"
        if name in self.global_namespace._data:
            raise NotImplementedError  # TODO choose a different name if there is a name collision
        return name

    def build_quantumdeclarations(self):
        """Builds a single QuantumDeclaration for the base register and a list of aliases

        The Base register name is the way the exporter handle registers.
        All the qubits are part of a long flat register and the QuantumRegisters are aliases
        """
        ret = []

        # Base register
        ret.append(
            QuantumDeclaration(
                Identifier(self.base_register_name),
                Designator(Integer(self.circuit_ctx[-1].num_qubits)),
            )
        )
        # aliases
        for qreg in self.circuit_ctx[-1].qregs:
            qubits = []
            for qubit in qreg:
                qubits.append(
                    IndexIdentifier2(
                        Identifier(self.base_register_name),
                        [Integer(self.circuit_ctx[-1].find_bit(qubit).index)],
                    )
                )
            ret.append(AliasStatement(Identifier(qreg.name), qubits))
        return ret

    def build_quantuminstructions(self, instructions):
        """Builds a list of call statements"""
        ret = []
        for instruction in instructions:
            if isinstance(instruction[0], Gate):
                if instruction[0].condition:
                    eqcondition = self.build_eqcondition(instruction[0].condition)
                    instruciton_without_condition = instruction[0].copy()
                    instruciton_without_condition.condition = None
                    programTrue = self.build_programblock(
                        [(instruciton_without_condition, instruction[1], instruction[2])]
                    )
                    ret.append(BranchingStatement(eqcondition, programTrue))
                else:
                    ret.append(self.build_quantumgatecall(instruction))
            elif isinstance(instruction[0], Barrier):
                indexIdentifierList = self.build_indexIdentifierlist(instruction[1])
                ret.append(QuantumBarrier(indexIdentifierList))
            elif isinstance(instruction[0], Measure):
                quantumMeasurement = QuantumMeasurement(
                    self.build_indexIdentifierlist(instruction[1])
                )
                indexIdentifierList = self.build_indexidentifier(instruction[2][0])
                ret.append(QuantumMeasurementAssignment(indexIdentifierList, quantumMeasurement))
            else:
                ret.append(self.build_subroutinecall(instruction))
        return ret

    def build_programblock(self, instructions):
        """Builds a ProgramBlock"""
        return ProgramBlock(self.build_quantuminstructions(instructions))

    def build_eqcondition(self, condition):
        """Classical Conditional condition from a instruction.condition"""
        if isinstance(condition[0], Clbit):
            condition_on = self.build_indexidentifier(condition[0])
        else:
            condition_on = Identifier(condition[0].name)
        return ComparisonExpression(condition_on, EqualsOperator(), Integer(int(condition[1])))

    def build_quantumArgumentList(self, qregs: [QuantumRegister], circuit=None):
        """Builds a quantumArgumentList"""
        quantumArgumentList = []
        for qreg in qregs:
            if self._flat_reg:
                for qubit in qreg:
                    if circuit is None:
                        raise Exception
                    reg, idx = self.find_bit(qubit)
                    qubit_name = f"{reg.name}_{idx}"
                    quantumArgumentList.append(QuantumArgument(Identifier(qubit_name)))
            else:
                quantumArgumentList.append(
                    QuantumArgument(Identifier(qreg.name), Designator(Integer(qreg.size)))
                )
        return quantumArgumentList

    def build_indexIdentifierlist(self, bitlist: [Bit]):
        """Builds a indexIdentifierList"""
        indexIdentifierList = []
        for bit in bitlist:
            indexIdentifierList.append(self.build_indexidentifier(bit))
        return indexIdentifierList

    def build_quantumgatecall(self, instruction):
        """Builds a QuantumGateCall"""
        if isinstance(instruction[0], UGate):
            quantumGateName = Identifier("U")
        else:
            quantumGateName = Identifier(self.global_namespace[instruction[0]])
        indexIdentifierList = self.build_indexIdentifierlist(instruction[1])
        if self.disable_constants:
            parameters = [Expression(param) for param in instruction[0].params]
        else:
            parameters = [
                Expression(pi_check(param, output="qasm")) for param in instruction[0].params
            ]

        return QuantumGateCall(quantumGateName, indexIdentifierList, parameters=parameters)

    def build_subroutinecall(self, instruction):
        """Builds a SubroutineCall"""
        identifier = Identifier(self.global_namespace[instruction[0]])
        expressionList = [Expression(param) for param in instruction[0].params]
        indexIdentifierList = self.build_indexIdentifierlist(instruction[1])

        return SubroutineCall(identifier, indexIdentifierList, expressionList)

    def build_indexidentifier(self, bit: Bit):
        """Build a IndexIdentifier2"""
        reg, idx = self.find_bit(bit)
        if self._physical_qubit and isinstance(bit, Qubit):
            return PhysicalQubitIdentifier(
                Identifier(str(self.circuit_ctx[-1].find_bit(bit).index))
            )
        if self._flat_reg:
            bit_name = f"{reg.name}_{idx}"
            return IndexIdentifier2(Identifier(bit_name))
        else:
            return IndexIdentifier2(Identifier(reg.name), [Integer(idx)])

    def find_bit(self, bit):
        """Returns the register and the index in that register for a particular bit"""
        bit_location = self.circuit_ctx[-1].find_bit(bit)
        return bit_location.registers[0]
