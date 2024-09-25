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

from __future__ import annotations

import collections
import contextlib
import dataclasses
import io
import itertools
import math
import numbers
import re
from typing import Iterable, List, Sequence, Union

from qiskit._accelerate.circuit import StandardGate
from qiskit.circuit import (
    library,
    Barrier,
    CircuitInstruction,
    Clbit,
    ControlledGate,
    Gate,
    Measure,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
    Qubit,
    Reset,
    Delay,
    Store,
)
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
    IfElseOp,
    ForLoopOp,
    WhileLoopOp,
    SwitchCaseOp,
    ControlFlowOp,
    BreakLoopOp,
    ContinueLoopOp,
    CASE_DEFAULT,
)
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check

from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter


# Reserved keywords that gates and variables cannot be named.  It is possible that some of these
# _could_ be accepted as variable names by OpenQASM 3 parsers, but it's safer for us to just be very
# conservative.
_RESERVED_KEYWORDS = frozenset(
    {
        "OPENQASM",
        "angle",
        "array",
        "barrier",
        "bit",
        "bool",
        "box",
        "break",
        "cal",
        "complex",
        "const",
        "continue",
        "creg",
        "ctrl",
        "def",
        "defcal",
        "defcalgrammar",
        "delay",
        "duration",
        "durationof",
        "else",
        "end",
        "extern",
        "float",
        "for",
        "gate",
        "gphase",
        "if",
        "in",
        "include",
        "input",
        "int",
        "inv",
        "let",
        "measure",
        "mutable",
        "negctrl",
        "output",
        "pow",
        "qreg",
        "qubit",
        "reset",
        "return",
        "sizeof",
        "stretch",
        "uint",
        "while",
    }
)

# This probably isn't precisely the same as the OQ3 spec, but we'd need an extra dependency to fully
# handle all Unicode character classes, and this should be close enough for users who aren't
# actively _trying_ to break us (fingers crossed).
_VALID_DECLARABLE_IDENTIFIER = re.compile(r"([\w][\w\d]*)", flags=re.U)
_VALID_HARDWARE_QUBIT = re.compile(r"\$[\d]+", flags=re.U)
_BAD_IDENTIFIER_CHARACTERS = re.compile(r"[^\w\d]", flags=re.U)


class Exporter:
    """QASM3 exporter main class."""

    def __init__(
        self,
        includes: Sequence[str] = ("stdgates.inc",),
        basis_gates: Sequence[str] = ("U",),
        disable_constants: bool = False,
        alias_classical_registers: bool = None,
        allow_aliasing: bool = None,
        indent: str = "  ",
        experimental: ExperimentalFeatures = ExperimentalFeatures(0),
    ):
        """
        Args:
            includes: the filenames that should be emitted as includes.

                .. note::

                    At present, only the standard-library file ``stdgates.inc`` is properly
                    understood by the exporter, in the sense that it knows the gates it defines.
                    You can specify other includes, but you will need to pass the names of the gates
                    they define in the ``basis_gates`` argument to avoid the exporter outputting a
                    separate ``gate`` definition.

            basis_gates: the basic defined gate set of the backend.
            disable_constants: if ``True``, always emit floating-point constants for numeric
                parameter values.  If ``False`` (the default), then values close to multiples of
                OpenQASM 3 constants (``pi``, ``euler``, and ``tau``) will be emitted in terms of those
                constants instead, potentially improving accuracy in the output.
            alias_classical_registers: If ``True``, then bits may be contained in more than one
                register.  If so, the registers will be emitted using "alias" definitions, which
                might not be well supported by consumers of OpenQASM 3.

                .. seealso::
                    Parameter ``allow_aliasing``
                        A value for ``allow_aliasing`` overrides any value given here, and
                        supersedes this parameter.
            allow_aliasing: If ``True``, then bits may be contained in more than one register.  If
                so, the registers will be emitted using "alias" definitions, which might not be
                well supported by consumers of OpenQASM 3.  Defaults to ``False`` or the value of
                ``alias_classical_registers``.

                .. versionadded:: 0.25.0
            indent: the indentation string to use for each level within an indented block.  Can be
                set to the empty string to disable indentation.
            experimental: any experimental features to enable during the export.  See
                :class:`ExperimentalFeatures` for more details.
        """
        self.basis_gates = basis_gates
        self.disable_constants = disable_constants
        self.allow_aliasing = (
            allow_aliasing if allow_aliasing is not None else (alias_classical_registers or False)
        )
        self.includes = list(includes)
        self.indent = indent
        self.experimental = experimental

    def dumps(self, circuit):
        """Convert the circuit to OpenQASM 3, returning the result as a string."""
        with io.StringIO() as stream:
            self.dump(circuit, stream)
            return stream.getvalue()

    def dump(self, circuit, stream):
        """Convert the circuit to OpenQASM 3, dumping the result to a file or text stream."""
        builder = QASM3Builder(
            circuit,
            includeslist=self.includes,
            basis_gates=self.basis_gates,
            disable_constants=self.disable_constants,
            allow_aliasing=self.allow_aliasing,
            experimental=self.experimental,
        )
        BasicPrinter(stream, indent=self.indent, experimental=self.experimental).visit(
            builder.build_program()
        )


# Just needs to have enough parameters to support the largest standard (non-controlled) gate in our
# standard library.  We have to use the same `Parameter` instances each time so the equality
# comparisons will work.
_FIXED_PARAMETERS = (Parameter("p0"), Parameter("p1"), Parameter("p2"), Parameter("p3"))

# Mapping of symbols defined by `stdgates.inc` to their gate definition source.
_KNOWN_INCLUDES = {
    "stdgates.inc": {
        "p": library.PhaseGate(*_FIXED_PARAMETERS[:1]),
        "x": library.XGate(),
        "y": library.YGate(),
        "z": library.ZGate(),
        "h": library.HGate(),
        "s": library.SGate(),
        "sdg": library.SdgGate(),
        "t": library.TGate(),
        "tdg": library.TdgGate(),
        "sx": library.SXGate(),
        "rx": library.RXGate(*_FIXED_PARAMETERS[:1]),
        "ry": library.RYGate(*_FIXED_PARAMETERS[:1]),
        "rz": library.RZGate(*_FIXED_PARAMETERS[:1]),
        "cx": library.CXGate(),
        "cy": library.CYGate(),
        "cz": library.CZGate(),
        "cp": library.CPhaseGate(*_FIXED_PARAMETERS[:1]),
        "crx": library.CRXGate(*_FIXED_PARAMETERS[:1]),
        "cry": library.CRYGate(*_FIXED_PARAMETERS[:1]),
        "crz": library.CRZGate(*_FIXED_PARAMETERS[:1]),
        "ch": library.CHGate(),
        "swap": library.SwapGate(),
        "ccx": library.CCXGate(),
        "cswap": library.CSwapGate(),
        "cu": library.CUGate(*_FIXED_PARAMETERS[:4]),
        "CX": library.CXGate(),
        "phase": library.PhaseGate(*_FIXED_PARAMETERS[:1]),
        "cphase": library.CPhaseGate(*_FIXED_PARAMETERS[:1]),
        "id": library.IGate(),
        "u1": library.U1Gate(*_FIXED_PARAMETERS[:1]),
        "u2": library.U2Gate(*_FIXED_PARAMETERS[:2]),
        "u3": library.U3Gate(*_FIXED_PARAMETERS[:3]),
    },
}

_BUILTIN_GATES = {
    "U": library.UGate(*_FIXED_PARAMETERS[:3]),
}


@dataclasses.dataclass
class GateInfo:
    """Symbol-table information on a gate."""

    canonical: Gate | None
    """The canonical object for the gate.  This is a Qiskit object that is not necessarily equal to
    any usage, but is the canonical form in terms of its parameter usage, such as a standard-library
    gate being defined in terms of the `_FIXED_PARAMETERS` objects.  A call-site gate whose
    canonical form equals this can use the corresponding symbol as the callee.

    This can be ``None`` if the gate was an overridden "basis gate" for this export, so no canonical
    form is known."""
    node: ast.QuantumGateDefinition | None
    """An AST node containing the gate definition.  This can be ``None`` if the gate came from an
    included file, or is an overridden "basis gate" of the export."""


class SymbolTable:
    """Track Qiskit objects and the OQ3 identifiers used to refer to them."""

    def __init__(self):
        self.gates: collections.OrderedDict[str, GateInfo | None] = {}
        """Mapping of the symbol name to the "definition source" of the gate, which provides its
        signature and decomposition.  The definition source can be `None` if the user set the gate
        as a custom "basis gate".

        Gates can only be declared in the global scope, so there is just a single look-up for this.

        This is insertion ordered, and that can be relied on for iteration later."""
        self.standard_gate_idents: dict[StandardGate, ast.Identifier] = {}
        """Mapping of standard gate enumeration values to the identifier we represent that as."""
        self.user_gate_idents: dict[int, ast.Identifier] = {}
        """Mapping of `id`s of user gates to the identifier we use for it."""

        self.variables: list[dict[str, object]] = [{}]
        """Stack of mappings of variable names to the Qiskit object that represents them.

        The zeroth index corresponds to the global scope, the highest index to the current scope."""
        self.objects: list[dict[object, ast.Identifier]] = [{}]
        """Stack of mappings of Qiskit objects to the identifier (or subscripted identifier) that
        refers to them.  This is similar to the inverse mapping of ``variables``.

        The zeroth index corresponds to the global scope, the highest index to the current scope."""

        # Quick-and-dirty method of getting unique salts for names.
        self._counter = itertools.count()

    def push_scope(self):
        """Enter a new variable scope."""
        self.variables.append({})
        self.objects.append({})

    def pop_scope(self):
        """Exit the current scope, returning to a previous scope."""
        self.objects.pop()
        self.variables.pop()

    def new_context(self) -> SymbolTable:
        """Create a new context, such as for a gate definition.

        Contexts share the same set of globally defined gates, but have no access to other variables
        defined in any scope."""
        out = SymbolTable()
        out.gates = self.gates
        out.standard_gate_idents = self.standard_gate_idents
        out.user_gate_idents = self.user_gate_idents
        return out

    def symbol_defined(self, name: str) -> bool:
        """Whether this identifier has a defined meaning already."""
        return (
            name in _RESERVED_KEYWORDS
            or name in self.gates
            or name in itertools.chain.from_iterable(reversed(self.variables))
        )

    def can_shadow_symbol(self, name: str) -> bool:
        """Whether a new definition of this symbol can be made within the OpenQASM 3 shadowing
        rules."""
        return (
            name not in self.variables[-1]
            and name not in self.gates
            and name not in _RESERVED_KEYWORDS
        )

    def escaped_declarable_name(self, name: str, *, allow_rename: bool, unique: bool = False):
        """Get an identifier based on ``name`` that can be safely shadowed within this scope.

        If ``unique`` is ``True``, then the name is required to be unique across all live scopes,
        not just able to be redefined."""
        name_allowed = (
            (lambda name: not self.symbol_defined(name)) if unique else self.can_shadow_symbol
        )
        valid_identifier = _VALID_DECLARABLE_IDENTIFIER
        if allow_rename:
            if not valid_identifier.fullmatch(name):
                name = "_" + _BAD_IDENTIFIER_CHARACTERS.sub("_", name)
            base = name
            while not name_allowed(name):
                name = f"{base}_{next(self._counter)}"
            return name
        if not valid_identifier.fullmatch(name):
            raise QASM3ExporterError(f"cannot use '{name}' as a name; it is not a valid identifier")
        if name in _RESERVED_KEYWORDS:
            raise QASM3ExporterError(f"cannot use the keyword '{name}' as a variable name")
        if not name_allowed(name):
            if self.gates.get(name) is not None:
                raise QASM3ExporterError(
                    f"cannot shadow variable '{name}', as it is already defined as a gate"
                )
            for scope in reversed(self.variables):
                if (other := scope.get(name)) is not None:
                    break
            else:  # pragma: no cover
                raise RuntimeError(f"internal error: could not locate unshadowable '{name}'")
            raise QASM3ExporterError(
                f"cannot shadow variable '{name}', as it is already defined as '{other}'"
            )
        return name

    def register_variable(
        self,
        name: str,
        variable: object,
        *,
        allow_rename: bool,
        force_global: bool = False,
        allow_hardware_qubit: bool = False,
    ) -> ast.Identifier:
        """Register a variable in the symbol table for the given scope, returning the name that
        should be used to refer to the variable.  The same name will be returned by subsequent calls
        to :meth:`get_variable` within the same scope.

        Args:
            name: the name to base the identifier on.
            variable: the Qiskit object this refers to.  This can be ``None`` in the case of
                reserving a dummy variable name that does not actually have a Qiskit object backing
                it.
            allow_rename: whether to allow the name to be mutated to escape it and/or make it safe
                to define (avoiding keywords, subject to shadowing rules, etc).
            force_global: force this declaration to be in the global scope.
            allow_hardware_qubit: whether to allow hardware qubits to pass through as identifiers.
                Hardware qubits are a dollar sign followed by a non-negative integer, and cannot be
                declared, so are not suitable identifiers for most objects.
        """
        scope_index = 0 if force_global else -1
        # We still need to do this escaping and shadow checking if `force_global`, because we don't
        # want a previous variable declared in the currently active scope to shadow the global.
        # This logic would be cleaner if we made the naming choices later, after AST generation
        # (e.g. by using only indices as the identifiers until we're outputting the program).
        if allow_hardware_qubit and _VALID_HARDWARE_QUBIT.fullmatch(name):
            if self.symbol_defined(name):  # pragma: no cover
                raise QASM3ExporterError(f"internal error: cannot redeclare hardware qubit {name}")
        else:
            name = self.escaped_declarable_name(
                name, allow_rename=allow_rename, unique=force_global
            )
        identifier = ast.Identifier(name)
        self.variables[scope_index][name] = variable
        if variable is not None:
            self.objects[scope_index][variable] = identifier
        return identifier

    def set_object_ident(self, ident: ast.Identifier, variable: object):
        """Set the identifier used to refer to a given object for this scope.

        This overwrites any previously set identifier, such as during the original registration.

        This is generally only useful for tracking "sub" objects, like bits out of a register, which
        will have an `SubscriptedIdentifier` as their identifier."""
        self.objects[-1][variable] = ident

    def get_variable(self, variable: object) -> ast.Identifier:
        """Lookup a non-gate variable in the symbol table."""
        for scope in reversed(self.objects):
            if (out := scope.get(variable)) is not None:
                return out
        raise KeyError(f"'{variable}' is not defined in the current context")

    def register_gate_without_definition(self, name: str, gate: Gate | None) -> ast.Identifier:
        """Register a gate that does not require an OQ3 definition.

        If the ``gate`` is given, it will be used to validate that a call to it is compatible (such
        as a known gate from an included file).  If it is not given, it is treated as a user-defined
        "basis gate" that assumes that all calling signatures are valid and that all gates of this
        name are exactly compatible, which is somewhat dangerous."""
        # Validate the name is usable.
        name = self.escaped_declarable_name(name, allow_rename=False, unique=False)
        ident = ast.Identifier(name)
        if gate is None:
            self.gates[name] = GateInfo(None, None)
        else:
            canonical = _gate_canonical_form(gate)
            self.gates[name] = GateInfo(canonical, None)
            if canonical._standard_gate is not None:
                self.standard_gate_idents[canonical._standard_gate] = ident
            else:
                self.user_gate_idents[id(canonical)] = ident
        return ident

    def register_gate(
        self,
        name: str,
        source: Gate,
        params: Iterable[ast.Identifier],
        qubits: Iterable[ast.Identifier],
        body: ast.QuantumBlock,
    ) -> ast.Identifier:
        """Register the given gate in the symbol table, using the given components to build up the
        full AST definition."""
        name = self.escaped_declarable_name(name, allow_rename=True, unique=False)
        ident = ast.Identifier(name)
        self.gates[name] = GateInfo(
            source, ast.QuantumGateDefinition(ident, tuple(params), tuple(qubits), body)
        )
        # Add the gate object with a magic lookup keep to the objects dictionary so we can retrieve
        # it later.  Standard gates are not guaranteed to have stable IDs (they're preferentially
        # not even created in Python space), but user gates are.
        if source._standard_gate is not None:
            self.standard_gate_idents[source._standard_gate] = ident
        else:
            self.user_gate_idents[id(source)] = ident
        return ident

    def get_gate(self, gate: Gate) -> ast.Identifier | None:
        """Lookup the identifier for a given `Gate`, if it exists."""
        canonical = _gate_canonical_form(gate)
        # `our_defn.canonical is None` means a basis gate that we should assume is always valid.
        if (our_defn := self.gates.get(gate.name)) is not None and (
            our_defn.canonical is None or our_defn.canonical == canonical
        ):
            return ast.Identifier(gate.name)
        if canonical._standard_gate is not None:
            if (our_ident := self.standard_gate_idents.get(canonical._standard_gate)) is None:
                return None
            return our_ident if self.gates[our_ident.string].canonical == canonical else None
        # No need to check equality if we're looking up by `id`; we must have the same object.
        return self.user_gate_idents.get(id(canonical))


def _gate_canonical_form(gate: Gate) -> Gate:
    """Get the canonical form of a gate.

    This is the gate object that should be used to provide the OpenQASM 3 definition of a gate (but
    not the call site; that's the input object).  This lets us return a re-parametrised gate in
    terms of general parameters, in cases where we can be sure that that is valid.  This is
    currently only Qiskit standard gates.  This lets multiple call-site gates match the same symbol,
    in the case of parametric gates.

    The definition source provides the number of qubits, the parameter signature and the body of the
    `gate` statement.  It does not provide the name of the symbol being defined."""
    # If a gate is part of the Qiskit standard-library gates, we know we can safely produce a
    # reparameterised gate by passing the parameters positionally to the standard-gate constructor
    # (and control state, if appropriate).
    if gate._standard_gate and not isinstance(gate, ControlledGate):
        return gate.base_class(*_FIXED_PARAMETERS[: len(gate.params)])
    elif gate._standard_gate:
        return gate.base_class(*_FIXED_PARAMETERS[: len(gate.params)], ctrl_state=gate.ctrl_state)
    return gate


@dataclasses.dataclass
class BuildScope:
    """The structure used in the builder to store the contexts and re-mappings of bits from the
    top-level scope where the bits were actually defined."""

    circuit: QuantumCircuit
    """The circuit block that we're currently working on exporting."""
    bit_map: dict[Bit, Bit]
    """Mapping of bit objects in ``circuit`` to the bit objects in the global-scope program
    :class:`.QuantumCircuit` that they are bound to."""


class QASM3Builder:
    """QASM3 builder constructs an AST from a QuantumCircuit."""

    builtins = (Barrier, Measure, Reset, Delay, BreakLoopOp, ContinueLoopOp, Store)
    loose_bit_prefix = "_bit"
    loose_qubit_prefix = "_qubit"
    gate_parameter_prefix = "_gate_p"
    gate_qubit_prefix = "_gate_q"

    def __init__(
        self,
        quantumcircuit,
        includeslist,
        basis_gates,
        disable_constants,
        allow_aliasing,
        experimental=ExperimentalFeatures(0),
    ):
        self.scope = BuildScope(
            quantumcircuit,
            {x: x for x in itertools.chain(quantumcircuit.qubits, quantumcircuit.clbits)},
        )
        self.symbols = SymbolTable()
        # `_global_io_declarations` and `_global_classical_declarations` are stateful, and any
        # operation that needs a parameter can append to them during the build.  We make all
        # classical declarations global because the IBM qe-compiler stack (our initial consumer of
        # OQ3 strings) prefers declarations to all be global, and it's valid OQ3, so it's not vendor
        # lock-in.  It's possibly slightly memory inefficient, but that's not likely to be a problem
        # in the near term.
        self._global_io_declarations = []
        self._global_classical_forward_declarations = []
        self.disable_constants = disable_constants
        self.allow_aliasing = allow_aliasing
        self.includes = includeslist
        self.basis_gates = basis_gates
        self.experimental = experimental

    @contextlib.contextmanager
    def new_scope(self, circuit: QuantumCircuit, qubits: Iterable[Qubit], clbits: Iterable[Clbit]):
        """Context manager that pushes a new scope (like a ``for`` or ``while`` loop body) onto the
        current context stack."""
        current_map = self.scope.bit_map
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
        self.symbols.push_scope()
        old_scope, self.scope = self.scope, BuildScope(circuit, mapping)
        yield self.scope
        self.scope = old_scope
        self.symbols.pop_scope()

    @contextlib.contextmanager
    def new_context(self, body: QuantumCircuit):
        """Push a new context (like for a ``gate`` or ``def`` body) onto the stack."""
        mapping = {bit: bit for bit in itertools.chain(body.qubits, body.clbits)}

        old_symbols, self.symbols = self.symbols, self.symbols.new_context()
        old_scope, self.scope = self.scope, BuildScope(body, mapping)
        yield self.scope
        self.scope = old_scope
        self.symbols = old_symbols

    def _lookup_variable(self, variable) -> ast.Identifier:
        """Lookup a Qiskit object within the current context, and return the name that should be
        used to represent it in OpenQASM 3 programmes."""
        if isinstance(variable, Bit):
            variable = self.scope.bit_map[variable]
        return self.symbols.get_variable(variable)

    def build_program(self):
        """Builds a Program"""
        circuit = self.scope.circuit
        if circuit.num_captured_vars:
            raise QASM3ExporterError(
                "cannot export an inner scope with captured variables as a top-level program"
            )

        # The order we build parts of the AST has an effect on which names will get escaped to avoid
        # collisions.  The current ideas are:
        #
        # * standard-library include files _must_ define symbols of the correct name.
        # * classical registers, IO variables and `Var` nodes are likely to be referred to by name
        #   by a user, so they get very high priority - we search for them before doing anything.
        # * qubit registers are not typically referred to by name by users, so they get a lower
        #   priority than the classical variables.
        # * we often have to escape user-defined gate names anyway because of our dodgy parameter
        #   handling, so they get the lowest priority; they get defined as they are encountered.
        #
        # An alternative approach would be to defer naming decisions until we are outputting the
        # AST, and using some UUID for each symbol we're going to define in the interrim.  This
        # would require relatively large changes to the symbol-table and AST handling, however.

        for builtin, gate in _BUILTIN_GATES.items():
            self.symbols.register_gate_without_definition(builtin, gate)
        for builtin in self.basis_gates:
            if builtin in _BUILTIN_GATES:
                # It's built into the langauge; we don't need to re-add it.
                continue
            try:
                self.symbols.register_gate_without_definition(builtin, None)
            except QASM3ExporterError as exc:
                raise QASM3ExporterError(
                    f"Cannot use '{builtin}' as a basis gate for the reason in the prior exception."
                    " Consider renaming the gate if needed, or omitting this basis gate if not."
                ) from exc

        header = ast.Header(ast.Version("3.0"), list(self.build_includes()))

        # Early IBM runtime parametrization uses unbound `Parameter` instances as `input` variables,
        # not the explicit realtime `Var` variables, so we need this explicit scan.
        self.hoist_global_parameter_declarations()
        # Qiskit's clbits and classical registers need to get mapped to implicit OQ3 variables, but
        # only if they're in the top-level circuit.  The QuantumCircuit data model is that inner
        # clbits are bound to outer bits, and inner registers must be closing over outer ones.
        self.hoist_classical_register_declarations()
        # We hoist registers before new-style vars because registers are an older part of the data
        # model (and used implicitly in PrimitivesV2 outputs) so they get the first go at reserving
        # names in the symbol table.
        self.hoist_classical_io_var_declarations()

        # Similarly, QuantumCircuit qubits/registers are only new variables in the global scope.
        quantum_declarations = self.build_quantum_declarations()

        # This call has side-effects - it can populate `self._global_io_declarations` and
        # `self._global_classical_declarations` as a courtesy to the qe-compiler that prefers our
        # hacky temporary `switch` target variables to be globally defined.  It also populates the
        # symbol table with encountered gates that weren't previously defined.
        main_statements = self.build_current_scope()

        statements = [
            statement
            for source in (
                # In older versions of the reference OQ3 grammar, IO declarations had to come before
                # anything else, so we keep doing that as a courtesy.
                self._global_io_declarations,
                (gate.node for gate in self.symbols.gates.values() if gate.node is not None),
                self._global_classical_forward_declarations,
                quantum_declarations,
                main_statements,
            )
            for statement in source
        ]
        return ast.Program(header, statements)

    def build_includes(self):
        """Builds a list of included files."""
        for filename in self.includes:
            # Note: unknown include files have a corresponding `include` statement generated, but do
            # not actually define any gates; we rely on the user to pass those in `basis_gates`.
            for name, gate in _KNOWN_INCLUDES.get(filename, {}).items():
                self.symbols.register_gate_without_definition(name, gate)
            yield ast.Include(filename)

    def define_gate(self, gate: Gate) -> ast.Identifier:
        """Define a gate in the symbol table, including building the gate-definition statement for
        it.

        This recurses through gate-definition statements."""
        if issubclass(gate.base_class, library.CXGate) and gate.ctrl_state == 1:
            # CX gets super duper special treatment because it's the base of Qiskit's definition
            # tree, but isn't an OQ3 built-in (it was in OQ2).  We use `issubclass` because we
            # haven't fully fixed what the name/class distinction is (there's a test from the
            # original OQ3 exporter that tries a naming collision with 'cx').
            control, target = ast.Identifier("c"), ast.Identifier("t")
            body = ast.QuantumBlock(
                [
                    ast.QuantumGateCall(
                        self.symbols.get_gate(library.UGate(math.pi, 0, math.pi)),
                        [control, target],
                        parameters=[ast.Constant.PI, ast.IntegerLiteral(0), ast.Constant.PI],
                        modifiers=[ast.QuantumGateModifier(ast.QuantumGateModifierName.CTRL)],
                    )
                ]
            )
            return self.symbols.register_gate(gate.name, gate, (), (control, target), body)
        if gate.definition is None:
            raise QASM3ExporterError(f"failed to export gate '{gate.name}' that has no definition")
        canonical = _gate_canonical_form(gate)
        with self.new_context(canonical.definition):
            defn = self.scope.circuit
            # If `defn.num_parameters == 0` but `gate.params` is non-empty, we are likely in the
            # case where the gate's circuit definition is fully bound (so we can't detect its inputs
            # anymore).  This is a problem in our data model - for arbitrary user gates, there's no
            # way we can reliably get a parametric version of the gate through our interfaces.  In
            # this case, we output a gate that has dummy parameters, and rely on it being a
            # different `id` each time to avoid duplication.  We assume that the parametrisation
            # order matches (which is a _big_ assumption).
            #
            # If `defn.num_parameters > 0`, we enforce that it must match how it's called.
            if defn.num_parameters > 0:
                if defn.num_parameters != len(gate.params):
                    raise QASM3ExporterError(
                        "parameter mismatch in definition of '{gate}':"
                        f" call has {len(gate.params)}, definition has {defn.num_parameters}"
                    )
                params = [
                    self.symbols.register_variable(param.name, param, allow_rename=True)
                    for param in defn.parameters
                ]
            else:
                # Fill with dummy parameters. The name is unimportant, because they're not actually
                # used in the definition.
                params = [
                    self.symbols.register_variable(
                        f"{self.gate_parameter_prefix}_{i}", None, allow_rename=True
                    )
                    for i in range(len(gate.params))
                ]
            qubits = [
                self.symbols.register_variable(
                    f"{self.gate_qubit_prefix}_{i}", qubit, allow_rename=True
                )
                for i, qubit in enumerate(defn.qubits)
            ]
            body = ast.QuantumBlock(self.build_current_scope())
        # We register the gate only after building its body so that any gates we needed for that in
        # turn are registered in the correct order.  Gates can't be recursive in OQ3, so there's no
        # problem with delaying this.
        return self.symbols.register_gate(canonical.name, canonical, params, qubits, body)

    def assert_global_scope(self):
        """Raise an error if we are not in the global scope, as a defensive measure."""
        if len(self.symbols.variables) > 1:  # pragma: no cover
            raise RuntimeError("not currently in the global scope")

    def hoist_global_parameter_declarations(self):
        """Extend ``self._global_io_declarations`` and ``self._global_classical_declarations`` with
        any implicit declarations used to support the early IBM efforts to use :class:`.Parameter`
        as an input variable."""
        self.assert_global_scope()
        circuit = self.scope.circuit
        for parameter in circuit.parameters:
            parameter_name = self.symbols.register_variable(
                parameter.name, parameter, allow_rename=True
            )
            declaration = _infer_variable_declaration(circuit, parameter, parameter_name)
            if declaration is None:
                continue
            if isinstance(declaration, ast.IODeclaration):
                self._global_io_declarations.append(declaration)
            else:
                self._global_classical_forward_declarations.append(declaration)

    def hoist_classical_register_declarations(self):
        """Extend the global classical declarations with AST nodes declaring all the global-scope
        circuit :class:`.Clbit` and :class:`.ClassicalRegister` instances.  Qiskit's data model
        doesn't involve the declaration of *new* bits or registers in inner scopes; only the
        :class:`.expr.Var` mechanism allows that.

        The behavior of this function depends on the setting ``allow_aliasing``. If this
        is ``True``, then the output will be in the same form as the output of
        :meth:`.build_classical_declarations`, with the registers being aliases.  If ``False``, it
        will instead return a :obj:`.ast.ClassicalDeclaration` for each classical register, and one
        for the loose :obj:`.Clbit` instances, and will raise :obj:`QASM3ExporterError` if any
        registers overlap.
        """
        self.assert_global_scope()
        circuit = self.scope.circuit
        if any(len(circuit.find_bit(q).registers) > 1 for q in circuit.clbits):
            # There are overlapping registers, so we need to use aliases to emit the structure.
            if not self.allow_aliasing:
                raise QASM3ExporterError(
                    "Some classical registers in this circuit overlap and need aliases to express,"
                    " but 'allow_aliasing' is false."
                )
            clbits = (
                ast.ClassicalDeclaration(
                    ast.BitType(),
                    self.symbols.register_variable(
                        f"{self.loose_bit_prefix}{i}", clbit, allow_rename=True
                    ),
                )
                for i, clbit in enumerate(circuit.clbits)
            )
            self._global_classical_forward_declarations.extend(clbits)
            self._global_classical_forward_declarations.extend(self.build_aliases(circuit.cregs))
            return
        # If we're here, we're in the clbit happy path where there are no clbits that are in more
        # than one register.  We can output things very naturally.
        self._global_classical_forward_declarations.extend(
            ast.ClassicalDeclaration(
                ast.BitType(),
                self.symbols.register_variable(
                    f"{self.loose_bit_prefix}{i}", clbit, allow_rename=True
                ),
            )
            for i, clbit in enumerate(circuit.clbits)
            if not circuit.find_bit(clbit).registers
        )
        for register in circuit.cregs:
            name = self.symbols.register_variable(register.name, register, allow_rename=True)
            for i, bit in enumerate(register):
                self.symbols.set_object_ident(
                    ast.SubscriptedIdentifier(name.string, ast.IntegerLiteral(i)), bit
                )
            self._global_classical_forward_declarations.append(
                ast.ClassicalDeclaration(ast.BitArrayType(len(register)), name)
            )

    def hoist_classical_io_var_declarations(self):
        """Hoist the declarations of classical IO :class:`.expr.Var` nodes into the global state.

        Local :class:`.expr.Var` declarations are handled by the regular local-block scope builder,
        and the :class:`.QuantumCircuit` data model ensures that the only time an IO variable can
        occur is in an outermost block."""
        self.assert_global_scope()
        circuit = self.scope.circuit
        for var in circuit.iter_input_vars():
            self._global_io_declarations.append(
                ast.IODeclaration(
                    ast.IOModifier.INPUT,
                    _build_ast_type(var.type),
                    self.symbols.register_variable(var.name, var, allow_rename=True),
                )
            )

    def build_quantum_declarations(self):
        """Return a list of AST nodes declaring all the qubits in the current scope, and all the
        alias declarations for these qubits."""
        self.assert_global_scope()
        circuit = self.scope.circuit
        if circuit.layout is not None:
            # We're referring to physical qubits.  These can't be declared in OQ3, but we need to
            # track the bit -> expression mapping in our symbol table.
            for i, bit in enumerate(circuit.qubits):
                self.symbols.register_variable(
                    f"${i}", bit, allow_rename=False, allow_hardware_qubit=True
                )
            return []
        if any(len(circuit.find_bit(q).registers) > 1 for q in circuit.qubits):
            # There are overlapping registers, so we need to use aliases to emit the structure.
            if not self.allow_aliasing:
                raise QASM3ExporterError(
                    "Some quantum registers in this circuit overlap and need aliases to express,"
                    " but 'allow_aliasing' is false."
                )
            qubits = [
                ast.QuantumDeclaration(
                    self.symbols.register_variable(
                        f"{self.loose_qubit_prefix}{i}", qubit, allow_rename=True
                    )
                )
                for i, qubit in enumerate(circuit.qubits)
            ]
            return qubits + self.build_aliases(circuit.qregs)
        # If we're here, we're in the virtual-qubit happy path where there are no qubits that are in
        # more than one register.  We can output things very naturally.
        loose_qubits = [
            ast.QuantumDeclaration(
                self.symbols.register_variable(
                    f"{self.loose_qubit_prefix}{i}", qubit, allow_rename=True
                )
            )
            for i, qubit in enumerate(circuit.qubits)
            if not circuit.find_bit(qubit).registers
        ]
        registers = []
        for register in circuit.qregs:
            name = self.symbols.register_variable(register.name, register, allow_rename=True)
            for i, bit in enumerate(register):
                self.symbols.set_object_ident(
                    ast.SubscriptedIdentifier(name.string, ast.IntegerLiteral(i)), bit
                )
            registers.append(
                ast.QuantumDeclaration(name, ast.Designator(ast.IntegerLiteral(len(register))))
            )
        return loose_qubits + registers

    def build_aliases(self, registers: Iterable[Register]) -> List[ast.AliasStatement]:
        """Return a list of alias declarations for the given registers.  The registers can be either
        classical or quantum."""
        out = []
        for register in registers:
            name = self.symbols.register_variable(register.name, register, allow_rename=True)
            elements = [self._lookup_variable(bit) for bit in register]
            for i, bit in enumerate(register):
                # This might shadow previous definitions, but that's not a problem.
                self.symbols.set_object_ident(
                    ast.SubscriptedIdentifier(name.string, ast.IntegerLiteral(i)), bit
                )
            out.append(ast.AliasStatement(name, ast.IndexSet(elements)))
        return out

    def build_current_scope(self) -> List[ast.Statement]:
        """Build the instructions that occur in the current scope.

        In addition to everything literally in the circuit's ``data`` field, this also includes
        declarations for any local :class:`.expr.Var` nodes.
        """

        # We forward-declare all local variables uninitialised at the top of their scope. It would
        # be nice to declare the variable at the point of first store (so we can write things like
        # `uint[8] a = 12;`), but there's lots of edge-case logic to catch with that around
        # use-before-definition errors in the OQ3 output, for example if the user has side-stepped
        # the `QuantumCircuit` API protection to produce a circuit that uses an uninitialised
        # variable, or the initial write to a variable is within a control-flow scope.  (It would be
        # easier to see the def/use chain needed to do this cleanly if we were using `DAGCircuit`.)
        statements = [
            ast.ClassicalDeclaration(
                _build_ast_type(var.type),
                self.symbols.register_variable(var.name, var, allow_rename=True),
            )
            for var in self.scope.circuit.iter_declared_vars()
        ]
        for instruction in self.scope.circuit.data:
            if isinstance(instruction.operation, ControlFlowOp):
                if isinstance(instruction.operation, ForLoopOp):
                    statements.append(self.build_for_loop(instruction))
                elif isinstance(instruction.operation, WhileLoopOp):
                    statements.append(self.build_while_loop(instruction))
                elif isinstance(instruction.operation, IfElseOp):
                    statements.append(self.build_if_statement(instruction))
                elif isinstance(instruction.operation, SwitchCaseOp):
                    statements.extend(self.build_switch_statement(instruction))
                else:
                    raise RuntimeError(f"unhandled control-flow construct: {instruction.operation}")
                continue
            # Build the node, ignoring any condition.
            if isinstance(instruction.operation, Gate):
                nodes = [self.build_gate_call(instruction)]
            elif isinstance(instruction.operation, Barrier):
                operands = [self._lookup_variable(operand) for operand in instruction.qubits]
                nodes = [ast.QuantumBarrier(operands)]
            elif isinstance(instruction.operation, Measure):
                measurement = ast.QuantumMeasurement(
                    [self._lookup_variable(operand) for operand in instruction.qubits]
                )
                qubit = self._lookup_variable(instruction.clbits[0])
                nodes = [ast.QuantumMeasurementAssignment(qubit, measurement)]
            elif isinstance(instruction.operation, Reset):
                nodes = [
                    ast.QuantumReset(self._lookup_variable(operand))
                    for operand in instruction.qubits
                ]
            elif isinstance(instruction.operation, Delay):
                nodes = [self.build_delay(instruction)]
            elif isinstance(instruction.operation, Store):
                nodes = [
                    ast.AssignmentStatement(
                        self.build_expression(instruction.operation.lvalue),
                        self.build_expression(instruction.operation.rvalue),
                    )
                ]
            elif isinstance(instruction.operation, BreakLoopOp):
                nodes = [ast.BreakStatement()]
            elif isinstance(instruction.operation, ContinueLoopOp):
                nodes = [ast.ContinueStatement()]
            else:
                raise QASM3ExporterError(
                    "non-unitary subroutine calls are not yet supported,"
                    f" but received '{instruction.operation}'"
                )

            if instruction.operation.condition is None:
                statements.extend(nodes)
            else:
                body = ast.ProgramBlock(nodes)
                statements.append(
                    ast.BranchingStatement(
                        self.build_expression(_lift_condition(instruction.operation.condition)),
                        body,
                    )
                )
        return statements

    def build_if_statement(self, instruction: CircuitInstruction) -> ast.BranchingStatement:
        """Build an :obj:`.IfElseOp` into a :obj:`.ast.BranchingStatement`."""
        condition = self.build_expression(_lift_condition(instruction.operation.condition))

        true_circuit = instruction.operation.blocks[0]
        with self.new_scope(true_circuit, instruction.qubits, instruction.clbits):
            true_body = ast.ProgramBlock(self.build_current_scope())
        if len(instruction.operation.blocks) == 1:
            return ast.BranchingStatement(condition, true_body, None)

        false_circuit = instruction.operation.blocks[1]
        with self.new_scope(false_circuit, instruction.qubits, instruction.clbits):
            false_body = ast.ProgramBlock(self.build_current_scope())
        return ast.BranchingStatement(condition, true_body, false_body)

    def build_switch_statement(self, instruction: CircuitInstruction) -> Iterable[ast.Statement]:
        """Build a :obj:`.SwitchCaseOp` into a :class:`.ast.SwitchStatement`."""
        real_target = self.build_expression(expr.lift(instruction.operation.target))
        target = self.symbols.register_variable(
            "switch_dummy", None, allow_rename=True, force_global=True
        )
        self._global_classical_forward_declarations.append(
            ast.ClassicalDeclaration(ast.IntType(), target, None)
        )

        if ExperimentalFeatures.SWITCH_CASE_V1 in self.experimental:
            # In this case, defaults can be folded in with other cases (useless as that is).

            def case(values, case_block):
                values = [
                    ast.DefaultCase() if v is CASE_DEFAULT else self.build_integer(v)
                    for v in values
                ]
                with self.new_scope(case_block, instruction.qubits, instruction.clbits):
                    case_body = ast.ProgramBlock(self.build_current_scope())
                return values, case_body

            return [
                ast.AssignmentStatement(target, real_target),
                ast.SwitchStatementPreview(
                    target,
                    (
                        case(values, block)
                        for values, block in instruction.operation.cases_specifier()
                    ),
                ),
            ]

        # Handle the stabilized syntax.
        cases = []
        default = None
        for values, block in instruction.operation.cases_specifier():
            with self.new_scope(block, instruction.qubits, instruction.clbits):
                case_body = ast.ProgramBlock(self.build_current_scope())
            if CASE_DEFAULT in values:
                # Even if it's mixed in with other cases, we can skip them and only output the
                # `default` since that's valid and execution will be the same; the evaluation of
                # case labels can't have side effects.
                default = case_body
                continue
            cases.append(([self.build_integer(value) for value in values], case_body))

        return [
            ast.AssignmentStatement(target, real_target),
            ast.SwitchStatement(target, cases, default=default),
        ]

    def build_while_loop(self, instruction: CircuitInstruction) -> ast.WhileLoopStatement:
        """Build a :obj:`.WhileLoopOp` into a :obj:`.ast.WhileLoopStatement`."""
        condition = self.build_expression(_lift_condition(instruction.operation.condition))
        loop_circuit = instruction.operation.blocks[0]
        with self.new_scope(loop_circuit, instruction.qubits, instruction.clbits):
            loop_body = ast.ProgramBlock(self.build_current_scope())
        return ast.WhileLoopStatement(condition, loop_body)

    def build_for_loop(self, instruction: CircuitInstruction) -> ast.ForLoopStatement:
        """Build a :obj:`.ForLoopOp` into a :obj:`.ast.ForLoopStatement`."""
        indexset, loop_parameter, loop_circuit = instruction.operation.params
        with self.new_scope(loop_circuit, instruction.qubits, instruction.clbits):
            name = "_" if loop_parameter is None else loop_parameter.name
            loop_parameter_ast = self.symbols.register_variable(
                name, loop_parameter, allow_rename=True
            )
            if isinstance(indexset, range):
                # OpenQASM 3 uses inclusive ranges on both ends, unlike Python.
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
                        "The values in OpenQASM 3 'for' loops must all be integers, but received"
                        f" '{indexset}'."
                    ) from None
            body_ast = ast.ProgramBlock(self.build_current_scope())
        return ast.ForLoopStatement(indexset_ast, loop_parameter_ast, body_ast)

    def build_expression(self, node: expr.Expr) -> ast.Expression:
        """Build an expression."""
        return node.accept(_ExprBuilder(self._lookup_variable))

    def build_delay(self, instruction: CircuitInstruction) -> ast.QuantumDelay:
        """Build a built-in delay statement."""
        if instruction.clbits:
            raise QASM3ExporterError(
                f"Found a delay instruction acting on classical bits: {instruction}"
            )
        duration_value, unit = instruction.operation.duration, instruction.operation.unit
        if unit == "ps":
            duration = ast.DurationLiteral(1000 * duration_value, ast.DurationUnit.NANOSECOND)
        else:
            unit_map = {
                "ns": ast.DurationUnit.NANOSECOND,
                "us": ast.DurationUnit.MICROSECOND,
                "ms": ast.DurationUnit.MILLISECOND,
                "s": ast.DurationUnit.SECOND,
                "dt": ast.DurationUnit.SAMPLE,
            }
            duration = ast.DurationLiteral(duration_value, unit_map[unit])
        return ast.QuantumDelay(
            duration, [self._lookup_variable(qubit) for qubit in instruction.qubits]
        )

    def build_integer(self, value) -> ast.IntegerLiteral:
        """Build an integer literal, raising a :obj:`.QASM3ExporterError` if the input is not
        actually an
        integer."""
        if not isinstance(value, numbers.Integral):
            # This is meant to be purely defensive, in case a non-integer slips into the logic
            # somewhere, but no valid Terra object should trigger this.
            raise QASM3ExporterError(f"'{value}' is not an integer")  # pragma: no cover
        return ast.IntegerLiteral(int(value))

    def _rebind_scoped_parameters(self, expression):
        """If the input is a :class:`.ParameterExpression`, rebind any internal
        :class:`.Parameter`\\ s so that their names match their names in the scope.  Other inputs
        are returned unchanged."""
        # This is a little hacky, but the entirety of the Expression handling is essentially
        # missing, pending a new system in Terra to replace it (2022-03-07).
        if not isinstance(expression, ParameterExpression):
            return expression
        return expression.subs(
            {
                param: Parameter(self._lookup_variable(param).string)
                for param in expression.parameters
            }
        )

    def build_gate_call(self, instruction: CircuitInstruction):
        """Builds a gate-call AST node.

        This will also push the gate into the symbol table (if required), including recursively
        defining the gate blocks."""
        ident = self.symbols.get_gate(instruction.operation)
        if ident is None:
            ident = self.define_gate(instruction.operation)
        qubits = [self._lookup_variable(qubit) for qubit in instruction.qubits]
        if self.disable_constants:
            parameters = [
                ast.StringifyAndPray(self._rebind_scoped_parameters(param))
                for param in instruction.operation.params
            ]
        else:
            parameters = [
                ast.StringifyAndPray(pi_check(self._rebind_scoped_parameters(param), output="qasm"))
                for param in instruction.operation.params
            ]

        return ast.QuantumGateCall(ident, qubits, parameters=parameters)


def _infer_variable_declaration(
    circuit: QuantumCircuit, parameter: Parameter, parameter_name: ast.Identifier
) -> Union[ast.ClassicalDeclaration, None]:
    """Attempt to infer what type a parameter should be declared as to work with a circuit.

    This is very simplistic; it assumes all parameters are real numbers that need to be input to the
    program, unless one is used as a loop variable, in which case it shouldn't be declared at all,
    because the ``for`` loop declares it implicitly (per the Qiskit/qe-compiler reading of the
    OpenQASM spec at openqasm/openqasm@8ee55ec).

    .. note::

        This is a hack around not having a proper type system implemented in Terra, and really this
        whole function should be removed in favour of proper symbol-table building and lookups.
        This function is purely to try and hack the parameters for ``for`` loops into the exporter
        for now.

    Args:
        circuit: The global-scope circuit, which is the base of the exported program.
        parameter: The parameter to infer the type of.
        parameter_name: The name of the parameter to use in the declaration.

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
        for instr_index, index in circuit._data._raw_parameter_table_entry(parameter):
            if instr_index is None:
                continue
            instruction = circuit.data[instr_index].operation
            if isinstance(instruction, ForLoopOp):
                # The parameters of ForLoopOp are (indexset, loop_parameter, body).
                if index == 1:
                    return True
            if isinstance(instruction, ControlFlowOp):
                if is_loop_variable(instruction.params[index], parameter):
                    return True
        return False

    if is_loop_variable(circuit, parameter):
        return None
    # Arbitrary choice of double-precision float for all other parameters, but it's what we actually
    # expect people to be binding to their Parameters right now.
    return ast.IODeclaration(ast.IOModifier.INPUT, ast.FloatType.DOUBLE, parameter_name)


def _lift_condition(condition):
    if isinstance(condition, expr.Expr):
        return condition
    return expr.lift_legacy_condition(condition)


def _build_ast_type(type_: types.Type) -> ast.ClassicalType:
    if type_.kind is types.Bool:
        return ast.BoolType()
    if type_.kind is types.Uint:
        return ast.UintType(type_.width)
    raise RuntimeError(f"unhandled expr type '{type_}'")


class _ExprBuilder(expr.ExprVisitor[ast.Expression]):
    __slots__ = ("lookup",)

    # This is a very simple, non-contextual converter.  As the type system expands, we may well end
    # up with some places where Terra's abstract type system needs to be lowered to OQ3 rather than
    # mapping 100% directly, which might need a more contextual visitor.

    def __init__(self, lookup):
        self.lookup = lookup

    def visit_var(self, node, /):
        return self.lookup(node) if node.standalone else self.lookup(node.var)

    def visit_value(self, node, /):
        if node.type.kind is types.Bool:
            return ast.BooleanLiteral(node.value)
        if node.type.kind is types.Uint:
            return ast.IntegerLiteral(node.value)
        raise RuntimeError(f"unhandled Value type '{node}'")

    def visit_cast(self, node, /):
        if node.implicit:
            return node.operand.accept(self)
        return ast.Cast(_build_ast_type(node.type), node.operand.accept(self))

    def visit_unary(self, node, /):
        return ast.Unary(ast.Unary.Op[node.op.name], node.operand.accept(self))

    def visit_binary(self, node, /):
        return ast.Binary(
            ast.Binary.Op[node.op.name], node.left.accept(self), node.right.accept(self)
        )

    def visit_index(self, node, /):
        return ast.Index(node.target.accept(self), node.index.accept(self))
