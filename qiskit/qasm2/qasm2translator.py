"""
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

Created on Fri May  7 20:50:54 2021

@author: jax jwoehr@softwoehr.com
"""

from collections import OrderedDict
import re
import traceback

# from typing import Callable, Union
from qiskit.circuit import (
    Gate,
    Instruction,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Parameter,
)
from qiskit.circuit.library.standard_gates.x import CCXGate
from qiskit.circuit.library.standard_gates.swap import CSwapGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.y import CYGate
from qiskit.circuit.library.standard_gates.z import CZGate
from qiskit.circuit.library.standard_gates.swap import SwapGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.i import IGate
from qiskit.circuit.library.standard_gates.s import SGate
from qiskit.circuit.library.standard_gates.s import SdgGate
from qiskit.circuit.library.standard_gates.sx import SXGate
from qiskit.circuit.library.standard_gates.sx import SXdgGate
from qiskit.circuit.library.standard_gates.t import TGate
from qiskit.circuit.library.standard_gates.t import TdgGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.u2 import U2Gate
from qiskit.circuit.library.standard_gates.u3 import U3Gate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.library.standard_gates.y import YGate
from qiskit.circuit.library.standard_gates.z import ZGate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.library.standard_gates.rxx import RXXGate
from qiskit.circuit.library.standard_gates.rzz import RZZGate
from qiskit.circuit.library.standard_gates.p import CPhaseGate
from qiskit.circuit.library.standard_gates.u import CUGate
from qiskit.circuit.library.standard_gates.u1 import CU1Gate
from qiskit.circuit.library.standard_gates.u3 import CU3Gate
from qiskit.circuit.library.standard_gates.h import CHGate
from qiskit.circuit.library.standard_gates.rx import CRXGate
from qiskit.circuit.library.standard_gates.ry import CRYGate
from qiskit.circuit.library.standard_gates.rz import CRZGate
from qiskit.circuit.library.standard_gates.sx import CSXGate
from qiskit.circuit.register import Register
from qiskit.qasm2 import Qasm2AST, QasmError, qasm2Parser, Qasm2ExpressionListener

STANDARD_GATES = {
    "u1": U1Gate,
    "u2": U2Gate,
    "u3": U3Gate,
    "u": UGate,
    "p": PhaseGate,
    "x": XGate,
    "y": YGate,
    "z": ZGate,
    "t": TGate,
    "tdg": TdgGate,
    "s": SGate,
    "sdg": SdgGate,
    "sx": SXGate,
    "sxdg": SXdgGate,
    "swap": SwapGate,
    "rx": RXGate,
    "rxx": RXXGate,
    "ry": RYGate,
    "rz": RZGate,
    "rzz": RZZGate,
    "id": IGate,
    "h": HGate,
    "cx": CXGate,
    "cy": CYGate,
    "cz": CZGate,
    "ch": CHGate,
    "crx": CRXGate,
    "cry": CRYGate,
    "crz": CRZGate,
    "csx": CSXGate,
    "cu1": CU1Gate,
    "cp": CPhaseGate,
    "cu": CUGate,
    "cu3": CU3Gate,
    "ccx": CCXGate,
    "cswap": CSwapGate,
}


# def flatten(list_of_lists: list) -> list:
#     """
#     Flatten recursively a list of lists

#     Parameters
#     ----------
#     list_of_lists : list
#         Input list of nested lists.

#     Returns
#     -------
#     list
#         Flat list

#     """
#     if len(list_of_lists) == 0:
#         return list_of_lists
#     if isinstance(list_of_lists[0], list):
#         return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
#     return list_of_lists[:1] + flatten(list_of_lists[1:])


def indices_from_str(str_indices: str) -> tuple:
    """
    Convert the text indices for a bit operation to a range-ish tuple
    We'll return a range even if it's a single bit index, that works
    with QuantumCircuit,  e.g, qc.measure([0:1], [0:1])

    Parameters
    ----------
    str_indices : str
        string input of bit index/indices of form n or n:m

    Returns
    -------
    tuple
        (start, stop)

    """
    _i = str_indices.split(":")
    _a = int(_i[0])
    _b = _a + 1
    if len(_i) > 1:
        _b = int(_i[1])
    return (_a, _b)


def find_register_named(regs: list, named: str) -> Register:
    """
    Find the named quantum or classical register

    Parameters
    ----------
    regs : list
        list of regs, e.g., qc.qregs or qc.cregs
    named : str
        name of sought register.

    Returns
    -------
    Register
        The register with that name or None.

    """
    _r = None
    for reg in regs:
        if reg.name == named:
            _r = reg
            break
    return _r


def prepare_param_list(params: list, param_dict: OrderedDict = None) -> list:
    """
    Take list of string parameter expressions and convert them to values
    that can be applied to gate operations.

    Parameters
    ----------
    params : list of string
        The parameters expressed as string expressions in the AST.
    param_dict : OrderedDict of str, Parameter
        Ordered dictionary mapping names to Parameters for substitution.
        This facility is used to compile gate definitions with unbound
        parameters which get bound at Qasm2Translator.apply_defined_gate().

    Raises
    ------
    QasmError
        If parameter expressions unconvertible.

    Returns
    -------
    list
        numeric values applicable to gate operations.

    """
    _paramlist = []
    for p in params:
        if param_dict and p in param_dict:
            _paramlist.append(param_dict[p])
        else:
            if p.isnumeric():
                _paramlist.append(float(p))
            else:
                _paramlist.append(Qasm2ExpressionListener(p).do_expr())
        # raise QasmError(
        #     "prepare_param_list could not translate parameter {} ".format(p)
        # )
    return _paramlist


def prepare_target_list(regs: list, targets: list) -> list:
    """
    Process the register target list by identifying the registers.

    Parameters
    ----------
    regs : list
        list of registers for circuit.
    targets : list
        list of string of target register/bit identifiers to be found in
        the list of registers.

    Raises
    ------
    QasmError
        If register and bit could not be identified.

    Returns
    -------
    list
        list of target regs/bits.

    """
    _targlist = []
    for _t in targets:
        _targsplit = Q2TRegEx.BITSPEC_SQBRACK.findall(_t)
        _regname = None
        _indices = None
        if _targsplit:
            _regname = _targsplit[0][0]
            _indices = indices_from_str(_targsplit[0][1])
        else:
            _regname = _t
        _reg = find_register_named(regs, _regname)
        if not _reg:
            raise QasmError(
                "prepare_target_list could not find register {} in {} ".format(_regname, regs)
            )
        if _indices:
            _targlist.append(_reg[_indices[0] : _indices[1]])
        else:
            _targlist.append(_reg[0 : len(_reg)])
    return _targlist


def prepare_measure_target_tuple(qregs: list, cregs: list, targets: list) -> tuple:
    """
    Create a tuple with the qreg(s) and creg(s) of a ``measure()`` operation

    Parameters
    ----------
    qregs : list
        list of quantum registers for the circuit being processed
    cregs : list
        list of classical registers for the circuit being processed
    targets : list
        list of quantum & classical registers (one entry of each), possibly with
        indices, that are targets.

    Returns
    -------
    tuple
        A tuple of register bit specs (quantum_targets, classical_targets).

    """
    _qs = prepare_target_list(qregs, targets[0])
    _cs = prepare_target_list(cregs, targets[1])
    return (_qs, _cs)


class Q2TRegEx:
    """Compiled regular expressions for last-level parsing."""

    BITSPEC = re.compile(r"(\d+\:*\d*)")
    BITSPEC_SQBRACK = re.compile(r"(\w*)\[(\d+\:*\d*)\]")


class Qasm2Translator:
    """
    Translate AST to Quantum Circuit
    """

    def __init__(self, ast: Qasm2AST) -> None:
        """
        Create an empty circuit and save the AST in preparation for translation.

        Parameters
        ----------
        ast : Qasm2AST
            The AST being translated

        Returns
        -------
        None

        """
        self.ast = ast
        self.qc = QuantumCircuit()

    @staticmethod
    def apply_metacomment(qc: QuantumCircuit, entry: dict) -> None:
        """
        Apply a metacomment. Currently stubbed out

        Parameters
        ----------
        qc : QuantumCircuit
            the circuit for the metacomment
        entry : dict
            entry in the given Section

        Returns
        -------
        None

        """
        print(
            "Stubbed out apply_metacomment {} to QuantumCircuit\n{}".format(
                entry["metacomment_list"], qc
            )
        )

    @staticmethod
    def apply_quantum_declaration(
        qc: QuantumCircuit, quantum_type: str, reg_name: str, reg_width: int
    ) -> None:
        """
        From the index identifier list in a quantum declaration,
            - Parse the reg name
            - Parse the bit count
            - Create the quantum reg
            - Add to passedd-in circuit

        Parameters
        ----------
        qc : QuantumCircuit

        index_identifier_list : list


        Returns
        -------
        None

        Parameters
        ----------
        qc : QuantumCircuit
            The circuit to which to apply the definition.
        quantum_type : str
            The type of declaration, e.g., qreg, qubit ...
        reg_name : str
            The name of the register or qubit (if any)
        reg_width : int
            The bit count.

        Raises
        ------
        QasmError
           If unknown or unsupported quantum type.

        Returns
        -------
        None

        """

        if quantum_type == "qreg":
            _qr = QuantumRegister(size=reg_width, name=reg_name)
            qc.add_register(_qr)
        else:
            raise QasmError(
                "Quantum declaration of unknown or unsuppored quantum type {}".format(quantum_type)
            )

    @staticmethod
    def apply_classical_declaration(
        qc: QuantumCircuit, bit_type: str, reg_name: str, reg_width: int
    ) -> None:
        """
        From the index identifier list in a quantum declaration,
            - Parse the reg name
            - Parse the bit count
            - Create the quantum reg
            - Add to passedd-in circuit

        Parameters
        ----------
        qc : QuantumCircuit

        index_identifier_list : list


        Returns
        -------
        None

        Parameters
        ----------
        qc : QuantumCircuit
            The circuit to which to apply the definition.
        bit_type : str
            The type of declaration, e.g., creg, bit ...
        reg_name : str
            The name of the register or bit (if any)
        reg_width : int
            The bit count.

        Raises
        ------
        QasmError
           If unknown or unsupported quantum type.

        Returns
        -------
        None

        """

        if bit_type == "creg":
            _cr = ClassicalRegister(size=reg_width, name=reg_name)
            qc.add_register(_cr)
        else:
            raise QasmError(
                "Quantum declaration of unknown or unsuppored quantum type {}".format(bit_type)
            )

    @staticmethod
    def apply_gate(
        qc: QuantumCircuit,
        gate: Gate,
        params: list,
        targets: list,
        param_dict: OrderedDict = None,
    ) -> Instruction:
        """
        Apply a gate to a circuit
        e.g.
         - myQuantumTranslator.apply_gate(my_qc, STANDARD_GATES['cx'], [], ['q1', 'q0'])
         - myQuantumTranslator.apply_gate(my_qc, STANDARD_GATES['u3'], ['pi/2', 'pi/2', '0'], ['q1'])

        Parameters
        ----------
        qc : QuantumCircuit
            the QuantumCircuit instance under construction
        gate : Gate
            the Gate to apply
        params: list of str
            list of parameters as parsed, not yet evaluated
        targets : list of str
            list of qreg (qubit) or creg (bit) args.
        param_dict : OrderedDict, optional
            OrderedDict of ``{parameter_str_name: parameter_str_expansion}``
            The default is None.
            Ordered dictionary mapping names to Parameters for substitution
            coming from the parameter mapping for defined gates

        Raises
        ------
        QasmError
           If error applying gate to circuit.

        Returns
        -------
        Instruction
            The Instruction returned by ``qc.append()`` which allows us
            to apply c_if()
        """

        _paramlist = prepare_param_list(params, param_dict=param_dict)
        _targlist = prepare_target_list(qc.qregs, targets)
        _instruction = None
        try:
            if _paramlist:
                _instruction = qc.append(gate(*_paramlist), _targlist)
            else:
                # This is a hack
                # but it works
                try:
                    # isinstance(gate, Gate):
                    _instruction = qc.append(gate, _targlist)
                except:  # pylint: disable=bare-except  # hack hack hack
                    # elif isinstance(gate, Instruction):
                    _instruction = qc.append(gate(), _targlist)
        except Exception as ex:
            print(
                "In Qasm2Translator.apply_gate gate {} params: {} targets: {} yielding paramlist {} targlist {} param_dict {}".format(  # pylint: disable=line-too-long
                    gate, params, targets, _paramlist, _targlist, param_dict
                )
            )
            traceback.print_tb(ex.__traceback__)
            raise QasmError(  # pylint: disable=raise-missing-from
                "Error caught in Qasm2Translator.apply_gate(): Message: {} Cause: {} Context: {}".format(
                    ex, ex.__cause__, ex.__context__
                )
            )
        return _instruction

    def gate_for_gate_def(self, gate_def: dict) -> Gate:
        """
        Returns the Gate for an included gate def. If it has not already been
        compiled, calls gate_def_to_gate() to compile the def on the fly and
        store it in the ['gate'] member of the GSect entry .

        Parameters
        ----------
        gate_def : dict
            A GSect entry

        Raises
        ------
        QasmError
            If not available or not compilable.

        Returns
        -------
        Gate
            The Gate object either freshly compiled or from previous compilation.

        """
        gate = gate_def["gate"]
        if not gate:
            gate = self.gate_def_to_gate(gate_def)
            if not gate:
                raise QasmError(
                    "Gate def for gate definition \n{}\n could not be fetched nor compiled.".format(
                        gate_def
                    )
                )
        return gate

    def gate_def_to_gate(self, gate_def: dict) -> Gate:
        """
        Compiles the Gate for an included gate def, i.e., a GSect entry,
        and stashes it in the ['gate'] member of the GSect entry. Searches
        recursively for standard gates and gate definitions if necessary.

        Parameters
        ----------
        gate_def : dict
            The GSect entry

        Raises
        ------
        QasmError
            If not compilable.

        Returns
        -------
        Gate
            The Gate object as freshly compiled whose own parameters are unbound.
            The binding will occur in when applied.
        """
        gate = None
        _qr = QuantumRegister(len(gate_def["target_list"]), name="q")
        _qc = QuantumCircuit(_qr)

        _param_dict = OrderedDict()
        for _pname in gate_def["parameter_list"]:
            _param_dict[_pname] = Parameter(_pname)

        for _step in gate_def["definition"]:
            _targs = []

            for _t in _step["target_list"]:
                _targs.append("q[" + str(gate_def["target_list"].index(_t)) + "]")
            _op = _step["op"]
            _param_list = _step["parameter_list"]

            _gate = None

            if _op in STANDARD_GATES:
                _gate = STANDARD_GATES[_op]
            elif _op == "U":
                _gate = UGate
            elif _op == "CX":
                _gate = CXGate

            else:
                _gatedef = self.ast.find_latest_gatedef(_op)
                if _gatedef:
                    try:
                        _gate = self.gate_for_gate_def(_op)

                    except Exception as ex:
                        traceback.print_tb(ex.__traceback__)
                        raise QasmError(  # pylint: disable=raise-missing-from
                            "gate_def_to_gate: Translation of code step {} failed with Exception {}".format(  # pylint: disable=line-too-long
                                _step, ex
                            )
                        )

            if _gate:
                Qasm2Translator.apply_gate(
                    _qc,
                    _gate,
                    _param_list,
                    _targs,
                    _param_dict,
                )

            else:
                raise QasmError(
                    "gate_def_to_gate: Not a standard gate, not a Qasm2 keyword gate, and no def not found for {} in {}.".format(  # pylint: disable=line-too-long
                        _op, _step
                    )
                )

        gate = gate_def["gate"] = _qc.to_gate(label=gate_def["op"])
        return gate

    @staticmethod
    def apply_barrier(qc: QuantumCircuit, targets: list) -> None:
        """
        Add a ``barrier()`` to a circuit.

        Parameters
        ----------
        qc : QuantumCircuit
            The circuit for the measure
        targets : list
            List of qreg targs

        Returns
        -------
        None

        """
        _targlist = prepare_target_list(qc.qregs, targets)
        qc.barrier(*_targlist)

    @staticmethod
    def apply_measure(qc: QuantumCircuit, targets: list) -> None:
        """
        Add a ``measure()`` to a circuit.

        Parameters
        ----------
        qc : QuantumCircuit
            The circuit for the measure
        targets : list
            List of lists [qregs, cregs]

        Returns
        -------
        None

        """
        _targ_tuple = prepare_measure_target_tuple(qc.qregs, qc.cregs, targets)
        qc.measure(*_targ_tuple[0], *_targ_tuple[1])

    def translate_code_entry(self, qc: QuantumCircuit, entry: dict) -> None:
        """
        Translate a single entry from the AST CSect

        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit to which code is to be compiled/appended

        entry : dict
            An entry from the CSect of the AST

        Raises
        ------
        QasmError
            On not implemented or unknown entry type

        Returns
        -------
        None

        """
        ctx = entry["ctx"]
        if isinstance(ctx, qasm2Parser.SubroutineCallContext):
            _op = entry["op"]
            _gate = None
            if _op in STANDARD_GATES:
                _gate = STANDARD_GATES[_op]
            elif _op == "U":
                _gate = UGate
            elif _op == "CX":
                _gate = CXGate
            else:
                _gatedef = self.ast.find_latest_gatedef(_op)
                if _gatedef:
                    try:
                        _gate = self.gate_for_gate_def(_gatedef)
                    except Exception as ex:
                        traceback.print_tb(ex.__traceback__)
                        raise QasmError(  # pylint: disable=raise-missing-from
                            "translate_code_entry {} failed with Exception {}".format(entry, ex)
                        )
            if _gate:
                Qasm2Translator.apply_gate(
                    qc,
                    _gate,
                    entry["parameter_list"],
                    entry["target_list"],
                )
            else:
                raise QasmError(
                    "translate_code_entry: Not a standard gate, not a Qasm2 keyword gate, and no def not found for {} in {}.".format(  # pylint: disable=line-too-long
                        _op, entry
                    )
                )

        elif isinstance(ctx, qasm2Parser.QuantumDeclarationContext):
            Qasm2Translator.apply_quantum_declaration(
                qc, entry["quantum_type"], entry["reg_name"], entry["reg_width"]
            )

        elif isinstance(ctx, qasm2Parser.ClassicalDeclarationContext):
            Qasm2Translator.apply_classical_declaration(
                qc, entry["bit_type"], entry["reg_name"], entry["reg_width"]
            )

        elif isinstance(ctx, qasm2Parser.QuantumBarrierContext):
            Qasm2Translator.apply_barrier(qc, entry["index_identifier_list"])

        elif isinstance(ctx, qasm2Parser.QuantumMeasurementAssignmentContext):
            Qasm2Translator.apply_measure(
                qc,
                [
                    [entry["index_identifier_list"][0]],
                    [entry["index_identifier_list"][1]],
                ],
            )

        elif isinstance(ctx, qasm2Parser.BranchingStatementContext):
            _comparison_operator = entry["comparison_expression_list"][1]
            if _comparison_operator != "==":
                raise QasmError(
                    "translate_code_entry {} failed for an unsupported branch comparison operator {}".format(  # pylint: disable=line-too-long
                        entry, _comparison_operator
                    )
                )
            _op = entry["op"]
            _gate = None
            if _op in STANDARD_GATES:
                _gate = STANDARD_GATES[_op]
            elif _op == "U":
                _gate = UGate
            elif _op == "CX":
                _gate = CXGate
            else:
                _gatedef = self.ast.find_latest_gatedef(_op)
                if _gatedef:
                    try:
                        _gate = self.gate_for_gate_def(_op)
                    except Exception as ex:
                        traceback.print_tb(ex.__traceback__)
                        raise QasmError(  # pylint: disable=raise-missing-from
                            "translate_code_entry {} failed with Exception {}".format(entry, ex)
                        )
            if _gate:
                instruction = Qasm2Translator.apply_gate(
                    qc,
                    _gate,
                    entry["parameter_list"],
                    entry["target_list"],
                )
                instruction.c_if(
                    find_register_named(qc.cregs, entry["comparison_expression_list"][0]),
                    int(entry["comparison_expression_list"][2]),
                )
            else:
                raise QasmError(
                    "translate_code_entry: Not a standard gate, not a Qasm2 keyword gate, and no def not found for {} in {}.".format(  # pylint: disable=line-too-long
                        _op, entry
                    )
                )

        elif isinstance(ctx, qasm2Parser.MetaCommentContext):
            Qasm2Translator.apply_metacomment(qc, entry)

        else:
            raise QasmError(
                "Translation of code entry {} failed due to unknown code entry type.".format(entry)
            )

    def translate(self) -> QuantumCircuit:
        """
        Translate entire AST to circuit.

        Raises
        ------
        QasmError
            On any Error encountered in translation.

        Returns
        -------
        QuantumCircuit
            The circuit which was the subject of translation.

        """
        try:
            for entry in self.ast["CSect"]:
                self.translate_code_entry(self.qc, entry)
        except Exception as ex:
            print(
                "Error caught in Qasm2Translator.translate():\nMessage:\n{}\nCause:\n{}\nContext:\n{}".format(  # pylint: disable=line-too-long
                    ex, ex.__cause__, ex.__context__
                )
            )
            traceback.print_tb(ex.__traceback__)
            raise QasmError(  # pylint disable=raise-missing-from
                "Error encountered in translation: {}".format(ex)
            )  # pylint: disable=raise-missing-from
        return self.qc
