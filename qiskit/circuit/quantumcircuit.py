# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=bad-docstring-quotes,invalid-name

"""Quantum circuit object."""

import copy
import itertools
import functools
import numbers
import multiprocessing as mp
from collections import OrderedDict, defaultdict
from typing import Union
import numpy as np
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.qasm.qasm import Qasm
from qiskit.qasm.exceptions import QasmError
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_function, deprecate_arguments
from .parameterexpression import ParameterExpression
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterTable, ParameterView
from .parametervector import ParameterVector, ParameterVectorElement
from .instructionset import InstructionSet
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData
from .delay import Delay

try:
    import pygments
    from pygments.formatters import Terminal256Formatter  # pylint: disable=no-name-in-module
    from qiskit.qasm.pygments import OpenQASMLexer  # pylint: disable=ungrouped-imports
    from qiskit.qasm.pygments import QasmTerminalStyle  # pylint: disable=ungrouped-imports

    HAS_PYGMENTS = True
except Exception:  # pylint: disable=broad-except
    HAS_PYGMENTS = False


class QuantumCircuit:
    """Create a new circuit.

    A circuit is a list of instructions bound to some registers.

    Args:
        regs (list(:class:`Register`) or list(``int``) or list(list(:class:`Bit`))): The
            registers to be included in the circuit.

            * If a list of :class:`Register` objects, represents the :class:`QuantumRegister`
              and/or :class:`ClassicalRegister` objects to include in the circuit.

              For example:

                * ``QuantumCircuit(QuantumRegister(4))``
                * ``QuantumCircuit(QuantumRegister(4), ClassicalRegister(3))``
                * ``QuantumCircuit(QuantumRegister(4, 'qr0'), QuantumRegister(2, 'qr1'))``

            * If a list of ``int``, the amount of qubits and/or classical bits to include in
              the circuit. It can either be a single int for just the number of quantum bits,
              or 2 ints for the number of quantum bits and classical bits, respectively.

              For example:

                * ``QuantumCircuit(4) # A QuantumCircuit with 4 qubits``
                * ``QuantumCircuit(4, 3) # A QuantumCircuit with 4 qubits and 3 classical bits``

            * If a list of python lists containing :class:`Bit` objects, a collection of
              :class:`Bit` s to be added to the circuit.


        name (str): the name of the quantum circuit. If not set, an
            automatically generated string will be assigned.
        global_phase (float or ParameterExpression): The global phase of the circuit in radians.
        metadata (dict): Arbitrary key value metadata to associate with the
            circuit. This gets stored as free-form data in a dict in the
            :attr:`~qiskit.circuit.QuantumCircuit.metadata` attribute. It will
            not be directly used in the circuit.

    Raises:
        CircuitError: if the circuit name, if given, is not valid.

    Examples:

        Construct a simple Bell state circuit.

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            qc.draw()

        Construct a 5-qubit GHZ circuit.

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            qc = QuantumCircuit(5)
            qc.h(0)
            qc.cx(0, range(1, 5))
            qc.measure_all()

        Construct a 4-qubit Bernstein-Vazirani circuit using registers.

        .. jupyter-execute::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

            qr = QuantumRegister(3, 'q')
            anc = QuantumRegister(1, 'ancilla')
            cr = ClassicalRegister(3, 'c')
            qc = QuantumCircuit(qr, anc, cr)

            qc.x(anc[0])
            qc.h(anc[0])
            qc.h(qr[0:3])
            qc.cx(qr[0:3], anc[0])
            qc.h(qr[0:3])
            qc.barrier(qr)
            qc.measure(qr, cr)

            qc.draw()
    """

    instances = 0
    prefix = "circuit"

    # Class variable OPENQASM header
    header = "OPENQASM 2.0;"
    extension_lib = 'include "qelib1.inc";'

    def __init__(self, *regs, name=None, global_phase=0, metadata=None):
        if any(not isinstance(reg, (list, QuantumRegister, ClassicalRegister)) for reg in regs):
            # check if inputs are integers, but also allow e.g. 2.0

            try:
                valid_reg_size = all(reg == int(reg) for reg in regs)
            except (ValueError, TypeError):
                valid_reg_size = False

            if not valid_reg_size:
                raise CircuitError(
                    "Circuit args must be Registers or integers. (%s '%s' was "
                    "provided)" % ([type(reg).__name__ for reg in regs], regs)
                )

            regs = tuple(int(reg) for reg in regs)  # cast to int
        self._base_name = None
        if name is None:
            self._base_name = self.cls_prefix()
            self._name_update()
        elif not isinstance(name, str):
            raise CircuitError(
                "The circuit name should be a string " "(or None to auto-generate a name)."
            )
        else:
            self._base_name = name
            self.name = name
        self._increment_instances()

        # Data contains a list of instructions and their contexts,
        # in the order they were applied.
        self._data = []

        # This is a map of registers bound to this circuit, by name.
        self.qregs = []
        self.cregs = []
        self._qubits = []
        self._qubit_set = set()
        self._clbits = []
        self._clbit_set = set()

        self._ancillas = []
        self._calibrations = defaultdict(dict)
        self.add_register(*regs)

        # Parameter table tracks instructions with variable parameters.
        self._parameter_table = ParameterTable()

        # Cache to avoid re-sorting parameters
        self._parameters = None

        self._layout = None
        self._global_phase = 0
        self.global_phase = global_phase

        self.duration = None
        self.unit = "dt"
        if not isinstance(metadata, dict) and metadata is not None:
            raise TypeError("Only a dictionary or None is accepted for circuit metadata")
        self._metadata = metadata

    @property
    def data(self):
        """Return the circuit data (instructions and context).

        Returns:
            QuantumCircuitData: a list-like object containing the tuples for the circuit's data.

            Each tuple is in the format ``(instruction, qargs, cargs)``, where instruction is an
            Instruction (or subclass) object, qargs is a list of Qubit objects, and cargs is a
            list of Clbit objects.
        """
        return QuantumCircuitData(self)

    @property
    def calibrations(self):
        """Return calibration dictionary.

        The custom pulse definition of a given gate is of the form
            {'gate_name': {(qubits, params): schedule}}
        """
        return dict(self._calibrations)

    @calibrations.setter
    def calibrations(self, calibrations):
        """Set the circuit calibration data from a dictionary of calibration definition.

        Args:
            calibrations (dict): A dictionary of input in the format
                {'gate_name': {(qubits, gate_params): schedule}}
        """
        self._calibrations = defaultdict(dict, calibrations)

    @data.setter
    def data(self, data_input):
        """Sets the circuit data from a list of instructions and context.

        Args:
            data_input (list): A list of instructions with context
                in the format (instruction, qargs, cargs), where Instruction
                is an Instruction (or subclass) object, qargs is a list of
                Qubit objects, and cargs is a list of Clbit objects.
        """

        # If data_input is QuantumCircuitData(self), clearing self._data
        # below will also empty data_input, so make a shallow copy first.
        data_input = data_input.copy()
        self._data = []
        self._parameter_table = ParameterTable()

        for inst, qargs, cargs in data_input:
            self.append(inst, qargs, cargs)

    @property
    def metadata(self):
        """The user provided metadata associated with the circuit

        The metadata for the circuit is a user provided ``dict`` of metadata
        for the circuit. It will not be used to influence the execution or
        operation of the circuit, but it is expected to be passed between
        all transforms of the circuit (ie transpilation) and that providers will
        associate any circuit metadata with the results it returns from
        execution of that circuit.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """Update the circuit metadata"""
        if not isinstance(metadata, dict) and metadata is not None:
            raise TypeError("Only a dictionary or None is accepted for circuit metadata")
        self._metadata = metadata

    def __str__(self):
        return str(self.draw(output="text"))

    def __eq__(self, other):
        if not isinstance(other, QuantumCircuit):
            return False

        # TODO: remove the DAG from this function
        from qiskit.converters import circuit_to_dag

        return circuit_to_dag(self) == circuit_to_dag(other)

    @classmethod
    def _increment_instances(cls):
        cls.instances += 1

    @classmethod
    def cls_instances(cls):
        """Return the current number of instances of this class,
        useful for auto naming."""
        return cls.instances

    @classmethod
    def cls_prefix(cls):
        """Return the prefix to use for auto naming."""
        return cls.prefix

    def _name_update(self):
        """update name of instance using instance number"""
        if not is_main_process():
            pid_name = f"-{mp.current_process().pid}"
        else:
            pid_name = ""

        self.name = f"{self._base_name}-{self.cls_instances()}{pid_name}"

    def has_register(self, register):
        """
        Test if this circuit has the register r.

        Args:
            register (Register): a quantum or classical register.

        Returns:
            bool: True if the register is contained in this circuit.
        """
        has_reg = False
        if isinstance(register, QuantumRegister) and register in self.qregs:
            has_reg = True
        elif isinstance(register, ClassicalRegister) and register in self.cregs:
            has_reg = True
        return has_reg

    def reverse_ops(self):
        """Reverse the circuit by reversing the order of instructions.

        This is done by recursively reversing all instructions.
        It does not invert (adjoint) any gate.

        Returns:
            QuantumCircuit: the reversed circuit.

        Examples:

            input:
                 ┌───┐
            q_0: ┤ H ├─────■──────
                 └───┘┌────┴─────┐
            q_1: ─────┤ RX(1.57) ├
                      └──────────┘

            output:
                             ┌───┐
            q_0: ─────■──────┤ H ├
                 ┌────┴─────┐└───┘
            q_1: ┤ RX(1.57) ├─────
                 └──────────┘
        """
        reverse_circ = QuantumCircuit(
            self.qubits, self.clbits, *self.qregs, *self.cregs, name=self.name + "_reverse"
        )

        for inst, qargs, cargs in reversed(self.data):
            reverse_circ._append(inst.reverse_ops(), qargs, cargs)

        reverse_circ.duration = self.duration
        reverse_circ.unit = self.unit
        return reverse_circ

    def reverse_bits(self):
        """Return a circuit with the opposite order of wires.

        The circuit is "vertically" flipped. If a circuit is
        defined over multiple registers, the resulting circuit will have
        the same registers but with their order flipped.

        This method is useful for converting a circuit written in little-endian
        convention to the big-endian equivalent, and vice versa.

        Returns:
            QuantumCircuit: the circuit with reversed bit order.

        Examples:

            input:
                 ┌───┐
            q_0: ┤ H ├─────■──────
                 └───┘┌────┴─────┐
            q_1: ─────┤ RX(1.57) ├
                      └──────────┘

            output:
                      ┌──────────┐
            q_0: ─────┤ RX(1.57) ├
                 ┌───┐└────┬─────┘
            q_1: ┤ H ├─────■──────
                 └───┘
        """
        circ = QuantumCircuit(
            *reversed(self.qregs),
            *reversed(self.cregs),
            name=self.name,
            global_phase=self.global_phase,
        )
        num_qubits = self.num_qubits
        num_clbits = self.num_clbits
        old_qubits = self.qubits
        old_clbits = self.clbits
        new_qubits = circ.qubits
        new_clbits = circ.clbits

        for inst, qargs, cargs in self.data:
            new_qargs = [new_qubits[num_qubits - old_qubits.index(q) - 1] for q in qargs]
            new_cargs = [new_clbits[num_clbits - old_clbits.index(c) - 1] for c in cargs]
            circ._append(inst, new_qargs, new_cargs)
        return circ

    def inverse(self):
        """Invert (take adjoint of) this circuit.

        This is done by recursively inverting all gates.

        Returns:
            QuantumCircuit: the inverted circuit

        Raises:
            CircuitError: if the circuit cannot be inverted.

        Examples:

            input:
                 ┌───┐
            q_0: ┤ H ├─────■──────
                 └───┘┌────┴─────┐
            q_1: ─────┤ RX(1.57) ├
                      └──────────┘

            output:
                              ┌───┐
            q_0: ──────■──────┤ H ├
                 ┌─────┴─────┐└───┘
            q_1: ┤ RX(-1.57) ├─────
                 └───────────┘
        """
        inverse_circ = QuantumCircuit(
            self.qubits,
            self.clbits,
            *self.qregs,
            *self.cregs,
            name=self.name + "_dg",
            global_phase=-self.global_phase,
        )

        for inst, qargs, cargs in reversed(self._data):
            inverse_circ._append(inst.inverse(), qargs, cargs)
        return inverse_circ

    def repeat(self, reps):
        """Repeat this circuit ``reps`` times.

        Args:
            reps (int): How often this circuit should be repeated.

        Returns:
            QuantumCircuit: A circuit containing ``reps`` repetitions of this circuit.
        """
        repeated_circ = QuantumCircuit(
            self.qubits, self.clbits, *self.qregs, *self.cregs, name=self.name + f"**{reps}"
        )

        # benefit of appending instructions: decomposing shows the subparts, i.e. the power
        # is actually `reps` times this circuit, and it is currently much faster than `compose`.
        if reps > 0:
            try:  # try to append as gate if possible to not disallow to_gate
                inst = self.to_gate()
            except QiskitError:
                inst = self.to_instruction()
            for _ in range(reps):
                repeated_circ._append(inst, self.qubits, self.clbits)

        return repeated_circ

    def power(self, power, matrix_power=False):
        """Raise this circuit to the power of ``power``.

        If ``power`` is a positive integer and ``matrix_power`` is ``False``, this implementation
        defaults to calling ``repeat``. Otherwise, if the circuit is unitary, the matrix is
        computed to calculate the matrix power.

        Args:
            power (int): The power to raise this circuit to.
            matrix_power (bool): If True, the circuit is converted to a matrix and then the
                matrix power is computed. If False, and ``power`` is a positive integer,
                the implementation defaults to ``repeat``.

        Raises:
            CircuitError: If the circuit needs to be converted to a gate but it is not unitary.

        Returns:
            QuantumCircuit: A circuit implementing this circuit raised to the power of ``power``.
        """
        if power >= 0 and isinstance(power, numbers.Integral) and not matrix_power:
            return self.repeat(power)

        # attempt conversion to gate
        if self.num_parameters > 0:
            raise CircuitError(
                "Cannot raise a parameterized circuit to a non-positive power "
                "or matrix-power, please bind the free parameters: "
                "{}".format(self.parameters)
            )

        try:
            gate = self.to_gate()
        except QiskitError as ex:
            raise CircuitError(
                "The circuit contains non-unitary operations and cannot be "
                "controlled. Note that no qiskit.circuit.Instruction objects may "
                "be in the circuit for this operation."
            ) from ex

        power_circuit = QuantumCircuit(self.qubits, self.clbits, *self.qregs, *self.cregs)
        power_circuit.append(gate.power(power), list(range(gate.num_qubits)))
        return power_circuit

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Control this circuit on ``num_ctrl_qubits`` qubits.

        Args:
            num_ctrl_qubits (int): The number of control qubits.
            label (str): An optional label to give the controlled operation for visualization.
            ctrl_state (str or int): The control state in decimal or as a bitstring
                (e.g. '111'). If None, use ``2**num_ctrl_qubits - 1``.

        Returns:
            QuantumCircuit: The controlled version of this circuit.

        Raises:
            CircuitError: If the circuit contains a non-unitary operation and cannot be controlled.
        """
        try:
            gate = self.to_gate()
        except QiskitError as ex:
            raise CircuitError(
                "The circuit contains non-unitary operations and cannot be "
                "controlled. Note that no qiskit.circuit.Instruction objects may "
                "be in the circuit for this operation."
            ) from ex

        controlled_gate = gate.control(num_ctrl_qubits, label, ctrl_state)
        control_qreg = QuantumRegister(num_ctrl_qubits)
        controlled_circ = QuantumCircuit(
            control_qreg, self.qubits, *self.qregs, name=f"c_{self.name}"
        )
        controlled_circ.append(controlled_gate, controlled_circ.qubits)

        return controlled_circ

    @deprecate_function(
        "The QuantumCircuit.combine() method is being deprecated. "
        "Use the compose() method which is more flexible w.r.t "
        "circuit register compatibility."
    )
    def combine(self, rhs):
        """DEPRECATED - Returns rhs appended to self if self contains compatible registers.

        Two circuits are compatible if they contain the same registers
        or if they contain different registers with unique names. The
        returned circuit will contain all unique registers between both
        circuits.

        Return self + rhs as a new object.

        Args:
            rhs (QuantumCircuit): The quantum circuit to append to the right hand side.

        Returns:
            QuantumCircuit: Returns a new QuantumCircuit object

        Raises:
            QiskitError: if the rhs circuit is not compatible
        """
        # Check registers in LHS are compatible with RHS
        self._check_compatible_regs(rhs)

        # Make new circuit with combined registers
        combined_qregs = copy.deepcopy(self.qregs)
        combined_cregs = copy.deepcopy(self.cregs)

        for element in rhs.qregs:
            if element not in self.qregs:
                combined_qregs.append(element)
        for element in rhs.cregs:
            if element not in self.cregs:
                combined_cregs.append(element)
        circuit = QuantumCircuit(*combined_qregs, *combined_cregs)
        for instruction_context in itertools.chain(self.data, rhs.data):
            circuit._append(*instruction_context)
        circuit.global_phase = self.global_phase + rhs.global_phase

        for gate, cals in rhs.calibrations.items():
            for key, sched in cals.items():
                circuit.add_calibration(gate, qubits=key[0], schedule=sched, params=key[1])

        for gate, cals in self.calibrations.items():
            for key, sched in cals.items():
                circuit.add_calibration(gate, qubits=key[0], schedule=sched, params=key[1])

        return circuit

    @deprecate_function(
        "The QuantumCircuit.extend() method is being deprecated. Use the "
        "compose() (potentially with the inplace=True argument) and tensor() "
        "methods which are more flexible w.r.t circuit register compatibility."
    )
    def extend(self, rhs):
        """DEPRECATED - Append QuantumCircuit to the RHS if it contains compatible registers.

        Two circuits are compatible if they contain the same registers
        or if they contain different registers with unique names. The
        returned circuit will contain all unique registers between both
        circuits.

        Modify and return self.

        Args:
            rhs (QuantumCircuit): The quantum circuit to append to the right hand side.

        Returns:
            QuantumCircuit: Returns this QuantumCircuit object (which has been modified)

        Raises:
            QiskitError: if the rhs circuit is not compatible
        """
        # Check registers in LHS are compatible with RHS
        self._check_compatible_regs(rhs)

        # Add new registers
        for element in rhs.qregs:
            if element not in self.qregs:
                self.qregs.append(element)
                self._qubits += element[:]
                self._qubit_set.update(element[:])
        for element in rhs.cregs:
            if element not in self.cregs:
                self.cregs.append(element)
                self._clbits += element[:]
                self._clbit_set.update(element[:])

        # Copy the circuit data if rhs and self are the same, otherwise the data of rhs is
        # appended to both self and rhs resulting in an infinite loop
        data = rhs.data.copy() if rhs is self else rhs.data

        # Add new gates
        for instruction_context in data:
            self._append(*instruction_context)
        self.global_phase += rhs.global_phase

        for gate, cals in rhs.calibrations.items():
            for key, sched in cals.items():
                self.add_calibration(gate, qubits=key[0], schedule=sched, params=key[1])

        return self

    def compose(self, other, qubits=None, clbits=None, front=False, inplace=False, wrap=False):
        """Compose circuit with ``other`` circuit or instruction, optionally permuting wires.

        ``other`` can be narrower or of equal width to ``self``.

        Args:
            other (qiskit.circuit.Instruction or QuantumCircuit or BaseOperator):
                (sub)circuit to compose onto self.
            qubits (list[Qubit|int]): qubits of self to compose onto.
            clbits (list[Clbit|int]): clbits of self to compose onto.
            front (bool): If True, front composition will be performed (not implemented yet).
            inplace (bool): If True, modify the object. Otherwise return composed circuit.
            wrap (bool): If True, wraps the other circuit into a gate (or instruction, depending on
                whether it contains only unitary instructions) before composing it onto self.

        Returns:
            QuantumCircuit: the composed circuit (returns None if inplace==True).

        Raises:
            CircuitError: if composing on the front.
            QiskitError: if ``other`` is wider or there are duplicate edge mappings.

        Examples::

            lhs.compose(rhs, qubits=[3, 2], inplace=True)

            .. parsed-literal::

                            ┌───┐                   ┌─────┐                ┌───┐
                lqr_1_0: ───┤ H ├───    rqr_0: ──■──┤ Tdg ├    lqr_1_0: ───┤ H ├───────────────
                            ├───┤              ┌─┴─┐└─────┘                ├───┤
                lqr_1_1: ───┤ X ├───    rqr_1: ┤ X ├───────    lqr_1_1: ───┤ X ├───────────────
                         ┌──┴───┴──┐           └───┘                    ┌──┴───┴──┐┌───┐
                lqr_1_2: ┤ U1(0.1) ├  +                     =  lqr_1_2: ┤ U1(0.1) ├┤ X ├───────
                         └─────────┘                                    └─────────┘└─┬─┘┌─────┐
                lqr_2_0: ─────■─────                           lqr_2_0: ─────■───────■──┤ Tdg ├
                            ┌─┴─┐                                          ┌─┴─┐        └─────┘
                lqr_2_1: ───┤ X ├───                           lqr_2_1: ───┤ X ├───────────────
                            └───┘                                          └───┘
                lcr_0: 0 ═══════════                           lcr_0: 0 ═══════════════════════

                lcr_1: 0 ═══════════                           lcr_1: 0 ═══════════════════════

        """

        if inplace:
            dest = self
        else:
            dest = self.copy()

        if wrap:
            try:
                other = other.to_gate()
            except QiskitError:
                other = other.to_instruction()

        if not isinstance(other, QuantumCircuit):
            if qubits is None:
                qubits = list(range(other.num_qubits))

            if clbits is None:
                clbits = list(range(other.num_clbits))

            if front:
                dest.data.insert(0, (other, qubits, clbits))
            else:
                dest.append(other, qargs=qubits, cargs=clbits)

            if inplace:
                return None
            return dest

        instrs = other.data

        if other.num_qubits > self.num_qubits or other.num_clbits > self.num_clbits:
            raise CircuitError(
                "Trying to compose with another QuantumCircuit " "which has more 'in' edges."
            )

        # number of qubits and clbits must match number in circuit or None
        identity_qubit_map = dict(zip(other.qubits, self.qubits))
        identity_clbit_map = dict(zip(other.clbits, self.clbits))

        if qubits is None:
            qubit_map = identity_qubit_map
        elif len(qubits) != len(other.qubits):
            raise CircuitError(
                "Number of items in qubits parameter does not"
                " match number of qubits in the circuit."
            )
        else:
            qubit_map = {
                other.qubits[i]: (self.qubits[q] if isinstance(q, int) else q)
                for i, q in enumerate(qubits)
            }
        if clbits is None:
            clbit_map = identity_clbit_map
        elif len(clbits) != len(other.clbits):
            raise CircuitError(
                "Number of items in clbits parameter does not"
                " match number of clbits in the circuit."
            )
        else:
            clbit_map = {
                other.clbits[i]: (self.clbits[c] if isinstance(c, int) else c)
                for i, c in enumerate(clbits)
            }

        edge_map = {**qubit_map, **clbit_map} or {**identity_qubit_map, **identity_clbit_map}

        mapped_instrs = []
        for instr, qargs, cargs in instrs:
            n_qargs = [edge_map[qarg] for qarg in qargs]
            n_cargs = [edge_map[carg] for carg in cargs]
            n_instr = instr.copy()

            if instr.condition is not None:
                from qiskit.dagcircuit import DAGCircuit  # pylint: disable=cyclic-import

                n_instr.condition = DAGCircuit._map_condition(edge_map, instr.condition, self.cregs)

            mapped_instrs.append((n_instr, n_qargs, n_cargs))

        if front:
            dest._data = mapped_instrs + dest._data
        else:
            dest._data += mapped_instrs

        if front:
            dest._parameter_table.clear()
            for instr, _, _ in dest._data:
                dest._update_parameter_table(instr)
        else:
            # just append new parameters
            for instr, _, _ in mapped_instrs:
                dest._update_parameter_table(instr)

        for gate, cals in other.calibrations.items():
            dest._calibrations[gate].update(cals)

        dest.global_phase += other.global_phase

        if inplace:
            return None

        return dest

    def tensor(self, other, inplace=False):
        """Tensor ``self`` with ``other``.

        Remember that in the little-endian convention the leftmost operation will be at the bottom
        of the circuit. See also
        [the docs](qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html)
        for more information.

        .. parsed-literal::

                 ┌────────┐        ┌─────┐          ┌─────┐
            q_0: ┤ bottom ├ ⊗ q_0: ┤ top ├  = q_0: ─┤ top ├──
                 └────────┘        └─────┘         ┌┴─────┴─┐
                                              q_1: ┤ bottom ├
                                                   └────────┘

        Args:
            other (QuantumCircuit): The other circuit to tensor this circuit with.
            inplace (bool): If True, modify the object. Otherwise return composed circuit.

        Examples:

            .. jupyter-execute::

                from qiskit import QuantumCircuit
                top = QuantumCircuit(1)
                top.x(0);
                bottom = QuantumCircuit(2)
                bottom.cry(0.2, 0, 1);
                tensored = bottom.tensor(top)
                print(tensored.draw())

        Returns:
            QuantumCircuit: The tensored circuit (returns None if inplace==True).
        """
        num_qubits = self.num_qubits + other.num_qubits
        num_clbits = self.num_clbits + other.num_clbits

        # If a user defined both circuits with via register sizes and not with named registers
        # (e.g. QuantumCircuit(2, 2)) then we have a naming collision, as the registers are by
        # default called "q" resp. "c". To still allow tensoring we define new registers of the
        # correct sizes.
        if (
            len(self.qregs) == len(other.qregs) == 1
            and self.qregs[0].name == other.qregs[0].name == "q"
        ):
            # check if classical registers are in the circuit
            if num_clbits > 0:
                dest = QuantumCircuit(num_qubits, num_clbits)
            else:
                dest = QuantumCircuit(num_qubits)

        # handle case if ``measure_all`` was called on both circuits, in which case the
        # registers are both named "meas"
        elif (
            len(self.cregs) == len(other.cregs) == 1
            and self.cregs[0].name == other.cregs[0].name == "meas"
        ):
            cr = ClassicalRegister(self.num_clbits + other.num_clbits, "meas")
            dest = QuantumCircuit(*other.qregs, *self.qregs, cr)

        # Now we don't have to handle any more cases arising from special implicit naming
        else:
            dest = QuantumCircuit(
                other.qubits,
                self.qubits,
                other.clbits,
                self.clbits,
                *other.qregs,
                *self.qregs,
                *other.cregs,
                *self.cregs,
            )

        # compose self onto the output, and then other
        dest.compose(other, range(other.num_qubits), range(other.num_clbits), inplace=True)
        dest.compose(
            self,
            range(other.num_qubits, num_qubits),
            range(other.num_clbits, num_clbits),
            inplace=True,
        )

        # Replace information from tensored circuit into self when inplace = True
        if inplace:
            self.__dict__.update(dest.__dict__)
            return None
        return dest

    @property
    def qubits(self):
        """
        Returns a list of quantum bits in the order that the registers were added.
        """
        return self._qubits

    @property
    def clbits(self):
        """
        Returns a list of classical bits in the order that the registers were added.
        """
        return self._clbits

    @property
    def ancillas(self):
        """
        Returns a list of ancilla bits in the order that the registers were added.
        """
        return self._ancillas

    @deprecate_function(
        "The QuantumCircuit.__add__() method is being deprecated."
        "Use the compose() method which is more flexible w.r.t "
        "circuit register compatibility."
    )
    def __add__(self, rhs):
        """Overload + to implement self.combine."""
        return self.combine(rhs)

    @deprecate_function(
        "The QuantumCircuit.__iadd__() method is being deprecated. Use the "
        "compose() (potentially with the inplace=True argument) and tensor() "
        "methods which are more flexible w.r.t circuit register compatibility."
    )
    def __iadd__(self, rhs):
        """Overload += to implement self.extend."""
        return self.extend(rhs)

    def __and__(self, rhs):
        """Overload & to implement self.compose."""
        return self.compose(rhs)

    def __iand__(self, rhs):
        """Overload &= to implement self.compose in place."""
        self.compose(rhs, inplace=True)
        return self

    def __xor__(self, top):
        """Overload ^ to implement self.tensor."""
        return self.tensor(top)

    def __ixor__(self, top):
        """Overload ^= to implement self.tensor in place."""
        self.tensor(top, inplace=True)
        return self

    def __len__(self):
        """Return number of operations in circuit."""
        return len(self._data)

    def __getitem__(self, item):
        """Return indexed operation."""
        return self._data[item]

    @staticmethod
    def cast(value, _type):
        """Best effort to cast value to type. Otherwise, returns the value."""
        try:
            return _type(value)
        except (ValueError, TypeError):
            return value

    @staticmethod
    def _bit_argument_conversion(bit_representation, in_array):
        ret = None
        try:
            if isinstance(bit_representation, Bit):
                # circuit.h(qr[0]) -> circuit.h([qr[0]])
                ret = [bit_representation]
            elif isinstance(bit_representation, Register):
                # circuit.h(qr) -> circuit.h([qr[0], qr[1]])
                ret = bit_representation[:]
            elif isinstance(QuantumCircuit.cast(bit_representation, int), int):
                # circuit.h(0) -> circuit.h([qr[0]])
                ret = [in_array[bit_representation]]
            elif isinstance(bit_representation, slice):
                # circuit.h(slice(0,2)) -> circuit.h([qr[0], qr[1]])
                ret = in_array[bit_representation]
            elif isinstance(bit_representation, list) and all(
                isinstance(bit, Bit) for bit in bit_representation
            ):
                # circuit.h([qr[0], qr[1]]) -> circuit.h([qr[0], qr[1]])
                ret = bit_representation
            elif isinstance(QuantumCircuit.cast(bit_representation, list), (range, list)):
                # circuit.h([0, 1])     -> circuit.h([qr[0], qr[1]])
                # circuit.h(range(0,2)) -> circuit.h([qr[0], qr[1]])
                # circuit.h([qr[0],1])  -> circuit.h([qr[0], qr[1]])
                ret = [
                    index if isinstance(index, Bit) else in_array[index]
                    for index in bit_representation
                ]
            else:
                raise CircuitError(
                    f"Not able to expand a {bit_representation} ({type(bit_representation)})"
                )
        except IndexError as ex:
            raise CircuitError("Index out of range.") from ex
        except TypeError as ex:
            raise CircuitError(
                f"Type error handling {bit_representation} ({type(bit_representation)})"
            ) from ex
        return ret

    def qbit_argument_conversion(self, qubit_representation):
        """
        Converts several qubit representations (such as indexes, range, etc.)
        into a list of qubits.

        Args:
            qubit_representation (Object): representation to expand

        Returns:
            List(tuple): Where each tuple is a qubit.
        """
        return QuantumCircuit._bit_argument_conversion(qubit_representation, self.qubits)

    def cbit_argument_conversion(self, clbit_representation):
        """
        Converts several classical bit representations (such as indexes, range, etc.)
        into a list of classical bits.

        Args:
            clbit_representation (Object): representation to expand

        Returns:
            List(tuple): Where each tuple is a classical bit.
        """
        return QuantumCircuit._bit_argument_conversion(clbit_representation, self.clbits)

    def append(self, instruction, qargs=None, cargs=None):
        """Append one or more instructions to the end of the circuit, modifying
        the circuit in place. Expands qargs and cargs.

        Args:
            instruction (qiskit.circuit.Instruction): Instruction instance to append
            qargs (list(argument)): qubits to attach instruction to
            cargs (list(argument)): clbits to attach instruction to

        Returns:
            qiskit.circuit.Instruction: a handle to the instruction that was just added

        Raises:
            CircuitError: if object passed is a subclass of Instruction
            CircuitError: if object passed is neither subclass nor an instance of Instruction
        """
        # Convert input to instruction
        if not isinstance(instruction, Instruction) and not hasattr(instruction, "to_instruction"):
            if issubclass(instruction, Instruction):
                raise CircuitError(
                    "Object is a subclass of Instruction, please add () to "
                    "pass an instance of this object."
                )

            raise CircuitError(
                "Object to append must be an Instruction or " "have a to_instruction() method."
            )
        if not isinstance(instruction, Instruction) and hasattr(instruction, "to_instruction"):
            instruction = instruction.to_instruction()

        # Make copy of parameterized gate instances
        if hasattr(instruction, "params"):
            is_parameter = any(isinstance(param, Parameter) for param in instruction.params)
            if is_parameter:
                instruction = copy.deepcopy(instruction)

        expanded_qargs = [self.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self.cbit_argument_conversion(carg) for carg in cargs or []]

        instructions = InstructionSet()
        for (qarg, carg) in instruction.broadcast_arguments(expanded_qargs, expanded_cargs):
            instructions.add(self._append(instruction, qarg, carg), qarg, carg)
        return instructions

    def _append(self, instruction, qargs, cargs):
        """Append an instruction to the end of the circuit, modifying
        the circuit in place.

        Args:
            instruction (Instruction or Operator): Instruction instance to append
            qargs (list(tuple)): qubits to attach instruction to
            cargs (list(tuple)): clbits to attach instruction to

        Returns:
            Instruction: a handle to the instruction that was just added

        Raises:
            CircuitError: if the gate is of a different shape than the wires
                it is being attached to.
        """
        if not isinstance(instruction, Instruction):
            raise CircuitError("object is not an Instruction.")

        # do some compatibility checks
        self._check_dups(qargs)
        self._check_qargs(qargs)
        self._check_cargs(cargs)

        # add the instruction onto the given wires
        instruction_context = instruction, qargs, cargs
        self._data.append(instruction_context)

        self._update_parameter_table(instruction)

        # mark as normal circuit if a new instruction is added
        self.duration = None
        self.unit = "dt"

        return instruction

    def _update_parameter_table(self, instruction):

        for param_index, param in enumerate(instruction.params):
            if isinstance(param, ParameterExpression):
                current_parameters = self._parameter_table

                for parameter in param.parameters:
                    if parameter in current_parameters:
                        if not self._check_dup_param_spec(
                            self._parameter_table[parameter], instruction, param_index
                        ):
                            self._parameter_table[parameter].append((instruction, param_index))
                    else:
                        if parameter.name in self._parameter_table.get_names():
                            raise CircuitError(
                                f"Name conflict on adding parameter: {parameter.name}"
                            )
                        self._parameter_table[parameter] = [(instruction, param_index)]

                        # clear cache if new parameter is added
                        self._parameters = None

        return instruction

    def _check_dup_param_spec(self, parameter_spec_list, instruction, param_index):
        for spec in parameter_spec_list:
            if spec[0] is instruction and spec[1] == param_index:
                return True
        return False

    def add_register(self, *regs):
        """Add registers."""
        if not regs:
            return

        if any(isinstance(reg, int) for reg in regs):
            # QuantumCircuit defined without registers
            if len(regs) == 1 and isinstance(regs[0], int):
                # QuantumCircuit with anonymous quantum wires e.g. QuantumCircuit(2)
                regs = (QuantumRegister(regs[0], "q"),)
            elif len(regs) == 2 and all(isinstance(reg, int) for reg in regs):
                # QuantumCircuit with anonymous wires e.g. QuantumCircuit(2, 3)
                regs = (QuantumRegister(regs[0], "q"), ClassicalRegister(regs[1], "c"))
            else:
                raise CircuitError(
                    "QuantumCircuit parameters can be Registers or Integers."
                    " If Integers, up to 2 arguments. QuantumCircuit was called"
                    " with %s." % (regs,)
                )

        for register in regs:
            if isinstance(register, Register) and any(
                register.name == reg.name for reg in self.qregs + self.cregs
            ):
                raise CircuitError('register name "%s" already exists' % register.name)

            if isinstance(register, AncillaRegister):
                self._ancillas.extend(register)

            if isinstance(register, QuantumRegister):
                self.qregs.append(register)
                new_bits = [bit for bit in register if bit not in self._qubit_set]
                self._qubits.extend(new_bits)
                self._qubit_set.update(new_bits)
            elif isinstance(register, ClassicalRegister):
                self.cregs.append(register)
                new_bits = [bit for bit in register if bit not in self._clbit_set]
                self._clbits.extend(new_bits)
                self._clbit_set.update(new_bits)
            elif isinstance(register, list):
                self.add_bits(register)
            else:
                raise CircuitError("expected a register")

    def add_bits(self, bits):
        """Add Bits to the circuit."""
        duplicate_bits = set(self.qubits + self.clbits).intersection(bits)
        if duplicate_bits:
            raise CircuitError(
                "Attempted to add bits found already in circuit: " "{}".format(duplicate_bits)
            )

        for bit in bits:
            if isinstance(bit, AncillaQubit):
                self._ancillas.append(bit)

            if isinstance(bit, Qubit):
                self._qubits.append(bit)
                self._qubit_set.add(bit)
            elif isinstance(bit, Clbit):
                self._clbits.append(bit)
                self._clbit_set.add(bit)
            else:
                raise CircuitError(
                    "Expected an instance of Qubit, Clbit, or "
                    "AncillaQubit, but was passed {}".format(bit)
                )

    def _check_dups(self, qubits):
        """Raise exception if list of qubits contains duplicates."""
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise CircuitError("duplicate qubit arguments")

    def _check_qargs(self, qargs):
        """Raise exception if a qarg is not in this circuit or bad format."""
        if not all(isinstance(i, Qubit) for i in qargs):
            raise CircuitError("qarg is not a Qubit")
        if not set(qargs).issubset(self._qubit_set):
            raise CircuitError("qargs not in this circuit")

    def _check_cargs(self, cargs):
        """Raise exception if clbit is not in this circuit or bad format."""
        if not all(isinstance(i, Clbit) for i in cargs):
            raise CircuitError("carg is not a Clbit")
        if not set(cargs).issubset(self._clbit_set):
            raise CircuitError("cargs not in this circuit")

    def to_instruction(self, parameter_map=None, label=None):
        """Create an Instruction out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               instruction. If None, existing circuit parameters will also
               parameterize the instruction.
            label (str): Optional gate label.

        Returns:
            qiskit.circuit.Instruction: a composite instruction encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_instruction import circuit_to_instruction

        return circuit_to_instruction(self, parameter_map, label=label)

    def to_gate(self, parameter_map=None, label=None):
        """Create a Gate out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               gate. If None, existing circuit parameters will also
               parameterize the gate.
            label (str): Optional gate label.

        Returns:
            Gate: a composite gate encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_gate import circuit_to_gate

        return circuit_to_gate(self, parameter_map, label=label)

    def decompose(self):
        """Call a decomposition pass on this circuit,
        to decompose one level (shallow decompose).

        Returns:
            QuantumCircuit: a circuit one level decomposed
        """
        # pylint: disable=cyclic-import
        from qiskit.transpiler.passes.basis.decompose import Decompose
        from qiskit.converters.circuit_to_dag import circuit_to_dag
        from qiskit.converters.dag_to_circuit import dag_to_circuit

        pass_ = Decompose()
        decomposed_dag = pass_.run(circuit_to_dag(self))
        return dag_to_circuit(decomposed_dag)

    def _check_compatible_regs(self, rhs):
        """Raise exception if the circuits are defined on incompatible registers"""
        list1 = self.qregs + self.cregs
        list2 = rhs.qregs + rhs.cregs
        for element1 in list1:
            for element2 in list2:
                if element2.name == element1.name:
                    if element1 != element2:
                        raise CircuitError(
                            "circuits are not compatible:"
                            f" registers {element1} and {element2} not compatible"
                        )

    def qasm(self, formatted=False, filename=None, encoding=None):
        """Return OpenQASM string.

        Args:
            formatted (bool): Return formatted Qasm string.
            filename (str): Save Qasm to file with name 'filename'.
            encoding (str): Optionally specify the encoding to use for the
                output file if ``filename`` is specified. By default this is
                set to the system's default encoding (ie whatever
                ``locale.getpreferredencoding()`` returns) and can be set to
                any valid codec or alias from stdlib's
                `codec module <https://docs.python.org/3/library/codecs.html#standard-encodings>`__

        Returns:
            str: If formatted=False.

        Raises:
            MissingOptionalLibraryError: If pygments is not installed and ``formatted`` is
                ``True``.
            QasmError: If circuit has free parameters.
        """

        if self.num_parameters > 0:
            raise QasmError("Cannot represent circuits with unbound parameters in OpenQASM 2.")

        existing_gate_names = [
            "barrier",
            "measure",
            "reset",
            "u3",
            "u2",
            "u1",
            "cx",
            "id",
            "u0",
            "u",
            "p",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
            "sx",
            "sxdg",
            "cz",
            "cy",
            "swap",
            "ch",
            "ccx",
            "cswap",
            "crx",
            "cry",
            "crz",
            "cu1",
            "cp",
            "cu3",
            "csx",
            "cu",
            "rxx",
            "rzz",
            "rccx",
            "rc3x",
            "c3x",
            "c3sx",
            "c4x",
        ]

        existing_composite_circuits = []

        string_temp = self.header + "\n"
        string_temp += self.extension_lib + "\n"
        for register in self.qregs:
            string_temp += register.qasm() + "\n"
        for register in self.cregs:
            string_temp += register.qasm() + "\n"

        qreg_bits = {bit for reg in self.qregs for bit in reg}
        creg_bits = {bit for reg in self.cregs for bit in reg}
        regless_qubits = []
        regless_clbits = []

        if set(self.qubits) != qreg_bits:
            regless_qubits = [bit for bit in self.qubits if bit not in qreg_bits]
            string_temp += "qreg %s[%d];\n" % ("regless", len(regless_qubits))

        if set(self.clbits) != creg_bits:
            regless_clbits = [bit for bit in self.clbits if bit not in creg_bits]
            string_temp += "creg %s[%d];\n" % ("regless", len(regless_clbits))

        bit_labels = {
            bit: "%s[%d]" % (reg.name, idx)
            for reg in self.qregs + self.cregs
            for (idx, bit) in enumerate(reg)
        }

        bit_labels.update(
            {
                bit: "regless[%d]" % idx
                for reg in (regless_qubits, regless_clbits)
                for idx, bit in enumerate(reg)
            }
        )

        for instruction, qargs, cargs in self._data:
            if instruction.name == "measure":
                qubit = qargs[0]
                clbit = cargs[0]
                string_temp += "{} {} -> {};\n".format(
                    instruction.qasm(),
                    bit_labels[qubit],
                    bit_labels[clbit],
                )
            else:
                # decompose gate using definitions if they are not defined in OpenQASM2
                if (
                    instruction.name not in existing_gate_names
                    and instruction not in existing_composite_circuits
                ):
                    if instruction.name in [
                        instruction.name for instruction in existing_composite_circuits
                    ]:
                        # append instruction id to name to make it unique
                        instruction.name += f"_{id(instruction)}"

                    existing_composite_circuits.append(instruction)
                    _add_sub_instruction_to_existing_composite_circuits(
                        instruction, existing_gate_names, existing_composite_circuits
                    )

                # Insert qasm representation of the original instruction
                string_temp += "{} {};\n".format(
                    instruction.qasm(),
                    ",".join([bit_labels[j] for j in qargs + cargs]),
                )

        # insert gate definitions
        string_temp = _insert_composite_gate_definition_qasm(
            string_temp, existing_composite_circuits, self.extension_lib
        )

        if filename:
            with open(filename, "w+", encoding=encoding) as file:
                file.write(string_temp)
            file.close()

        if formatted:
            if not HAS_PYGMENTS:
                raise MissingOptionalLibraryError(
                    libname="pygments>2.4",
                    name="formatted QASM output",
                    pip_install="pip install pygments",
                )
            code = pygments.highlight(
                string_temp, OpenQASMLexer(), Terminal256Formatter(style=QasmTerminalStyle)
            )
            print(code)
            return None
        else:
            return string_temp

    def draw(
        self,
        output=None,
        scale=None,
        filename=None,
        style=None,
        interactive=False,
        plot_barriers=True,
        reverse_bits=False,
        justify=None,
        vertical_compression="medium",
        idle_wires=True,
        with_layout=True,
        fold=None,
        ax=None,
        initial_state=False,
        cregbundle=True,
    ):
        """Draw the quantum circuit. Use the output parameter to choose the drawing format:

        **text**: ASCII art TextDrawing that can be printed in the console.

        **matplotlib**: images with color rendered purely in Python.

        **latex**: high-quality images compiled via latex.

        **latex_source**: raw uncompiled latex output.

        Args:
            output (str): select the output method to use for drawing the circuit.
                Valid choices are ``text``, ``mpl``, ``latex``, ``latex_source``.
                By default the `text` drawer is used unless the user config file
                (usually ``~/.qiskit/settings.conf``) has an alternative backend set
                as the default. For example, ``circuit_drawer = latex``. If the output
                kwarg is set, that backend will always be used over the default in
                the user config file.
            scale (float): scale of image to draw (shrink if < 1.0). Only used by
                the `mpl`, `latex` and `latex_source` outputs. Defaults to 1.0.
            filename (str): file path to save image to. Defaults to None.
            style (dict or str): dictionary of style or file name of style json file.
                This option is only used by the `mpl` or `latex` output type.
                If `style` is a str, it is used as the path to a json file
                which contains a style dict. The file will be opened, parsed, and
                then any style elements in the dict will replace the default values
                in the input dict. A file to be loaded must end in ``.json``, but
                the name entered here can omit ``.json``. For example,
                ``style='iqx.json'`` or ``style='iqx'``.
                If `style` is a dict and the ``'name'`` key is set, that name
                will be used to load a json file, followed by loading the other
                items in the style dict. For example, ``style={'name': 'iqx'}``.
                If `style` is not a str and `name` is not a key in the style dict,
                then the default value from the user config file (usually
                ``~/.qiskit/settings.conf``) will be used, for example,
                ``circuit_mpl_style = iqx``.
                If none of these are set, the `default` style will be used.
                The search path for style json files can be specified in the user
                config, for example,
                ``circuit_mpl_style_path = /home/user/styles:/home/user``.
                See: :class:`~qiskit.visualization.qcstyle.DefaultStyle` for more
                information on the contents.
            interactive (bool): when set to true, show the circuit in a new window
                (for `mpl` this depends on the matplotlib backend being used
                supporting this). Note when used with either the `text` or the
                `latex_source` output type this has no effect and will be silently
                ignored. Defaults to False.
            reverse_bits (bool): when set to True, reverse the bit order inside
                registers for the output visualization. Defaults to False.
            plot_barriers (bool): enable/disable drawing barriers in the output
                circuit. Defaults to True.
            justify (string): options are ``left``, ``right`` or ``none``. If
                anything else is supplied, it defaults to left justified. It refers
                to where gates should be placed in the output circuit if there is
                an option. ``none`` results in each gate being placed in its own
                column.
            vertical_compression (string): ``high``, ``medium`` or ``low``. It
                merges the lines generated by the `text` output so the drawing
                will take less vertical room.  Default is ``medium``. Only used by
                the `text` output, will be silently ignored otherwise.
            idle_wires (bool): include idle wires (wires with no circuit elements)
                in output visualization. Default is True.
            with_layout (bool): include layout information, with labels on the
                physical layout. Default is True.
            fold (int): sets pagination. It can be disabled using -1. In `text`,
                sets the length of the lines. This is useful when the drawing does
                not fit in the console. If None (default), it will try to guess the
                console width using ``shutil.get_terminal_size()``. However, if
                running in jupyter, the default line length is set to 80 characters.
                In `mpl`, it is the number of (visual) layers before folding.
                Default is 25.
            ax (matplotlib.axes.Axes): Only used by the `mpl` backend. An optional
                Axes object to be used for the visualization output. If none is
                specified, a new matplotlib Figure will be created and used.
                Additionally, if specified there will be no returned Figure since
                it is redundant.
            initial_state (bool): optional. Adds ``|0>`` in the beginning of the wire.
                Default is False.
            cregbundle (bool): optional. If set True, bundle classical registers.
                Default is True.

        Returns:
            :class:`TextDrawing` or :class:`matplotlib.figure` or :class:`PIL.Image` or
            :class:`str`:

            * `TextDrawing` (output='text')
                A drawing that can be printed as ascii art.
            * `matplotlib.figure.Figure` (output='mpl')
                A matplotlib figure object for the circuit diagram.
            * `PIL.Image` (output='latex')
                An in-memory representation of the image of the circuit diagram.
            * `str` (output='latex_source')
                The LaTeX source code for visualizing the circuit diagram.

        Raises:
            VisualizationError: when an invalid output method is selected
            ImportError: when the output methods requires non-installed libraries.

        Example:
            .. jupyter-execute::

                from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
                from qiskit.tools.visualization import circuit_drawer
                q = QuantumRegister(1)
                c = ClassicalRegister(1)
                qc = QuantumCircuit(q, c)
                qc.h(q)
                qc.measure(q, c)
                qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
        """

        # pylint: disable=cyclic-import
        from qiskit.visualization import circuit_drawer

        return circuit_drawer(
            self,
            scale=scale,
            filename=filename,
            style=style,
            output=output,
            interactive=interactive,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            vertical_compression=vertical_compression,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            ax=ax,
            initial_state=initial_state,
            cregbundle=cregbundle,
        )

    def size(self):
        """Returns total number of gate operations in circuit.

        Returns:
            int: Total number of gate operations.
        """
        gate_ops = 0
        for instr, _, _ in self._data:
            if not instr._directive:
                gate_ops += 1
        return gate_ops

    def depth(self):
        """Return circuit depth (i.e., length of critical path).
        This does not include compiler or simulator directives
        such as 'barrier' or 'snapshot'.

        Returns:
            int: Depth of circuit.

        Notes:
            The circuit depth and the DAG depth need not be the
            same.
        """
        # Assign each bit in the circuit a unique integer
        # to index into op_stack.
        bit_indices = {bit: idx for idx, bit in enumerate(self.qubits + self.clbits)}

        # If no bits, return 0
        if not bit_indices:
            return 0

        # A list that holds the height of each qubit
        # and classical bit.
        op_stack = [0] * len(bit_indices)

        # Here we are playing a modified version of
        # Tetris where we stack gates, but multi-qubit
        # gates, or measurements have a block for each
        # qubit or cbit that are connected by a virtual
        # line so that they all stacked at the same depth.
        # Conditional gates act on all cbits in the register
        # they are conditioned on.
        # We treat barriers or snapshots different as
        # They are transpiler and simulator directives.
        # The max stack height is the circuit depth.
        for instr, qargs, cargs in self._data:
            levels = []
            reg_ints = []
            # If count then add one to stack heights
            count = True
            if instr._directive:
                count = False
            for ind, reg in enumerate(qargs + cargs):
                # Add to the stacks of the qubits and
                # cbits used in the gate.
                reg_ints.append(bit_indices[reg])
                if count:
                    levels.append(op_stack[reg_ints[ind]] + 1)
                else:
                    levels.append(op_stack[reg_ints[ind]])
            # Assuming here that there is no conditional
            # snapshots or barriers ever.
            if instr.condition:
                # Controls operate over all bits of a classical register
                # or over a single bit
                if isinstance(instr.condition[0], Clbit):
                    condition_bits = [instr.condition[0]]
                else:
                    condition_bits = instr.condition[0]
                for cbit in condition_bits:
                    idx = bit_indices[cbit]
                    if idx not in reg_ints:
                        reg_ints.append(idx)
                        levels.append(op_stack[idx] + 1)

            max_level = max(levels)
            for ind in reg_ints:
                op_stack[ind] = max_level

        return max(op_stack)

    def width(self):
        """Return number of qubits plus clbits in circuit.

        Returns:
            int: Width of circuit.

        """
        return len(self.qubits) + len(self.clbits)

    @property
    def num_qubits(self):
        """Return number of qubits."""
        return len(self.qubits)

    @property
    def num_ancillas(self):
        """Return the number of ancilla qubits."""
        return len(self.ancillas)

    @property
    def num_clbits(self):
        """Return number of classical bits."""
        return len(self.clbits)

    def count_ops(self):
        """Count each operation kind in the circuit.

        Returns:
            OrderedDict: a breakdown of how many operations of each kind, sorted by amount.
        """
        count_ops = {}
        for instr, _, _ in self._data:
            count_ops[instr.name] = count_ops.get(instr.name, 0) + 1
        return OrderedDict(sorted(count_ops.items(), key=lambda kv: kv[1], reverse=True))

    def num_nonlocal_gates(self):
        """Return number of non-local gates (i.e. involving 2+ qubits).

        Conditional nonlocal gates are also included.
        """
        multi_qubit_gates = 0
        for instr, _, _ in self._data:
            if instr.num_qubits > 1 and not instr._directive:
                multi_qubit_gates += 1
        return multi_qubit_gates

    def get_instructions(self, name):
        """Get instructions matching name.

        Args:
            name (str): The name of instruction to.

        Returns:
            list(tuple): list of (instruction, qargs, cargs).
        """
        return [match for match in self._data if match[0].name == name]

    def num_connected_components(self, unitary_only=False):
        """How many non-entangled subcircuits can the circuit be factored to.

        Args:
            unitary_only (bool): Compute only unitary part of graph.

        Returns:
            int: Number of connected components in circuit.
        """
        # Convert registers to ints (as done in depth).
        bits = self.qubits if unitary_only else (self.qubits + self.clbits)
        bit_indices = {bit: idx for idx, bit in enumerate(bits)}

        # Start with each qubit or cbit being its own subgraph.
        sub_graphs = [[bit] for bit in range(len(bit_indices))]

        num_sub_graphs = len(sub_graphs)

        # Here we are traversing the gates and looking to see
        # which of the sub_graphs the gate joins together.
        for instr, qargs, cargs in self._data:
            if unitary_only:
                args = qargs
                num_qargs = len(args)
            else:
                args = qargs + cargs
                num_qargs = len(args) + (1 if instr.condition else 0)

            if num_qargs >= 2 and not instr._directive:
                graphs_touched = []
                num_touched = 0
                # Controls necessarily join all the cbits in the
                # register that they use.
                if instr.condition and not unitary_only:
                    if isinstance(instr.condition[0], Clbit):
                        condition_bits = [instr.condition[0]]
                    else:
                        condition_bits = instr.condition[0]
                    for bit in condition_bits:
                        idx = bit_indices[bit]
                        for k in range(num_sub_graphs):
                            if idx in sub_graphs[k]:
                                graphs_touched.append(k)
                                break

                for item in args:
                    reg_int = bit_indices[item]
                    for k in range(num_sub_graphs):
                        if reg_int in sub_graphs[k]:
                            if k not in graphs_touched:
                                graphs_touched.append(k)
                                break

                graphs_touched = list(set(graphs_touched))
                num_touched = len(graphs_touched)

                # If the gate touches more than one subgraph
                # join those graphs together and return
                # reduced number of subgraphs
                if num_touched > 1:
                    connections = []
                    for idx in graphs_touched:
                        connections.extend(sub_graphs[idx])
                    _sub_graphs = []
                    for idx in range(num_sub_graphs):
                        if idx not in graphs_touched:
                            _sub_graphs.append(sub_graphs[idx])
                    _sub_graphs.append(connections)
                    sub_graphs = _sub_graphs
                    num_sub_graphs -= num_touched - 1
            # Cannot go lower than one so break
            if num_sub_graphs == 1:
                break
        return num_sub_graphs

    def num_unitary_factors(self):
        """Computes the number of tensor factors in the unitary
        (quantum) part of the circuit only.
        """
        return self.num_connected_components(unitary_only=True)

    def num_tensor_factors(self):
        """Computes the number of tensor factors in the unitary
        (quantum) part of the circuit only.

        Notes:
            This is here for backwards compatibility, and will be
            removed in a future release of Qiskit. You should call
            `num_unitary_factors` instead.
        """
        return self.num_unitary_factors()

    def copy(self, name=None):
        """Copy the circuit.

        Args:
          name (str): name to be given to the copied circuit. If None, then the name stays the same

        Returns:
          QuantumCircuit: a deepcopy of the current circuit, with the specified name
        """
        cpy = copy.copy(self)
        # copy registers correctly, in copy.copy they are only copied via reference
        cpy.qregs = self.qregs.copy()
        cpy.cregs = self.cregs.copy()
        cpy._qubits = self._qubits.copy()
        cpy._clbits = self._clbits.copy()
        cpy._qubit_set = self._qubit_set.copy()
        cpy._clbit_set = self._clbit_set.copy()

        instr_instances = {id(instr): instr for instr, _, __ in self._data}

        instr_copies = {id_: instr.copy() for id_, instr in instr_instances.items()}

        cpy._parameter_table = ParameterTable(
            {
                param: [
                    (instr_copies[id(instr)], param_index)
                    for instr, param_index in self._parameter_table[param]
                ]
                for param in self._parameter_table
            }
        )

        cpy._data = [
            (instr_copies[id(inst)], qargs.copy(), cargs.copy())
            for inst, qargs, cargs in self._data
        ]

        cpy._calibrations = copy.deepcopy(self._calibrations)
        cpy._metadata = copy.deepcopy(self._metadata)

        if name:
            cpy.name = name
        return cpy

    def _create_creg(self, length, name):
        """Creates a creg, checking if ClassicalRegister with same name exists"""
        if name in [creg.name for creg in self.cregs]:
            save_prefix = ClassicalRegister.prefix
            ClassicalRegister.prefix = name
            new_creg = ClassicalRegister(length)
            ClassicalRegister.prefix = save_prefix
        else:
            new_creg = ClassicalRegister(length, name)
        return new_creg

    def _create_qreg(self, length, name):
        """Creates a qreg, checking if QuantumRegister with same name exists"""
        if name in [qreg.name for qreg in self.qregs]:
            save_prefix = QuantumRegister.prefix
            QuantumRegister.prefix = name
            new_qreg = QuantumRegister(length)
            QuantumRegister.prefix = save_prefix
        else:
            new_qreg = QuantumRegister(length, name)
        return new_qreg

    def measure_active(self, inplace=True):
        """Adds measurement to all non-idle qubits. Creates a new ClassicalRegister with
        a size equal to the number of non-idle qubits being measured.

        Returns a new circuit with measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements when `inplace = False`.
        """
        from qiskit.converters.circuit_to_dag import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()
        dag = circuit_to_dag(circ)
        qubits_to_measure = [qubit for qubit in circ.qubits if qubit not in dag.idle_wires()]
        new_creg = circ._create_creg(len(qubits_to_measure), "measure")
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(qubits_to_measure, new_creg)

        if not inplace:
            return circ
        else:
            return None

    def measure_all(self, inplace=True):
        """Adds measurement to all qubits. Creates a new ClassicalRegister with a
        size equal to the number of qubits being measured.

        Returns a new circuit with measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements when `inplace = False`.
        """
        if inplace:
            circ = self
        else:
            circ = self.copy()

        new_creg = circ._create_creg(len(circ.qubits), "meas")
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(circ.qubits, new_creg)

        if not inplace:
            return circ
        else:
            return None

    def remove_final_measurements(self, inplace=True):
        """Removes final measurement on all qubits if they are present.
        Deletes the ClassicalRegister that was used to store the values from these measurements
        if it is idle.

        Returns a new circuit without measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements removed inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements removed when `inplace = False`.
        """
        # pylint: disable=cyclic-import
        from qiskit.transpiler.passes import RemoveFinalMeasurements
        from qiskit.converters import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()

        dag = circuit_to_dag(circ)
        remove_final_meas = RemoveFinalMeasurements()
        new_dag = remove_final_meas.run(dag)

        # Set circ cregs and instructions to match the new DAGCircuit's
        circ.data.clear()
        circ._parameter_table.clear()
        circ.cregs = list(new_dag.cregs.values())

        for node in new_dag.topological_op_nodes():
            # Get arguments for classical condition (if any)
            inst = node.op.copy()
            circ.append(inst, node.qargs, node.cargs)

        circ.clbits.clear()

        if not inplace:
            return circ
        else:
            return None

    @staticmethod
    def from_qasm_file(path):
        """Take in a QASM file and generate a QuantumCircuit object.

        Args:
          path (str): Path to the file for a QASM program
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM
        """
        qasm = Qasm(filename=path)
        return _circuit_from_qasm(qasm)

    @staticmethod
    def from_qasm_str(qasm_str):
        """Take in a QASM string and generate a QuantumCircuit object.

        Args:
          qasm_str (str): A QASM program string
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM
        """
        qasm = Qasm(data=qasm_str)
        return _circuit_from_qasm(qasm)

    @property
    def global_phase(self):
        """Return the global phase of the circuit in radians."""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle):
        """Set the phase of the circuit.

        Args:
            angle (float, ParameterExpression): radians
        """
        if isinstance(angle, ParameterExpression) and angle.parameters:
            self._global_phase = angle
        else:
            # Set the phase to the [0, 2π) interval
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * np.pi)

    @property
    def parameters(self):
        """Convenience function to get the parameters defined in the parameter table."""
        # parameters from gates
        if self._parameters is None:
            unsorted = self._unsorted_parameters()
            self._parameters = sorted(unsorted, key=functools.cmp_to_key(_compare_parameters))

        # return as parameter view, which implements the set and list interface
        return ParameterView(self._parameters)

    @property
    def num_parameters(self):
        """Convenience function to get the number of parameter objects in the circuit."""
        return len(self._unsorted_parameters())

    def _unsorted_parameters(self):
        """Efficiently get all parameters in the circuit, without any sorting overhead."""
        parameters = set(self._parameter_table)
        if isinstance(self.global_phase, ParameterExpression):
            parameters.update(self.global_phase.parameters)

        return parameters

    @deprecate_arguments({"param_dict": "parameters"})
    def assign_parameters(
        self, parameters, inplace=False, param_dict=None
    ):  # pylint: disable=unused-argument
        """Assign parameters to new parameters or values.

        The keys of the parameter dictionary must be Parameter instances in the current circuit. The
        values of the dictionary can either be numeric values or new parameter objects.
        The values can be assigned to the current circuit object or to a copy of it.

        Args:
            parameters (dict or iterable): Either a dictionary or iterable specifying the new
                parameter values. If a dict, it specifies the mapping from ``current_parameter`` to
                ``new_parameter``, where ``new_parameter`` can be a new parameter object or a
                numeric value. If an iterable, the elements are assigned to the existing parameters
                in the order they were inserted. You can call ``QuantumCircuit.parameters`` to check
                this order.
            inplace (bool): If False, a copy of the circuit with the bound parameters is
                returned. If True the circuit instance itself is modified.
            param_dict (dict): Deprecated, use ``parameters`` instead.

        Raises:
            CircuitError: If parameters is a dict and contains parameters not present in the
                circuit.
            ValueError: If parameters is a list/array and the length mismatches the number of free
                parameters in the circuit.

        Returns:
            Optional(QuantumCircuit): A copy of the circuit with bound parameters, if
            ``inplace`` is False, otherwise None.

        Examples:

            Create a parameterized circuit and assign the parameters in-place.

            .. jupyter-execute::

                from qiskit.circuit import QuantumCircuit, Parameter

                circuit = QuantumCircuit(2)
                params = [Parameter('A'), Parameter('B'), Parameter('C')]
                circuit.ry(params[0], 0)
                circuit.crx(params[1], 0, 1)

                print('Original circuit:')
                print(circuit.draw())

                circuit.assign_parameters({params[0]: params[2]}, inplace=True)

                print('Assigned in-place:')
                print(circuit.draw())

            Bind the values out-of-place and get a copy of the original circuit.

            .. jupyter-execute::

                from qiskit.circuit import QuantumCircuit, ParameterVector

                circuit = QuantumCircuit(2)
                params = ParameterVector('P', 2)
                circuit.ry(params[0], 0)
                circuit.crx(params[1], 0, 1)

                bound_circuit = circuit.assign_parameters({params[0]: 1, params[1]: 2})
                print('Bound circuit:')
                print(bound_circuit.draw())

                print('The original circuit is unchanged:')
                print(circuit.draw())

        """
        # replace in self or in a copy depending on the value of in_place
        if inplace:
            bound_circuit = self
        else:
            bound_circuit = self.copy()
            self._increment_instances()
            bound_circuit._name_update()

        if isinstance(parameters, dict):
            # unroll the parameter dictionary (needed if e.g. it contains a ParameterVector)
            unrolled_param_dict = self._unroll_param_dict(parameters)
            unsorted_parameters = self._unsorted_parameters()

            # check that all param_dict items are in the _parameter_table for this circuit
            params_not_in_circuit = [
                param_key
                for param_key in unrolled_param_dict
                if param_key not in unsorted_parameters
            ]
            if len(params_not_in_circuit) > 0:
                raise CircuitError(
                    "Cannot bind parameters ({}) not present in the circuit.".format(
                        ", ".join(map(str, params_not_in_circuit))
                    )
                )

            # replace the parameters with a new Parameter ("substitute") or numeric value ("bind")
            for parameter, value in unrolled_param_dict.items():
                bound_circuit._assign_parameter(parameter, value)
        else:
            if len(parameters) != self.num_parameters:
                raise ValueError(
                    "Mismatching number of values and parameters. For partial binding "
                    "please pass a dictionary of {parameter: value} pairs."
                )
            for i, value in enumerate(parameters):
                bound_circuit._assign_parameter(self.parameters[i], value)
        return None if inplace else bound_circuit

    @deprecate_arguments({"value_dict": "values"})
    def bind_parameters(self, values, value_dict=None):  # pylint: disable=unused-argument
        """Assign numeric parameters to values yielding a new circuit.

        To assign new Parameter objects or bind the values in-place, without yielding a new
        circuit, use the :meth:`assign_parameters` method.

        Args:
            values (dict or iterable): {parameter: value, ...} or [value1, value2, ...]
            value_dict (dict): Deprecated, use ``values`` instead.

        Raises:
            CircuitError: If values is a dict and contains parameters not present in the circuit.
            TypeError: If values contains a ParameterExpression.

        Returns:
            QuantumCircuit: copy of self with assignment substitution.
        """
        if isinstance(values, dict):
            if any(isinstance(value, ParameterExpression) for value in values.values()):
                raise TypeError(
                    "Found ParameterExpression in values; use assign_parameters() instead."
                )
            return self.assign_parameters(values)
        else:
            if any(isinstance(value, ParameterExpression) for value in values):
                raise TypeError(
                    "Found ParameterExpression in values; use assign_parameters() instead."
                )
            return self.assign_parameters(values)

    def _unroll_param_dict(self, value_dict):
        unrolled_value_dict = {}
        for (param, value) in value_dict.items():
            if isinstance(param, ParameterVector):
                if not len(param) == len(value):
                    raise CircuitError(
                        "ParameterVector {} has length {}, which "
                        "differs from value list {} of "
                        "len {}".format(param, len(param), value, len(value))
                    )
                unrolled_value_dict.update(zip(param, value))
            # pass anything else except number through. error checking is done in assign_parameter
            elif isinstance(param, (ParameterExpression, str)) or param is None:
                unrolled_value_dict[param] = value
        return unrolled_value_dict

    def _assign_parameter(self, parameter, value):
        """Update this circuit where instances of ``parameter`` are replaced by ``value``, which
        can be either a numeric value or a new parameter expression.

        Args:
            parameter (ParameterExpression): Parameter to be bound
            value (Union(ParameterExpression, float, int)): A numeric or parametric expression to
                replace instances of ``parameter``.
        """
        # parameter might be in global phase only
        if parameter in self._parameter_table.keys():
            for instr, param_index in self._parameter_table[parameter]:
                new_param = instr.params[param_index].assign(parameter, value)
                # if fully bound, validate
                if len(new_param.parameters) == 0:
                    instr.params[param_index] = instr.validate_parameter(new_param)
                else:
                    instr.params[param_index] = new_param

                self._rebind_definition(instr, parameter, value)

            if isinstance(value, ParameterExpression):
                entry = self._parameter_table.pop(parameter)
                for new_parameter in value.parameters:
                    if new_parameter in self._parameter_table:
                        self._parameter_table[new_parameter].extend(entry)
                    else:
                        self._parameter_table[new_parameter] = entry
            else:
                del self._parameter_table[parameter]  # clear evaluated expressions

        if (
            isinstance(self.global_phase, ParameterExpression)
            and parameter in self.global_phase.parameters
        ):
            self.global_phase = self.global_phase.assign(parameter, value)

        # clear parameter cache
        self._parameters = None
        self._assign_calibration_parameters(parameter, value)

    def _assign_calibration_parameters(self, parameter, value):
        """Update parameterized pulse gate calibrations, if there are any which contain
        ``parameter``. This updates the calibration mapping as well as the gate definition
        ``Schedule``s, which also may contain ``parameter``.
        """
        for cals in self.calibrations.values():
            for (qubit, cal_params), schedule in copy.copy(cals).items():
                if any(
                    isinstance(p, ParameterExpression) and parameter in p.parameters
                    for p in cal_params
                ):
                    del cals[(qubit, cal_params)]
                    new_cal_params = []
                    for p in cal_params:
                        if isinstance(p, ParameterExpression) and parameter in p.parameters:
                            new_param = p.assign(parameter, value)
                            if not new_param.parameters:
                                new_param = float(new_param)
                            new_cal_params.append(new_param)
                        else:
                            new_cal_params.append(p)
                    schedule.assign_parameters({parameter: value})
                    cals[(qubit, tuple(new_cal_params))] = schedule

    def _rebind_definition(self, instruction, parameter, value):
        if instruction._definition:
            for op, _, _ in instruction._definition:
                for idx, param in enumerate(op.params):
                    if isinstance(param, ParameterExpression) and parameter in param.parameters:
                        if isinstance(value, ParameterExpression):
                            op.params[idx] = param.subs({parameter: value})
                        else:
                            op.params[idx] = param.bind({parameter: value})
                        self._rebind_definition(op, parameter, value)

    def barrier(self, *qargs):
        """Apply :class:`~qiskit.circuit.Barrier`. If qargs is None, applies to all."""
        from .barrier import Barrier

        qubits = []

        if not qargs:  # None
            qubits.extend(self.qubits)

        for qarg in qargs:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, range):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        return self.append(Barrier(len(qubits)), qubits, [])

    def delay(self, duration, qarg=None, unit="dt"):
        """Apply :class:`~qiskit.circuit.Delay`. If qarg is None, applies to all qubits.
        When applying to multiple qubits, delays with the same duration will be created.

        Args:
            duration (int or float or ParameterExpression): duration of the delay.
            qarg (Object): qubit argument to apply this delay.
            unit (str): unit of the duration. Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
                Default is ``dt``, i.e. integer time unit depending on the target backend.

        Returns:
            qiskit.Instruction: the attached delay instruction.

        Raises:
            CircuitError: if arguments have bad format.
        """
        qubits = []
        if qarg is None:  # -> apply delays to all qubits
            for q in self.qubits:
                qubits.append(q)
        else:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, (range, tuple)):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        instructions = InstructionSet()
        for q in qubits:
            inst = (Delay(duration, unit), [q], [])
            self.append(*inst)
            instructions.add(*inst)
        return instructions

    def h(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.HGate`."""
        from .library.standard_gates.h import HGate

        return self.append(HGate(), [qubit], [])

    def ch(
        self,
        control_qubit,
        target_qubit,  # pylint: disable=invalid-name
        label=None,
        ctrl_state=None,
    ):
        """Apply :class:`~qiskit.circuit.library.CHGate`."""
        from .library.standard_gates.h import CHGate

        return self.append(
            CHGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def i(self, qubit):
        """Apply :class:`~qiskit.circuit.library.IGate`."""
        from .library.standard_gates.i import IGate

        return self.append(IGate(), [qubit], [])

    def id(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.IGate`."""
        return self.i(qubit)

    def ms(self, theta, qubits):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.MSGate`."""
        # pylint: disable=cyclic-import
        from .library.generalized_gates.gms import MSGate

        return self.append(MSGate(len(qubits), theta), qubits)

    def p(self, theta, qubit):
        """Apply :class:`~qiskit.circuit.library.PhaseGate`."""
        from .library.standard_gates.p import PhaseGate

        return self.append(PhaseGate(theta), [qubit], [])

    def cp(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CPhaseGate`."""
        from .library.standard_gates.p import CPhaseGate

        return self.append(
            CPhaseGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def mcp(self, lam, control_qubits, target_qubit):
        """Apply :class:`~qiskit.circuit.library.MCPhaseGate`."""
        from .library.standard_gates.p import MCPhaseGate

        num_ctrl_qubits = len(control_qubits)
        return self.append(
            MCPhaseGate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], []
        )

    def r(self, theta, phi, qubit):
        """Apply :class:`~qiskit.circuit.library.RGate`."""
        from .library.standard_gates.r import RGate

        return self.append(RGate(theta, phi), [qubit], [])

    def rv(self, vx, vy, vz, qubit):
        """Apply :class:`~qiskit.circuit.library.RVGate`."""
        from .library.generalized_gates.rv import RVGate

        return self.append(RVGate(vx, vy, vz), [qubit], [])

    def rccx(self, control_qubit1, control_qubit2, target_qubit):
        """Apply :class:`~qiskit.circuit.library.RCCXGate`."""
        from .library.standard_gates.x import RCCXGate

        return self.append(RCCXGate(), [control_qubit1, control_qubit2, target_qubit], [])

    def rcccx(self, control_qubit1, control_qubit2, control_qubit3, target_qubit):
        """Apply :class:`~qiskit.circuit.library.RC3XGate`."""
        from .library.standard_gates.x import RC3XGate

        return self.append(
            RC3XGate(), [control_qubit1, control_qubit2, control_qubit3, target_qubit], []
        )

    def rx(self, theta, qubit, label=None):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.RXGate`."""
        from .library.standard_gates.rx import RXGate

        return self.append(RXGate(theta, label=label), [qubit], [])

    def crx(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CRXGate`."""
        from .library.standard_gates.rx import CRXGate

        return self.append(
            CRXGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def rxx(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RXXGate`."""
        from .library.standard_gates.rxx import RXXGate

        return self.append(RXXGate(theta), [qubit1, qubit2], [])

    def ry(self, theta, qubit, label=None):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.RYGate`."""
        from .library.standard_gates.ry import RYGate

        return self.append(RYGate(theta, label=label), [qubit], [])

    def cry(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CRYGate`."""
        from .library.standard_gates.ry import CRYGate

        return self.append(
            CRYGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def ryy(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RYYGate`."""
        from .library.standard_gates.ryy import RYYGate

        return self.append(RYYGate(theta), [qubit1, qubit2], [])

    def rz(self, phi, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.RZGate`."""
        from .library.standard_gates.rz import RZGate

        return self.append(RZGate(phi), [qubit], [])

    def crz(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CRZGate`."""
        from .library.standard_gates.rz import CRZGate

        return self.append(
            CRZGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def rzx(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RZXGate`."""
        from .library.standard_gates.rzx import RZXGate

        return self.append(RZXGate(theta), [qubit1, qubit2], [])

    def rzz(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RZZGate`."""
        from .library.standard_gates.rzz import RZZGate

        return self.append(RZZGate(theta), [qubit1, qubit2], [])

    def ecr(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.ECRGate`."""
        from .library.standard_gates.ecr import ECRGate

        return self.append(ECRGate(), [qubit1, qubit2], [])

    def s(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.SGate`."""
        from .library.standard_gates.s import SGate

        return self.append(SGate(), [qubit], [])

    def sdg(self, qubit):
        """Apply :class:`~qiskit.circuit.library.SdgGate`."""
        from .library.standard_gates.s import SdgGate

        return self.append(SdgGate(), [qubit], [])

    def swap(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.SwapGate`."""
        from .library.standard_gates.swap import SwapGate

        return self.append(SwapGate(), [qubit1, qubit2], [])

    def iswap(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.iSwapGate`."""
        from .library.standard_gates.iswap import iSwapGate

        return self.append(iSwapGate(), [qubit1, qubit2], [])

    def cswap(self, control_qubit, target_qubit1, target_qubit2, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CSwapGate`."""
        from .library.standard_gates.swap import CSwapGate

        return self.append(
            CSwapGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit1, target_qubit2],
            [],
        )

    def fredkin(self, control_qubit, target_qubit1, target_qubit2):
        """Apply :class:`~qiskit.circuit.library.CSwapGate`."""
        return self.cswap(control_qubit, target_qubit1, target_qubit2)

    def sx(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.SXGate`."""
        from .library.standard_gates.sx import SXGate

        return self.append(SXGate(), [qubit], [])

    def sxdg(self, qubit):
        """Apply :class:`~qiskit.circuit.library.SXdgGate`."""
        from .library.standard_gates.sx import SXdgGate

        return self.append(SXdgGate(), [qubit], [])

    def csx(self, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CSXGate`."""
        from .library.standard_gates.sx import CSXGate

        return self.append(
            CSXGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    def t(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.TGate`."""
        from .library.standard_gates.t import TGate

        return self.append(TGate(), [qubit], [])

    def tdg(self, qubit):
        """Apply :class:`~qiskit.circuit.library.TdgGate`."""
        from .library.standard_gates.t import TdgGate

        return self.append(TdgGate(), [qubit], [])

    def u(self, theta, phi, lam, qubit):
        """Apply :class:`~qiskit.circuit.library.UGate`."""
        from .library.standard_gates.u import UGate

        return self.append(UGate(theta, phi, lam), [qubit], [])

    def cu(self, theta, phi, lam, gamma, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CUGate`."""
        from .library.standard_gates.u import CUGate

        return self.append(
            CUGate(theta, phi, lam, gamma, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    @deprecate_function(
        "The QuantumCircuit.u1 method is deprecated as of "
        "0.16.0. It will be removed no earlier than 3 months "
        "after the release date. You should use the "
        "QuantumCircuit.p method instead, which acts "
        "identically."
    )
    def u1(self, theta, qubit):
        """Apply :class:`~qiskit.circuit.library.U1Gate`."""
        from .library.standard_gates.u1 import U1Gate

        return self.append(U1Gate(theta), [qubit], [])

    @deprecate_function(
        "The QuantumCircuit.cu1 method is deprecated as of "
        "0.16.0. It will be removed no earlier than 3 months "
        "after the release date. You should use the "
        "QuantumCircuit.cp method instead, which acts "
        "identically."
    )
    def cu1(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CU1Gate`."""
        from .library.standard_gates.u1 import CU1Gate

        return self.append(
            CU1Gate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    @deprecate_function(
        "The QuantumCircuit.mcu1 method is deprecated as of "
        "0.16.0. It will be removed no earlier than 3 months "
        "after the release date. You should use the "
        "QuantumCircuit.mcp method instead, which acts "
        "identically."
    )
    def mcu1(self, lam, control_qubits, target_qubit):
        """Apply :class:`~qiskit.circuit.library.MCU1Gate`."""
        from .library.standard_gates.u1 import MCU1Gate

        num_ctrl_qubits = len(control_qubits)
        return self.append(MCU1Gate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], [])

    @deprecate_function(
        "The QuantumCircuit.u2 method is deprecated as of "
        "0.16.0. It will be removed no earlier than 3 months "
        "after the release date. You can use the general 1-"
        "qubit gate QuantumCircuit.u instead: u2(φ,λ) = "
        "u(π/2, φ, λ). Alternatively, you can decompose it in"
        "terms of QuantumCircuit.p and QuantumCircuit.sx: "
        "u2(φ,λ) = p(π/2+φ) sx p(λ-π/2) (1 pulse on hardware)."
    )
    def u2(self, phi, lam, qubit):
        """Apply :class:`~qiskit.circuit.library.U2Gate`."""
        from .library.standard_gates.u2 import U2Gate

        return self.append(U2Gate(phi, lam), [qubit], [])

    @deprecate_function(
        "The QuantumCircuit.u3 method is deprecated as of 0.16.0. It will be "
        "removed no earlier than 3 months after the release date. You should use "
        "QuantumCircuit.u instead, which acts identically. Alternatively, you can "
        "decompose u3 in terms of QuantumCircuit.p and QuantumCircuit.sx: "
        "u3(ϴ,φ,λ) = p(φ+π) sx p(ϴ+π) sx p(λ) (2 pulses on hardware)."
    )
    def u3(self, theta, phi, lam, qubit):
        """Apply :class:`~qiskit.circuit.library.U3Gate`."""
        from .library.standard_gates.u3 import U3Gate

        return self.append(U3Gate(theta, phi, lam), [qubit], [])

    @deprecate_function(
        "The QuantumCircuit.cu3 method is deprecated as of 0.16.0. It will be "
        "removed no earlier than 3 months after the release date. You should "
        "use the QuantumCircuit.cu method instead, where "
        "cu3(ϴ,φ,λ) = cu(ϴ,φ,λ,0)."
    )
    def cu3(self, theta, phi, lam, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CU3Gate`."""
        from .library.standard_gates.u3 import CU3Gate

        return self.append(
            CU3Gate(theta, phi, lam, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    def x(self, qubit, label=None):
        """Apply :class:`~qiskit.circuit.library.XGate`."""
        from .library.standard_gates.x import XGate

        return self.append(XGate(label=label), [qubit], [])

    def cx(
        self,
        control_qubit,
        target_qubit,  # pylint: disable=invalid-name
        label=None,
        ctrl_state=None,
    ):
        """Apply :class:`~qiskit.circuit.library.CXGate`."""
        from .library.standard_gates.x import CXGate

        return self.append(
            CXGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def cnot(self, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CXGate`."""
        self.cx(control_qubit, target_qubit, label, ctrl_state)

    def dcx(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.DCXGate`."""
        from .library.standard_gates.dcx import DCXGate

        return self.append(DCXGate(), [qubit1, qubit2], [])

    def ccx(
        self,
        control_qubit1,
        control_qubit2,
        target_qubit,
        ctrl_state=None,
    ):
        """Apply :class:`~qiskit.circuit.library.CCXGate`."""
        from .library.standard_gates.x import CCXGate

        return self.append(
            CCXGate(ctrl_state=ctrl_state),
            [control_qubit1, control_qubit2, target_qubit],
            [],
        )

    def toffoli(self, control_qubit1, control_qubit2, target_qubit):
        """Apply :class:`~qiskit.circuit.library.CCXGate`."""
        self.ccx(control_qubit1, control_qubit2, target_qubit)

    def mcx(self, control_qubits, target_qubit, ancilla_qubits=None, mode="noancilla"):
        """Apply :class:`~qiskit.circuit.library.MCXGate`.

        The multi-cX gate can be implemented using different techniques, which use different numbers
        of ancilla qubits and have varying circuit depth. These modes are:
        - 'noancilla': Requires 0 ancilla qubits.
        - 'recursion': Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.
        - 'v-chain': Requires 2 less ancillas than the number of control qubits.
        - 'v-chain-dirty': Same as for the clean ancillas (but the circuit will be longer).
        """
        from .library.standard_gates.x import MCXGrayCode, MCXRecursive, MCXVChain

        num_ctrl_qubits = len(control_qubits)

        available_implementations = {
            "noancilla": MCXGrayCode(num_ctrl_qubits),
            "recursion": MCXRecursive(num_ctrl_qubits),
            "v-chain": MCXVChain(num_ctrl_qubits, False),
            "v-chain-dirty": MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
            # outdated, previous names
            "advanced": MCXRecursive(num_ctrl_qubits),
            "basic": MCXVChain(num_ctrl_qubits, dirty_ancillas=False),
            "basic-dirty-ancilla": MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
        }

        # check ancilla input
        if ancilla_qubits:
            _ = self.qbit_argument_conversion(ancilla_qubits)

        try:
            gate = available_implementations[mode]
        except KeyError as ex:
            all_modes = list(available_implementations.keys())
            raise ValueError(
                f"Unsupported mode ({mode}) selected, choose one of {all_modes}"
            ) from ex

        if hasattr(gate, "num_ancilla_qubits") and gate.num_ancilla_qubits > 0:
            required = gate.num_ancilla_qubits
            if ancilla_qubits is None:
                raise AttributeError(f"No ancillas provided, but {required} are needed!")

            # convert ancilla qubits to a list if they were passed as int or qubit
            if not hasattr(ancilla_qubits, "__len__"):
                ancilla_qubits = [ancilla_qubits]

            if len(ancilla_qubits) < required:
                actually = len(ancilla_qubits)
                raise ValueError(f"At least {required} ancillas required, but {actually} given.")
            # size down if too many ancillas were provided
            ancilla_qubits = ancilla_qubits[:required]
        else:
            ancilla_qubits = []

        return self.append(gate, control_qubits[:] + [target_qubit] + ancilla_qubits[:], [])

    def mct(self, control_qubits, target_qubit, ancilla_qubits=None, mode="noancilla"):
        """Apply :class:`~qiskit.circuit.library.MCXGate`."""
        return self.mcx(control_qubits, target_qubit, ancilla_qubits, mode)

    def y(self, qubit):
        """Apply :class:`~qiskit.circuit.library.YGate`."""
        from .library.standard_gates.y import YGate

        return self.append(YGate(), [qubit], [])

    def cy(
        self,
        control_qubit,
        target_qubit,  # pylint: disable=invalid-name
        label=None,
        ctrl_state=None,
    ):
        """Apply :class:`~qiskit.circuit.library.CYGate`."""
        from .library.standard_gates.y import CYGate

        return self.append(
            CYGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def z(self, qubit):
        """Apply :class:`~qiskit.circuit.library.ZGate`."""
        from .library.standard_gates.z import ZGate

        return self.append(ZGate(), [qubit], [])

    def cz(self, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CZGate`."""
        from .library.standard_gates.z import CZGate

        return self.append(
            CZGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def pauli(self, pauli_string, qubits):
        """Apply :class:`~qiskit.circuit.library.PauliGate`."""
        from qiskit.circuit.library.generalized_gates.pauli import PauliGate

        return self.append(PauliGate(pauli_string), qubits, [])

    def add_calibration(self, gate, qubits, schedule, params=None):
        """Register a low-level, custom pulse definition for the given gate.

        Args:
            gate (Union[Gate, str]): Gate information.
            qubits (Union[int, Tuple[int]]): List of qubits to be measured.
            schedule (Schedule): Schedule information.
            params (Optional[List[Union[float, Parameter]]]): A list of parameters.

        Raises:
            Exception: if the gate is of type string and params is None.
        """
        if isinstance(gate, Gate):
            self._calibrations[gate.name][(tuple(qubits), tuple(gate.params))] = schedule
        else:
            self._calibrations[gate][(tuple(qubits), tuple(params or []))] = schedule

    # Functions only for scheduled circuits
    def qubit_duration(self, *qubits: Union[Qubit, int]) -> float:
        """Return the duration between the start and stop time of the first and last instructions,
        excluding delays, over the supplied qubits. Its time unit is ``self.unit``.

        Args:
            *qubits: Qubits within ``self`` to include.

        Returns:
            Return the duration between the first start and last stop time of non-delay instructions
        """
        return self.qubit_stop_time(*qubits) - self.qubit_start_time(*qubits)

    def qubit_start_time(self, *qubits: Union[Qubit, int]) -> float:
        """Return the start time of the first instruction, excluding delays,
        over the supplied qubits. Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the start time of the first instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if self.duration is None:
            # circuit has only delays, this is kind of scheduled
            for inst, _, _ in self.data:
                if not isinstance(inst, Delay):
                    raise CircuitError(
                        "qubit_start_time undefined. " "Circuit must be scheduled first."
                    )
            return 0

        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]

        starts = {q: 0 for q in qubits}
        dones = {q: False for q in qubits}
        for inst, qargs, _ in self.data:
            for q in qubits:
                if q in qargs:
                    if isinstance(inst, Delay):
                        if not dones[q]:
                            starts[q] += inst.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):  # all done
                return min(start for start in starts.values())

        return 0  # If there are no instructions over bits

    def qubit_stop_time(self, *qubits: Union[Qubit, int]) -> float:
        """Return the stop time of the last instruction, excluding delays, over the supplied qubits.
        Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the stop time of the last instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if self.duration is None:
            # circuit has only delays, this is kind of scheduled
            for inst, _, _ in self.data:
                if not isinstance(inst, Delay):
                    raise CircuitError(
                        "qubit_stop_time undefined. " "Circuit must be scheduled first."
                    )
            return 0

        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]

        stops = {q: self.duration for q in qubits}
        dones = {q: False for q in qubits}
        for inst, qargs, _ in reversed(self.data):
            for q in qubits:
                if q in qargs:
                    if isinstance(inst, Delay):
                        if not dones[q]:
                            stops[q] -= inst.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):  # all done
                return max(stop for stop in stops.values())

        return 0  # If there are no instructions over bits


def _circuit_from_qasm(qasm):
    # pylint: disable=cyclic-import
    from qiskit.converters import ast_to_dag
    from qiskit.converters import dag_to_circuit

    ast = qasm.parse()
    dag = ast_to_dag(ast)
    return dag_to_circuit(dag)


def _standard_compare(value1, value2):
    if value1 < value2:
        return -1
    if value1 > value2:
        return 1
    return 0


def _compare_parameters(param1, param2):
    if isinstance(param1, ParameterVectorElement) and isinstance(param2, ParameterVectorElement):
        # if they belong to a vector with the same name, sort by index
        if param1.vector.name == param2.vector.name:
            return _standard_compare(param1.index, param2.index)

    # else sort by name
    return _standard_compare(param1.name, param2.name)


def _add_sub_instruction_to_existing_composite_circuits(
    instruction, existing_gate_names, existing_composite_circuits
):
    """Recursively add undefined sub-instructions in the definition of the given
    instruction to existing_composite_circuit list.
    """
    for sub_instruction, _, _ in instruction.definition:
        if (
            sub_instruction.name not in existing_gate_names
            and sub_instruction not in existing_composite_circuits
        ):
            existing_composite_circuits.insert(0, sub_instruction)
            _add_sub_instruction_to_existing_composite_circuits(
                sub_instruction, existing_gate_names, existing_composite_circuits
            )


def _get_composite_circuit_qasm_from_instruction(instruction):
    """Returns OpenQASM string composite circuit given an instruction.
    The given instruction should be the result of composite_circuit.to_instruction()."""

    if instruction.definition is None:
        raise ValueError(f'Instruction "{instruction.name}" is not defined.')

    gate_parameters = ",".join(["param%i" % num for num in range(len(instruction.params))])
    qubit_parameters = ",".join(["q%i" % num for num in range(instruction.num_qubits)])
    composite_circuit_gates = ""

    definition = instruction.definition
    definition_bit_labels = {
        bit: idx for bits in (definition.qubits, definition.clbits) for idx, bit in enumerate(bits)
    }
    for sub_instruction, qargs, _ in definition:
        gate_qargs = ",".join(
            ["q%i" % index for index in [definition_bit_labels[qubit] for qubit in qargs]]
        )
        composite_circuit_gates += "%s %s; " % (sub_instruction.qasm(), gate_qargs)

    if composite_circuit_gates:
        composite_circuit_gates = composite_circuit_gates.rstrip(" ")

    if gate_parameters:
        qasm_string = "gate %s(%s) %s { %s }" % (
            instruction.name,
            gate_parameters,
            qubit_parameters,
            composite_circuit_gates,
        )
    else:
        qasm_string = "gate %s %s { %s }" % (
            instruction.name,
            qubit_parameters,
            composite_circuit_gates,
        )

    return qasm_string


def _insert_composite_gate_definition_qasm(string_temp, existing_composite_circuits, extension_lib):
    """Insert composite gate definition QASM code right after extension library in the header"""

    gate_definition_string = ""

    # Generate gate definition string
    for instruction in existing_composite_circuits:
        if hasattr(instruction, "_qasm_definition"):
            qasm_string = instruction._qasm_definition
        else:
            qasm_string = _get_composite_circuit_qasm_from_instruction(instruction)
        gate_definition_string += "\n" + qasm_string

    string_temp = string_temp.replace(
        extension_lib, "%s%s" % (extension_lib, gate_definition_string)
    )
    return string_temp
