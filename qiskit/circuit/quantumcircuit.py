# -*- coding: utf-8 -*-

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

"""Quantum circuit object."""

import copy
import itertools
import sys
import warnings
import numbers
import multiprocessing as mp
from collections import OrderedDict
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.util import is_main_process
from qiskit.util import deprecate_arguments
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.qasm.qasm import Qasm
from qiskit.circuit.exceptions import CircuitError
from .parameterexpression import ParameterExpression
from .quantumregister import QuantumRegister, Qubit, AncillaRegister
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterTable
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData

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
        regs: list(:class:`Register`) or list(``int``) The registers to be
            included in the circuit.

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


        name (str): the name of the quantum circuit. If not set, an
            automatically generated string will be assigned.
        global_phase (float): The global phase of the circuit in radians.

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

        Construct a 4-qubit Berstein-Vazirani circuit using registers.

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
    prefix = 'circuit'

    # Class variable OPENQASM header
    header = "OPENQASM 2.0;"
    extension_lib = "include \"qelib1.inc\";"

    def __init__(self, *regs, name=None, global_phase=0):
        if any([not isinstance(reg, (QuantumRegister, ClassicalRegister)) for reg in regs]):
            try:
                regs = tuple(int(reg) for reg in regs)
            except Exception:
                raise CircuitError("Circuit args must be Registers or be castable to an int" +
                                   "(%s '%s' was provided)"
                                   % ([type(reg).__name__ for reg in regs], regs))
        if name is None:
            name = self.cls_prefix() + str(self.cls_instances())
            if sys.platform != "win32" and not is_main_process():
                name += '-{}'.format(mp.current_process().pid)
        self._increment_instances()

        if not isinstance(name, str):
            raise CircuitError("The circuit name should be a string "
                               "(or None to auto-generate a name).")

        self.name = name

        # Data contains a list of instructions and their contexts,
        # in the order they were applied.
        self._data = []

        # This is a map of registers bound to this circuit, by name.
        self.qregs = []
        self.cregs = []
        self._qubits = []
        self._clbits = []
        self._ancillas = []
        self.add_register(*regs)

        # Parameter table tracks instructions with variable parameters.
        self._parameter_table = ParameterTable()

        self._layout = None
        self._global_phase = 0
        self.global_phase = global_phase

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

    def __str__(self):
        return str(self.draw(output='text'))

    def __eq__(self, other):
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

    def has_register(self, register):
        """
        Test if this circuit has the register r.

        Args:
            register (Register): a quantum or classical register.

        Returns:
            bool: True if the register is contained in this circuit.
        """
        has_reg = False
        if (isinstance(register, QuantumRegister) and
                register in self.qregs):
            has_reg = True
        elif (isinstance(register, ClassicalRegister) and
              register in self.cregs):
            has_reg = True
        return has_reg

    def mirror(self):
        """DEPRECATED: use circuit.reverse_ops().

        Returns:
            QuantumCircuit: the reversed circuit.
        """
        warnings.warn('circuit.mirror() is deprecated. Use circuit.reverse_ops() to '
                      'reverse the order of gates.', DeprecationWarning)
        return self.reverse_ops()

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
        reverse_circ = QuantumCircuit(*self.qregs, *self.cregs,
                                      name=self.name + '_reverse')

        for inst, qargs, cargs in reversed(self.data):
            reverse_circ._append(inst.reverse_ops(), qargs, cargs)
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
        circ = QuantumCircuit(*reversed(self.qregs), *reversed(self.cregs),
                              name=self.name)
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
        inverse_circ = QuantumCircuit(*self.qregs, *self.cregs,
                                      name=self.name + '_dg', global_phase=-self.global_phase)

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
        repeated_circ = QuantumCircuit(*self.qregs, *self.cregs,
                                       name=self.name + '**{}'.format(reps),
                                       global_phase=reps * self.global_phase)

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
        if len(self.parameters) > 0:
            raise CircuitError('Cannot raise a parameterized circuit to a non-positive power '
                               'or matrix-power, please bind the free parameters: '
                               '{}'.format(self.parameters))

        try:
            gate = self.to_gate()
        except QiskitError:
            raise CircuitError('The circuit contains non-unitary operations and cannot be '
                               'controlled. Note that no qiskit.circuit.Instruction objects may '
                               'be in the circuit for this operation.')

        power_circuit = QuantumCircuit(*self.qregs, *self.cregs)
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
        except QiskitError:
            raise CircuitError('The circuit contains non-unitary operations and cannot be '
                               'controlled. Note that no qiskit.circuit.Instruction objects may '
                               'be in the circuit for this operation.')

        controlled_gate = gate.control(num_ctrl_qubits, label, ctrl_state)
        control_qreg = QuantumRegister(num_ctrl_qubits)
        controlled_circ = QuantumCircuit(control_qreg, *self.qregs,
                                         name='c_{}'.format(self.name))
        controlled_circ.append(controlled_gate, controlled_circ.qubits)

        return controlled_circ

    def combine(self, rhs):
        """Append rhs to self if self contains compatible registers.

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
        return circuit

    def extend(self, rhs):
        """Append QuantumCircuit to the right hand side if it contains compatible registers.

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
        for element in rhs.cregs:
            if element not in self.cregs:
                self.cregs.append(element)
                self._clbits += element[:]

        # Copy the circuit data if rhs and self are the same, otherwise the data of rhs is
        # appended to both self and rhs resulting in an infinite loop
        data = rhs.data.copy() if rhs is self else rhs.data

        # Add new gates
        for instruction_context in data:
            self._append(*instruction_context)
        self.global_phase += rhs.global_phase
        return self

    def compose(self, other, qubits=None, clbits=None, front=False, inplace=False):
        """Compose circuit with ``other`` circuit or instruction, optionally permuting wires.

        ``other`` can be narrower or of equal width to ``self``.

        Args:
            other (qiskit.circuit.Instruction or QuantumCircuit or BaseOperator):
                (sub)circuit to compose onto self.
            qubits (list[Qubit|int]): qubits of self to compose onto.
            clbits (list[Clbit|int]): clbits of self to compose onto.
            front (bool): If True, front composition will be performed (not implemented yet).
            inplace (bool): If True, modify the object. Otherwise return composed circuit.

        Returns:
            QuantumCircuit: the composed circuit (returns None if inplace==True).

        Raises:
            CircuitError: if composing on the front.
            QiskitError: if ``other`` is wider or there are duplicate edge mappings.

        Examples:

            >>> lhs.compose(rhs, qubits=[3, 2], inplace=True)

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

        if not isinstance(other, QuantumCircuit):
            if front:
                dest.data.insert(0, (other, qubits, clbits))
            else:
                dest.append(other, qargs=qubits, cargs=clbits)

            if inplace:
                return None
            return dest

        instrs = other.data

        if other.num_qubits > self.num_qubits or \
           other.num_clbits > self.num_clbits:
            raise CircuitError("Trying to compose with another QuantumCircuit "
                               "which has more 'in' edges.")

        # number of qubits and clbits must match number in circuit or None
        identity_qubit_map = dict(zip(other.qubits, self.qubits))
        identity_clbit_map = dict(zip(other.clbits, self.clbits))

        if qubits is None:
            qubit_map = identity_qubit_map
        elif len(qubits) != len(other.qubits):
            raise CircuitError("Number of items in qubits parameter does not"
                               " match number of qubits in the circuit.")
        else:
            qubit_map = {other.qubits[i]: (self.qubits[q] if isinstance(q, int) else q)
                         for i, q in enumerate(qubits)}
        if clbits is None:
            clbit_map = identity_clbit_map
        elif len(clbits) != len(other.clbits):
            raise CircuitError("Number of items in clbits parameter does not"
                               " match number of clbits in the circuit.")
        else:
            clbit_map = {other.clbits[i]: (self.clbits[c] if isinstance(c, int) else c)
                         for i, c in enumerate(clbits)}

        edge_map = {**qubit_map, **clbit_map} or {**identity_qubit_map, **identity_clbit_map}

        mapped_instrs = []
        for instr, qargs, cargs in instrs:
            n_qargs = [edge_map[qarg] for qarg in qargs]
            n_cargs = [edge_map[carg] for carg in cargs]
            n_instr = instr.copy()

            if instr.condition is not None:
                from qiskit.dagcircuit import DAGCircuit  # pylint: disable=cyclic-import
                n_instr.condition = DAGCircuit._map_condition(edge_map, instr.condition)

            mapped_instrs.append((n_instr, n_qargs, n_cargs))

        if front:
            dest._data = mapped_instrs + dest._data
        else:
            dest._data += mapped_instrs

        for instr, _, _ in mapped_instrs:
            dest._update_parameter_table(instr)

        dest.global_phase += other.global_phase

        if inplace:
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

    def __add__(self, rhs):
        """Overload + to implement self.combine."""
        return self.combine(rhs)

    def __iadd__(self, rhs):
        """Overload += to implement self.extend."""
        return self.extend(rhs)

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
            elif isinstance(bit_representation, list) and \
                    all(isinstance(bit, Bit) for bit in bit_representation):
                # circuit.h([qr[0], qr[1]]) -> circuit.h([qr[0], qr[1]])
                ret = bit_representation
            elif isinstance(QuantumCircuit.cast(bit_representation, list), (range, list)):
                # circuit.h([0, 1])     -> circuit.h([qr[0], qr[1]])
                # circuit.h(range(0,2)) -> circuit.h([qr[0], qr[1]])
                # circuit.h([qr[0],1])  -> circuit.h([qr[0], qr[1]])
                ret = [index if isinstance(index, Bit) else in_array[
                    index] for index in bit_representation]
            else:
                raise CircuitError('Not able to expand a %s (%s)' % (bit_representation,
                                                                     type(bit_representation)))
        except IndexError:
            raise CircuitError('Index out of range.')
        except TypeError:
            raise CircuitError('Type error handling %s (%s)' % (bit_representation,
                                                                type(bit_representation)))
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
        if not isinstance(instruction, Instruction) and not hasattr(instruction, 'to_instruction'):
            if issubclass(instruction, Instruction):
                raise CircuitError('Object is a subclass of Instruction, please add () to '
                                   'pass an instance of this object.')

            raise CircuitError('Object to append must be an Instruction or '
                               'have a to_instruction() method.')
        if not isinstance(instruction, Instruction) and hasattr(instruction, "to_instruction"):
            instruction = instruction.to_instruction()

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
            raise CircuitError('object is not an Instruction.')

        # do some compatibility checks
        self._check_dups(qargs)
        self._check_qargs(qargs)
        self._check_cargs(cargs)

        # add the instruction onto the given wires
        instruction_context = instruction, qargs, cargs
        self._data.append(instruction_context)

        self._update_parameter_table(instruction)

        return instruction

    def _update_parameter_table(self, instruction):
        for param_index, param in enumerate(instruction.params):
            if isinstance(param, ParameterExpression):
                current_parameters = self._parameter_table

                for parameter in param.parameters:
                    if parameter in current_parameters:
                        if not self._check_dup_param_spec(self._parameter_table[parameter],
                                                          instruction, param_index):
                            self._parameter_table[parameter].append((instruction, param_index))
                    else:
                        if parameter.name in self._parameter_table.get_names():
                            raise CircuitError(
                                'Name conflict on adding parameter: {}'.format(parameter.name))
                        self._parameter_table[parameter] = [(instruction, param_index)]

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

        if any([isinstance(reg, int) for reg in regs]):
            # QuantumCircuit defined without registers
            if len(regs) == 1 and isinstance(regs[0], int):
                # QuantumCircuit with anonymous quantum wires e.g. QuantumCircuit(2)
                regs = (QuantumRegister(regs[0], 'q'),)
            elif len(regs) == 2 and all([isinstance(reg, int) for reg in regs]):
                # QuantumCircuit with anonymous wires e.g. QuantumCircuit(2, 3)
                regs = (QuantumRegister(regs[0], 'q'), ClassicalRegister(regs[1], 'c'))
            else:
                raise CircuitError("QuantumCircuit parameters can be Registers or Integers."
                                   " If Integers, up to 2 arguments. QuantumCircuit was called"
                                   " with %s." % (regs,))

        for register in regs:
            if register.name in [reg.name for reg in self.qregs + self.cregs]:
                raise CircuitError("register name \"%s\" already exists"
                                   % register.name)

            if isinstance(register, AncillaRegister):
                self._ancillas.extend(register)

            if isinstance(register, QuantumRegister):
                self.qregs.append(register)
                self._qubits.extend(register)
            elif isinstance(register, ClassicalRegister):
                self.cregs.append(register)
                self._clbits.extend(register)
            else:
                raise CircuitError("expected a register")

    def _check_dups(self, qubits):
        """Raise exception if list of qubits contains duplicates."""
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise CircuitError("duplicate qubit arguments")

    def _check_qargs(self, qargs):
        """Raise exception if a qarg is not in this circuit or bad format."""
        if not all(isinstance(i, Qubit) for i in qargs):
            raise CircuitError("qarg is not a Qubit")
        if not all(self.has_register(i.register) for i in qargs):
            raise CircuitError("register not in this circuit")

    def _check_cargs(self, cargs):
        """Raise exception if clbit is not in this circuit or bad format."""
        if not all(isinstance(i, Clbit) for i in cargs):
            raise CircuitError("carg is not a Clbit")
        if not all(self.has_register(i.register) for i in cargs):
            raise CircuitError("register not in this circuit")

    def to_instruction(self, parameter_map=None):
        """Create an Instruction out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               instruction. If None, existing circuit parameters will also
               parameterize the instruction.

        Returns:
            qiskit.circuit.Instruction: a composite instruction encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_instruction import circuit_to_instruction
        return circuit_to_instruction(self, parameter_map)

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
                        raise CircuitError("circuits are not compatible")

    @staticmethod
    def _get_composite_circuit_qasm_from_instruction(instruction):
        """Returns OpenQASM string composite circuit given an instruction.
        The given instruction should be the result of composite_circuit.to_instruction()."""

        qubit_parameters = ",".join(["q%i" % num for num in range(instruction.num_qubits)])
        composite_circuit_gates = ""

        for data, qargs, _ in instruction.definition:
            gate_qargs = ",".join(["q%i" % index for index in [qubit.index for qubit in qargs]])
            composite_circuit_gates += "%s %s; " % (data.qasm(), gate_qargs)

        qasm_string = "gate %s %s {%s}" % (instruction.name, qubit_parameters,
                                           composite_circuit_gates)

        return qasm_string

    def qasm(self, formatted=False, filename=None):
        """Return OpenQASM string.

        Parameters:
            formatted (bool): Return formatted Qasm string.
            filename (str): Save Qasm to file with name 'filename'.

        Returns:
            str: If formatted=False.

        Raises:
            ImportError: If pygments is not installed and ``formatted`` is
                ``True``.
        """
        existing_gate_names = ['ch', 'cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'ccx', 'cswap',
                               'cu1', 'cu3', 'dcx', 'h', 'i', 'id', 'iden', 'iswap', 'ms',
                               'r', 'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzx', 'rzz', 's', 'sdg',
                               'swap', 'x', 'y', 'z', 't', 'tdg', 'u1', 'u2', 'u3']

        existing_composite_circuits = []

        string_temp = self.header + "\n"
        string_temp += self.extension_lib + "\n"
        for register in self.qregs:
            string_temp += register.qasm() + "\n"
        for register in self.cregs:
            string_temp += register.qasm() + "\n"
        unitary_gates = []
        for instruction, qargs, cargs in self._data:
            if instruction.name == 'measure':
                qubit = qargs[0]
                clbit = cargs[0]
                string_temp += "%s %s[%d] -> %s[%d];\n" % (instruction.qasm(),
                                                           qubit.register.name, qubit.index,
                                                           clbit.register.name, clbit.index)
            # If instruction is a composite circuit
            elif not isinstance(instruction, Gate) and (instruction.name not in ['barrier',
                                                                                 'reset']):
                if instruction not in existing_composite_circuits:
                    if instruction.name in existing_gate_names:
                        old_name = instruction.name
                        instruction.name += "_" + str(id(instruction))

                        warnings.warn("A gate named {} already exists. "
                                      "We have renamed "
                                      "your gate to {}".format(old_name, instruction.name))

                    # Get qasm of composite circuit
                    qasm_string = self._get_composite_circuit_qasm_from_instruction(instruction)

                    # Insert composite circuit qasm definition right after header and extension lib
                    string_temp = string_temp.replace(self.extension_lib,
                                                      "%s\n%s" % (self.extension_lib,
                                                                  qasm_string))

                    existing_composite_circuits.append(instruction)
                    existing_gate_names.append(instruction.name)

                # Insert qasm representation of the original instruction
                string_temp += "%s %s;\n" % (instruction.qasm(),
                                             ",".join(["%s[%d]" % (j.register.name, j.index)
                                                       for j in qargs + cargs]))
            else:
                string_temp += "%s %s;\n" % (instruction.qasm(),
                                             ",".join(["%s[%d]" % (j.register.name, j.index)
                                                       for j in qargs + cargs]))
            if instruction.name == 'unitary':
                unitary_gates.append(instruction)

        # this resets them, so if another call to qasm() is made the gate def is added again
        for gate in unitary_gates:
            gate._qasm_def_written = False

        if filename:
            with open(filename, 'w+') as file:
                file.write(string_temp)
            file.close()

        if formatted:
            if not HAS_PYGMENTS:
                raise ImportError("To use the formatted output pygments>2.4 "
                                  "must be installed. To install pygments run "
                                  '"pip install pygments".')
            code = pygments.highlight(string_temp,
                                      OpenQASMLexer(),
                                      Terminal256Formatter(style=QasmTerminalStyle))
            print(code)
            return None
        else:
            return string_temp

    def draw(self, output=None, scale=None, filename=None, style=None,
             interactive=False, plot_barriers=True,
             reverse_bits=False, justify=None, vertical_compression='medium', idle_wires=True,
             with_layout=True, fold=None, ax=None, initial_state=False, cregbundle=True):
        """Draw the quantum circuit.

        **text**: ASCII art TextDrawing that can be printed in the console.

        **latex**: high-quality images compiled via LaTeX.

        **latex_source**: raw uncompiled LaTeX output.

        **matplotlib**: images with color rendered purely in Python.

        Args:
            output (str): Select the output method to use for drawing the
                circuit. Valid choices are ``text``, ``latex``,
                ``latex_source``, or ``mpl``. By default the `'text`' drawer is
                used unless a user config file has an alternative backend set
                as the default. If the output kwarg is set, that backend
                will always be used over the default in a user config file.
            scale (float): scale of image to draw (shrink if < 1)
            filename (str): file path to save image to
            style (dict or str): dictionary of style or file name of style
                file. This option is only used by the ``mpl`` output type. If a
                str is passed in that is the path to a json file which contains
                a dictionary of style, then that will be opened, parsed, and used
                as the input dict. See: :ref:`Style Dict Doc <style-dict-circ-doc>` for more
                information on the contents.

            interactive (bool): when set true show the circuit in a new window
                (for `mpl` this depends on the matplotlib backend being used
                supporting this). Note when used with either the `text` or the
                `latex_source` output type this has no effect and will be
                silently ignored.
            reverse_bits (bool): When set to True, reverse the bit order inside
                registers for the output visualization.
            plot_barriers (bool): Enable/disable drawing barriers in the output
                circuit. Defaults to True.
            justify (string): Options are ``left``, ``right`` or
                ``none``. If anything else is supplied it defaults to left
                justified. It refers to where gates should be placed in the
                output circuit if there is an option. ``none`` results in
                each gate being placed in its own column.
            vertical_compression (string): ``high``, ``medium`` or ``low``. It
                merges the lines generated by the ``text`` output so the
                drawing will take less vertical room.  Default is ``medium``.
                Only used by the ``text`` output, will be silently ignored
                otherwise.
            idle_wires (bool): Include idle wires (wires with no circuit
                elements) in output visualization. Default is True.
            with_layout (bool): Include layout information, with labels on the
                physical layout. Default is True.
            fold (int): Sets pagination. It can be disabled using -1.
                In `text`, sets the length of the lines. This is useful when the
                drawing does not fit in the console. If None (default), it will
                try to guess the console width using ``shutil.
                get_terminal_size()``. However, if running in jupyter, the
                default line length is set to 80 characters. In ``mpl`` is the
                number of (visual) layers before folding. Default is 25.
            ax (matplotlib.axes.Axes): An optional Axes object to be used for
                the visualization output. If none is specified, a new matplotlib
                Figure will be created and used. Additionally, if specified,
                there will be no returned Figure since it is redundant. This is
                only used when the ``output`` kwarg is set to use the ``mpl``
                backend. It will be silently ignored with all other outputs.
            initial_state (bool): Optional. Adds ``|0>`` in the beginning of the wire.
                Only used by the ``text``, ``latex`` and ``latex_source`` outputs.
                Default: ``False``.
            cregbundle (bool): Optional. If set True bundle classical registers. Not used by
                the ``matplotlib`` output. Default: ``True``.

        Returns:
            :class:`PIL.Image` or :class:`matplotlib.figure` or :class:`str` or
            :class:`TextDrawing`:

            * `PIL.Image` (output='latex')
                an in-memory representation of the image of the circuit
                diagram.
            * `matplotlib.figure.Figure` (output='mpl')
                a matplotlib figure object for the circuit diagram.
            * `str` (output='latex_source')
                The LaTeX source code for visualizing the circuit diagram.
            * `TextDrawing` (output='text')
                A drawing that can be printed as ASCII art.

        Raises:
            VisualizationError: when an invalid output method is selected
            ImportError: when the output methods require non-installed
                libraries

        .. _style-dict-circ-doc:

        **Style Dict Details**

        The style dict kwarg contains numerous options that define the style of
        the output circuit visualization. The style dict is only used by the
        ``mpl`` output. The options available in the style dict are defined
        below:

        Args:
            textcolor (str): The color code to use for text. Defaults to
                `'#000000'`
            subtextcolor (str): The color code to use for subtext. Defaults to
                `'#000000'`
            linecolor (str): The color code to use for lines. Defaults to
                `'#000000'`
            creglinecolor (str): The color code to use for classical register
                lines. Defaults to `'#778899'`
            gatetextcolor (str): The color code to use for gate text. Defaults
                to `'#000000'`
            gatefacecolor (str): The color code to use for gates. Defaults to
                `'#ffffff'`
            barrierfacecolor (str): The color code to use for barriers.
                Defaults to `'#bdbdbd'`
            backgroundcolor (str): The color code to use for the background.
                Defaults to `'#ffffff'`
            fontsize (int): The font size to use for text. Defaults to 13.
            subfontsize (int): The font size to use for subtext. Defaults to 8.
            displaytext (dict): A dictionary of the text to use for each
                element type in the output visualization. The default values
                are::

                    {
                        'id': 'id',
                        'u0': 'U_0',
                        'u1': 'U_1',
                        'u2': 'U_2',
                        'u3': 'U_3',
                        'x': 'X',
                        'y': 'Y',
                        'z': 'Z',
                        'h': 'H',
                        's': 'S',
                        'sdg': 'S^\\dagger',
                        't': 'T',
                        'tdg': 'T^\\dagger',
                        'rx': 'R_x',
                        'ry': 'R_y',
                        'rz': 'R_z',
                        'reset': '\\left|0\\right\\rangle'
                    }

                You must specify all the necessary values if using this. There
                is no provision for passing an incomplete dict in.
            displaycolor (dict): The color codes to use for each circuit
                element. The default values are::

                    {
                        'id': '#F0E442',
                        'u0': '#E7AB3B',
                        'u1': '#E7AB3B',
                        'u2': '#E7AB3B',
                        'u3': '#E7AB3B',
                        'x': '#58C698',
                        'y': '#58C698',
                        'z': '#58C698',
                        'h': '#70B7EB',
                        's': '#E0722D',
                        'sdg': '#E0722D',
                        't': '#E0722D',
                        'tdg': '#E0722D',
                        'rx': '#ffffff',
                        'ry': '#ffffff',
                        'rz': '#ffffff',
                        'reset': '#D188B4',
                        'target': '#70B7EB',
                        'meas': '#D188B4'
                    }

               Also, just like  `displaytext` there is no provision for an
               incomplete dict passed in.

            latexdrawerstyle (bool): When set to True, enable LaTeX mode, which
                will draw gates like the `latex` output modes.
            usepiformat (bool): When set to True, use radians for output.
            fold (int): The number of circuit elements to fold the circuit at.
                Defaults to 20.
            cregbundle (bool): If set True, bundle classical registers
            showindex (bool): If set True, draw an index.
            compress (bool): If set True, draw a compressed circuit.
            figwidth (int): The maximum width (in inches) for the output figure.
            dpi (int): The DPI to use for the output image. Defaults to 150.
            margin (list): A list of margin values to adjust spacing around
                output image. Takes a list of 4 ints:
                [x left, x right, y bottom, y top].
            creglinestyle (str): The style of line to use for classical
                registers. Choices are `'solid'`, `'doublet'`, or any valid
                matplotlib `linestyle` kwarg value. Defaults to `doublet`
        """

        # pylint: disable=cyclic-import
        from qiskit.visualization import circuit_drawer
        if isinstance(output, (int, float, np.number)):
            warnings.warn("Setting 'scale' as the first argument is deprecated. "
                          "Use scale=%s instead." % output,
                          DeprecationWarning)
            scale = output
            output = None

        return circuit_drawer(self, scale=scale,
                              filename=filename, style=style,
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
                              cregbundle=cregbundle)

    def size(self):
        """Returns total number of gate operations in circuit.

        Returns:
            int: Total number of gate operations.
        """
        gate_ops = 0
        for instr, _, _ in self._data:
            if instr.name not in ['barrier', 'snapshot']:
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
        # Labels the registers by ints
        # and then the qubit position in
        # a register is given by reg_int+qubit_num
        reg_offset = 0
        reg_map = {}
        for reg in self.qregs + self.cregs:
            reg_map[reg.name] = reg_offset
            reg_offset += reg.size

        # If no registers return 0
        if reg_offset == 0:
            return 0

        # A list that holds the height of each qubit
        # and classical bit.
        op_stack = [0] * reg_offset
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
            if instr.name in ['barrier', 'snapshot']:
                count = False
            for ind, reg in enumerate(qargs + cargs):
                # Add to the stacks of the qubits and
                # cbits used in the gate.
                reg_ints.append(reg_map[reg.register.name] + reg.index)
                if count:
                    levels.append(op_stack[reg_ints[ind]] + 1)
                else:
                    levels.append(op_stack[reg_ints[ind]])
            # Assuming here that there is no conditional
            # snapshots or barriers ever.
            if instr.condition:
                # Controls operate over all bits in the
                # classical register they use.
                cint = reg_map[instr.condition[0].name]
                for off in range(instr.condition[0].size):
                    if cint + off not in reg_ints:
                        reg_ints.append(cint + off)
                        levels.append(op_stack[cint + off] + 1)

            max_level = max(levels)
            for ind in reg_ints:
                op_stack[ind] = max_level

        return max(op_stack)

    def width(self):
        """Return number of qubits plus clbits in circuit.

        Returns:
            int: Width of circuit.

        """
        return sum(reg.size for reg in self.qregs + self.cregs)

    @property
    def num_qubits(self):
        """Return number of qubits."""
        qubits = 0
        for reg in self.qregs:
            qubits += reg.size
        return qubits

    @property
    def num_ancillas(self):
        """Return the number of ancilla qubits."""
        return len(self.ancillas)

    @property
    def n_qubits(self):
        """Deprecated, use ``num_qubits`` instead. Return number of qubits."""
        warnings.warn('The QuantumCircuit.n_qubits method is deprecated as of 0.13.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the QuantumCircuit.num_qubits method instead.',
                      DeprecationWarning, stacklevel=2)
        return self.num_qubits

    @property
    def num_clbits(self):
        """Return number of classical bits."""
        return sum(len(reg) for reg in self.cregs)

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
            if instr.num_qubits > 1 and instr.name not in ['barrier', 'snapshot']:
                multi_qubit_gates += 1
        return multi_qubit_gates

    def num_connected_components(self, unitary_only=False):
        """How many non-entangled subcircuits can the circuit be factored to.

        Args:
            unitary_only (bool): Compute only unitary part of graph.

        Returns:
            int: Number of connected components in circuit.
        """
        # Convert registers to ints (as done in depth).
        reg_offset = 0
        reg_map = {}

        if unitary_only:
            regs = self.qregs
        else:
            regs = self.qregs + self.cregs

        for reg in regs:
            reg_map[reg.name] = reg_offset
            reg_offset += reg.size
        # Start with each qubit or cbit being its own subgraph.
        sub_graphs = [[bit] for bit in range(reg_offset)]

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

            if num_qargs >= 2 and instr.name not in ['barrier', 'snapshot']:
                graphs_touched = []
                num_touched = 0
                # Controls necessarily join all the cbits in the
                # register that they use.
                if instr.condition and not unitary_only:
                    creg = instr.condition[0]
                    creg_int = reg_map[creg.name]
                    for coff in range(creg.size):
                        temp_int = creg_int + coff
                        for k in range(num_sub_graphs):
                            if temp_int in sub_graphs[k]:
                                graphs_touched.append(k)
                                num_touched += 1
                                break

                for item in args:
                    reg_int = reg_map[item.register.name] + item.index
                    for k in range(num_sub_graphs):
                        if reg_int in sub_graphs[k]:
                            if k not in graphs_touched:
                                graphs_touched.append(k)
                                num_touched += 1
                                break

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
                    num_sub_graphs -= (num_touched - 1)
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

        instr_instances = {id(instr): instr
                           for instr, _, __ in self._data}

        instr_copies = {id_: instr.copy()
                        for id_, instr in instr_instances.items()}

        cpy._parameter_table = ParameterTable({
            param: [(instr_copies[id(instr)], param_index)
                    for instr, param_index in self._parameter_table[param]]
            for param in self._parameter_table
        })

        cpy._data = [(instr_copies[id(inst)], qargs.copy(), cargs.copy())
                     for inst, qargs, cargs in self._data]

        if name:
            cpy.name = name
        return cpy

    def _create_creg(self, length, name):
        """ Creates a creg, checking if ClassicalRegister with same name exists
        """
        if name in [creg.name for creg in self.cregs]:
            save_prefix = ClassicalRegister.prefix
            ClassicalRegister.prefix = name
            new_creg = ClassicalRegister(length)
            ClassicalRegister.prefix = save_prefix
        else:
            new_creg = ClassicalRegister(length, name)
        return new_creg

    def _create_qreg(self, length, name):
        """ Creates a qreg, checking if QuantumRegister with same name exists
        """
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

        Parameters:
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
        new_creg = circ._create_creg(len(qubits_to_measure), 'measure')
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

        Parameters:
            inplace (bool): All measurements inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements when `inplace = False`.
        """
        if inplace:
            circ = self
        else:
            circ = self.copy()

        new_creg = circ._create_creg(len(circ.qubits), 'meas')
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

        Parameters:
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
        circ.cregs = list(new_dag.cregs.values())

        for node in new_dag.topological_op_nodes():
            qubits = []
            for qubit in node.qargs:
                qubits.append(new_dag.qregs[qubit.register.name][qubit.index])

            clbits = []
            for clbit in node.cargs:
                clbits.append(new_dag.cregs[clbit.register.name][clbit.index])

            # Get arguments for classical condition (if any)
            inst = node.op.copy()
            inst.condition = node.condition
            circ.append(inst, qubits, clbits)

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
        if isinstance(angle, ParameterExpression):
            self._global_phase = angle
        else:
            # Set the phase to the [-2 * pi, 2 * pi] interval
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            elif angle < 0:
                self._global_phase = angle % (-2 * np.pi)
            else:
                self._global_phase = angle % (2 * np.pi)

    @property
    def parameters(self):
        """Convenience function to get the parameters defined in the parameter table."""
        return self._parameter_table.get_keys()

    @property
    def num_parameters(self):
        """Convenience function to get the number of parameter objects in the circuit."""
        return len(self.parameters)

    def assign_parameters(self, param_dict, inplace=False):
        """Assign parameters to new parameters or values.

        The keys of the parameter dictionary must be Parameter instances in the current circuit. The
        values of the dictionary can either be numeric values or new parameter objects.
        The values can be assigned to the current circuit object or to a copy of it.

        Args:
            param_dict (dict): A dictionary specifying the mapping from ``current_parameter``
                to ``new_parameter``, where ``new_parameter`` can be a new parameter object
                or a numeric value.
            inplace (bool): If False, a copy of the circuit with the bound parameters is
                returned. If True the circuit instance itself is modified.

        Raises:
            CircuitError: If param_dict contains parameters not present in the circuit

        Returns:
            optional(QuantumCircuit): A copy of the circuit with bound parameters, if
                ``inplace`` is True, otherwise None.

        Examples:

            >>> from qiskit.circuit import QuantumCircuit, Parameter
            >>> circuit = QuantumCircuit(2)
            >>> params = [Parameter('A'), Parameter('B'), Parameter('C')]
            >>> circuit.ry(params[0], 0)
            >>> circuit.crx(params[1], 0, 1)
            >>> circuit.draw()
                    ┌───────┐
            q_0: |0>┤ Ry(A) ├────■────
                    └───────┘┌───┴───┐
            q_1: |0>─────────┤ Rx(B) ├
                             └───────┘
            >>> circuit.assign_parameters({params[0]: params[2]}, inplace=True)
            >>> circuit.draw()
                    ┌───────┐
            q_0: |0>┤ Ry(C) ├────■────
                    └───────┘┌───┴───┐
            q_1: |0>─────────┤ Rx(B) ├
                             └───────┘
            >>> bound_circuit = circuit.assign_parameters({params[1]: 1, params[2]: 2})
            >>> bound_circuit.draw()
                    ┌───────┐
            q_0: |0>┤ Ry(2) ├────■────
                    └───────┘┌───┴───┐
            q_1: |0>─────────┤ Rx(1) ├
                             └───────┘
            >>> bound_circuit.parameters  # this one has no free parameters anymore
            set()
            >>> circuit.parameters  # the original one is still parameterized
            {Parameter(A), Parameter(C)}
        """
        # replace in self or in a copy depending on the value of in_place
        bound_circuit = self if inplace else self.copy()

        # unroll the parameter dictionary (needed if e.g. it contains a ParameterVector)
        unrolled_param_dict = self._unroll_param_dict(param_dict)

        # check that only existing parameters are in the parameter dictionary
        if unrolled_param_dict.keys() > self._parameter_table.keys():
            raise CircuitError('Cannot bind parameters ({}) not present in the circuit.'.format(
                [str(p) for p in param_dict.keys() - self._parameter_table]))

        # replace the parameters with a new Parameter ("substitute") or numeric value ("bind")
        for parameter, value in unrolled_param_dict.items():
            if isinstance(value, ParameterExpression):
                bound_circuit._substitute_parameter(parameter, value)
            else:
                bound_circuit._bind_parameter(parameter, value)
                del bound_circuit._parameter_table[parameter]  # clear evaluated expressions

        return None if inplace else bound_circuit

    def bind_parameters(self, value_dict):
        """Assign numeric parameters to values yielding a new circuit.

        To assign new Parameter objects or bind the values in-place, without yielding a new
        circuit, use the assign_parameters method.

        Args:
            value_dict (dict): {parameter: value, ...}

        Raises:
            CircuitError: If value_dict contains parameters not present in the circuit
            TypeError: If value_dict contains a ParameterExpression in the values.

        Returns:
            QuantumCircuit: copy of self with assignment substitution.
        """
        bound_circuit = self.copy()

        # unroll the parameter dictionary (needed if e.g. it contains a ParameterVector)
        unrolled_value_dict = self._unroll_param_dict(value_dict)

        # check that only existing parameters are in the parameter dictionary
        if len(unrolled_value_dict) > len(self._parameter_table):
            raise CircuitError('Cannot bind parameters ({}) not present in the circuit.'.format(
                [str(p) for p in value_dict.keys() - self._parameter_table.keys()]))

        # replace the parameters with a new Parameter ("substitute") or numeric value ("bind")
        for parameter, value in unrolled_value_dict.items():
            bound_circuit._bind_parameter(parameter, value)
            del bound_circuit._parameter_table[parameter]  # clear evaluated expressions

        return bound_circuit

    def _unroll_param_dict(self, value_dict):
        unrolled_value_dict = {}
        for (param, value) in value_dict.items():
            if isinstance(param, ParameterExpression):
                unrolled_value_dict[param] = value
            if isinstance(param, ParameterVector):
                if not len(param) == len(value):
                    raise CircuitError('ParameterVector {} has length {}, which '
                                       'differs from value list {} of '
                                       'len {}'.format(param, len(param), value, len(value)))
                unrolled_value_dict.update(zip(param, value))
        return unrolled_value_dict

    def _bind_parameter(self, parameter, value):
        """Assigns a parameter value to matching instructions in-place."""
        for (instr, param_index) in self._parameter_table[parameter]:
            instr.params[param_index] = instr.params[param_index].bind({parameter: value})

            # For instructions which have already been defined (e.g. composite
            # instructions), search the definition for instances of the
            # parameter which also need to be bound.
            self._rebind_definition(instr, parameter, value)
        # bind circuit's phase
        if (isinstance(self.global_phase, ParameterExpression) and
                parameter in self.global_phase.parameters):
            self.global_phase = self.global_phase.bind({parameter: value})

    def _substitute_parameter(self, old_parameter, new_parameter_expr):
        """Substitute an existing parameter in all circuit instructions and the parameter table."""
        for instr, param_index in self._parameter_table[old_parameter]:
            new_param = instr.params[param_index].subs({old_parameter: new_parameter_expr})
            instr.params[param_index] = new_param
            self._rebind_definition(instr, old_parameter, new_parameter_expr)

        entry = self._parameter_table.pop(old_parameter)
        for new_parameter in new_parameter_expr.parameters:
            self._parameter_table[new_parameter] = entry
        if (isinstance(self.global_phase, ParameterExpression)
                and old_parameter in self.global_phase.parameters):
            self.global_phase = self.global_phase.subs({old_parameter: new_parameter_expr})

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
            for qreg in self.qregs:
                for j in range(qreg.size):
                    qubits.append(qreg[j])

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

    @deprecate_arguments({'q': 'qubit'})
    def h(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.HGate`."""
        from .library.standard_gates.h import HGate
        return self.append(HGate(), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit', 'tgt': 'target_qubit'})
    def ch(self, control_qubit, target_qubit,  # pylint: disable=invalid-name
           *, label=None, ctrl_state=None, ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CHGate`."""
        from .library.standard_gates.h import CHGate
        return self.append(CHGate(label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def i(self, qubit, *, q=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.IGate`."""
        from .library.standard_gates.i import IGate
        return self.append(IGate(), [qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def id(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.IGate`."""
        return self.i(qubit)

    @deprecate_arguments({'q': 'qubit'})
    def iden(self, qubit, *, q=None):  # pylint: disable=unused-argument
        """Deprecated identity gate."""
        warnings.warn('The QuantumCircuit.iden() method is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the QuantumCircuit.i() method instead.',
                      DeprecationWarning, stacklevel=2)
        return self.i(qubit)

    def ms(self, theta, qubits):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.MSGate`."""
        from .library.standard_gates.ms import MSGate
        return self.append(MSGate(len(qubits), theta), qubits)

    def p(self, theta, qubit):
        """Apply :class:`~qiskit.circuit.library.PhaseGate`."""
        from .library.standard_gates.p import PhaseGate
        return self.append(PhaseGate(theta), [qubit], [])

    def cp(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CPhaseGate`."""
        from .library.standard_gates.p import CPhaseGate
        return self.append(CPhaseGate(theta, label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def r(self, theta, phi, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.RGate`."""
        from .library.standard_gates.r import RGate
        return self.append(RGate(theta, phi), [qubit], [])

    def rccx(self, control_qubit1, control_qubit2, target_qubit):
        """Apply :class:`~qiskit.circuit.library.RCCXGate`."""
        from .library.standard_gates.x import RCCXGate
        return self.append(RCCXGate(), [control_qubit1, control_qubit2, target_qubit], [])

    def rcccx(self, control_qubit1, control_qubit2, control_qubit3, target_qubit):
        """Apply :class:`~qiskit.circuit.library.RC3XGate`."""
        from .library.standard_gates.x import RC3XGate
        return self.append(RC3XGate(),
                           [control_qubit1, control_qubit2, control_qubit3, target_qubit],
                           [])

    @deprecate_arguments({'q': 'qubit'})
    # pylint: disable=invalid-name,unused-argument
    def rx(self, theta, qubit, *, label=None, q=None):
        """Apply :class:`~qiskit.circuit.library.RXGate`."""
        from .library.standard_gates.rx import RXGate
        return self.append(RXGate(theta, label=label), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt': 'target_qubit'})
    def crx(self, theta, control_qubit, target_qubit, *, label=None, ctrl_state=None,
            ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CRXGate`."""
        from .library.standard_gates.rx import CRXGate
        return self.append(CRXGate(theta, label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    def rxx(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RXXGate`."""
        from .library.standard_gates.rxx import RXXGate
        return self.append(RXXGate(theta), [qubit1, qubit2], [])

    # pylint: disable=invalid-name,unused-argument
    @deprecate_arguments({'q': 'qubit'})
    def ry(self, theta, qubit, *, label=None, q=None):
        """Apply :class:`~qiskit.circuit.library.RYGate`."""
        from .library.standard_gates.ry import RYGate
        return self.append(RYGate(theta, label=label), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt': 'target_qubit'})
    def cry(self, theta, control_qubit, target_qubit, *, label=None, ctrl_state=None,
            ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CRYGate`."""
        from .library.standard_gates.ry import CRYGate
        return self.append(CRYGate(theta, label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    def ryy(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RYYGate`."""
        from .library.standard_gates.ryy import RYYGate
        return self.append(RYYGate(theta), [qubit1, qubit2], [])

    @deprecate_arguments({'q': 'qubit'})
    def rz(self, phi, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.RZGate`."""
        from .library.standard_gates.rz import RZGate
        return self.append(RZGate(phi), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit', 'tgt': 'target_qubit'})
    def crz(self, theta, control_qubit, target_qubit, *, label=None, ctrl_state=None,
            ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CRZGate`."""
        from .library.standard_gates.rz import CRZGate
        return self.append(CRZGate(theta, label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    def rzx(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RZXGate`."""
        from .library.standard_gates.rzx import RZXGate
        return self.append(RZXGate(theta), [qubit1, qubit2], [])

    def rzz(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RZZGate`."""
        from .library.standard_gates.rzz import RZZGate
        return self.append(RZZGate(theta), [qubit1, qubit2], [])

    @deprecate_arguments({'q': 'qubit'})
    def s(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.SGate`."""
        from .library.standard_gates.s import SGate
        return self.append(SGate(), [qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def sdg(self, qubit, *, q=None):  # pylint: disable=unused-argument
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

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt1': 'target_qubit1',
                          'tgt2': 'target_qubit2'})
    def cswap(self, control_qubit, target_qubit1, target_qubit2, *, label=None, ctrl_state=None,
              ctl=None, tgt1=None, tgt2=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CSwapGate`."""
        from .library.standard_gates.swap import CSwapGate
        return self.append(CSwapGate(label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit1, target_qubit2], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt1': 'target_qubit1',
                          'tgt2': 'target_qubit2'})
    def fredkin(self, control_qubit, target_qubit1, target_qubit2,
                *, ctl=None, tgt1=None, tgt2=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CSwapGate`."""
        return self.cswap(control_qubit, target_qubit1, target_qubit2)

    def sx(self, qubit):
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
        return self.append(CSXGate(label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def t(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.TGate`."""
        from .library.standard_gates.t import TGate
        return self.append(TGate(), [qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def tdg(self, qubit, *, q=None):  # pylint: disable=unused-argument
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
        return self.append(CUGate(theta, phi, lam, gamma, label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def u1(self, theta, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.U1Gate`."""
        from .library.standard_gates.u1 import U1Gate
        return self.append(U1Gate(theta), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt': 'target_qubit'})
    def cu1(self, theta, control_qubit, target_qubit, *, label=None, ctrl_state=None,
            ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CU1Gate`."""
        from .library.standard_gates.u1 import CU1Gate
        return self.append(CU1Gate(theta, label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    def mcu1(self, lam, control_qubits, target_qubit):
        """Apply :class:`~qiskit.circuit.library.MCU1Gate`."""
        from .library.standard_gates.u1 import MCU1Gate
        num_ctrl_qubits = len(control_qubits)
        return self.append(MCU1Gate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def u2(self, phi, lam, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.U2Gate`."""
        from .library.standard_gates.u2 import U2Gate
        return self.append(U2Gate(phi, lam), [qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def u3(self, theta, phi, lam, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
        """Apply :class:`~qiskit.circuit.library.U3Gate`."""
        from .library.standard_gates.u3 import U3Gate
        return self.append(U3Gate(theta, phi, lam), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt': 'target_qubit'})
    def cu3(self, theta, phi, lam, control_qubit, target_qubit, *, label=None, ctrl_state=None,
            ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CU3Gate`."""
        from .library.standard_gates.u3 import CU3Gate
        return self.append(CU3Gate(theta, phi, lam, label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def x(self, qubit, *, label=None, ctrl_state=None, q=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.XGate`."""
        from .library.standard_gates.x import XGate
        return self.append(XGate(label=label), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt': 'target_qubit'})
    def cx(self, control_qubit, target_qubit, *, label=None, ctrl_state=None,
           ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CXGate`."""
        from .library.standard_gates.x import CXGate
        return self.append(CXGate(label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt': 'target_qubit'})
    def cnot(self, control_qubit, target_qubit, *, label=None, ctrl_state=None,
             ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CXGate`."""
        self.cx(control_qubit, target_qubit)

    def dcx(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.DCXGate`."""
        from .library.standard_gates.dcx import DCXGate
        return self.append(DCXGate(), [qubit1, qubit2], [])

    @deprecate_arguments({'ctl1': 'control_qubit1',
                          'ctl2': 'control_qubit2',
                          'tgt': 'target_qubit'})
    def ccx(self, control_qubit1, control_qubit2, target_qubit,
            *, ctl1=None, ctl2=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CCXGate`."""
        from .library.standard_gates.x import CCXGate
        return self.append(CCXGate(),
                           [control_qubit1, control_qubit2, target_qubit], [])

    @deprecate_arguments({'ctl1': 'control_qubit1',
                          'ctl2': 'control_qubit2',
                          'tgt': 'target_qubit'})
    def toffoli(self, control_qubit1, control_qubit2, target_qubit,
                *, ctl1=None, ctl2=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CCXGate`."""
        self.ccx(control_qubit1, control_qubit2, target_qubit)

    def mcx(self, control_qubits, target_qubit, ancilla_qubits=None, mode='noancilla'):
        """Apply :class:`~qiskit.circuit.library.MCXGate`.

        The multi-cX gate can be implemented using different techniques, which use different numbers
        of ancilla qubits and have varying circuit depth. These modes are:
        - 'no-ancilla': Requires 0 ancilla qubits.
        - 'recursion': Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.
        - 'v-chain': Requires 2 less ancillas than the number of control qubits.
        - 'v-chain-dirty': Same as for the clean ancillas (but the circuit will be longer).
        """
        from .library.standard_gates.x import MCXGrayCode, MCXRecursive, MCXVChain
        num_ctrl_qubits = len(control_qubits)

        available_implementations = {
            'noancilla': MCXGrayCode(num_ctrl_qubits),
            'recursion': MCXRecursive(num_ctrl_qubits),
            'v-chain': MCXVChain(num_ctrl_qubits, False),
            'v-chain-dirty': MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
            # outdated, previous names
            'advanced': MCXRecursive(num_ctrl_qubits),
            'basic': MCXVChain(num_ctrl_qubits, dirty_ancillas=False),
            'basic-dirty-ancilla': MCXVChain(num_ctrl_qubits, dirty_ancillas=True)
        }

        # check ancilla input
        if ancilla_qubits:
            _ = self.qbit_argument_conversion(ancilla_qubits)

        try:
            gate = available_implementations[mode]
        except KeyError:
            all_modes = list(available_implementations.keys())
            raise ValueError('Unsupported mode ({}) selected, choose one of {}'.format(mode,
                                                                                       all_modes))

        if hasattr(gate, 'num_ancilla_qubits') and gate.num_ancilla_qubits > 0:
            required = gate.num_ancilla_qubits
            if ancilla_qubits is None:
                raise AttributeError('No ancillas provided, but {} are needed!'.format(required))

            # convert ancilla qubits to a list if they were passed as int or qubit
            if not hasattr(ancilla_qubits, '__len__'):
                ancilla_qubits = [ancilla_qubits]

            if len(ancilla_qubits) < required:
                actually = len(ancilla_qubits)
                raise ValueError('At least {} ancillas required, but {} given.'.format(required,
                                                                                       actually))
            # size down if too many ancillas were provided
            ancilla_qubits = ancilla_qubits[:required]
        else:
            ancilla_qubits = []

        return self.append(gate, control_qubits[:] + [target_qubit] + ancilla_qubits[:], [])

    def mct(self, control_qubits, target_qubit, ancilla_qubits=None, mode='noancilla'):
        """Apply :class:`~qiskit.circuit.library.MCXGate`."""
        return self.mcx(control_qubits, target_qubit, ancilla_qubits, mode)

    @deprecate_arguments({'q': 'qubit'})
    def y(self, qubit, *, q=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.YGate`."""
        from .library.standard_gates.y import YGate
        return self.append(YGate(), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt': 'target_qubit'})
    def cy(self, control_qubit, target_qubit, *, label=None, ctrl_state=None,
           ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CYGate`."""
        from .library.standard_gates.y import CYGate
        return self.append(CYGate(label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])

    @deprecate_arguments({'q': 'qubit'})
    def z(self, qubit, *, q=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.ZGate`."""
        from .library.standard_gates.z import ZGate
        return self.append(ZGate(), [qubit], [])

    @deprecate_arguments({'ctl': 'control_qubit',
                          'tgt': 'target_qubit'})
    def cz(self, control_qubit, target_qubit, *, label=None, ctrl_state=None,
           ctl=None, tgt=None):  # pylint: disable=unused-argument
        """Apply :class:`~qiskit.circuit.library.CZGate`."""
        from .library.standard_gates.z import CZGate
        return self.append(CZGate(label=label, ctrl_state=ctrl_state),
                           [control_qubit, target_qubit], [])


def _circuit_from_qasm(qasm):
    # pylint: disable=cyclic-import
    from qiskit.converters import ast_to_dag
    from qiskit.converters import dag_to_circuit
    ast = qasm.parse()
    dag = ast_to_dag(ast)
    return dag_to_circuit(dag)
