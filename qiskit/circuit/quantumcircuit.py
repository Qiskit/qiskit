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

from copy import deepcopy
import itertools
import sys
import multiprocessing as mp

from qiskit.circuit.instruction import Instruction
from qiskit.qasm.qasm import Qasm
from qiskit.exceptions import QiskitError
from qiskit.circuit.parameter import Parameter
from .quantumregister import QuantumRegister
from .classicalregister import ClassicalRegister
from .parametertable import ParameterTable
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .register import Register


def _is_bit(obj):
    """Determine if obj is a bit"""
    # If there is a bit type this could be replaced by isinstance.
    if isinstance(obj, tuple) and len(obj) == 2:
        if isinstance(obj[0], Register) and isinstance(obj[1], int) and obj[1] < len(obj[0]):
            return True
    return False


class QuantumCircuit:
    """Quantum circuit."""
    instances = 0
    prefix = 'circuit'

    # Class variable OPENQASM header
    header = "OPENQASM 2.0;"
    extension_lib = "include \"qelib1.inc\";"

    def __init__(self, *regs, name=None):
        """Create a new circuit.
        A circuit is a list of instructions bound to some registers.
        Args:
            *regs (list(Register) or list(Int)): To be included in the circuit.
                  - If [Register], the QuantumRegister and/or ClassicalRegister
                    to include in the circuit.
                    E.g.: QuantumCircuit(QuantumRegister(4))
                          QuantumCircuit(QuantumRegister(4), ClassicalRegister(3))
                          QuantumCircuit(QuantumRegister(4, 'qr0'), QuantumRegister(2, 'qr1'))
                  - If [Int], the amount of qubits and/or classical bits to include
                  in the circuit. It can be (Int, ) or (Int, Int).
                    E.g.: QuantumCircuit(4) # A QuantumCircuit with 4 qubits
                          QuantumCircuit(4, 3) # A QuantumCircuit with 4 qubits and 3 classical bits
            name (str or None): the name of the quantum circuit. If
                None, an automatically generated string will be assigned.

        Raises:
            QiskitError: if the circuit name, if given, is not valid.
        """
        if name is None:
            name = self.cls_prefix() + str(self.cls_instances())
            # pylint: disable=not-callable
            # (known pylint bug: https://github.com/PyCQA/pylint/issues/1699)
            if sys.platform != "win32" and isinstance(mp.current_process(), mp.context.ForkProcess):
                name += '-{}'.format(mp.current_process().pid)
        self._increment_instances()

        if not isinstance(name, str):
            raise QiskitError("The circuit name should be a string "
                              "(or None to auto-generate a name).")

        self.name = name

        # Data contains a list of instructions and their contexts,
        # in the order they were applied.
        self.data = []

        # This is a map of registers bound to this circuit, by name.
        self.qregs = []
        self.cregs = []
        self.add_register(*regs)

        # Parameter table tracks instructions with variable parameters.
        self._parameter_table = ParameterTable()

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
        """Mirror the circuit by reversing the instructions.

        This is done by recursively mirroring all instructions.
        It does not invert any gate.

        Returns:
            QuantumCircuit: the mirrored circuit
        """
        reverse_circ = self.copy(name=self.name + '_mirror')
        reverse_circ.data = []
        for inst, qargs, cargs in reversed(self.data):
            reverse_circ.data.append((inst.mirror(), qargs, cargs))
        return reverse_circ

    def inverse(self):
        """Invert this circuit.

        This is done by recursively inverting all gates.

        Returns:
            QuantumCircuit: the inverted circuit

        Raises:
            QiskitError: if the circuit cannot be inverted.
        """
        inverse_circ = self.copy(name=self.name + '_dg')
        inverse_circ.data = []
        for inst, qargs, cargs in reversed(self.data):
            inverse_circ.data.append((inst.inverse(), qargs, cargs))
        return inverse_circ

    def combine(self, rhs):
        """
        Append rhs to self if self contains compatible registers.

        Two circuits are compatible if they contain the same registers
        or if they contain different registers with unique names. The
        returned circuit will contain all unique registers between both
        circuits.

        Return self + rhs as a new object.
        """
        # Check registers in LHS are compatible with RHS
        self._check_compatible_regs(rhs)

        # Make new circuit with combined registers
        combined_qregs = deepcopy(self.qregs)
        combined_cregs = deepcopy(self.cregs)

        for element in rhs.qregs:
            if element not in self.qregs:
                combined_qregs.append(element)
        for element in rhs.cregs:
            if element not in self.cregs:
                combined_cregs.append(element)
        circuit = QuantumCircuit(*combined_qregs, *combined_cregs)
        for instruction_context in itertools.chain(self.data, rhs.data):
            circuit.append(*instruction_context)
        return circuit

    def extend(self, rhs):
        """
        Append rhs to self if self contains compatible registers.

        Two circuits are compatible if they contain the same registers
        or if they contain different registers with unique names. The
        returned circuit will contain all unique registers between both
        circuits.

        Modify and return self.
        """
        # Check registers in LHS are compatible with RHS
        self._check_compatible_regs(rhs)

        # Add new registers
        for element in rhs.qregs:
            if element not in self.qregs:
                self.qregs.append(element)
        for element in rhs.cregs:
            if element not in self.cregs:
                self.cregs.append(element)

        # Add new gates
        for instruction_context in rhs.data:
            self.append(*instruction_context)
        return self

    @property
    def qubits(self):
        """
        Returns a list of quantum bits in the order that the registers had been added.
        """
        return [qbit for qreg in self.qregs for qbit in qreg]

    @property
    def clbits(self):
        """
        Returns a list of classical bits in the order that the registers had been added.
        """
        return [cbit for creg in self.cregs for cbit in creg]

    def __add__(self, rhs):
        """Overload + to implement self.combine."""
        return self.combine(rhs)

    def __iadd__(self, rhs):
        """Overload += to implement self.extend."""
        return self.extend(rhs)

    def __len__(self):
        """Return number of operations in circuit."""
        return len(self.data)

    def __getitem__(self, item):
        """Return indexed operation."""
        return self.data[item]

    @staticmethod
    def cast(value, _type):
        """Best effort to cast value to type. Otherwise, returns the value."""
        try:
            return _type(value)
        except (ValueError, TypeError):
            return value

    @staticmethod
    def _bit_argument_conversion(bit_representation, in_array):
        try:
            if _is_bit(bit_representation):
                # circuit.h(qr[0]) -> circuit.h([qr[0]])
                return [bit_representation]
            elif isinstance(bit_representation, Register):
                # circuit.h(qr) -> circuit.h([qr[0], qr[1]])
                return bit_representation[:]
            elif isinstance(QuantumCircuit.cast(bit_representation, int), int):
                # circuit.h(0) -> circuit.h([qr[0]])
                return [in_array[bit_representation]]
            elif isinstance(bit_representation, slice):
                # circuit.h(slice(0,2)) -> circuit.h([qr[0], qr[1]])
                return in_array[bit_representation]
            elif isinstance(bit_representation, list) and \
                    all(_is_bit(bit) for bit in bit_representation):
                # circuit.h([qr[0], qr[1]]) -> circuit.h([qr[0], qr[1]])
                return bit_representation
            elif isinstance(QuantumCircuit.cast(bit_representation, list), (range, list)):
                # circuit.h([0, 1])     -> circuit.h([qr[0], qr[1]])
                # circuit.h(range(0,2)) -> circuit.h([qr[0], qr[1]])
                return [in_array[index] for index in bit_representation]
            else:
                raise QiskitError('Not able to expand a %s (%s)' % (bit_representation,
                                                                    type(bit_representation)))
        except IndexError:
            raise QiskitError('Index out of range.')
        except TypeError:
            raise QiskitError('Type error handeling %s (%s)' % (bit_representation,
                                                                type(bit_representation)))

    def qbit_argument_conversion(self, qubit_representation):
        """
        Converts several qubit representations (such as indexes, range, etc)
        into a list of qubits.

        Args:
            qubit_representation (Object): representation to expand

        Returns:
            List(tuple): Where each tuple is a qubit.
        """
        return QuantumCircuit._bit_argument_conversion(qubit_representation, self.qubits)

    def cbit_argument_conversion(self, clbit_representation):
        """
        Converts several classical bit representations (such as indexes, range, etc)
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
            instruction (Instruction or Operation): Instruction instance to append
            qargs (list(argument)): qubits to attach instruction to
            cargs (list(argument)): clbits to attach instruction to

        Returns:
            Instruction: a handle to the instruction that was just added
        """
        # Convert input to instruction
        if not isinstance(instruction, Instruction) and hasattr(instruction, 'to_instruction'):
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
            QiskitError: if the gate is of a different shape than the wires
                it is being attached to.
        """
        if not isinstance(instruction, Instruction):
            raise QiskitError('object is not an Instruction.')

        # do some compatibility checks
        self._check_dups(qargs)
        self._check_qargs(qargs)
        self._check_cargs(cargs)

        # add the instruction onto the given wires
        instruction_context = instruction, qargs, cargs
        self.data.append(instruction_context)

        # track variable parameters in instruction
        for param_index, param in enumerate(instruction.params):
            if isinstance(param, Parameter):
                current_symbols = self.parameters

                if param in current_symbols:
                    self._parameter_table[param].append((instruction, param_index))
                else:
                    if param.name in {p.name for p in current_symbols}:
                        raise QiskitError(
                            'Name conflict on adding parameter: {}'.format(param.name))
                    self._parameter_table[param] = [(instruction, param_index)]

        return instruction

    def _attach(self, instruction, qargs, cargs):
        """DEPRECATED after 0.8"""
        self.append(instruction, qargs, cargs)

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
                raise QiskitError("QuantumCircuit parameters can be Registers or Integers."
                                  " If Integers, up to 2 arguments. QuantumCircuit was called"
                                  " with %s." % (regs,))

        for register in regs:
            if register in self.qregs or register in self.cregs:
                raise QiskitError("register name \"%s\" already exists"
                                  % register.name)
            if isinstance(register, QuantumRegister):
                self.qregs.append(register)
            elif isinstance(register, ClassicalRegister):
                self.cregs.append(register)
            else:
                raise QiskitError("expected a register")

    def _check_dups(self, qubits):
        """Raise exception if list of qubits contains duplicates."""
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise QiskitError("duplicate qubit arguments")

    def _check_qargs(self, qargs):
        """Raise exception if a qarg is not in this circuit or bad format."""
        if not all(isinstance(i, tuple) and
                   isinstance(i[0], QuantumRegister) and
                   isinstance(i[1], int) for i in qargs):
            raise QiskitError("qarg not (QuantumRegister, int) tuple")
        if not all(self.has_register(i[0]) for i in qargs):
            raise QiskitError("register not in this circuit")
        for qubit in qargs:
            qubit[0].check_range(qubit[1])

    def _check_cargs(self, cargs):
        """Raise exception if clbit is not in this circuit or bad format."""
        if not all(isinstance(i, tuple) and
                   isinstance(i[0], ClassicalRegister) and
                   isinstance(i[1], int) for i in cargs):
            raise QiskitError("carg not (ClassicalRegister, int) tuple")
        if not all(self.has_register(i[0]) for i in cargs):
            raise QiskitError("register not in this circuit")
        for clbit in cargs:
            clbit[0].check_range(clbit[1])

    def to_instruction(self, parameter_map=None):
        """Create an Instruction out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               instruction. If None, existing circuit parameters will also
               parameterize the instruction.

        Returns:
            Instruction: a composite instruction encapsulating this circuit
                (can be decomposed back)
        """
        from qiskit.converters.circuit_to_instruction import circuit_to_instruction
        return circuit_to_instruction(self, parameter_map)

    def decompose(self):
        """Call a decomposition pass on this circuit,
        to decompose one level (shallow decompose).

        Returns:
            QuantumCircuit: a circuit one level decomposed
        """
        from qiskit.transpiler.passes.decompose import Decompose
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
                        raise QiskitError("circuits are not compatible")

    def qasm(self):
        """Return OpenQASM string."""
        string_temp = self.header + "\n"
        string_temp += self.extension_lib + "\n"
        for register in self.qregs:
            string_temp += register.qasm() + "\n"
        for register in self.cregs:
            string_temp += register.qasm() + "\n"
        for instruction, qargs, cargs in self.data:
            if instruction.name == 'measure':
                qubit = qargs[0]
                clbit = cargs[0]
                string_temp += "%s %s[%d] -> %s[%d];\n" % (instruction.qasm(),
                                                           qubit[0].name, qubit[1],
                                                           clbit[0].name, clbit[1])
            else:
                string_temp += "%s %s;\n" % (instruction.qasm(),
                                             ",".join(["%s[%d]" % (j[0].name, j[1])
                                                       for j in qargs + cargs]))
        return string_temp

    def draw(self, scale=0.7, filename=None, style=None, output=None,
             interactive=False, line_length=None, plot_barriers=True,
             reverse_bits=False, justify=None):
        """Draw the quantum circuit

        Using the output parameter you can specify the format. The choices are:
        0. text: ASCII art string
        1. latex: high-quality images, but heavy external software dependencies
        2. matplotlib: purely in Python with no external dependencies

        Defaults to an overcomplete basis, in order to not alter gates.

        Args:
            scale (float): scale of image to draw (shrink if < 1)
            filename (str): file path to save image to
            style (dict or str): dictionary of style or file name of style
                file. You can refer to the
                :ref:`Style Dict Doc <style-dict-doc>` for more information
                on the contents.
            output (str): Select the output method to use for drawing the
                circuit. Valid choices are `text`, `latex`, `latex_source`,
                `mpl`. By default the 'text' drawer is used unless a user
                config file has an alternative backend set as the default. If
                the output is passed in that backend will always be used.
            interactive (bool): when set true show the circuit in a new window
                (for `mpl` this depends on the matplotlib backend being used
                supporting this). Note when used with either the `text` or the
                `latex_source` output type this has no effect and will be
                silently ignored.
            line_length (int): sets the length of the lines generated by `text`
            reverse_bits (bool): When set to True reverse the bit order inside
                registers for the output visualization.
            plot_barriers (bool): Enable/disable drawing barriers in the output
                circuit. Defaults to True.
            justify (string): Options are `left`, `right` or `none`, if anything
                else is supplied it defaults to left justified. It refers to where
                gates should be placed in the output circuit if there is an option.
                `none` results in each gate being placed in its own column. Currently
                only supported by text drawer.

        Returns:
            PIL.Image or matplotlib.figure or str or TextDrawing:
                * PIL.Image: (output `latex`) an in-memory representation of the
                  image of the circuit diagram.
                * matplotlib.figure: (output `mpl`) a matplotlib figure object
                  for the circuit diagram.
                * str: (output `latex_source`). The LaTeX source code.
                * TextDrawing: (output `text`). A drawing that can be printed as
                  ascii art

        Raises:
            VisualizationError: when an invalid output method is selected
        """
        # pylint: disable=cyclic-import
        from qiskit.visualization import circuit_drawer
        return circuit_drawer(self, scale=scale,
                              filename=filename, style=style,
                              output=output,
                              interactive=interactive,
                              line_length=line_length,
                              plot_barriers=plot_barriers,
                              reverse_bits=reverse_bits,
                              justify=justify)

    def size(self):
        """Returns total number of gate operations in circuit.

        Returns:
            int: Total number of gate operations.
        """
        gate_ops = 0
        for instr, _, _ in self.data:
            if instr.name not in ['barrier', 'snapshot']:
                gate_ops += 1
        return gate_ops

    def depth(self):
        """Return circuit depth (i.e. length of critical path).
        This does not include compiler or simulator directives
        such as 'barrier' or 'snapshot'.

        Returns:
            int: Depth of circuit.

        Notes:
            The circuit depth and the DAG depth need not bt the
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
        # We do not consider barriers or snapshots as
        # They are transpiler and simulator directives.
        # The max stack height is the circuit depth.
        for instr, qargs, cargs in self.data:
            if instr.name not in ['barrier', 'snapshot']:
                levels = []
                reg_ints = []
                for ind, reg in enumerate(qargs + cargs):
                    # Add to the stacks of the qubits and
                    # cbits used in the gate.
                    reg_ints.append(reg_map[reg[0].name] + reg[1])
                    levels.append(op_stack[reg_ints[ind]] + 1)
                if instr.control:
                    # Controls operate over all bits in the
                    # classical register they use.
                    cint = reg_map[instr.control[0].name]
                    for off in range(instr.control[0].size):
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

    def count_ops(self):
        """Count each operation kind in the circuit.

        Returns:
            dict: a breakdown of how many operations of each kind.
        """
        count_ops = {}
        for instr, _, _ in self.data:
            if instr.name in count_ops.keys():
                count_ops[instr.name] += 1
            else:
                count_ops[instr.name] = 1
        return count_ops

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
        for instr, qargs, cargs in self.data:
            if unitary_only:
                args = qargs
                num_qargs = len(args)
            else:
                args = qargs + cargs
                num_qargs = len(args) + (1 if instr.control else 0)

            if num_qargs >= 2 and instr.name not in ['barrier', 'snapshot']:
                graphs_touched = []
                num_touched = 0
                # Controls necessarily join all the cbits in the
                # register that they use.
                if instr.control and not unitary_only:
                    creg = instr.control[0]
                    creg_int = reg_map[creg.name]
                    for coff in range(creg.size):
                        temp_int = creg_int + coff
                        for k in range(num_sub_graphs):
                            if temp_int in sub_graphs[k]:
                                graphs_touched.append(k)
                                num_touched += 1
                                break

                for item in args:
                    reg_int = reg_map[item[0].name] + item[1]
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
            removed in a future release of qiskit. You should call
            `num_unitary_factors` instead.
        """
        return self.num_unitary_factors()

    def copy(self, name=None):
        """
        Args:
          name (str): name to be given to the copied circuit, if None then the name stays the same
        Returns:
          QuantumCircuit: a deepcopy of the current circuit, with the name updated if
                          it was provided
        """
        cpy = deepcopy(self)
        if name:
            cpy.name = name
        return cpy

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
    def parameters(self):
        """convenience function to get the parameters defined in the parameter table"""
        return set(self._parameter_table.keys())

    def bind_parameters(self, value_dict):
        """Assign parameters to values yielding a new circuit.

        Args:
            value_dict (dict): {parameter: value, ...}

        Raises:
            QiskitError: If value_dict contains parameters not present in the circuit

        Returns:
            QuantumCircuit: copy of self with assignment substitution.
        """
        new_circuit = self.copy()
        unrolled_value_dict = self._unroll_param_dict(value_dict)

        if unrolled_value_dict.keys() > self.parameters:
            raise QiskitError('Cannot bind parameters ({}) not present in the circuit.'.format(
                [str(p) for p in value_dict.keys() - self.parameters]))

        for parameter, value in unrolled_value_dict.items():
            new_circuit._bind_parameter(parameter, value)
        # clear evaluated expressions
        for parameter in unrolled_value_dict:
            del new_circuit._parameter_table[parameter]
        return new_circuit

    def _unroll_param_dict(self, value_dict):
        unrolled_value_dict = {}
        for (param, value) in value_dict.items():
            if isinstance(param, Parameter):
                unrolled_value_dict[param] = value
            if isinstance(param, ParameterVector):
                if not len(param) == len(value):
                    raise QiskitError('ParameterVector {} has length {}, which '
                                      'differs from value list {} of '
                                      'len {}'.format(param, len(param), value, len(value)))
                unrolled_value_dict.update(zip(param, value))
        return unrolled_value_dict

    def _bind_parameter(self, parameter, value):
        """Assigns a parameter value to matching instructions in-place."""
        for (instr, param_index) in self._parameter_table[parameter]:
            instr.params[param_index] = value

    def _substitute_parameters(self, parameter_map):
        """For every {existing_parameter: replacement_parameter} pair in
        parameter_map, substitute replacement for existing in all
        circuit instructions and the parameter table.
        """
        for old_parameter, new_parameter in parameter_map.items():
            self._bind_parameter(old_parameter, new_parameter)
            self._parameter_table[new_parameter] = self._parameter_table.pop(old_parameter)


def _circuit_from_qasm(qasm):
    # pylint: disable=cyclic-import
    from qiskit.converters import ast_to_dag
    from qiskit.converters import dag_to_circuit
    ast = qasm.parse()
    dag = ast_to_dag(ast)
    return dag_to_circuit(dag)
