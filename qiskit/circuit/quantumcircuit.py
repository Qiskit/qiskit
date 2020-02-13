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
import warnings
import multiprocessing as mp
from collections import OrderedDict
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.qasm.qasm import Qasm
from qiskit.circuit.exceptions import CircuitError
from .parameterexpression import ParameterExpression
from .quantumregister import QuantumRegister, Qubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterTable
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData


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

    def __init__(self, *regs, name=None):
        if name is None:
            name = self.cls_prefix() + str(self.cls_instances())
            # pylint: disable=not-callable
            # (known pylint bug: https://github.com/PyCQA/pylint/issues/1699)
            if sys.platform != "win32":
                if isinstance(mp.current_process(),
                              (mp.context.ForkProcess, mp.context.SpawnProcess)):
                    name += '-{}'.format(mp.current_process().pid)
                elif sys.version_info[0] == 3 \
                    and (sys.version_info[1] == 5 or sys.version_info[1] == 6) \
                        and mp.current_process().name != 'MainProcess':
                    # It seems condition of if-statement doesn't work in python 3.5 and 3.6
                    # because processes created by "ProcessPoolExecutor" are not
                    # mp.context.ForkProcess or mp.context.SpawnProcess. As a workaround,
                    # "name" of the process is checked instead.
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
        self.add_register(*regs)

        # Parameter table tracks instructions with variable parameters.
        self._parameter_table = ParameterTable()

        self._layout = None

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
        """Mirror the circuit by reversing the instructions.

        This is done by recursively mirroring all instructions.
        It does not invert any gate.

        Returns:
            QuantumCircuit: the mirrored circuit
        """
        reverse_circ = self.copy(name=self.name + '_mirror')
        reverse_circ._data = []
        for inst, qargs, cargs in reversed(self.data):
            reverse_circ.append(inst.mirror(), qargs, cargs)
        return reverse_circ

    def inverse(self):
        """Invert this circuit.

        This is done by recursively inverting all gates.

        Returns:
            QuantumCircuit: the inverted circuit

        Raises:
            CircuitError: if the circuit cannot be inverted.
        """
        inverse_circ = self.copy(name=self.name + '_dg')
        inverse_circ._data = []
        for inst, qargs, cargs in reversed(self._data):
            inverse_circ._data.append((inst.inverse(), qargs, cargs))
        return inverse_circ

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
            circuit._append(*instruction_context)
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
        for element in rhs.cregs:
            if element not in self.cregs:
                self.cregs.append(element)

        # Add new gates
        for instruction_context in rhs.data:
            self._append(*instruction_context)
        return self

    @property
    def qubits(self):
        """
        Returns a list of quantum bits in the order that the registers were added.
        """
        return [qbit for qreg in self.qregs for qbit in qreg]

    @property
    def clbits(self):
        """
        Returns a list of classical bits in the order that the registers were added.
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
                current_parameters = self.parameters

                for parameter in param.parameters:
                    if parameter in current_parameters:
                        if not self._check_dup_param_spec(self._parameter_table[parameter],
                                                          instruction, param_index):
                            self._parameter_table[parameter].append((instruction, param_index))
                    else:
                        if parameter.name in {p.name for p in current_parameters}:
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
            if isinstance(register, QuantumRegister):
                self.qregs.append(register)
            elif isinstance(register, ClassicalRegister):
                self.cregs.append(register)
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
            Instruction: a composite instruction encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_instruction import circuit_to_instruction
        return circuit_to_instruction(self, parameter_map)

    def to_gate(self, parameter_map=None):
        """Create a Gate out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               gate. If None, existing circuit parameters will also
               parameterize the gate.

        Returns:
            Gate: a composite gate encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_gate import circuit_to_gate
        return circuit_to_gate(self, parameter_map)

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

    def qasm(self):
        """Return OpenQASM string."""
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
            else:
                string_temp += "%s %s;\n" % (instruction.qasm(),
                                             ",".join(["%s[%d]" % (j.register.name, j.index)
                                                       for j in qargs + cargs]))
            if instruction.name == 'unitary':
                unitary_gates.append(instruction)

        # this resets them, so if another call to qasm() is made the gate def is added again
        for gate in unitary_gates:
            gate._qasm_def_written = False
        return string_temp

    def draw(self, output=None, scale=0.7, filename=None, style=None,
             interactive=False, line_length=None, plot_barriers=True,
             reverse_bits=False, justify=None, vertical_compression='medium', idle_wires=True,
             with_layout=True, fold=None, ax=None):
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
            line_length (int): Deprecated; see `fold` which supersedes this
                option. Sets the length of the lines generated by `text` output
                type. This is useful when the drawing does not fit in the console.
                If None (default), it will try to guess the console width using
                ``shutil.get_terminal_size()``. However, if you're running in
                jupyter, the default line length is set to 80 characters. If you
                don't want pagination at all, set ``line_length=-1``.
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
                              line_length=line_length,
                              plot_barriers=plot_barriers,
                              reverse_bits=reverse_bits,
                              justify=justify,
                              vertical_compression=vertical_compression,
                              idle_wires=idle_wires,
                              with_layout=with_layout,
                              fold=fold,
                              ax=ax)

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
    def n_qubits(self):
        """
        Return number of qubits.
        """
        qubits = 0
        for reg in self.qregs:
            qubits += reg.size
        return qubits

    def count_ops(self):
        """Count each operation kind in the circuit.

        Returns:
            OrderedDict: a breakdown of how many operations of each kind, sorted by amount.
        """
        count_ops = {}
        for instr, _, _ in self._data:
            if instr.name in count_ops.keys():
                count_ops[instr.name] += 1
            else:
                count_ops[instr.name] = 1
        return OrderedDict(sorted(count_ops.items(), key=lambda kv: kv[1], reverse=True))

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
        cpy = deepcopy(self)
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

        new_creg = circ._create_creg(len(circ.qubits), 'measure')
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
    def parameters(self):
        """Convenience function to get the parameters defined in the parameter table."""
        return set(self._parameter_table.keys())

    def bind_parameters(self, value_dict):
        """Assign parameters to values yielding a new circuit.

        Args:
            value_dict (dict): {parameter: value, ...}

        Raises:
            CircuitError: If value_dict contains parameters not present in the circuit

        Returns:
            QuantumCircuit: copy of self with assignment substitution.
        """
        new_circuit = self.copy()
        unrolled_value_dict = self._unroll_param_dict(value_dict)

        if unrolled_value_dict.keys() > self.parameters:
            raise CircuitError('Cannot bind parameters ({}) not present in the circuit.'.format(
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

    def _rebind_definition(self, instruction, parameter, value):
        if instruction._definition:
            for op, _, _ in instruction._definition:
                for idx, param in enumerate(op.params):
                    if isinstance(param, ParameterExpression) and parameter in param.parameters:
                        op.params[idx] = param.bind({parameter: value})
                        self._rebind_definition(op, parameter, value)

    def _substitute_parameters(self, parameter_map):
        """For every {existing_parameter: replacement_parameter} pair in
        parameter_map, substitute replacement for existing in all
        circuit instructions and the parameter table.
        """
        for old_parameter, new_parameter in parameter_map.items():
            for (instr, param_index) in self._parameter_table[old_parameter]:
                new_param = instr.params[param_index].subs({old_parameter: new_parameter})
                instr.params[param_index] = new_param
            self._parameter_table[new_parameter] = self._parameter_table.pop(old_parameter)


def _circuit_from_qasm(qasm):
    # pylint: disable=cyclic-import
    from qiskit.converters import ast_to_dag
    from qiskit.converters import dag_to_circuit
    ast = qasm.parse()
    dag = ast_to_dag(ast)
    return dag_to_circuit(dag)
