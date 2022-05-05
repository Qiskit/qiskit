# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Common visualization utilities."""

import re
from collections import OrderedDict

import numpy as np

from qiskit.circuit import (
    BooleanExpression,
    Clbit,
    ControlledGate,
    Delay,
    Gate,
    Instruction,
    Measure,
    ControlFlowOp,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.tools import pi_check
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states import DensityMatrix
from qiskit.utils import optionals as _optionals
from qiskit.visualization.exceptions import VisualizationError


def get_gate_ctrl_text(op, drawer, style=None, calibrations=None):
    """Load the gate_text and ctrl_text strings based on names and labels"""
    op_label = getattr(op, "label", None)
    op_type = type(op)
    base_name = base_label = base_type = None
    if hasattr(op, "base_gate"):
        base_name = op.base_gate.name
        base_label = op.base_gate.label
        base_type = type(op.base_gate)
    ctrl_text = None

    if base_label:
        gate_text = base_label
        ctrl_text = op_label
    elif op_label and isinstance(op, ControlledGate):
        gate_text = base_name
        ctrl_text = op_label
    elif op_label:
        gate_text = op_label
    elif base_name:
        gate_text = base_name
    else:
        gate_text = op.name

    # raw_gate_text is used in color selection in mpl instead of op.name, since
    # if it's a controlled gate, the color will likely not be the base_name color
    raw_gate_text = op.name if gate_text == base_name else gate_text

    # For mpl and latex drawers, check style['disptex'] in qcstyle.py
    if drawer != "text" and gate_text in style["disptex"]:
        # First check if this entry is in the old style disptex that
        # included "$\\mathrm{  }$". If so, take it as is.
        if style["disptex"][gate_text][0] == "$" and style["disptex"][gate_text][-1] == "$":
            gate_text = style["disptex"][gate_text]
        else:
            gate_text = f"$\\mathrm{{{style['disptex'][gate_text]}}}$"

    elif drawer == "latex":
        # Special formatting for Booleans in latex (due to '~' causing crash)
        if (gate_text == op.name and op_type is BooleanExpression) or (
            gate_text == base_name and base_type is BooleanExpression
        ):
            gate_text = gate_text.replace("~", "$\\neg$").replace("&", "\\&")
            gate_text = f"$\\texttt{{{gate_text}}}$"
        # Capitalize if not a user-created gate or instruction
        elif (
            (gate_text == op.name and op_type not in (Gate, Instruction))
            or (gate_text == base_name and base_type not in (Gate, Instruction))
        ) and (op_type is not PauliEvolutionGate):
            gate_text = f"$\\mathrm{{{gate_text.capitalize()}}}$"
        else:
            gate_text = f"$\\mathrm{{{gate_text}}}$"
            # Remove mathmode _, ^, and - formatting from user names and labels
            gate_text = gate_text.replace("_", "\\_")
            gate_text = gate_text.replace("^", "\\string^")
            gate_text = gate_text.replace("-", "\\mbox{-}")
        ctrl_text = f"$\\mathrm{{{ctrl_text}}}$"

    # Only captitalize internally-created gate or instruction names
    elif (
        (gate_text == op.name and op_type not in (Gate, Instruction))
        or (gate_text == base_name and base_type not in (Gate, Instruction))
    ) and (op_type is not PauliEvolutionGate):
        gate_text = gate_text.capitalize()

    if drawer == "mpl" and op.name in calibrations:
        if isinstance(op, ControlledGate):
            ctrl_text = "" if ctrl_text is None else ctrl_text
            ctrl_text = "(cal)\n" + ctrl_text
        else:
            gate_text = gate_text + "\n(cal)"

    return gate_text, ctrl_text, raw_gate_text


def get_param_str(op, drawer, ndigits=3):
    """Get the params as a string to add to the gate text display"""
    if not hasattr(op, "params") or any(isinstance(param, np.ndarray) for param in op.params):
        return ""

    if isinstance(op, ControlFlowOp):
        return ""

    if isinstance(op, Delay):
        param_list = [f"{op.params[0]}[{op.unit}]"]
    else:
        param_list = []
        for count, param in enumerate(op.params):
            # Latex drawer will cause an xy-pic error and mpl drawer will overwrite
            # the right edge if param string too long, so limit params.
            if (drawer == "latex" and count > 3) or (drawer == "mpl" and count > 15):
                param_list.append("...")
                break
            try:
                param_list.append(pi_check(param, output=drawer, ndigits=ndigits))
            except TypeError:
                param_list.append(str(param))

    param_str = ""
    if param_list:
        if drawer == "latex":
            param_str = f"\\,(\\mathrm{{{','.join(param_list)}}})"
        elif drawer == "mpl":
            param_str = f"{', '.join(param_list)}".replace("-", "$-$")
        else:
            param_str = f"({','.join(param_list)})"

    return param_str


def get_wire_map(circuit, bits, cregbundle):
    """Map the bits and registers to the index from the top of the drawing.
    The key to the dict is either the (Qubit, Clbit) or if cregbundle True,
    the register that is being bundled.

    Args:
        circuit (QuantumCircuit): the circuit being drawn
        bits (list(Qubit, Clbit)): the Qubit's and Clbit's in the circuit
        cregbundle (bool): if True bundle classical registers. Default: ``True``.

    Returns:
        dict((Qubit, Clbit, ClassicalRegister): index): map of bits/registers
            to index
    """
    prev_reg = None
    wire_index = 0
    wire_map = {}
    for bit in bits:
        register = get_bit_register(circuit, bit)
        if register is None or not isinstance(bit, Clbit) or not cregbundle:
            wire_map[bit] = wire_index
            wire_index += 1
        elif register is not None and cregbundle and register != prev_reg:
            prev_reg = register
            wire_map[register] = wire_index
            wire_index += 1

    return wire_map


def get_bit_register(circuit, bit):
    """Get the register for a bit if there is one

    Args:
        circuit (QuantumCircuit): the circuit being drawn
        bit (Qubit, Clbit): the bit to use to find the register and indexes

    Returns:
        ClassicalRegister: register associated with the bit
    """
    bit_loc = circuit.find_bit(bit)
    return bit_loc.registers[0][0] if bit_loc.registers else None


def get_bit_reg_index(circuit, bit, reverse_bits):
    """Get the register for a bit if there is one, and the index of the bit
    from the top of the circuit, or the index of the bit within a register.

    Args:
        circuit (QuantumCircuit): the circuit being drawn
        bit (Qubit, Clbit): the bit to use to find the register and indexes
        reverse_bits (bool): if True reverse order of the bits. Default: ``False``.

    Returns:
        (ClassicalRegister, None): register associated with the bit
        int: index of the bit from the top of the circuit
        int: index of the bit within the register, if there is a register
    """
    bit_loc = circuit.find_bit(bit)
    bit_index = bit_loc.index
    register, reg_index = bit_loc.registers[0] if bit_loc.registers else (None, None)
    if register is not None and reverse_bits:
        bits_len = len(circuit.clbits) if isinstance(bit, Clbit) else len(circuit.qubits)
        bit_index = bits_len - bit_index - 1

    return register, bit_index, reg_index


def get_wire_label(drawer, register, index, layout=None, cregbundle=True):
    """Get the bit labels to display to the left of the wires.

    Args:
        drawer (str): which drawer is calling ("text", "mpl", or "latex")
        register (QuantumRegister or ClassicalRegister): get wire_label for this register
        index (int): index of bit in register
        layout (Layout): Optional. mapping of virtual to physical bits
        cregbundle (bool): Optional. if set True bundle classical registers.
            Default: ``True``.

    Returns:
        str: label to display for the register/index
    """
    index_str = f"{index}" if drawer == "text" else f"{{{index}}}"
    if register is None:
        wire_label = index_str
        return wire_label

    if drawer == "text":
        reg_name = f"{register.name}"
        reg_name_index = f"{register.name}_{index}"
    else:
        reg_name = f"{{{fix_special_characters(register.name)}}}"
        reg_name_index = f"{reg_name}_{{{index}}}"

    # Clbits
    if isinstance(register, ClassicalRegister):
        if cregbundle and drawer != "latex":
            wire_label = f"{register.name}"
            return wire_label

        if register.size == 1 or cregbundle:
            wire_label = reg_name
        else:
            wire_label = reg_name_index
        return wire_label

    # Qubits
    if register.size == 1:
        wire_label = reg_name
    elif layout is None:
        wire_label = reg_name_index
    elif layout[index]:
        virt_bit = layout[index]
        try:
            virt_reg = next(reg for reg in layout.get_registers() if virt_bit in reg)
            if drawer == "text":
                wire_label = f"{virt_reg.name}_{virt_reg[:].index(virt_bit)} -> {index}"
            else:
                wire_label = (
                    f"{{{virt_reg.name}}}_{{{virt_reg[:].index(virt_bit)}}} \\mapsto {{{index}}}"
                )
        except StopIteration:
            if drawer == "text":
                wire_label = f"{virt_bit} -> {index}"
            else:
                wire_label = f"{{{virt_bit}}} \\mapsto {{{index}}}"
        if drawer != "text":
            wire_label = wire_label.replace(" ", "\\;")  # use wider spaces
    else:
        wire_label = index_str

    return wire_label


def get_condition_label_val(condition, circuit, cregbundle, reverse_bits):
    """Get the label and value list to display a condition

    Args:
        condition (Union[Clbit, ClassicalRegister], int): classical condition
        circuit (QuantumCircuit): the circuit that is being drawn
        cregbundle (bool): if set True bundle classical registers
        reverse_bits (bool): if set True reverse the bit order

    Returns:
        str: label to display for the condition
        list(str): list of 1's and 0's indicating values of condition
    """
    cond_is_bit = bool(isinstance(condition[0], Clbit))
    cond_val = int(condition[1])

    # if condition on a register, return list of 1's and 0's indicating
    # closed or open, else only one element is returned
    if isinstance(condition[0], ClassicalRegister) and not cregbundle:
        val_bits = list(f"{cond_val:0{condition[0].size}b}")
        if not reverse_bits:
            val_bits = val_bits[::-1]
    else:
        val_bits = list(str(cond_val))

    label = ""
    if cond_is_bit and cregbundle:
        register, _, reg_index = get_bit_reg_index(circuit, condition[0], False)
        if register is not None:
            label = f"{register.name}_{reg_index}={hex(cond_val)}"
    elif not cond_is_bit:
        label = hex(cond_val)

    return label, val_bits


def fix_special_characters(label):
    """
    Convert any special characters for mpl and latex drawers.
    Currently only checks for multiple underscores in register names
    and uses wider space for mpl and latex drawers.

    Args:
        label (str): the label to fix

    Returns:
        str: label to display
    """
    label = label.replace("_", r"\_").replace(" ", "\\;")
    return label


@_optionals.HAS_PYLATEX.require_in_call("the latex and latex_source circuit drawers")
def generate_latex_label(label):
    """Convert a label to a valid latex string."""
    from pylatexenc.latexencode import utf8tolatex

    regex = re.compile(r"(?<!\\)\$(.*)(?<!\\)\$")
    match = regex.search(label)
    if not match:
        label = label.replace(r"\$", "$")
        final_str = utf8tolatex(label, non_ascii_only=True)
    else:
        mathmode_string = match.group(1).replace(r"\$", "$")
        before_match = label[: match.start()]
        before_match = before_match.replace(r"\$", "$")
        after_match = label[match.end() :]
        after_match = after_match.replace(r"\$", "$")
        final_str = (
            utf8tolatex(before_match, non_ascii_only=True)
            + mathmode_string
            + utf8tolatex(after_match, non_ascii_only=True)
        )
    return final_str.replace(" ", "\\,")  # Put in proper spaces


@_optionals.HAS_PIL.require_in_call("the latex circuit drawer")
def _trim(image):
    """Trim a PIL image and remove white space."""
    import PIL

    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image


def _get_layered_instructions(circuit, reverse_bits=False, justify=None, idle_wires=True):
    """
    Given a circuit, return a tuple (qubits, clbits, nodes) where
    qubits and clbits are the quantum and classical registers
    in order (based on reverse_bits) and nodes is a list
    of DAG nodes whose type is "operation".

    Args:
        circuit (QuantumCircuit): From where the information is extracted.
        reverse_bits (bool): If true the order of the bits in the registers is
            reversed.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
    Returns:
        Tuple(list,list,list): To be consumed by the visualizer directly.
    """
    if justify:
        justify = justify.lower()

    # default to left
    justify = justify if justify in ("right", "none") else "left"

    dag = circuit_to_dag(circuit)

    nodes = []
    qubits = dag.qubits
    clbits = dag.clbits

    # Create a mapping of each register to the max layer number for all measure ops
    # with that register as the target. Then when an op with condition is seen,
    # it will be placed to the right of the measure op if the register matches.
    measure_map = OrderedDict([(c, -1) for c in clbits])

    if justify == "none":
        for node in dag.topological_op_nodes():
            nodes.append([node])
    else:
        nodes = _LayerSpooler(dag, justify, measure_map, reverse_bits)

    if reverse_bits:
        qubits.reverse()
        clbits.reverse()

    # Optionally remove all idle wires and instructions that are on them and
    # on them only.
    if not idle_wires:
        for wire in dag.idle_wires(ignore=["barrier", "delay"]):
            if wire in qubits:
                qubits.remove(wire)
            if wire in clbits:
                clbits.remove(wire)

    nodes = [[node for node in layer if any(q in qubits for q in node.qargs)] for layer in nodes]

    return qubits, clbits, nodes


def _sorted_nodes(dag_layer):
    """Convert DAG layer into list of nodes sorted by node_id
    qiskit-terra #2802
    """
    nodes = dag_layer["graph"].op_nodes()
    # sort into the order they were input
    nodes.sort(key=lambda nd: nd._node_id)
    return nodes


def _get_gate_span(qubits, node, reverse_bits):
    """Get the list of qubits drawing this gate would cover
    qiskit-terra #2802
    """
    min_index = len(qubits)
    max_index = 0
    for qreg in node.qargs:
        index = qubits.index(qreg)

        if index < min_index:
            min_index = index
        if index > max_index:
            max_index = index

    if node.cargs or node.op.condition:
        if reverse_bits:
            return qubits[: max_index + 1]
        else:
            return qubits[min_index : len(qubits)]

    return qubits[min_index : max_index + 1]


def _any_crossover(qubits, node, nodes, reverse_bits):
    """Return True .IFF. 'node' crosses over any 'nodes'."""
    gate_span = _get_gate_span(qubits, node, reverse_bits)
    all_indices = []
    for check_node in nodes:
        if check_node != node:
            all_indices += _get_gate_span(qubits, check_node, reverse_bits)
    return any(i in gate_span for i in all_indices)


class _LayerSpooler(list):
    """Manipulate list of layer dicts for _get_layered_instructions."""

    def __init__(self, dag, justification, measure_map, reverse_bits):
        """Create spool"""
        super().__init__()
        self.dag = dag
        self.qubits = dag.qubits
        self.clbits = dag.clbits
        self.justification = justification
        self.measure_map = measure_map
        self.cregs = [self.dag.cregs[reg] for reg in self.dag.cregs]
        self.reverse_bits = reverse_bits

        if self.justification == "left":
            for dag_layer in dag.layers():
                current_index = len(self) - 1
                dag_nodes = _sorted_nodes(dag_layer)
                for node in dag_nodes:
                    self.add(node, current_index)
        else:
            dag_layers = []
            for dag_layer in dag.layers():
                dag_layers.append(dag_layer)

            # going right to left!
            dag_layers.reverse()

            for dag_layer in dag_layers:
                current_index = 0
                dag_nodes = _sorted_nodes(dag_layer)
                for node in dag_nodes:
                    self.add(node, current_index)

    def is_found_in(self, node, nodes):
        """Is any qreq in node found in any of nodes?"""
        all_qargs = []
        for a_node in nodes:
            for qarg in a_node.qargs:
                all_qargs.append(qarg)
        return any(i in node.qargs for i in all_qargs)

    def insertable(self, node, nodes):
        """True .IFF. we can add 'node' to layer 'nodes'"""
        return not _any_crossover(self.qubits, node, nodes, self.reverse_bits)

    def slide_from_left(self, node, index):
        """Insert node into first layer where there is no conflict going l > r"""
        measure_layer = None
        if isinstance(node.op, Measure):
            measure_bit = next(bit for bit in self.measure_map if node.cargs[0] == bit)

        if not self:
            inserted = True
            self.append([node])
        else:
            inserted = False
            curr_index = index
            last_insertable_index = -1
            index_stop = -1
            if node.op.condition:
                if isinstance(node.op.condition[0], Clbit):
                    cond_bit = [clbit for clbit in self.clbits if node.op.condition[0] == clbit]
                    index_stop = self.measure_map[cond_bit[0]]
                else:
                    for bit in node.op.condition[0]:
                        max_index = -1
                        if bit in self.measure_map:
                            if self.measure_map[bit] > max_index:
                                index_stop = max_index = self.measure_map[bit]
            if node.cargs:
                for carg in node.cargs:
                    try:
                        carg_bit = next(bit for bit in self.measure_map if carg == bit)
                        if self.measure_map[carg_bit] > index_stop:
                            index_stop = self.measure_map[carg_bit]
                    except StopIteration:
                        pass
            while curr_index > index_stop:
                if self.is_found_in(node, self[curr_index]):
                    break
                if self.insertable(node, self[curr_index]):
                    last_insertable_index = curr_index
                curr_index = curr_index - 1

            if last_insertable_index >= 0:
                inserted = True
                self[last_insertable_index].append(node)
                measure_layer = last_insertable_index
            else:
                inserted = False
                curr_index = index
                while curr_index < len(self):
                    if self.insertable(node, self[curr_index]):
                        self[curr_index].append(node)
                        measure_layer = curr_index
                        inserted = True
                        break
                    curr_index = curr_index + 1

        if not inserted:
            self.append([node])

        if isinstance(node.op, Measure):
            if not measure_layer:
                measure_layer = len(self) - 1
            if measure_layer > self.measure_map[measure_bit]:
                self.measure_map[measure_bit] = measure_layer

    def slide_from_right(self, node, index):
        """Insert node into rightmost layer as long there is no conflict."""
        if not self:
            self.insert(0, [node])
            inserted = True
        else:
            inserted = False
            curr_index = index
            last_insertable_index = None

            while curr_index < len(self):
                if self.is_found_in(node, self[curr_index]):
                    break
                if self.insertable(node, self[curr_index]):
                    last_insertable_index = curr_index
                curr_index = curr_index + 1

            if last_insertable_index:
                self[last_insertable_index].append(node)
                inserted = True
            else:
                curr_index = index
                while curr_index > -1:
                    if self.insertable(node, self[curr_index]):
                        self[curr_index].append(node)
                        inserted = True
                        break
                    curr_index = curr_index - 1

        if not inserted:
            self.insert(0, [node])

    def add(self, node, index):
        """Add 'node' where it belongs, starting the try at 'index'."""
        if self.justification == "left":
            self.slide_from_left(node, index)
        else:
            self.slide_from_right(node, index)


def _bloch_multivector_data(state):
    """Return list of Bloch vectors for each qubit

    Args:
        state (DensityMatrix or Statevector): an N-qubit state.

    Returns:
        list: list of Bloch vectors (x, y, z) for each qubit.

    Raises:
        VisualizationError: if input is not an N-qubit state.
    """
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    pauli_singles = PauliList(["X", "Y", "Z"])
    bloch_data = []
    for i in range(num):
        if num > 1:
            paulis = PauliList.from_symplectic(
                np.zeros((3, (num - 1)), dtype=bool), np.zeros((3, (num - 1)), dtype=bool)
            ).insert(i, pauli_singles, qubit=True)
        else:
            paulis = pauli_singles
        bloch_state = [np.real(np.trace(np.dot(mat, rho.data))) for mat in paulis.matrix_iter()]
        bloch_data.append(bloch_state)
    return bloch_data


def _paulivec_data(state):
    """Return paulivec data for plotting.

    Args:
        state (DensityMatrix or Statevector): an N-qubit state.

    Returns:
        tuple: (labels, values) for Pauli vector.

    Raises:
        VisualizationError: if input is not an N-qubit state.
    """
    rho = SparsePauliOp.from_operator(DensityMatrix(state))
    if rho.num_qubits is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    return rho.paulis.to_labels(), np.real(rho.coeffs)


MATPLOTLIB_INLINE_BACKENDS = {
    "module://ipykernel.pylab.backend_inline",
    "module://matplotlib_inline.backend_inline",
    "nbAgg",
}


def matplotlib_close_if_inline(figure):
    """Close the given matplotlib figure if the backend in use draws figures inline.

    If the backend does not draw figures inline, this does nothing.  This function is to prevent
    duplicate images appearing; the inline backends will capture the figure in preparation and
    display it as well, whereas the drawers want to return the figure to be displayed."""
    # This can only called if figure has already been created, so matplotlib must exist.
    import matplotlib.pyplot

    if matplotlib.get_backend() in MATPLOTLIB_INLINE_BACKENDS:
        matplotlib.pyplot.close(figure)
