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
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators.symplectic import PauliTable, SparsePauliOp
from qiskit.visualization.exceptions import VisualizationError
from qiskit.circuit import Measure

try:
    import PIL
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from pylatexenc.latexencode import utf8tolatex

    HAS_PYLATEX = True
except ImportError:
    HAS_PYLATEX = False


def generate_latex_label(label):
    """Convert a label to a valid latex string."""
    if not HAS_PYLATEX:
        raise ImportError('The latex and latex_source drawers need '
                          'pylatexenc installed. Run "pip install '
                          'pylatexenc" before using the latex or '
                          'latex_source drawers.')

    regex = re.compile(r"(?<!\\)\$(.*)(?<!\\)\$")
    match = regex.search(label)
    if not match:
        label = label.replace(r'\$', '$')
        final_str = utf8tolatex(label, non_ascii_only=True)
    else:
        mathmode_string = match.group(1).replace(r'\$', '$')
        before_match = label[:match.start()]
        before_match = before_match.replace(r'\$', '$')
        after_match = label[match.end():]
        after_match = after_match.replace(r'\$', '$')
        final_str = (utf8tolatex(before_match, non_ascii_only=True) + mathmode_string
                     + utf8tolatex(after_match, non_ascii_only=True))
    return final_str.replace(' ', '\\,')   # Put in proper spaces


def _trim(image):
    """Trim a PIL image and remove white space."""
    if not HAS_PIL:
        raise ImportError('The latex drawer needs pillow installed. '
                          'Run "pip install pillow" before using the '
                          'latex drawer.')
    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image


def _get_layered_instructions(circuit, reverse_bits=False,
                              justify=None, idle_wires=True):
    """
    Given a circuit, return a tuple (qubits, clbits, ops) where
    qubits and clbits are the quantum and classical registers
    in order (based on reverse_bits) and ops is a list
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
    justify = justify if justify in ('right', 'none') else 'left'

    dag = circuit_to_dag(circuit)

    ops = []
    qubits = dag.qubits
    clbits = dag.clbits

    # Create a mapping of each register to the max layer number for all measure ops
    # with that register as the target. Then when a node with condition is seen,
    # it will be placed to the right of the measure op if the register matches.
    measure_map = OrderedDict([(c, -1) for c in circuit.cregs])

    if justify == 'none':
        for node in dag.topological_op_nodes():
            ops.append([node])
    else:
        ops = _LayerSpooler(dag, justify, measure_map)

    if reverse_bits:
        qubits.reverse()
        clbits.reverse()

    # Optionally remove all idle wires and instructions that are on them and
    # on them only.
    if not idle_wires:
        for wire in dag.idle_wires(ignore=['barrier', 'delay']):
            if wire in qubits:
                qubits.remove(wire)
            if wire in clbits:
                clbits.remove(wire)

    ops = [[op for op in layer if any(q in qubits for q in op.qargs)]
           for layer in ops]

    return qubits, clbits, ops


def _sorted_nodes(dag_layer):
    """Convert DAG layer into list of nodes sorted by node_id
    qiskit-terra #2802
    """
    dag_instructions = dag_layer['graph'].op_nodes()
    # sort into the order they were input
    dag_instructions.sort(key=lambda nd: nd._node_id)
    return dag_instructions


def _get_gate_span(qubits, instruction):
    """Get the list of qubits drawing this gate would cover
    qiskit-terra #2802
    """
    min_index = len(qubits)
    max_index = 0
    for qreg in instruction.qargs:
        index = qubits.index(qreg)

        if index < min_index:
            min_index = index
        if index > max_index:
            max_index = index

    if instruction.cargs:
        return qubits[min_index:]
    if instruction.condition:
        return qubits[min_index:]

    return qubits[min_index:max_index + 1]


def _any_crossover(qubits, node, nodes):
    """Return True .IFF. 'node' crosses over any in 'nodes',"""
    gate_span = _get_gate_span(qubits, node)
    all_indices = []
    for check_node in nodes:
        if check_node != node:
            all_indices += _get_gate_span(qubits, check_node)
    return any(i in gate_span for i in all_indices)


class _LayerSpooler(list):
    """Manipulate list of layer dicts for _get_layered_instructions."""

    def __init__(self, dag, justification, measure_map):
        """Create spool"""
        super().__init__()
        self.dag = dag
        self.qubits = dag.qubits
        self.justification = justification
        self.measure_map = measure_map

        if self.justification == 'left':
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
        return not _any_crossover(self.qubits, node, nodes)

    def slide_from_left(self, node, index):
        """Insert node into first layer where there is no conflict going l > r"""
        measure_layer = None
        if isinstance(node.op, Measure):
            measure_reg = next(reg for reg in self.measure_map
                               if node.cargs[0] in reg)

        if not self:
            inserted = True
            self.append([node])
        else:
            inserted = False
            curr_index = index
            last_insertable_index = -1
            index_stop = -1
            if node.condition:
                index_stop = self.measure_map[node.condition[0]]
            elif node.cargs:
                for carg in node.cargs:
                    try:
                        carg_reg = next(reg for reg in self.measure_map
                                        if carg in reg)
                        if self.measure_map[carg_reg] > index_stop:
                            index_stop = self.measure_map[carg_reg]
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
            if measure_layer > self.measure_map[measure_reg]:
                self.measure_map[measure_reg] = measure_layer

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
    pauli_singles = PauliTable.from_labels(['X', 'Y', 'Z'])
    bloch_data = []
    for i in range(num):
        if num > 1:
            paulis = PauliTable(np.zeros((3, 2 * (num-1)), dtype=bool)).insert(
                i, pauli_singles, qubit=True)
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
    return rho.table.to_labels(), np.real(rho.coeffs)
