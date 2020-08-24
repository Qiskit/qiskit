# -*- coding: utf-8 -*-

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

# pylint: disable=anomalous-backslash-in-string

"""Common visualization utilities."""

import re

import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators import PauliTable, SparsePauliOp
from qiskit.visualization.exceptions import VisualizationError

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
        return utf8tolatex(label)
    else:
        mathmode_string = match.group(1).replace(r'\$', '$')
        before_match = label[:match.start()]
        before_match = before_match.replace(r'\$', '$')
        after_match = label[match.end():]
        after_match = after_match.replace(r'\$', '$')
        return utf8tolatex(before_match) + mathmode_string + utf8tolatex(
            after_match)


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
    Given a circuit, return a tuple (qregs, cregs, ops) where
    qregs and cregs are the quantum and classical registers
    in order (based on reverse_bits) and ops is a list
    of DAG nodes which type is "operation".

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
    qregs = dag.qubits
    cregs = dag.clbits

    if justify == 'none':
        for node in dag.topological_op_nodes():
            ops.append([node])

    else:
        ops = _LayerSpooler(dag, justify)

    if reverse_bits:
        qregs.reverse()
        cregs.reverse()

    if not idle_wires:
        for wire in dag.idle_wires(ignore=['barrier']):
            if wire in qregs:
                qregs.remove(wire)
            if wire in cregs:
                cregs.remove(wire)

    return qregs, cregs, ops


def _sorted_nodes(dag_layer):
    """Convert DAG layer into list of nodes sorted by node_id
    qiskit-terra #2802
    """
    dag_instructions = dag_layer['graph'].op_nodes()
    # sort into the order they were input
    dag_instructions.sort(key=lambda nd: nd._node_id)
    return dag_instructions


def _get_gate_span(qregs, instruction):
    """Get the list of qubits drawing this gate would cover
    qiskit-terra #2802
    """
    min_index = len(qregs)
    max_index = 0
    for qreg in instruction.qargs:
        index = qregs.index(qreg)

        if index < min_index:
            min_index = index
        if index > max_index:
            max_index = index

    if instruction.cargs:
        return qregs[min_index:]
    if instruction.condition:
        return qregs[min_index:]

    return qregs[min_index:max_index + 1]


def _any_crossover(qregs, node, nodes):
    """Return True .IFF. 'node' crosses over any in 'nodes',"""
    gate_span = _get_gate_span(qregs, node)
    all_indices = []
    for check_node in nodes:
        if check_node != node:
            all_indices += _get_gate_span(qregs, check_node)
    return any(i in gate_span for i in all_indices)


class _LayerSpooler(list):
    """Manipulate list of layer dicts for _get_layered_instructions."""

    def __init__(self, dag, justification):
        """Create spool"""
        super(_LayerSpooler, self).__init__()
        self.dag = dag
        self.qregs = dag.qubits
        self.justification = justification

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
        return not _any_crossover(self.qregs, node, nodes)

    def slide_from_left(self, node, index):
        """Insert node into first layer where there is no conflict going l > r"""
        if not self:
            self.append([node])
            inserted = True

        else:
            inserted = False
            curr_index = index
            last_insertable_index = None

            while curr_index > -1:
                if self.is_found_in(node, self[curr_index]):
                    break
                if self.insertable(node, self[curr_index]):
                    last_insertable_index = curr_index
                curr_index = curr_index - 1

            if last_insertable_index:
                self[last_insertable_index].append(node)
                inserted = True

            else:
                inserted = False
                curr_index = index
                while curr_index < len(self):
                    if self.insertable(node, self[curr_index]):
                        self[curr_index].append(node)
                        inserted = True
                        break
                    curr_index = curr_index + 1

        if not inserted:
            self.append([node])

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
    """Return list of bloch vectors for each qubit

    Args:
        state (DensityMatrix or Statevector): an N-qubit state.

    Returns:
        list: list of bloch vectors (x, y, z) for each qubit.

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
            paulis = PauliTable(np.zeros((3, 2 * (num-1)), dtype=np.bool)).insert(
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
        tuple: (labels, values) for Pauli vec.

    Raises:
        VisualizationError: if input is not an N-qubit state.
    """
    rho = SparsePauliOp.from_operator(DensityMatrix(state))
    if rho.num_qubits is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    return rho.table.to_labels(), np.real(rho.coeffs)
