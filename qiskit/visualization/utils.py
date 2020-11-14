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
from collections import OrderedDict

import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators import PauliTable, SparsePauliOp
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

    for x in dag.nodes():
        print('Node', x._op)
    for x in dag.layers():
        print('\npart', x['partition'])
        print('layer', x['graph'])
        print("DAG**********", x['graph'].node_counter, '\n')
        for y in x['graph'].op_nodes():
            print('Node', y.type, y._op)
    print()

    ops = []
    qregs = dag.qubits
    cregs = dag.clbits

    print(cregs)
    measure_map = OrderedDict([(i.index, -1) for i in cregs])
    print(measure_map)
    print(measure_map[1])

    if justify == 'none':
        for node in dag.topological_op_nodes():
            ops.append([node])

    else:
        ops = _LayerSpooler(dag, justify, measure_map)

    print(ops)
    print(measure_map)
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
    for x in dag_instructions:
        print('Sorted nodes', x.op.name, x._node_id)
    dag_instructions.sort(key=lambda nd: nd._node_id)
    for x in dag_instructions:
        print(x.op.name, x._node_id)
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
    #print('INSTRUCTION-gate span', instruction.op, min_index, max_index)

    return qregs[min_index:max_index + 1]


def _any_crossover(qregs, node, nodes):
    """Return True .IFF. 'node' crosses over any in 'nodes',"""
    gate_span = _get_gate_span(qregs, node)
    #print('crossover', node.op.name, gate_span)
    all_indices = []
    for check_node in nodes:
        if check_node != node:
            all_indices += _get_gate_span(qregs, check_node)
            #print('cross check', check_node.op.name, _get_gate_span(qregs, check_node))
    return any(i in gate_span for i in all_indices)


class _LayerSpooler(list):
    """Manipulate list of layer dicts for _get_layered_instructions."""

    def __init__(self, dag, justification, measure_map):
        """Create spool"""
        super().__init__()
        self.dag = dag
        self.qregs = dag.qubits
        self.justification = justification
        self.measure_map = measure_map

        print('len self', len(self))
        if self.justification == 'left':

            for dag_layer in dag.layers():
                #print(dag_layer['graph'].__dict__)
                current_index = len(self) - 1
                dag_nodes = _sorted_nodes(dag_layer)
                for node in dag_nodes:
                    print('curr_index', current_index, node.op.name)
                    self.add(node, current_index)
                    #print('LEN SelF', len(self))
                    for i, s in enumerate(self):
                        print('Layer', i)
                        for ss in s:
                            print('List object', ss._op)

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
        print('node.qargs', node.qargs)
        for a_node in nodes:
            print('a_node', a_node.op.name)
            for qarg in a_node.qargs:
                print('qarg', qarg)
                all_qargs.append(qarg)
        print('all_qargs', all_qargs)
        print('any', any(i in node.qargs for i in all_qargs))
        return any(i in node.qargs for i in all_qargs)

    def insertable(self, node, nodes):
        """True .IFF. we can add 'node' to layer 'nodes'"""
        print('insertable', node.op.name, nodes, not _any_crossover(self.qregs, node, nodes))
        return not _any_crossover(self.qregs, node, nodes)

    def slide_from_left(self, node, index):
        """Insert node into first layer where there is no conflict going l > r"""
        print("Sliding node", node.op, node.cargs)
        measure_pos = None
        if isinstance(node.op, Measure):
            print("FOUND MEASURE ********************", node.op, node.cargs[0].index)
            measure_index = node.cargs[0].index
            #self.measure_map[measure_index][0] = True
        if not self:
            print('First insert', node._op)
            self.append([node])
            inserted = True

        else:
            inserted = False
            curr_index = index
            index_stop = -1
            if node.condition:
                print('NODE CONDITION ************', int(node.condition[1] / 2))
                index_stop = self.measure_map[int(node.condition[1] / 2)]
            last_insertable_index = None
            print('in slide current_index', curr_index)
            while curr_index > index_stop:
                print('In while before is found in', node.op.name, curr_index)
                if self.is_found_in(node, self[curr_index]):
                    break
                print("Before insertable 1")
                if self.insertable(node, self[curr_index]):
                    last_insertable_index = curr_index
                curr_index = curr_index - 1

            if last_insertable_index is not None:
                print('LAST INS True', last_insertable_index)
                self[last_insertable_index].append(node)
                measure_pos = last_insertable_index
                inserted = True

            else:
                print('Last Ins False')
                inserted = False
                curr_index = index
                while curr_index < len(self):
                    print('curr, len', curr_index, len(self))
                    print("Before insertable 2")
                    if self.insertable(node, self[curr_index]):
                        self[curr_index].append(node)
                        measure_pos = curr_index
                        inserted = True
                        print('Pre-break')
                        break
                    curr_index = curr_index + 1

        if not inserted:
            print("Appended", node._op)
            self.append([node])
        if isinstance(node.op, Measure):
            measure_pos = measure_pos if measure_pos else len(self) - 1
            self.measure_map[measure_index] = measure_pos

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
