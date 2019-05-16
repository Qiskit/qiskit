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

"""Common visualization utilities."""

import PIL
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.visualization.exceptions import VisualizationError


def _validate_input_state(quantum_state):
    """Validates the input to state visualization functions.

    Args:
        quantum_state (ndarray): Input state / density matrix.
    Returns:
        rho: A 2d numpy array for the density matrix.
    Raises:
        VisualizationError: Invalid input.
    """
    rho = np.asarray(quantum_state)
    if rho.ndim == 1:
        rho = np.outer(rho, np.conj(rho))
    # Check the shape of the input is a square matrix
    shape = np.shape(rho)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise VisualizationError("Input is not a valid quantum state.")
    # Check state is an n-qubit state
    num = int(np.log2(rho.shape[0]))
    if 2 ** num != rho.shape[0]:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    return rho


def _trim(image):
    """Trim a PIL image and remove white space."""
    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image


def _get_layered_instructions(circuit, reverse_bits=False, justify=None):
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
    Returns:
        Tuple(list,list,list): To be consumed by the visualizer directly.
    """
    if justify:
        justify = justify.lower()

    # default to left
    justify = justify if justify in ('right', 'none') else 'left'

    dag = circuit_to_dag(circuit)
    ops = []
    qregs = []
    cregs = []

    for qreg in dag.qregs.values():
        qregs += [(qreg, bitno) for bitno in range(qreg.size)]

    for creg in dag.cregs.values():
        cregs += [(creg, bitno) for bitno in range(creg.size)]

    if justify == 'none':
        for node in dag.topological_op_nodes():
            ops.append([node])

    if justify == 'left':
        for dag_layer in dag.layers():
            layers = []
            current_layer = []

            dag_nodes = dag_layer['graph'].op_nodes()
            dag_nodes.sort(key=lambda nd: nd._node_id)

            for node in dag_nodes:
                multibit_gate = len(node.qargs) + len(node.cargs) > 1

                if multibit_gate:
                    # need to see if it crosses over any other nodes
                    gate_span = _get_gate_span(qregs, node)

                    all_indices = []
                    for check_node in dag_nodes:
                        if check_node != node:
                            all_indices += _get_gate_span(qregs, check_node)

                    if any(i in gate_span for i in all_indices):
                        # needs to be a new layer
                        layers.append([node])
                    else:
                        # can be added
                        current_layer.append(node)
                else:
                    current_layer.append(node)

            if current_layer:
                layers.append(current_layer)
            ops += layers

    if justify == 'right':
        dag_layers = []

        for dag_layer in dag.layers():
            dag_layers.append(dag_layer)

        # Have to work from the end of the circuit
        dag_layers.reverse()

        # Dict per layer, keys are qubits and values are the gate
        layer_dicts = [{}]

        for dag_layer in dag_layers:

            dag_instructions = dag_layer['graph'].op_nodes()

            # sort into the order they were input
            dag_instructions.sort(key=lambda nd: nd._node_id)
            for instruction_node in dag_instructions:

                gate_span = _get_gate_span(qregs, instruction_node)

                added = False
                for i in range(len(layer_dicts)):
                    # iterate from the end
                    curr_dict = layer_dicts[-1 - i]

                    if any(index in curr_dict for index in gate_span):
                        added = True

                        if i == 0:
                            new_dict = {}

                            for index in gate_span:
                                new_dict[index] = instruction_node
                            layer_dicts.append(new_dict)
                        else:
                            curr_dict = layer_dicts[-i]
                            for index in gate_span:
                                curr_dict[index] = instruction_node

                        break

                if not added:
                    for index in gate_span:
                        layer_dicts[0][index] = instruction_node

        # need to convert from dict format to layers
        layer_dicts.reverse()
        ops = [list(layer.values()) for layer in layer_dicts]

    if reverse_bits:
        qregs.reverse()
        cregs.reverse()

    return qregs, cregs, ops


def _get_gate_span(qregs, instruction):
    """Get the list of qubits drawing this gate would cover"""

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

    return qregs[min_index:max_index + 1]
