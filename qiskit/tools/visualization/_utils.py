# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree

"""Common visualization utilities."""

import PIL
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization._error import VisualizationError


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


def _get_instructions(circuit, reversebits=False):
    """
    Given a circuit, return a tuple (qregs, cregs, ops) where
    qregs and cregs are the quantum and classical registers
    in order (based on reversebits) and ops is a list
    of DAG nodes which type is "operation".
    Args:
        circuit (QuantumCircuit): From where the information is extracted.
        reversebits (bool): If true the order of the bits in the registers is
            reversed.
    Returns:
        Tuple(list,list,list): To be consumed by the visualizer directly.
    """
    dag = circuit_to_dag(circuit)
    ops = []
    qregs = []
    cregs = []
    for node_no in dag.node_nums_in_topological_order():
        node = dag.multi_graph.node[node_no]
        if node['type'] == 'op':
            ops.append(node)

    for qreg in dag.qregs.values():
        qregs += [(qreg, bitno) for bitno in range(qreg.size)]

    for creg in dag.cregs.values():
        cregs += [(creg, bitno) for bitno in range(creg.size)]

    if reversebits:
        qregs.reverse()
        cregs.reverse()

    return qregs, cregs, ops
