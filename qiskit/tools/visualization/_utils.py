# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree

"""Common visualization utilities."""

import PIL

from qiskit import dagcircuit


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
    dag = dagcircuit.DAGCircuit.fromQuantumCircuit(circuit, expand_gates=False)
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

    if not reversebits:
        qregs.reverse()
        cregs.reverse()

    return qregs, cregs, ops
