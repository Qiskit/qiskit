# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint disable=missing-return-type-doc

"""Disassemble function for a qobj into a list of circuits and it's config"""

from qiskit.converters import qobj_to_circuits


def disassemble(qobj):
    """Dissasemble a qobj and return the circuits, run_config, and user header

    Args:
        qobj (Qobj): The input qobj object to dissasemble
    Returns:
        circuits (list): A list of quantum circuits
        run_config (dict): The dist of the run config
        user_qobj_header (dict): The dict of any user headers in the qobj

    """
    run_config = qobj.config.to_dict()
    user_qobj_header = qobj.header.to_dict()
    circuits = qobj_to_circuits(qobj)

    return circuits, run_config, user_qobj_header
