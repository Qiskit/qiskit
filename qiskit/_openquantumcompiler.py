# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tools for compiling Quantum Programs."""
from qiskit.unroll import DagUnroller, JsonBackend


# TODO: This is here for backward compatibility with QISKit Developer Challenge
# Once the challenge is finished, we have to remove this entire module.
def dag2json(dag_circuit, basis_gates='u1,u2,u3,cx,id'):
    """Make a Json representation of the circuit.
    Takes a circuit dag and returns json circuit obj. This is an internal
    function.
    Args:
        dag_circuit (QuantumCircuit): a dag representation of the circuit.
        basis_gates (str): a comma seperated string and are the base gates,
                               which by default are: u1,u2,u3,cx,id
    Returns:
        json: the json version of the dag
    """
    return DagUnroller(dag_circuit, JsonBackend(basis_gates.split(","))).execute()
