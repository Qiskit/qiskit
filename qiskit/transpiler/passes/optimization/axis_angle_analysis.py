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

"""Analysis pass to find commutation relations between DAG nodes."""

import math
import numpy as np
import pandas as pd
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import ControlledGate

_CUTOFF_PRECISION = 1E-10


class AxisAngleAnalysis(AnalysisPass):
    """Analysis pass to find rotation axes and angles of single qubit dag nodes.

    Property_set['axis_angle'] is a dictionary
    """

    def __init__(self):
        super().__init__()
        self.cache = {}
        self.property_set['axis-angle'] = {}
        self.property_set['var_gate_class'] = {}

    def run(self, dag):
        """Run the axis-angle analysis pass.

        Run the pass on the DAG, and write the discovered commutation relations
        into the property_set.

        Args:
            dag (DAGCircuit): circuit graph
        """
        props = list()
        dfprop = pd.DataFrame()
        var_gate_class = dict()
        for node in dag.gate_nodes():
            # TODO: cache angle-axis evaluation
            if len(node.qargs) == 1:
                aprop = _get_1q_gate_props(node.op)
                aprop['qubit0'] = node.qargs[0]
                aprop['qubit1'] = None
            elif len(node.qargs) == 2 and isinstance(node.op, ControlledGate):
                aprop = _get_1q_gate_props(node.op.base_gate, period=4*np.pi)
                aprop['qubit0'] = node.qargs[0]
                aprop['qubit1'] = node.qargs[1]
            else:
                continue
            aprop['id'] = node._node_id
            aprop['name'] = node.name
            aprop['num_qubits'] = len(node.qargs)
            props.append(aprop)
            if node.name not in var_gate_class and aprop['nparams'] == 1:
                var_gate_class[node.name] = node.op.__class__
        if props:
            dfprop = pd.DataFrame.from_dict(props).astype({'symmetry_order': int})
        self.property_set['axis-angle'] = dfprop
        self.property_set['var_gate_class'] = var_gate_class


def _get_1q_gate_props(gate, period=None):
    """
    Get single qubit operator properties
    """
    decimals = 12
    rel_tol = 1e-9
    abs_tol = 1e-9
    # TODO: avoid 'x' check
    if period is None or gate.name == 'x':
        period = 2 * np.pi
    try:
        # Operator does this too but maybe this is slightly more direct.
        mat = gate.to_matrix()
    except CircuitError:
        mat = Operator(gate.definition).data
    axis, angle, phase = _su2_axis_angle(mat)
    if angle > np.pi:
        sym_angle = 2 * np.pi - angle
    else:
        sym_angle = angle
    quotient, remainder = divmod(period, sym_angle)
    if math.isclose(remainder, 0, rel_tol=rel_tol, abs_tol=abs_tol):
        symmetry_order = int(quotient)
    elif math.isclose(remainder, sym_angle, rel_tol=rel_tol, abs_tol=abs_tol):
        # divmod had a rounding error
        symmetry_order = int(quotient + 1)
    else:
        symmetry_order = 1
    nparams = len(gate.params)
    prop = {'nparams': nparams,
            # needed for drop_duplicates; hashable type. Instead of rounding here
            # could consider putting the effect into the dot product test during
            # reduction.
            'axis': tuple(np.around(axis, decimals=decimals)),
            'angle': angle,
            'phase': phase,
            'symmetry_order': symmetry_order}
    return prop


def _su2_axis_angle(mat):
    """
    Convert 2x2 unitary matrix to axis-angle.

    Args:
        mat (ndarray): single qubit unitary matrix to convert

    Returns:
        tuple(axis, angle, phase): where axis is vector in SO(3), angle is the rotation
            angle in radians, and phase scales mat as exp(1j*phase)
            away from the symmetry of su2. If the matrix is a scalar operator the axis will
            be the zero vector, the angle will be zero, and the phase gives the scalar constant as
            exp(𝑖 * phase) · 𝕀.
    """
    axis = np.zeros(3)
    det = np.linalg.det(mat)
    umat = mat / np.sqrt(det)
    u00 = umat[0, 0]
    u10 = umat[1, 0]

    phase = (-1j * np.log(det)).real / 2
    angle = 2 * np.arccos(u00.real)
    if angle == 0:
        return axis, angle, phase
    sine = np.sin(angle / 2)
    axis[0] = -u10.imag / sine
    axis[1] = u10.real / sine
    axis[2] = -u00.imag / sine
    return axis, angle, phase
