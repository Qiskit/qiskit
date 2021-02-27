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

from collections import defaultdict
import math
import numpy as np
import pandas as pd
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.exceptions import CircuitError

_CUTOFF_PRECISION = 1E-10


class AxisAngleAnalysis(AnalysisPass):
    """Analysis pass to find rotation axes and angles of single qubit dag nodes.

    Property_set['axis_angle'] is a dictionary
    """

    def __init__(self):
        super().__init__()
        self.cache = {}

    def run(self, dag, global_basis=True, decimals=12):
        """Run the axis-angle analysis pass.

        Run the pass on the DAG, and write the discovered commutation relations
        into the property_set.

        Args: 
            dag (DAGCircuit): circuit graph
            global_basis (bool): If true, the discovered rotation axes are assummed to
                to be applicable to all qubits. If false, the operations on each qubit describe
                the basis.
        """
        props = []
        for node in dag.gate_nodes():
            # TODO: cache angle-axis evaluation
            if len(node.qargs) == 1:
                try:
                    mat = node.op.to_matrix()
                except CircuitError:
                    mat = Operator(node.op.definition).data
                axis, angle, phase = _su2_axis_angle(mat)
                quotient, remainder = divmod(2 * np.pi, angle)
                symmetry_order = quotient if math.isclose(remainder, 0) else 1
                props.append({'id': id(node.op),
                              'name': node.name,
                              'nparams': len(node.op.params),
                              'qubit': node.qargs[0],
                              # needed for drop_duplicates; hashable type. Instead of rounding here
                              # could consider putting the effect into the dot product test below.
                              'axis': tuple(np.around(axis, decimals=decimals)),  
                              'angle': angle,
                              'phase': phase,
                              'symmetry_order': symmetry_order})
        dfprop = pd.DataFrame.from_dict(props)
        self.property_set['axis-angle'] = dfprop
                          
        breakpoint()
        pass

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
    mat = mat / np.sqrt(det)
    u00 = mat[0, 0]
    u10 = mat[1, 0]

    phase = (-1j * np.log(det)).real / 2
    angle = 2 * np.arccos(u00.real)
    if angle == 0:
        return axis, angle, phase
    sine = np.sin(angle / 2)
    axis[0] = -u10.imag / sine
    axis[1] = u10.real / sine
    axis[2] = -u00.imag / sine
    return axis, angle, phase
