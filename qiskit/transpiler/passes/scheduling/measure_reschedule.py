# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rescheduling measurements to be compatible with the meas_map constraint."""
import warnings
from collections import defaultdict
from itertools import groupby

from qiskit.circuit import QuantumRegister
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError


class MeasureReschedule(TransformationPass):
    """Pass to reschedule circuit to be compatible with the meas_map constraint.
    Assume all measurements are done at once at the last of the circuit.
    """

    def __init__(self, meas_map):
        """MeasureReschedule initializer.

        Args:
            meas_map (list): .
        """
        super().__init__()
        self.meas_map = meas_map

    def run(self, dag):
        """Extend measurements to be compatible with the meas_map constraint.
        Assume all measurements are done at once at the last of the circuit.

        Args:
            dag (DAGCircuit): DAG to be converted.

        Returns:
            DAGCircuit: A converted DAG.

        Raises:
            TranspilerError: if ...
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('MeasureReschedule runs on physical circuits only')

        qubit = {}  # qubit-index (qidx) -> qubit
        for q in dag.qubits:
            qubit[q.index] = q

        measures = list(dag.op_nodes(op=Measure))
        if not measures:
            return dag

        # trace start times
        intervals = {}
        qubit_time_available = defaultdict(int)
        for node in dag.op_nodes():
            start = qubit_time_available[node.qargs[0]]
            stop = start + node.op.duration
            intervals[node] = (start, stop)
            for q in node.qargs:
                qubit_time_available[q] = stop

        measures = sorted([(intervals[n], n) for n in measures], key=lambda x: x[0])
        for interval, group in groupby(measures, key=lambda x: x[0]):
            group_qidxs = sorted([q.index for i, n in group for q in n.qargs])
            if group_qidxs in self.meas_map:
                continue

            def covering_map(meas_map, qidxs):
                for amap in meas_map:
                    if set(qidxs) <= set(amap):
                        return amap
                return []

            meas_qidxs = covering_map(self.meas_map, group_qidxs)
            if not meas_qidxs:
                raise TranspilerError('Not yet supported this case.')

            def overlap(lhs, rhs):
                # TODO: need to check again
                if rhs[0] < lhs[1] and lhs[0] < rhs[1]:
                    return True
                return False

            # find all nodes that overlaps the interval
            ovalapping_nodes = []
            for qidx in meas_qidxs:
                if qidx not in group_qidxs:
                    try:
                        q = qubit[qidx]
                        for node in dag.nodes_on_wire(q, only_ops=True):
                            if overlap(intervals[node], interval):
                                if not isinstance(node.op, Delay):
                                    raise TranspilerError('Not yet supported this case.')
                                ovalapping_nodes.append(node)
                    except KeyError:
                        # we can ignore qubits not in dag.qubits() (truncated by coupling_map)
                        continue

            for node in ovalapping_nodes:
                # split the delay node into a sequence of a delay and a measurement
                start_meas, end_meas = interval
                start_delay, end_delay = intervals[node]
                if end_meas != end_delay or start_meas < start_delay:
                    print("end_meas:{}, end_delay:{}".format(end_meas, end_delay))
                    raise TranspilerError('Not yet supported this case.')
                end_delay = start_meas
                qr = QuantumRegister(1)
                splitted = DAGCircuit()
                splitted.add_qreg(qr)
                delay = Delay(1, duration=end_delay-start_delay)
                meas = Measure()
                meas.duration = end_meas - start_meas
                splitted.apply_operation_back(delay, [qr[0]])
                splitted.apply_operation_back(meas, [qr[0]])
                dag.substitute_node_with_dag(node, splitted)

        return dag
