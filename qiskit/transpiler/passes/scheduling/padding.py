# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Padding pass to fill empty timeslot."""

from qiskit.circuit import Qubit
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class BasePadding(TransformationPass):
    """The base class of padding pass.

    This pass requires one of scheduling passes to be executed before itself.
    Since there are multiple scheduling strategies, the selection of scheduling
    pass is left in the hands of one designs the pass manager.
    Once schedule pass is called on the DAG circuit, ``time_slot`` is generated
    in the :attr:`property_set`.  The time slot is the dictionary of
    the expected instruction execution times keyed on the node instances.
    Entries in the dictionary are only created for non-delay nodes.
    The padding pass expects all ``DAGOpNode`` in the circuit to be scheduled in the time slot.

    This pass doesn't define any sequence to interleave, but the pass manages
    the location where the sequence is inserted, and provides set of information necessary
    to construct the proper sequence. Thus, a subclass of this pass just need to implement
    :meth:`_pad` method, in which the subclass constructs a circuit block to insert.

    This mechanism removes lots of boilerplate logic to manage whole DAG circuits.
    """

    def run(self, dag: DAGCircuit):
        """Run the measurement alignment pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to be checked.

        Returns:
            DAGCircuit: DAG with consistent timing and op nodes annotated with duration.

        Raises:
            TranspilerError: If the whole circuit or instruction is not scheduled.
        """
        if "time_slot" not in self.property_set:
            raise TranspilerError(
                f"The input circuit {dag.name} is not scheduled. Call one of scheduling passes "
                f"before running the {self.__class__.__name__} pass."
            )
        time_slot = self.property_set["time_slot"]

        new_dag = DAGCircuit()

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.duration = dag.duration
        new_dag.duration = self.property_set["duration"]
        new_dag.unit = self.property_set["time_unit"]
        new_dag.calibrations = dag.calibrations

        idle_after = {bit: 0 for bit in dag.qubits}
        for node in dag.topological_op_nodes():
            # Delays are ignored. Delays are not explicitly stored in the time slot.
            # The duration of delays might be corrected by alignment passes.
            # We don't keep user input delays.
            key = node.name, node.sort_key, getattr(node, "_node_id")
            if key in time_slot:
                t0 = time_slot[key]
                t1 = t0 + node.op.duration

                for bit in node.qargs:
                    # Find idle time from the latest instruction on the wire
                    idle_time = t0 - idle_after[bit]

                    # Fill idle time with some sequence
                    if idle_time > 0:
                        # Find previous node on the wire, i.e. always the latest node on the wire
                        prev_node = next(new_dag.predecessors(new_dag.output_map[bit]))
                        self._pad(
                            dag=new_dag,
                            qubit=bit,
                            time_interval=idle_time,
                            next_node=node,
                            prev_node=prev_node,
                        )

                    idle_after[bit] = t1

                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            else:
                # Note that delay is not scheduled in the scheduling pass.
                # Delay node is removed so that we can extract non-delay predecessors.
                if not isinstance(node.op, Delay):
                    raise TranspilerError(
                        f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                        "Schedule the circuit again if you transformed it."
                    )
                dag.remove_op_node(node)

        # Add delays until the end of circuit.
        for bit in new_dag.qubits:
            idle_time = new_dag.duration - idle_after[bit]
            node = new_dag.output_map[bit]
            prev_node = next(new_dag.predecessors(node))
            if idle_time > 0:
                self._pad(
                    dag=new_dag,
                    qubit=bit,
                    time_interval=idle_time,
                    next_node=node,
                    prev_node=prev_node,
                )

        return new_dag

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        time_interval: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        """Interleave instruction sequence in between two nodes.

        Args:
            dag: DAG circuit that sequence is applied.
            qubit: The wire that the sequence is applied on.
            time_interval: Duration of idle time in between two nodes.
            next_node: Node that follows the sequence.
            prev_node: Node ahead of the sequence.
        """
        raise NotImplementedError


class PadDelay(BasePadding):
    """Padding idle time with Delay instructions.

    Consecutive delays will be merged in the output of this pass.

    .. code-block::python

        durations = InstructionDurations([("x", None, 160), ("cx", None, 800)])

        qc = QuantumCircuit(2)
        qc.delay(100, 0)
        qc.x(1)
        qc.cx(0, 1)

    The ASAP-scheduled circuit output may become

    .. parsed-literal::

             ┌────────────────┐
        q_0: ┤ Delay(160[dt]) ├──■──
             └─────┬───┬──────┘┌─┴─┐
        q_1: ──────┤ X ├───────┤ X ├
                   └───┘       └───┘

    Note that the additional idle time of 60dt on the ``q_0`` wire coming from the duration difference
    between ``Delay`` of 100dt (``q_0``) and ``XGate`` of 160 dt (``q_1``) is absorbed in
    the delay instruction on the ``q_0`` wire, i.e. in total 160 dt.

    See :class:`BasePadding` pass for details.
    """

    def __init__(self, fill_very_end: bool = True):
        """Create new padding delay pass.

        Args:
            fill_very_end: Set ``True`` to fill the end of circuit with delay.
        """
        super().__init__()
        self.fill_very_end = fill_very_end

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        time_interval: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        if not self.fill_very_end and isinstance(next_node, DAGOutNode):
            return

        dag.apply_operation_back(Delay(time_interval, dag.unit), [qubit])
