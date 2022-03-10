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
    pass is left in the hands of the pass manager designer.
    Once a scheduling analysis pass is run, ``node_start_time`` is generated
    in the :attr:`property_set`.  This information is represented by a python dictionary of
    the expected instruction execution times keyed on the node instances.
    Entries in the dictionary are only created for non-delay nodes.
    The padding pass expects all ``DAGOpNode`` in the circuit to be scheduled.

    This base class doesn't define any sequence to interleave, but it manages
    the location where the sequence is inserted, and provides a set of information necessary
    to construct the proper sequence. Thus, a subclass of this pass just needs to implement
    :meth:`_pad` method, in which the subclass constructs a circuit block to insert.
    This mechanism removes lots of boilerplate logic to manage whole DAG circuits.

    Note that padding pass subclasses should define interleaving sequences satisfying:

        - Interleaved sequence does not change start time of other nodes
        - Interleaved sequence should have total duration of the provided ``time_interval``.

    Any manipulation violating these constraints may prevent this base pass from correctly
    tracking the start time of each instruction,
    which may result in violation of hardware alignment constraints.
    """

    def run(self, dag: DAGCircuit):
        """Run the padding pass on ``dag``.

        Args:
            dag (DAGCircuit): DAG to be checked.

        Returns:
            DAGCircuit: DAG with idle time filled with instructions.

        Raises:
            TranspilerError: If the whole circuit or instruction is not scheduled.
        """
        if "node_start_time" not in self.property_set:
            raise TranspilerError(
                f"The input circuit {dag.name} is not scheduled. Call one of scheduling passes "
                f"before running the {self.__class__.__name__} pass."
            )
        node_start_time = self.property_set["node_start_time"]

        new_dag = DAGCircuit()

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.unit = self.property_set["time_unit"]
        new_dag.calibrations = dag.calibrations

        idle_after = {bit: 0 for bit in dag.qubits}

        # Compute fresh circuit duration from the node start time dictionary and op duration.
        # Note that pre-scheduled duration may change within the alignment passes, i.e.
        # if some instruction time t0 violating the hardware alignment constraint,
        # the alignment pass may delay t0 and accordingly the circuit duration changes.
        circuit_duration = 0
        for node in dag.topological_op_nodes():
            if node in node_start_time:
                t0 = node_start_time[node]
                t1 = t0 + node.op.duration
                circuit_duration = max(circuit_duration, t1)

                if isinstance(node.op, Delay):
                    # The padding class considers a delay instruction as idle time
                    # rather than instruction. Delay node is removed so that
                    # we can extract non-delay predecessors.
                    dag.remove_op_node(node)
                    continue

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
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

        # Add delays until the end of circuit.
        for bit in new_dag.qubits:
            idle_time = circuit_duration - idle_after[bit]
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

        new_dag.duration = circuit_duration

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
