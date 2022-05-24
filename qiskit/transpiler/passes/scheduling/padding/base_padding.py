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

from typing import List, Optional, Union

from qiskit.circuit import Qubit, Clbit, Instruction
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode
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
            dag: DAG to be checked.

        Returns:
            DAGCircuit: DAG with idle time filled with instructions.

        Raises:
            TranspilerError: When a particular node is not scheduled, likely some transform pass
                is inserted before this node is called.
        """
        self._pre_runhook(dag)

        node_start_time = self.property_set["node_start_time"].copy()

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # Update start time dictionary for the new_dag.
        # This information may be used for further scheduling tasks,
        # but this is immediately invalidated becasue node id is updated in the new_dag.
        self.property_set["node_start_time"].clear()

        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.unit = self.property_set["time_unit"]
        new_dag.calibrations = dag.calibrations
        new_dag.global_phase = dag.global_phase

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

                    # Fill idle time with some sequence
                    if t0 - idle_after[bit] > 0:
                        # Find previous node on the wire, i.e. always the latest node on the wire
                        prev_node = next(new_dag.predecessors(new_dag.output_map[bit]))
                        self._pad(
                            dag=new_dag,
                            qubit=bit,
                            t_start=idle_after[bit],
                            t_end=t0,
                            next_node=node,
                            prev_node=prev_node,
                        )

                    idle_after[bit] = t1

                self._apply_scheduled_op(new_dag, t0, node.op, node.qargs, node.cargs)
            else:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

        # Add delays until the end of circuit.
        for bit in new_dag.qubits:
            if circuit_duration - idle_after[bit] > 0:
                node = new_dag.output_map[bit]
                prev_node = next(new_dag.predecessors(node))
                self._pad(
                    dag=new_dag,
                    qubit=bit,
                    t_start=idle_after[bit],
                    t_end=circuit_duration,
                    next_node=node,
                    prev_node=prev_node,
                )

        new_dag.duration = circuit_duration

        return new_dag

    def _pre_runhook(self, dag: DAGCircuit):
        """Extra routine inserted before running the padding pass.

        Args:
            dag: DAG circuit on which the sequence is applied.

        Raises:
            TranspilerError: If the whole circuit or instruction is not scheduled.
        """
        if "node_start_time" not in self.property_set:
            raise TranspilerError(
                f"The input circuit {dag.name} is not scheduled. Call one of scheduling passes "
                f"before running the {self.__class__.__name__} pass."
            )

    def _apply_scheduled_op(
        self,
        dag: DAGCircuit,
        t_start: int,
        oper: Instruction,
        qubits: Union[Qubit, List[Qubit]],
        clbits: Optional[Union[Clbit, List[Clbit]]] = None,
    ):
        """Add new operation to DAG with scheduled information.

        This is identical to apply_operation_back + updating the node_start_time propety.

        Args:
            dag: DAG circuit on which the sequence is applied.
            t_start: Start time of new node.
            oper: New operation that is added to the DAG circuit.
            qubits: The list of qubits that the operation acts on.
            clbits: The list of clbits that the operation acts on.
        """
        if isinstance(qubits, Qubit):
            qubits = [qubits]
        if isinstance(clbits, Clbit):
            clbits = [clbits]

        new_node = dag.apply_operation_back(oper, qargs=qubits, cargs=clbits)
        self.property_set["node_start_time"][new_node] = t_start

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        """Interleave instruction sequence in between two nodes.

        .. note::
            If a DAGOpNode is added here, it should update node_start_time property
            in the property set so that the added node is also scheduled.
            This is achieved by adding operation via :meth:`_apply_scheduled_op`.

        .. note::

            This method doesn't check if the total duration of new DAGOpNode added here
            is identical to the interval (``t_end - t_start``).
            A developer of the pass must guarantee this is satisfied.
            If the duration is greater than the interval, your circuit may be
            compiled down to the target code with extra duration on the backend compiler,
            which is then played normally without error. However, the outcome of your circuit
            might be unexpected due to erroneous scheduling.

        Args:
            dag: DAG circuit that sequence is applied.
            qubit: The wire that the sequence is applied on.
            t_start: Absolute start time of this interval.
            t_end: Absolute end time of this interval.
            next_node: Node that follows the sequence.
            prev_node: Node ahead of the sequence.
        """
        raise NotImplementedError
