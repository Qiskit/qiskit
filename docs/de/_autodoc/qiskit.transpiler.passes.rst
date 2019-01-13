qiskit.transpiler.passes package
================================


.. automodule:: qiskit.transpiler.passes
    
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       qiskit.transpiler.passes.commutation_analysis
       qiskit.transpiler.passes.commutation_transformation
       qiskit.transpiler.passes.cx_cancellation
       qiskit.transpiler.passes.decompose
       qiskit.transpiler.passes.fixed_point
       qiskit.transpiler.passes.optimize_1q_gates

    Subpackages
    -----------

    .. toctree::
       :maxdepth: 1

       qiskit.transpiler.passes.mapping

    Classes
    -------


    .. list-table::
    
       * - :class:`BarrierBeforeFinalMeasurements <qiskit.transpiler.passes.mapping.barrier_before_final_measurements.BarrierBeforeFinalMeasurements>`
         - Adds a barrier before final measurements.
       * - :class:`BasicSwap <qiskit.transpiler.passes.mapping.basic_swap.BasicSwap>`
         - Maps (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates.
       * - :class:`CXCancellation <qiskit.transpiler.passes.cx_cancellation.CXCancellation>`
         - Cancel back-to-back 'cx' gates in dag.
       * - :class:`CXDirection <qiskit.transpiler.passes.mapping.cx_direction.CXDirection>`
         - Rearranges the direction of the cx nodes to make the circuit compatible with the directed coupling map.
       * - :class:`CheckMap <qiskit.transpiler.passes.mapping.check_map.CheckMap>`
         - Checks if a DAGCircuit is mapped to `coupling_map`.
       * - :class:`CommutationAnalysis <qiskit.transpiler.passes.commutation_analysis.CommutationAnalysis>`
         - An analysis pass to find commutation relations between DAG nodes.
       * - :class:`CommutationTransformation <qiskit.transpiler.passes.commutation_transformation.CommutationTransformation>`
         - A transformation pass to change DAG edges depending on previously discovered commutation relations.
       * - :class:`Decompose <qiskit.transpiler.passes.decompose.Decompose>`
         - Expand a gate in a circle using its decomposition rules.
       * - :class:`FixedPoint <qiskit.transpiler.passes.fixed_point.FixedPoint>`
         - A dummy analysis pass that checks if a property reached a fixed point.
       * - :class:`LookaheadSwap <qiskit.transpiler.passes.mapping.lookahead_swap.LookaheadSwap>`
         - Map input circuit onto a backend topology via insertion of SWAPs.
       * - :class:`Optimize1qGates <qiskit.transpiler.passes.optimize_1q_gates.Optimize1qGates>`
         - Simplify runs of single qubit gates in the ["u1", "u2", "u3", "cx", "id"] basis.
       * - :class:`StochasticSwap <qiskit.transpiler.passes.mapping.stochastic_swap.StochasticSwap>`
         - Maps a DAGCircuit onto a `coupling_map` adding swap gates.
       * - :class:`Unroller <qiskit.transpiler.passes.mapping.unroller.Unroller>`
         - Unroll (expand) non-basis, non-opaque instructions recursively to a desired basis, using decomposition rules defined for each instruction.
    