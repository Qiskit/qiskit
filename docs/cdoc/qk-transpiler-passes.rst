=================
Transpiler Passes
=================

Transpilation is the process of rewriting a given input circuit to match the topology of a specific
quantum device, and/or to optimize the circuit for execution on a quantum system.

Most circuits must undergo a series of transformations that make them compatible with a given target
device, and optimize them to reduce the effects of noise on the resulting outcomes. Rewriting quantum
circuits to match hardware constraints and optimizing for performance can be far from trivial. The flow
of logic in the rewriting tool chain need not be linear, and can often have iterative sub-loops,
conditional branches, and other complex behaviors.

In Qiskit, the transpiler is built up by executing as a series of passes that each perform a singular task
to analyze or transform a quantum circuit. The Python :mod:`~qiskit.transpiler` documentation contains a
more detailed explanation of the transpilation process.

The Qiskit C API provides functions that execute transpiler passes in a standalone mode, where you
provide the pass with a ``QkCircuit`` and then any necessary configuration for the execution of the
pass, typically at least a ``QkTarget``. These functions return either a new ``QkCircuit`` pointer
or the analysis results of running the pass. While this can be used to create a custom workflow, the
functions following the naming convention ``qk_transpiler_pass_standalone_*`` will have higher overhead
as internally they're converting from the quantum circuit to the dag circuit IR on the input, and if
the function returns a new circuit it will convert back before returning. These standalone functions
are intended to execute single passes in isolation rather than building a custom transpilation pipeline.

DAG-based Passes
----------------

For users who are working directly with the DAG representation (``QkDag``), the C API also provides
functions that operate directly on the DAG without the conversion overhead. These functions follow
the naming convention ``qk_transpiler_pass_*`` and modify the DAG in place. This is
useful when you need to chain multiple passes together or when you're already working with a DAG
representation of your circuit. Using DAG-based passes avoids the repeated circuit-to-DAG and
DAG-to-circuit conversions that would otherwise occur when chaining multiple standalone passes.

Functions
=========

.. doxygengroup:: QkTranspilerPasses
    :members:
    :content-only:
