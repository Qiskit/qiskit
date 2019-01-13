qiskit.transpiler package
=========================


.. automodule:: qiskit.transpiler
    
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       
       
       
       
       
       

    Subpackages
    -----------

    .. toctree::
       :maxdepth: 1

       qiskit.transpiler.passes

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`MapperError <qiskit.transpiler._transpilererror.MapperError>`
         - Exception for cases where a mapper pass cannot map.
       * - :exc:`TranspilerAccessError <qiskit.transpiler._transpilererror.TranspilerAccessError>`
         - Exception of access error in the transpiler passes.
       * - :exc:`TranspilerError <qiskit.transpiler._transpilererror.TranspilerError>`
         - Exceptions raised during transpilation
    

    Classes
    -------


    .. list-table::
    
       * - :class:`AnalysisPass <qiskit.transpiler._basepasses.AnalysisPass>`
         - An analysis pass: change property set, not DAG.
       * - :class:`FencedDAGCircuit <qiskit.transpiler._fencedobjs.FencedDAGCircuit>`
         - A dag circuit that cannot be modified (via _remove_op_node)
       * - :class:`FencedPropertySet <qiskit.transpiler._fencedobjs.FencedPropertySet>`
         - A property set that cannot be written (via __setitem__)
       * - :class:`FlowController <qiskit.transpiler._passmanager.FlowController>`
         - This class is a base class for multiple types of working list.
       * - :class:`PassManager <qiskit.transpiler._passmanager.PassManager>`
         - A PassManager schedules the passes
       * - :class:`PropertySet <qiskit.transpiler._propertyset.PropertySet>`
         - A dictionary-like object
       * - :class:`TransformationPass <qiskit.transpiler._basepasses.TransformationPass>`
         - A transformation pass: change DAG, not property set.
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`transpile <qiskit.transpiler._transpiler.transpile>`
         - transpile one or more circuits.
       * - :func:`transpile_dag <qiskit.transpiler._transpiler.transpile_dag>`
         - Transform a dag circuit into another dag circuit (transpile), through consecutive passes on the dag.
    