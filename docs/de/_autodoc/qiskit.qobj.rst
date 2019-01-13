qiskit.qobj package
===================


.. automodule:: qiskit.qobj
    
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       
       
       
       
       

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`QobjValidationError <qiskit.qobj._validation.QobjValidationError>`
         - Represents an error during Qobj validation.
    

    Classes
    -------


    .. list-table::
    
       * - :class:`Qobj <qiskit.qobj._qobj.Qobj>`
         - Representation of a Qobj.
       * - :class:`QobjConfig <qiskit.qobj._qobj.QobjConfig>`
         - Configuration for a Qobj.
       * - :class:`QobjExperiment <qiskit.qobj._qobj.QobjExperiment>`
         - Quantum experiment represented inside a Qobj.
       * - :class:`QobjExperimentHeader <qiskit.qobj._qobj.QobjExperimentHeader>`
         - Header for a Qobj.
       * - :class:`QobjHeader <qiskit.qobj._qobj.QobjHeader>`
         - Header for a Qobj.
       * - :class:`QobjInstruction <qiskit.qobj._qobj.QobjInstruction>`
         - Quantum Instruction.
       * - :class:`QobjItem <qiskit.qobj._qobj.QobjItem>`
         - Generic Qobj structure.
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`qobj_to_dict <qiskit.qobj._converter.qobj_to_dict>`
         - Convert a Qobj to another version of the schema.
       * - :func:`validate_qobj_against_schema <qiskit.qobj._validation.validate_qobj_against_schema>`
         - Validates a QObj against a schema.
    