qiskit.mapper package
=====================


.. automodule:: qiskit.mapper
    
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       
       
       
       
       
       

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`CouplingError <qiskit.mapper._couplingerror.CouplingError>`
         - Base class for errors raised by the coupling graph object.
       * - :exc:`MapperError <qiskit.mapper._mappererror.MapperError>`
         - Base class for errors raised by mapper module.
    

    Classes
    -------


    .. list-table::
    
       * - :class:`CouplingMap <qiskit.mapper._coupling.CouplingMap>`
         - Directed graph specifying fixed coupling.
       * - :class:`Layout <qiskit.mapper._layout.Layout>`
         - Two-ways dict to represent a Layout.
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`euler_angles_1q <qiskit.mapper._compiling.euler_angles_1q>`
         - Compute Euler angles for a single-qubit gate.
       * - :func:`swap_mapper <qiskit.mapper._mapping.swap_mapper>`
         - Map a DAGCircuit onto a CouplingGraph using swap gates.
       * - :func:`two_qubit_kak <qiskit.mapper._compiling.two_qubit_kak>`
         - Decompose a two-qubit gate over CNOT + SU(2) using the KAK decomposition.
    