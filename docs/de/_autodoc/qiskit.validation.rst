qiskit.validation package
=========================


.. automodule:: qiskit.validation
    
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       qiskit.validation.base
       qiskit.validation.validate

    Subpackages
    -----------

    .. toctree::
       :maxdepth: 1

       qiskit.validation.fields

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`ValidationError <marshmallow.exceptions.ValidationError>`
         - Raised when validation fails on a field.
    

    Classes
    -------


    .. list-table::
    
       * - :class:`BaseModel <qiskit.validation.base.BaseModel>`
         - Base class for Models for validated Qiskit classes.
       * - :class:`BaseSchema <qiskit.validation.base.BaseSchema>`
         - Base class for Schemas for validated Qiskit classes.
       * - :class:`ModelTypeValidator <qiskit.validation.base.ModelTypeValidator>`
         - A field able to validate the correct type of a value.
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`bind_schema <qiskit.validation.base.bind_schema>`
         - Class decorator for adding schema validation to its instances.
    