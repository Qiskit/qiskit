qiskit.validation.fields package
================================


.. automodule:: qiskit.validation.fields
    :members: Boolean, Complex, Date, DateTime, Email, Float, Integer, Number, Raw, String, Url
    :undoc-members:
    :show-inheritance:
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       qiskit.validation.fields.containers
       qiskit.validation.fields.polymorphic

    Classes
    -------


    .. list-table::
    
       * - :class:`Boolean <qiskit.validation.fields.Boolean>`
         - A boolean field.
       * - :class:`ByAttribute <qiskit.validation.fields.polymorphic.ByAttribute>`
         - Polymorphic field that disambiguates based on an attribute's existence.
       * - :class:`ByType <qiskit.validation.fields.polymorphic.ByType>`
         - Polymorphic field that disambiguates based on an attribute's type.
       * - :class:`Complex <qiskit.validation.fields.Complex>`
         - Field for complex numbers.
       * - :class:`Date <qiskit.validation.fields.Date>`
         - ISO8601-formatted date string.
       * - :class:`DateTime <qiskit.validation.fields.DateTime>`
         - A formatted datetime string in UTC.
       * - :class:`Email <qiskit.validation.fields.Email>`
         - A validated email field.
       * - :class:`Float <qiskit.validation.fields.Float>`
         - A double as IEEE-754 double precision string.
       * - :class:`Integer <qiskit.validation.fields.Integer>`
         - An integer field.
       * - :class:`List <qiskit.validation.fields.containers.List>`
         - A list field, composed with another `Field` class or instance.
       * - :class:`ModelTypeValidator <qiskit.validation.base.ModelTypeValidator>`
         - A field able to validate the correct type of a value.
       * - :class:`Nested <qiskit.validation.fields.containers.Nested>`
         - Allows you to nest a :class:`Schema <marshmallow.Schema>` inside a field.
       * - :class:`Number <qiskit.validation.fields.Number>`
         - Base class for number fields.
       * - :class:`Raw <qiskit.validation.fields.Raw>`
         - A boolean field.
       * - :class:`String <qiskit.validation.fields.String>`
         - A string field.
       * - :class:`TryFrom <qiskit.validation.fields.polymorphic.TryFrom>`
         - Polymorphic field that returns the first candidate schema that matches.
       * - :class:`Url <qiskit.validation.fields.Url>`
         - A validated URL field.
       * - :class:`date <datetime.date>`
         - date(year, month, day) --> date object
       * - :class:`datetime <datetime.datetime>`
         - datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`is_collection <marshmallow.utils.is_collection>`
         - Return True if ``obj`` is a collection type, e.g list, tuple, queryset.
    