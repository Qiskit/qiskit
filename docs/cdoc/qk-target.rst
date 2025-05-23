========
QkTarget
========

.. code-block:: C

    typedef struct QkTarget QkTarget

A mapping of instructions and properties representing the partiucular constraints
of a backend. Its purpose is to provide the compiler with information that allows it
to compile an input circuit into another that is optimized taking in consideration the
``Target``'s specifications. This structure represents a low level interface to the main
Target data structure in Rust, which represents the base class for its Python
counterpart, :class:`.Target`.

Here's an example of how this structure works:

.. code-block:: C

    #include <qiskit.h>
    #include <math.h>

    // Create a Target with 2 qubits
    QkTarget *target = qk_target_new(2);
    // Create a property map with qargs (0, 1) and properties
    // duration = 0.0123
    // error = NaN
    uint32_t qargs[2] = {0, 1};
    QkPropsMap *property_map = qk_property_map_new();
    qk_property_map_add(property_map, qargs, 2, 0.0123, NAN);
    
    // Add a CX Gate to the target
    qk_target_add_instruction(target, QkGate_CX, property_map);

    // Add a global H gate
    qk_target_add_instruction(target, QkGate_H, NULL);

    // Free the created property_map and target.
    qk_property_map_free(property_map);
    qk_target_free(target);

The Target C API currently only supports additions of ``StandardGate`` instances as
``NormalOperation``. The functionality will be expanded over time as we improve our
Rust data model capabilities.

Functions
=========

.. doxygengroup:: QkTarget
    :members:
    :content-only:
