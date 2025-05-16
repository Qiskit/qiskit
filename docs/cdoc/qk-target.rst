========
QkTarget
========

.. code-block:: C

    typedef struct QkTarget QkTarget

A mapping of instructions and properties representing the partiucular constraints
of a backend. Its purpose is to provide the compiler with information that allows it
to compile an input circuit into another that is optimized taking in consideration the
``Target``'s specifications. This structure represents a low level interface to the main
:class:`.Target` data structure in Rust, which represents the base class for its Python
counterpart.

Here's an example of how this structure works:

.. code-block:: C

    #include <qiskit.h>
    #include "math.h"

    // Create a Target with 2 qubits
    QkTarget *target = qk_target_new(2);
    // Create a property map with qargs (0, 1) and properties
    // duration = 0.0123
    // error = NaN
    uint32_t qargs[2] = {0, 1};
    QkPropsMap *property_map = qk_property_map_new();
    qk_property_map_add(property_map, qargs, 2, 0.0123, NAN);
    
    // Add a CX Gate to the target
    qk_target_add_instruction(target, QkGate_CX, NULL, property_map);

    // Add a global H gate
    qk_target_add_instruction(target, QkGate_H, NULL, NULL);

    // Check if the Target is compatible with a CX operation on qargs (0,1)
    bool result = qk_target_instruction_supported(target, QkGate_CX, qargs);

    // Free the created property_map and target.
    qk_property_map_free(property_map);
    qk_target_free(target);

The Target C API currently only supports additions of ``StandardGate`` instances as
``NormalOperation`` and queries using ``StandardGate`` instead of gate names. The
functionality will be expanded over time as we improve our Rust data model capabilities.

Data Types
==========

.. doxygenstruct:: QkInstructionProps
    :members:

.. doxygenstruct:: QkTargetQargs
    :members:

.. doxygenstruct:: QkTargetQargsList
    :members:

.. doxygenstruct:: QkTargetOpsList
    :members:

Functions
=========

.. doxygengroup:: QkTarget
    :members:
    :content-only:

==========
QkPropsMap
==========

.. code-block:: C

    typedef struct QkPropsMap QkPropsMap

A mapping of qargs and instruction properties representing gate map of the 
Target. This feature is used due to not having valid native mappings available from
C.

Here's an example of how this structure works:

.. code-block:: C

    #include <qiskit.h>
    #include "math.h"

    // Create a Property Map
    QkPropsMap *property_map = qk_property_map_new();

    // Add mapping between (0,1) and InstructionProperties(10e-9, NAN)
    uint32_t qargs[2] = {0, 1};
    qk_property_map_add(property_map, qargs, 2, 10e-9, NAN);

    // Add mapping between Global, and InstructionProperties(NAN, 0.003)
    qk_property_map_add(property_map, NULL, 0, NAN, 0.003);

    // Check if (0,2) are present
    uint32_t check_qargs[2] = {0, 2};
    qk_property_map_contains_qargs(property_map, qargs, 2);

    // Free the pointer
    qk_property_map_free(property_map)

.. note::
    This feature might be removed in a future release as we move to more C native
    structure usage.

Functions
=========

.. doxygengroup:: QkPropsMap
    :members:
    :content-only: