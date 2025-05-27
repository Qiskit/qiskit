
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

    // Free the pointer
    qk_property_map_free(property_map)


Functions
=========

.. doxygengroup:: QkPropsMap
    :members:
    :content-only: