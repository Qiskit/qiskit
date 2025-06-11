========
Versions
========

Qiskit's version can be queried using a set of compiler macros. For example, 
``QISKIT_VERSION`` is a ``char*``, human-readable version and ``QISKIT_VERSION_HEX`` is 
a 4-byte HEX number encoding the version.

Concretely, the following definitions are available, which are here shown for version 2.1.0rc1:

.. code-block:: c

    #define QISKIT_RELEASE_LEVEL_DEV 0xA  // dev (or "a"lpha), has serial 0
    #define QISKIT_RELEASE_LEVEL_RC 0xC  // "c"andidate release
    #define QISKIT_RELEASE_LEVEL_FINAL 0xF // "f"inal release, has serial 0

    #define QISKIT_VERSION_MAJOR 2
    #define QISKIT_VERSION_MINOR 1
    #define QISKIT_VERSION_PATCH 0
    #define QISKIT_RELEASE_LEVEL QISKIT_RELEASE_LEVEL_RC
    #define QISKIT_RELEASE_SERIAL 1 // 0 for dev or final

    #define QISKIT_VERSION "2.1.0rc1" // human readable version of the string

    // macro to obtain a a numeric value for the version
    #define QISKIT_GET_VERSION_HEX(major, minor, patch, level, serial) (\
        (major & 0xff) << 24 | \ 
        (minor & 0xff) << 16 | \
        (patch & 0xff) << 8 | \
        (level & 0xf) << 4 | \ 
        (serial & 0xf)\
    )

    // for 2.1.0rc1 this is 0x020100C1 
    #define QISKIT_VERSION_HEX QISKIT_GET_VERSION_HEX(\
        QISKIT_VERSION_MAJOR, \
        QISKIT_VERSION_MINOR, \
        QISKIT_VERSION_PATCH, \
        QISKIT_RELEASE_LEVEL, \
        QISKIT_RELEASE_SERIAL \
    )

This can be used to check the current version, e.g. to ensure the version is at least 2.1.0 (final),
you can use

.. code-block:: c

    if (QISKIT_VERSION_HEX >= QISKIT_GET_VERSION_HEX(2, 1, 0, 0xF, 0)) {
        // Code for version 2.1.0 (final) or later
    }
