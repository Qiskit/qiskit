.. _capi-qpy:

===
QPY
===

QPY is Qiskit's portable, cross-platform binary format for storing quantum circuits. The QPY C
API can serialize a :c:struct:`QkCircuit` to a file or an in-memory buffer and load a circuit from
either representation. QPY data produced by this interface is compatible with the Python
:mod:`qiskit.qpy` interface.

The C API currently reads only the first circuit from QPY data, and can save only one circuit at a time. The user is responsible for freeing any loaded circuits with
A loaded circuit is newly allocated and must be released with :c:func:`qk_circuit_free`.

The following example writes a Bell circuit to a file and loads it again:

.. code-block:: c

   #include <qiskit.h>

   int main(void) {
       QkCircuit *source = qk_circuit_new(2, 0);
       qk_circuit_gate(source, QkGate_H, (uint32_t[]){0}, NULL);
       qk_circuit_gate(source, QkGate_CX, (uint32_t[]){0, 1}, NULL);

       if (qk_qpy_dump_file(source, "bell.qpy") != QkExitCode_Success) {
           qk_circuit_free(source);
           return 1;
       }

       QkCircuit *loaded = NULL;
       if (qk_qpy_load_file(&loaded, "bell.qpy") != QkExitCode_Success) {
           qk_circuit_free(source);
           return 1;
       }

       qk_circuit_free(loaded);
       qk_circuit_free(source);
       return 0;
   }

The buffer variants allocate the serialized data.  Pass the returned pointer and its exact size to
:c:func:`qk_qpy_free_buffer` when it is no longer needed. The ``*_with_version`` variants select a
specific output QPY format version; the native serializer currently supports writing version 17 or
later. All functions that can fail return :c:enum:`QkExitCode`, including ``QkExitCode_QpyError`` 
for serialization, deserialization, path, and file-I/O errors.

Functions
=========

.. doxygengroup:: QkQpy
   :members:
   :content-only:
