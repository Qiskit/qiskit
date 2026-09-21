.. _capi-qpy:

===
QPY
===

QPY is Qiskit's portable, cross-platform binary format for storing quantum circuits. The QPY C
API can serialize multiple :c:struct:`QkCircuit` objects to a file or an in-memory buffer and load
all the circuits from either representation. QPY data produced by this interface is compatible
with the Python :mod:`qiskit.qpy` interface.

:c:func:`qk_qpy_read_min_version` returns the oldest QPY format version that the loaded library
can read. This is equivalent to Python's :attr:`qiskit.qpy.QPY_COMPATIBILITY_VERSION`.
:c:func:`qk_qpy_write_min_version` returns the oldest version that the C API can write. The two
bounds differ because the C API uses the Rust QPY implementation, while Python can use its legacy
implementation to write older formats.

Loaded circuits are newly allocated and must be released with
:c:func:`qk_qpy_free_circuits`.

The standard dump functions accept native :c:struct:`QkCircuit` objects and do not interact with
Python. Circuits borrowed from Python must instead be passed to the corresponding
``*_from_python`` function:

* :c:func:`qk_qpy_dump_file_from_python`
* :c:func:`qk_qpy_dump_file_with_version_from_python`
* :c:func:`qk_qpy_dump_buffer_from_python`
* :c:func:`qk_qpy_dump_buffer_with_version_from_python`

The thread calling one of these Python-specific functions must already be attached to a Python
interpreter and hold the GIL. The functions register the call with PyO3 while cloning
Python-owned circuit data, but callers remain responsible for all GIL coordination. In
particular, do not call one from a child thread while another thread retains the GIL; arrange for
the calling thread to hold the GIL before entering the function.

The following example writes two circuits to a file and loads them again:

.. code-block:: c

   #include <qiskit.h>
   #include <stddef.h>

   int main(void) {
       QkCircuit *bell = qk_circuit_new(2, 0);
       qk_circuit_gate(bell, QkGate_H, (uint32_t[]){0}, NULL);
       qk_circuit_gate(bell, QkGate_CX, (uint32_t[]){0, 1}, NULL);

       QkCircuit *plus = qk_circuit_new(1, 0);
       qk_circuit_gate(plus, QkGate_H, (uint32_t[]){0}, NULL);

       const QkCircuit *circuits[] = {bell, plus};

       if (qk_qpy_dump_file(circuits, 2, "circuits.qpy", NULL) != QkExitCode_Success) {
           qk_circuit_free(plus);
           qk_circuit_free(bell);
           return 1;
       }

       QkCircuit **loaded = NULL;
       size_t num_loaded = 0;
       if (qk_qpy_load_file(&loaded, &num_loaded, "circuits.qpy", NULL) != QkExitCode_Success) {
           qk_circuit_free(plus);
           qk_circuit_free(bell);
           return 1;
       }

       int result = num_loaded == 2 ? 0 : 1;
       qk_qpy_free_circuits(loaded, num_loaded);
       qk_circuit_free(plus);
       qk_circuit_free(bell);
       return result;
   }

The buffer variants allocate the serialized data.  Pass the returned pointer and its exact size to
:c:func:`qk_qpy_free_buffer` when it is no longer needed. The ``*_with_version`` variants select a
specific output QPY format version; query :c:func:`qk_qpy_write_min_version` for the minimum
supported output version. All functions that can fail return :c:enum:`QkExitCode`, including
``QkExitCode_QpyError``
for serialization, deserialization, path, and file-I/O errors. Their final ``error`` argument may
be null to ignore diagnostic details. Otherwise, on a QPY error it is populated with an allocated,
nul-terminated description that must be released with :c:func:`qk_str_free`. The output pointer is
unchanged on success and for errors without an additional diagnostic.

Functions
=========

.. doxygengroup:: QkQpy
   :members:
   :content-only:
