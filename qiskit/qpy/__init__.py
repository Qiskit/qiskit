# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
###########################################################
QPY serialization (:mod:`qiskit.qpy`)
###########################################################

.. currentmodule:: qiskit.qpy

*********
Using QPY
*********

Using QPY is defined to be straightforward and mirror the user API of the
serializers in Python's standard library, ``pickle`` and ``json``. There are
2 user facing functions: :func:`qiskit.qpy.dump` and :func:`qiskit.qpy.load`
which are used to dump QPY data to a file object and load circuits from QPY data
in a file object respectively.
For example::

    from qiskit.circuit import QuantumCircuit
    from qiskit import qpy

    qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    with open('bell.qpy', 'wb') as fd:
        qpy.dump(qc, fd)

    with open('bell.qpy', 'rb') as fd:
        new_qc = qpy.load(fd)[0]

We can also write and read pulse programs to and from the file object.
For example::

    from qiskit.pulse import build, Play, Gaussian, DriveChannel
    from qiskit import qpy

    with build() as sched:
        Play(Gaussian(160, 0.1, 40), DriveChannel(0))

    with open('pi_sched.qpy', 'wb') as fd:
        qpy.dump(sched, fd)

    with open('pi_sched.qpy', 'rb') as fd:
        new_sched = qpy.load(fd)


API documentation
=================

.. autosummary::
   :toctree: ../stubs/

   load
   dump

QPY Compatibility
=================

The QPY format is designed to be backwards compatible moving forward. This means
you should be able to load a QPY with any newer Qiskit version than the one
that generated it. However, loading a QPY file with an older Qiskit version is
not supported and may not work.

For example, if you generated a QPY file using qiskit-terra 0.18.1 you could
load that QPY file with qiskit-terra 0.19.0 and a hypothetical qiskit-terra
0.29.0. However, loading that QPY file with 0.18.0 is not supported and may not
work.

**********
QPY Format
**********

The QPY serialization format is a portable cross-platform binary
serialization format for :class:`~qiskit.circuit.QuantumCircuit`
or :class:`~qiskit.pulse.schedule.ScheduleBlock` objects in Qiskit.
The basic file format is as follows:

A QPY file (or memory object) always starts with the following 7
byte UTF8 string: ``QISKIT`` which is immediately followed by the overall
file header. The contents of the file header as defined as a C struct are:

.. code-block:: c

    struct {
        uint8_t qpy_version;
        uint8_t qiskit_major_version;
        uint8_t qiskit_minor_version;
        uint8_t qiskit_patch_version;
        uint64_t num_programs;
    }

All values use network byte order [#f1]_ (big endian) for cross platform
compatibility.

The file header is immediately followed by the quantum program payloads.
Each individual circuit is composed of the following parts:

    ``HEADER | NAME | METADATA | REGISTERS | CUSTOM_DEFINITIONS | INSTRUCTIONS``

Each individual pulse schedule is composed of the following parts:

    ``HEADER | NAME | METADATA | ALIGNMENT | INSTRUCTIONS``

There is a circuit or pulse schedule payload for each programs (where the total number is dictated
by ``num_programs`` in the file header). There is no padding between the
programs in the data.

.. _version_3:

Version 3
=========

Version 3 of the QPY format is different from previous versions. The pulse program
:class:`~qiskit.pulse.schedule.ScheduleBlock` is newly supported from this version.
According to this change, the type identifier, which is a single character either ``"q"`` or ``"s"``
representing the quantum circuit or pulse schedule, respectively, is added in front of
each quantum program payload. Since this identifier information is missing in old QPY data,
the :func:`~qiskit.qpy.interface.load` function assumes the QPY binary data consists only of
:class:`~qiskit.circuit.QuantumCircuit` type payload when QPY version < 2.

Pulse schedule binary representation is described as follows. Note that quantum circuit
representation remains unchanged.

HEADER
------

The contents of :class:`~qiskit.pulse.schedule.ScheduleBlock` HEADER are defined as a C struct are:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint64_t metadata_size;
        uint64_t alignment_data_size;
        uint64_t num_elements;
    }

This is immediately followed by ``name_size`` bytes of utf8 data for the name of
the schedule. Following this is immediately ``metadata_size`` bytes of the json serialized
schedule metadata which takes the same format as the quantum circuit payload.

ALIGNMENT
---------

Alignment is the data field which is peculiar to :class:`~qiskit.pulse.schedule.ScheduleBlock`,
that represents soft scheduling constraints of the context instructions. See
:class:`~qiskit.pulse.transforms.alignments.AlignmentKind` for details.
The binary representation of this data consists of uft data for the name of
alignment class and the context parameter that is encoded to the dedicated mapping representation.

MAPPING
-------

This mapping representation can take :class:`~qiskit.circuit.parameterexpression.ParameterExpression`,
in addition to numerical values and strings as a value of the dictionary keyed on string
that will be encoded into utf8 data.
This data chunk begins with ``uint16_t`` number representing the number of mapping items,
followed by each dictionary item payload starting from below header:

.. code-block:: c

     struct {
        uint16_t name_size;
        char type;
        uint64_t data_size;
     }

This is immediately followed by the ``name_size`` bytes of utf8 data for the key string.
Following this is immediately ``type`` character representing the data format of the value,
and ``data_size`` bytes of arbitrary numerical or parameter expression object.

ELEMENTS
--------

Note that :class:`~qiskit.pulse.schedule.ScheduleBlock` can take pulse
:class:`~qiskit.pulse.instructions.instruction.Instruction` or another pulse program, i.e. nesting,
as a block. This block element is written in a data chunk begging with the header:

.. code-block:: c

     struct {
        char type;
        uint64_t size;
     }

with type key ``"i"`` for instructions and ``"b"`` for :class:`~qiskit.pulse.schedule.ScheduleBlock`.

INSTRUCTIONS
------------

Pulse program may consist of several different instruction kinds. However, these instructions
are designed to have common format ``Instruction(*operand)``, thus the pulse instruction
is formatted with the header:

.. code-block:: c

     struct {
        uint16_t name_size;
        uint16_t label_size;
        uint16_t num_operands;
     }

that is immediately followed by the ``name_size`` bytes of utf8 data for the instruction class name.
Following this is immediately the name associated with each instruction instance, i.e. label,
and the number of operands that might be different for each instruction kind.

OPERANDS
--------

Representative pulse instruction operands are

    - Numerical values and parameter expressions
    - :class:`~qiskit.pulse.channels.Channel` instance with index value that can take parameter
    - :class:`~qiskit.pulse.library.waveform.Waveform` mainly consists of
      a long numpy array of data points.
    - :class:`~qiskit.pulse.library.parametric_pulses.ParametricPulse` mainly consists of
      a parameter dictionary, and the class itself representing the envelope kind.
    - :class:`~qiskit.pulse.configuration.Kernel` and :class:`~qiskit.pulse.configuration.Discriminator`
      instance for the acquisition instruction configuration, with the name representing the
      hardware-supported processor kind and associated parameter dictionary.

These operands are encoded in the dedicated format with the binary header data.

CHANNEL
-------

.. code-block:: c

     struct {
        uint16_t name_size;
        char index_type;
        uint16_t index_size;
     }

followed by the channel instance class name, data format of the channel index value,
and binary representation of the index value.

WAVEFORM
--------

.. code-block:: c

     struct {
        uint16_t label_size;
        double epsilon;
        uint64_t data_size;
        _Bool amp_limited;
     }

followed by the utf8 data for the pulse name, tolerance for the maximum pulse amplitude limit,
``data_size`` bytes of complex numpy ``ndarray`` representing the waveform envelope,
and the boolean flag to validate maximum amplitude constraint on the hardware.

PARAMETRIC_PULSE
----------------

.. code-block:: c

     struct {
        uint16_t name_size;
        uint16_t module_path_size;
        uint16_t label_size;
        _Bool amp_limited;
     }

followed by the utf8 data for the parametric pulse class name, its module path when
the pulse is not Qiskit-defined (otherwise empty), another utf8 data for the pulse name,
the boolean flag to validate maximum amplitude constraint on the hardware,
and the data chunk for mapping shown above representing the parametric pulse parameters.

KERNEL and DISCRIMINATOR
------------------------

.. code-block:: c

     struct {
        uint16_t name_size;
        uint64_t params_size;
     }

followed by the utf data for the hardware-supported processor kind, and ``params_size`` bytes
of json serialized dictionary of the processor parameters.

.. _version_2:

Version 2
=========

Version 2 of the QPY format is identical to version 1 except for the HEADER
section is slightly different. You can refer to the :ref:`version_1` section
for the details on the rest of the payload format.

HEADER
------

The contents of HEADER are defined as a C struct are:

.. code-block:: c

    struct {
        uint16_t name_size;
        char global_phase_type;
        uint16_t global_phase_size;
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
        uint64_t num_custom_gates;
    }

This is immediately followed by ``name_size`` bytes of utf8 data for the name
of the circuit. Following this is immediately ``global_phase_size`` bytes
representing the global phase. The content of that data is dictated by the
value of ``global_phase_type``. If it's ``'f'`` the data is a float and is the
size of a ``double``. If it's ``'p'`` defines a :class:`~qiskit.circuit.Parameter`
object  which is represented by a PARAM struct (see below), ``e`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (that's not a
:class:`~qiskit.circuit.Parameter`) which is represented by a PARAM_EXPR struct
(see below).

.. _version_1:

Version 1
=========

HEADER
------

The contents of HEADER as defined as a C struct are:

.. code-block:: c

    struct {
        uint16_t name_size;
        double global_phase;
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
        uint64_t num_custom_gates;
    }

This is immediately followed by ``name_size`` bytes of utf8 data for the name
of the circuit.

METADATA
--------

The METADATA field is a UTF8 encoded JSON string. After reading the HEADER
(which is a fixed size at the start of the QPY file) and the ``name`` string
you then read the`metadata_size`` number of bytes and parse the JSON to get
the metadata for the circuit.

REGISTERS
---------

The contents of REGISTERS is a number of REGISTER object. If num_registers is
> 0 then after reading METADATA you read that number of REGISTER structs defined
as:

.. code-block:: c

    struct {
        char type;
        _Bool standalone;
        uint32_t size;
        uint16_t name_size;
    }

``type`` can be ``'q'`` or ``'c'``.

Immediately following the REGISTER struct is the utf8 encoded register name of
size ``name_size``. After the ``name`` utf8 bytes there is then an array of
uint32_t values of size ``size`` that contains a map of the register's index to
the circuit's qubit index. For example, array element 0's value is the index
of the ``register[0]``'s position in the containing circuit's qubits list.

The standalone boolean determines whether the register is constructed as a
standalone register that was added to the circuit or was created from existing
bits. A register is considered standalone if it has bits constructed solely
as part of it, for example::

    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)

the register ``qr`` would be a standalone register. While something like::

    bits = [Qubit(), Qubit()]
    qr = QuantumRegister(bits=bits)
    qc = QuantumCircuit(bits=bits)

``qr`` would have ``standalone`` set to ``False``.


CUSTOM_DEFINITIONS
------------------

This section specifies custom definitions for any of the instructions in the circuit.

CUSTOM_DEFINITION_HEADER contents are defined as:

.. code-block:: c

    struct {
        uint64_t size;
    }

If size is greater than 0 that means the circuit contains custom instruction(s).
Each custom instruction is defined with a CUSTOM_INSTRUCTION block defined as:

.. code-block:: c

    struct {
        uint16_t name_size;
        char type;
        _Bool custom_definition;
        uint64_t size;
    }

Immediately following the CUSTOM_INSTRUCTION struct is the utf8 encoded name
of size ``name_size``.

If ``custom_definition`` is ``True`` that means that the immediately following
``size`` bytes contains a QPY circuit data which can be used for the custom
definition of that gate. If ``custom_definition`` is ``False`` then the
instruction can be considered opaque (ie no definition).

INSTRUCTIONS
------------

The contents of INSTRUCTIONS is a list of INSTRUCTION metadata objects

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t label_size;
        uint16_t num_parameters;
        uint32_t num_qargs;
        uint32_t num_cargs;
        _Bool has_conditional;
        uint16_t conditional_reg_name_size;
        int64_t conditional_value;
    }

This metadata object is immediately followed by ``name_size`` bytes of utf8 bytes
for the ``name``. ``name`` here is the Qiskit class name for the Instruction
class if it's defined in Qiskit. Otherwise it falls back to the custom
instruction name. Following the ``name`` bytes there are ``label_size`` bytes of
utf8 data for the label if one was set on the instruction. Following the label
bytes if ``has_conditional`` is ``True`` then there are
``conditional_reg_name_size`` bytes of utf8 data for the name of the conditional
register name. In case of single classical bit conditions the register name
utf8 data will be prefixed with a null character "\\x00" and then a utf8 string
integer representing the classical bit index in the circuit that the condition
is on.

This is immediately followed by the INSTRUCTION_ARG structs for the list of
arguments of that instruction. These are in the order of all quantum arguments
(there are num_qargs of these) followed by all classical arguments (num_cargs
of these).

The contents of each INSTRUCTION_ARG is:

.. code-block:: c

    struct {
        char type;
        uint32_t index;
    }

``type`` can be ``'q'`` or ``'c'``.

After all arguments for an instruction the parameters are specified with
``num_parameters`` INSTRUCTION_PARAM structs.

The contents of each INSTRUCTION_PARAM is:

.. code-block:: c

    struct {
        char type;
        uint64_t size;
    }

After each INSTRUCTION_PARAM the next ``size`` bytes are the parameter's data.
The ``type`` field can be ``'i'``, ``'f'``, ``'p'``, ``'e'``, ``'s'``, ``'c'``
or ``'n'`` which dictate the format. For ``'i'`` it's an integer, ``'f'`` it's
a double, ``'s'`` if it's a string (encoded as utf8), ``'c'`` is a complex and
the data is represented by the struct format in the :ref:`param_expr` section.
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is
represented by a PARAM struct (see below), ``e`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (that's not a
:class:`~qiskit.circuit.Parameter`) which is represented by a PARAM_EXPR struct
(see below), and ``'n'`` represents an object from numpy (either an ``ndarray``
or a numpy type) which means the data is .npy format [#f2]_ data.

PARAMETER
---------

A PARAMETER represents a :class:`~qiskit.circuit.Parameter` object the data for
a INSTRUCTION_PARAM. The contents of the PARAMETER are defined as:

.. code-block:: c

    struct {
        uint16_t name_size;
        char uuid[16];
    }

which is immediately followed by ``name_size`` utf8 bytes representing the
parameter name.

.. _param_expr:

PARAMETER_EXPR
--------------

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The contents of a PARAMETER_EXPR
are defined as:

The PARAMETER_EXPR data starts with a header:

.. code-block:: c

    struct {
        uint64_t map_elements;
        uint64_t expr_size;
    }

Immediately following the header is ``expr_size`` bytes of utf8 data containing
the expression string, which is the sympy srepr of the expression for the
parameter expression. Follwing that is a symbol map which contains
``map_elements`` elements with the format

.. code-block:: c

    struct {
        char type;
        uint64_t size;
    }

Which is followed immediately by ``PARAMETER`` object (both the struct and utf8
name bytes) for the symbol map key. That is followed by ``size`` bytes for the
data of the symbol. The data format is dependent on the value of ``type``. If
``type`` is ``p`` then it represents a :class:`~qiskit.circuit.Parameter` and
size will be 0, the value will just be the same as the key. If
``type`` is ``f`` then it represents a double precision float. If ``type`` is
``c`` it represents a double precision complex, which is represented by:

.. code-block:: c

    struct {
        double real;
        double imag;
    }

this matches the internal C representation of Python's complex type. [#f3]_
Finally, if type is ``i`` it represents an integer which is an ``int64_t``.


.. [#f1] https://tools.ietf.org/html/rfc1700
.. [#f2] https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
.. [#f3] https://docs.python.org/3/c-api/complex.html#c.Py_complex
"""

from .interface import dump, load
