# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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

QPY is a binary serialization format for :class:`~.QuantumCircuit` and
:class:`~.ScheduleBlock` objects that is designed to be cross-platform,
Python version agnostic, and backwards compatible moving forward. QPY should
be used if you need a mechanism to save or copy between systems a
:class:`~.QuantumCircuit` or :class:`~.ScheduleBlock` that preserves the full
Qiskit object structure (except for custom attributes defined outside of
Qiskit code).

This differs from other serialization formats like
`OpenQASM <https://github.com/openqasm/openqasm>`__ (2.0 or 3.0) which has a
different abstraction model and can result in a loss of information contained
in the original circuit (or is unable to represent some aspects of the
Qiskit objects) or Python's `pickle <https://docs.python.org/3/library/pickle.html>`__
which will preserve the Qiskit object exactly but will only work for a single Qiskit
version (it is also
`potentially insecure <https://docs.python.org/3/library/pickle.html#module-pickle>`__).

*********
Using QPY
*********

Using QPY is defined to be straightforward and mirror the user API of the
serializers in Python's standard library, ``pickle`` and ``json``. There are
2 user facing functions: :func:`qiskit.qpy.dump` and
:func:`qiskit.qpy.load` which are used to dump QPY data
to a file object and load circuits from QPY data in a file object respectively.
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

The :func:`qiskit.qpy.dump` function also lets you
include multiple circuits in a single QPY file::

    with open('twenty_bells.qpy', 'wb') as fd:
        qpy.dump([qc] * 20, fd)

and then loading that file will return a list with all the circuits

    with open('twenty_bells.qpy', 'rb') as fd:
        twenty_new_bells = qpy.load(fd)

.. note::

    Different programs cannot be packed together in the same file.
    You must create different files for different program types.
    Multiple objects with the same type can be saved in a single file.

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

.. _qpy_format:

**********
QPY Format
**********

.. note::

    This section has been updated to the latest version of QPY (v.7).
    Older versions of QPY contain variations in the format (header and payload blocks).
    To review older formats, go to the "Older QPY versions" section. [find better name/phrasing].


The QPY serialization format is a portable cross-platform binary
serialization format for :class:`~qiskit.circuit.QuantumCircuit` and
:class:`~.ScheduleBlock` objects in Qiskit. The basic
file format is as follows:

A QPY file (or memory object) always starts with the following 7
byte UTF8 string: ``QISKIT`` which is immediately followed by the **overall
file header**. The file header is followed by the payload blocks.
These can be **circuit payloads** or **block schedule payloads** (introduced in v.5),
as defined by the **payload header** (also introduced in v.5).


``'QISKIT' | FILE_HEADER | PAYLOAD_HEADER | PAYLOAD_DATA_BLOCKS``


FILE_HEADER
===========

 The contents of the file header are defined as a C struct:

.. code-block:: c

    struct {
        uint8_t qpy_version;
        uint8_t qiskit_major_version;
        uint8_t qiskit_minor_version;
        uint8_t qiskit_patch_version;
        uint64_t num_circuits; // number of payload blocks
    }

.. note::
    All values use network byte order [#f1]_ (big endian) for cross platform
    compatibility.

``num_circuits`` defines the number of consecutive payload blocks that will be found in the file,
with no padding in between data blocks.

PAYLOAD_HEADER
==============

In QPY v.5 and above, the payload header:

.. code-block:: c

    struct {
        char type;
    }

immediately follows the file header block to represent the program type stored in the file:

- When ``type==c``, :class:`~.QuantumCircuit` payloads follow
- When ``type==s``, :class:`~.ScheduleBlock` payloads follow

.. note::

    Different types of payloads cannot be packed together in the same file.
    You must create different files for different payload types.
    However, multiple objects with the same payload type **can** be saved in a single file.

:class:`~.QuantumCircuit`  Payloads
===================================

The structure of the circuit payloads is as follows:

``CIRCUIT_HEADER | METADATA | REGISTERS | CUSTOM_DEFINITIONS | INSTRUCTIONS | CIRCUIT_CALIBRATIONS``

In QPY v.5 and above, the :ref:`qpy_circuit_calibrations` payload is
packed after the :ref:`qpy_instructions` block to support the serialization of calibration
gates/pulse gates.

CIRCUIT_HEADER
--------------

The contents of the CIRCUIT_HEADER block are:

``HEADER | NAME | GLOBAL_PHASE_DATA``

The HEADER is defined as a C struct:

.. code-block:: c

    struct {
        uint16_t name_size;
        char global_phase_type; // added in v.2, replacing global_phase
        uint16_t global_phase_size; // added in v.2, replacing global_phase
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
        uint64_t num_custom_gates;
    }

The HEADER is immediately followed by ``name_size`` bytes of UTF-8 data containing the name
of the circuit. Following this are ``global_phase_size`` bytes
representing the global phase. The type of the data contained in this block is dictated by the
value of ``global_phase_type``. If it's ``'f'``, the data is a float and is the
size of a ``double``. If it's ``'p'``, it defines a :class:`~qiskit.circuit.Parameter`
object  which is represented by a :ref:`qpy_param_struct`, ``'e'`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (note that this is different from
:class:`~qiskit.circuit.Parameter`), which is represented by a :ref:`qpy_param_expr`.

METADATA
--------

The next field to be parsed is the METADATA field, a UTF-8 encoded JSON string, of
``metadata_size`` number of bytes as defined by the HEADER.


.. _qpy_registers:

REGISTERS
---------

The following field contains a series of ``REGISTER`` blocks of format:

``HEADER | REGISTER_NAME | REGISTER_MAP``

The number of blocks is determined by the ``CIRCUIT_HEADER``'s ``num_registers`` field (if ``num_registers==0``, no registers
will be parsed).

Each ``REGISTER`` contains a header struct:

.. code-block:: c

    struct {
        char type;
        _Bool standalone;
        uint32_t size;
        uint16_t name_size;
        _bool in_circuit; // added in v.4
    }

Where ``type`` can be ``'q'`` (for quantum register) or ``'c'`` (for classical register).
The ``in_circuit`` field represents whether the register is part of the circuit or not.

The ``standalone`` boolean determines whether the register is constructed as a
standalone register that was added to the circuit or was created from existing
bits. A register is considered standalone if it has bits constructed solely
as part of it, for example::

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

the register ``qr`` would be a standalone register. While something like::

        bits = [Qubit(), Qubit()]
        qr2 = QuantumRegister(bits=bits)
        qc = QuantumCircuit(qr2)

``qr2`` would have ``standalone`` set to ``False``.

For each block, immediately following the HEADER struct is the UTF-8 encoded register name of
size ``name_size``. After the ``name``, there is then an array of
``int64_t`` values of size ``size`` that contains a map of the register's index to
the circuit's qubit index. For example, array element 0's value is the index
of the ``register[0]``'s position in the containing circuit's qubits list.

.. note::
    [NO IDEA WHERE THIS GOES YET]
    Prior to QPY v.4 the type of array elements was ``uint32_t``. This was changed
    to enable negative values which represent bits in the array not present in the
    circuit. Additionally, version 4 changes the type of register index mapping
    array from uint32_t to int64_t. If the values of any of the array elements
    are negative they represent a register bit that is not present in the circuit.

.. _qpy_custom_definition:

CUSTOM_DEFINITIONS
------------------

This block specifies custom definitions for any of the instructions in the circuit, it contains
a header followed by a serious of CUSTOM_INSTRUCTION blocks:

``CUSTOM_DEFINITION_HEADER | CUSTOM_INSTRUCTIONS``

CUSTOM_DEFINITION_HEADER contents are defined as:

.. code-block:: c

    struct {
        uint64_t size;
    }

If ``size`` is greater than 0, the circuit contains custom instructions.

Each ``CUSTOM_INSTRUCTION`` block contains a header followed by data
that depends on the instruction represented:

``HEADER | NAME | INSTRUCTION_DATA``

The``CUSTOM_INSTRUCTION`` header is a C struct defined as:

.. code-block:: c

    struct {
        uint16_t name_size;
        char type;
        uint32_t num_qubits;
        uint32_t num_clbits;
        _Bool custom_definition;
        uint64_t size;
        uint32_t num_ctrl_qubits; // added in QPY v.5
        uint32_t ctrl_state; // added in QPY v.5
        uint64_t base_gate_size // added in QPY v.5
    }

Immediately following the header struct is the UTF-8 encoded name
of size ``name_size``. It should be noted that :class:`~qiskit.circuit.library.PauliEvolutionGate`
entries (added in v.3) have a unique name that is generated with the string
``"###PauliEvolutionGate_"`` followed by a uuid string.
This gate name is reserved in QPY, and cannot be used for other custom
:class:`~qiskit.circuit.Instruction` objects.

If ``custom_definition`` is ``True``, the immediately following
``size`` bytes contain QPY circuit data that can be used for the custom
definition of that gate. If ``custom_definition`` is ``False``, then the
instruction can be considered opaque (i.e. no custom definition).

The ``type`` field determines what type of object will get
created with the custom definition.

If it's ``'g'``, it will be a :class:`~qiskit.circuit.Gate` object, with ``'i'``
it will be a :class:`~qiskit.circuit.Instruction` object (serialized as :ref:`qpy_instructions`). The type value
``'p'`` represents a
:class:`~qiskit.circuit.library.PauliEvolutionGate` (only after v.3), and the
data payload is defined in :ref:`pauli_evo_qpy`.

Additionally, if ``type`` is ``'c'``, the custom instruction is a custom :class:`~.ControlledGate`
and ``base_gate_size`` is used to define the size of the
:class:`qiskit.circuit.Instruction` object stored in the
:attr:`.ControlledGate.base_gate` attribute. Thus, the next ``base_gate_size`` bytes
will contain the ``INSTRUCTION`` payload for the :attr:`.ControlledGate.base_gate`.

.. _pauli_evo_qpy:

PAULI_EVOLUTION
^^^^^^^^^^^^^^^

This block represents the high level :class:`~qiskit.circuit.library.PauliEvolutionGate` data payload:

``HEADER | SPARSE_PAULI_OPS | TIME | SYNTHESIS_DATA``

The ``PAULI_EVOLUTION`` header is a C struct:

.. code-block:: c

    struct {
        uint64_t operator_count;
        _Bool standalone_op;
        char time_type;
        uint64_t time_size;
        uint64_t synthesis_size;
    }

The header is immediately followed by ``operator_count`` elements defined by the :ref:`qpy_sparse_pauli_op`
payload.  Following that we have ``time_size`` bytes representing the ``time`` attribute.

If ``standalone_op`` is ``True`` then there must only be a single operator. The
encoding of these bytes is determined by the value of ``time_type``. Possible values of
``time_type`` are ``'f'``, ``'p'``, and ``'e'``. If ``time_type`` is ``'f'`` it's a double,
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is represented by a
:ref:`qpy_param_struct`, ``e`` defines a :class:`~qiskit.circuit.ParameterExpression` object
(that's not a :class:`~qiskit.circuit.Parameter`) which is represented by a :ref:`qpy_param_expr`.

Following that are ``synthesis_size`` bytes with a UTF-8 encoded JSON payload representing
the :class:`.EvolutionSynthesis` class used by the gate.

.. _qpy_sparse_pauli_op:

SPARSE_PAULI_OP_LIST_ELEM
^^^^^^^^^^^^^^^^^^^^^^^^^

This block represents an instance of :class:`.SparsePauliOp`, with the format:

``HEADER | PAULI_DATA``

The header struct is quite simple:

.. code-block:: c

    struct {
        uint32_t pauli_op_size;
    }

and immediately followed by ``pauli_op_size`` bytes of ``.npy`` format [#f2]_
data that represents the underlying :class:`~qiskit.quantum_info.SparsePauliOp`.


.. _qpy_instructions:

INSTRUCTIONS
------------

The INSTRUCTIONS field contains a list of INSTRUCTION blocks of format:

``HEADER | NAME | LABEL | CONDITIONAL_REG_NAME | INSTRUCTION_ARGS | INSTRUCTION_PARAMS``

The ``INSTRUCTION`` header is a struct:

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
        uint32_t num_ctrl_qubits; // added in QPY v.5
        uint32_t ctrl_state;  // added in QPY v.5
    }

This object is immediately followed by ``name_size`` bytes of UTF-8 bytes
storing the ``name``. ``name`` here is the Qiskit class name for the Instruction
class if it's defined in Qiskit. Otherwise it falls back to the custom
instruction name. Following the ``name`` bytes there are ``label_size`` bytes of
UTF-8 data for the label if one was set on the instruction. Following the label
bytes, if ``has_conditional`` is ``True``, there are
``conditional_reg_name_size`` bytes of UTF-8 data for the name of the conditional
register name. In case of single classical bit conditions the register name
UTF-8 data will be prefixed with a null character "\\x00" and then a UTF-8 string
integer representing the classical bit index in the circuit that the condition
is on.

This is immediately followed by a list of INSTRUCTION_ARG that represent the
arguments of that instruction. They contain ``num_qargs`` quantum arguments followed
by ``num_cargs`` classical arguments

Each INSTRUCTION_ARG is a represented as a struct:

.. code-block:: c

    struct {
        char type;
        int64_t index; //changed after v.4??
    }

Where ``type`` can be ``'q'`` or ``'c'``.

[Not sure if here it's also changed to int64_t. Look up] If the values of any of the
array elements of ``index`` are negative, they represent a
register bit that is not present in the circuit.

After all INSTRUCTION_ARGS are defined, the parameters are specified with
``num_parameters`` INSTRUCTION_PARAM blocks of format:

``HEADER | DATA``

The header defines the format of each INSTRUCTION_PARAM:

.. code-block:: c

    struct {
        char type;
        uint64_t size;
    }

After the header, the next ``size`` bytes  of the INSTRUCTION_PARAM contain the parameter's data.
The ``type`` field can be ``'i'``, ``'f'``, ``'p'``, ``'e'``, ``'s'``, ``'c'``, ``'n'``,
``'v'``, ``'z'``, ``'q'``, ``'r'``, ``'t'``.

``'i'`` stands for integer, ``'f'`` for double, ``'s'`` for string
(encoded as UTF-8), ``'c'`` is a complex type (the data is represented
by the struct format in the :ref:`qpy_param_expr` section),
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is
represented by a :ref:`qpy_param_struct` struct, ``'e'`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (that is not a
:class:`~qiskit.circuit.Parameter`) which is represented by a :ref:`qpy_param_expr`
struct, ``'n'`` represents an object from numpy (either an
``ndarray`` or a numpy type) which means the data is .npy format [#f2]_ data,
and in QPY v.3 onward, ``'v'`` represents a
:class:`~qiskit.circuit.ParameterVectorElement` which is represented by a
:ref:`qpy_param_vector` struct. In QPY v.4 onward, ``'z'`` represents ``None``
(which is encoded as no data), ``'q'`` represents a :class:`.QuantumCircuit`
(which is encoded as a QPY circuit), ``'r'`` represents a ``range`` of
integers (which is encoded as a :ref:`qpy_range_pack`), and ``'t'``
represents a ``sequence`` (which is encoded as
defined by :ref:`qpy_sequence`).

QPY Version 5 added support for controlled gates, where ``num_ctrl_qubits``
and ``ctrl_state`` are used to model the :attr:`.ControlledGate.num_ctrl_qubits` and
:attr:`.ControlledGate.ctrl_state` attributes.

.. _qpy_param_struct:

PARAMETER
^^^^^^^^^

A PARAMETER represents a :class:`~qiskit.circuit.Parameter` object, it's one of the data types for
INSTRUCTION_PARAMs. The contents of the PARAMETER are defined by a header:

.. code-block:: c

    struct {
        uint16_t name_size;
        char uuid[16];
    }

which is immediately followed by ``name_size`` UTF-8 bytes representing the
parameter name.

.. _qpy_param_expr:

PARAMETER_EXPR
^^^^^^^^^^^^^^

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The contents of a PARAMETER_EXPR
are defined by a header:

.. code-block:: c

    struct {
        uint64_t map_elements;
        uint64_t expr_size;
    }

Immediately following the header are ``expr_size`` bytes of UTF-8 data containing
the expression string, which is the sympy srepr of the expression for the
parameter expression. Following that is a symbol map which contains
``map_elements`` elements with the format

.. code-block:: c

    struct {
        char symbol_type;
        char type;
        uint64_t size;
    }

The ``symbol_type`` key determines the payload type of the symbol representation
for the element. If it's ``p`` it represents a :class:`~qiskit.circuit.Parameter`
and if it's ``v`` it represents a :class:`~qiskit.circuit.ParameterVectorElement`.
The map element struct is immediately followed by the symbol map key payload, if
``symbol_type`` is ``p`` then it is followed immediately by a :ref:`qpy_param_struct`
object (both the struct and UTF-8 name bytes) and if ``symbol_type`` is ``v``
then the struct is imediately followed by :ref:`qpy_param_vector` (both the struct
and UTF-8 name bytes). That is followed by ``size`` bytes for the
data of the symbol. The data format is dependent on the value of ``type``. If
``type`` is ``p`` then it represents a :class:`~qiskit.circuit.Parameter` and
size will be 0, the value will just be the same as the key. Similarly if the
``type`` is ``v`` then it represents a :class:`~qiskit.circuit.ParameterVectorElement`
and size will be 0 as the value will just be the same as the key. If
``type`` is ``f`` then it represents a double precision float. If ``type`` is
``c`` it represents a double precision complex, which is represented by the
:ref:`qpy_complex`. Finally, if type is ``i`` it represents an integer which is an
``int64_t``.

.. _qpy_param_vector:

PARAMETER_VECTOR_ELEMENT
^^^^^^^^^^^^^^^^^^^^^^^^

A PARAMETER_VECTOR_ELEMENT represents a :class:`~qiskit.circuit.ParameterVectorElement`
object the data for a INSTRUCTION_PARAM. The contents of the PARAMETER_VECTOR_ELEMENT are
defined as:

.. code-block:: c

    struct {
        uint16_t vector_name_size;
        uint64_t vector_size;
        char uuid[16];
        uint64_t index;
    }

which is immediately followed by ``vector_name_size`` UTF-8 bytes representing
the parameter's vector name.

.. _qpy_complex:

COMPLEX
^^^^^^^

When representing a double precision complex value in QPY the following
struct is used:


.. code-block:: c

    struct {
        double real;
        double imag;
    }

this matches the internal C representation of Python's complex type. [#f3]_

.. _qpy_range_pack:

RANGE
^^^^^

A RANGE is a representation of a ``range`` object. It is defined as:

.. code-block:: c

    struct {
        int64_t start;
        int64_t stop;
        int64_t step;
    }

.. _qpy_sequence:

SEQUENCE
^^^^^^^^

A SEQUENCE is a representation of an arbitrary sequence object. As sequencse are just fixed length
containers of arbitrary python objects their QPY can't fully represent any sequence,
but as long as the contents in a sequence are other QPY serializable types for
the INSTRUCTION_PARAM payload the ``sequence`` object can be serialized.

A sequence instruction parameter starts with a header defined as:

.. code-block:: c

    struct {
        uint64_t size;
    }

followed by ``size`` elements that are INSTRUCTION_PARAM payloads, where each of
these define an element in the sequence. The sequence object will be typecasted
into proper type, e.g. ``tuple``, afterwards.

.. _qpy_circuit_calibrations:

CIRCUIT_CALIBRATIONS
--------------------

The CIRCUIT_CALIBRATIONS block is a dictionary to define pulse calibrations of the custom
instruction set. This block starts with the following CALIBRATION header:

.. code-block:: c

    struct {
        uint16_t num_cals;
    }

which is followed by the ``num_cals`` length of calibration entries, each starts with
the CALIBRATION_DEF header:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t num_qubits;
        uint16_t num_params;
        char type;
    }

The calibration definition header is then followed by ``name_size`` UTF-8 bytes of
the gate name, ``num_qubits`` length of integers representing a sequence of qubits,
and ``num_params`` length of INSTRUCTION_PARAM payload for parameters
associated to the custom instruction.
The ``type`` indicates the class of pulse program which is either, in pricinple,
:class:`~.ScheduleBlock` or :class:`~.Schedule`. As of QPY Version 5,
only :class:`~.ScheduleBlock` payload is supported.
Finally, a :ref:`qpy_schedule_block` payload is packed for each CALIBRATION_DEF entry.


.. _qpy_schedule_block:

:class:`.~ScheduleBlock`  Payloads
===================================

QPY versions 5 and above support serialization of :class:`.~ScheduleBlock` payloads.
With the support of :class:`.~ScheduleBlock`, :class:`~.QuantumCircuit` can be
serialized together with :attr:`~.QuantumCircuit.calibrations`, or
`Pulse Gates <https://qiskit.org/documentation/tutorials/circuits_advanced/05_pulse_gates.html>`_.

.. code-block:: python

    from qiskit import pulse, qpy

    with pulse.build() as schedule:
        pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

    with open('schedule.qpy', 'wb') as fd:
        qpy.dump(qc, fd)

    with open('schedule.qpy', 'rb') as fd:
        new_qc = qpy.load(fd)[0]

Note that circuit and schedule block are serialized and deserialized through
the same QPY interface. Input data type is implicitly analyzed and
no extra option is required to save the schedule block.

A :class:`.~ScheduleBlock` payload will follow:

``SCHEDULE_BLOCK_HEADER | METADATA | SCHEDULE_BLOCK_ALIGNMENTS | SCHEDULE_BLOCK_INSTRUCTIONS | MAPPING``

Version 7 adds support for :class:`.~Reference` instruction and serialization of
a :class:`.~ScheduleBlock` program while keeping its reference to subroutines::

    from qiskit import pulse
    from qiskit import qpy

    with pulse.build() as schedule:
        pulse.reference("cr45p", "q0", "q1")
        pulse.reference("x", "q0")
        pulse.reference("cr45p", "q0", "q1")

    with open('template_ecr.qpy', 'wb') as fd:
        qpy.dump(schedule, fd)

The conventional :ref:`qpy_schedule_block` data model is preserved, but in
version 7 it is immediately followed by an extra :ref:`qpy_mapping` UTF-8 bytes block
representing the data of the referenced subroutines.

.. _qpy_schedule_block_header:

SCHEDULE_BLOCK_HEADER
---------------------

:class:`~.ScheduleBlock` block starts with the following header:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint64_t metadata_size;
        uint16_t num_element;
    }

which is immediately followed by ``name_size`` UTF-8 bytes of schedule name and
``metadata_size`` UTF-8 bytes of the JSON serialized metadata dictionary
attached to the schedule.

.. _qpy_schedule_alignments:

SCHEDULE_BLOCK_ALIGNMENTS
-------------------------

Then, alignment context of the schedule block starts with ``char``
representing the supported context type followed by the :ref:`qpy_sequence` block representing
the parameters associated with the alignment context :attr:`AlignmentKind._context_params`.
The context type char is mapped to each alignment subclass as follows:

- ``l``: :class:`~.AlignLeft`
- ``r``: :class:`~.AlignRight`
- ``s``: :class:`~.AlignSequential`
- ``e``: :class:`~.AlignEquispaced`

Note that :class:`~.AlignFunc` context is not supported becasue of the callback function
stored in the context parameters.

.. _qpy_schedule_instructions:

SCHEDULE_BLOCK_INSTRUCTIONS
---------------------------

This alignment block is further followed by ``num_element`` length of block elements which may
consist of nested schedule blocks and schedule instructions.

Each schedule instruction starts with ``char`` representing the instruction type
followed by the :ref:`qpy_sequence` block representing the instruction
:attr:`~qiskit.pulse.instructions.Instruction.operands`.
Note that the data structure of pulse :class:`~qiskit.pulse.instructions.Instruction`
is unified so that instance can be uniquely determied by the class and a tuple of operands.
The mapping of type char to the instruction subclass is defined as follows:

- ``a``: :class:`~qiskit.pulse.instructions.Acquire` instruction
- ``p``: :class:`~qiskit.pulse.instructions.Play` instruction
- ``d``: :class:`~qiskit.pulse.instructions.Delay` instruction
- ``f``: :class:`~qiskit.pulse.instructions.SetFrequency` instruction
- ``g``: :class:`~qiskit.pulse.instructions.ShiftFrequency` instruction
- ``q``: :class:`~qiskit.pulse.instructions.SetPhase` instruction
- ``r``: :class:`~qiskit.pulse.instructions.ShiftPhase` instruction
- ``b``: :class:`~qiskit.pulse.instructions.RelativeBarrier` instruction
- ``t``: :class:`~qiskit.pulse.instructions.TimeBlockade` instruction
- ``y``: :class:`~qiskit.pulse.instructions.Reference` instruction (new in version 0.7)

.. _qpy_schedule_operands:

SCHEDULE_BLOCK_OPERANDS
-----------------------

The operands of these instances can be serialized through the standard QPY value serialization
mechanism, however there are special object types that only appear in the schedule operands.
Since the operands are serialized as :ref:`qpy_sequence`, each element must be packed with the
INSTRUCTION_PARAM pack struct, where each payload starts with a header block consists of
the char ``type`` and uint64_t ``size``.

Special objects start with the following type key:

- ``c``: :class:`~qiskit.pulse.channels.Channel`
- ``w``: :class:`~qiskit.pulse.library.Waveform`
- ``s``: :class:`~qiskit.pulse.library.SymbolicPulse`
- ``o``: string (operand string, new in version 0.7)
- ``d``: :data:`.CASE_DEFAULT` for switch-statement support (new in version 0.7)
- ``R``: :class:`.ClassicalRegister` or :class:`.Clbit` (new in version 0.7)

Note that ``o`` represents the operand string, which should, in theory, follow same encoding as the
built-in Python string ``s``. However, in the context of pulse instruction operands, this
conflicts with the :class:`~qiskit.pulse.library.SymbolicPulse`, so a different type
character ``o`` has been reserved for this purpose.

``d`` is followed by no data because it represents the literal value :data:`.CASE_DEFAULT`
for switch-statement support,
and ``R`` represents a :class:`.ClassicalRegister` or :class:`.Clbit`,
and is followed by the same format as the description of register or classical bit as used in the
first element of :ref:`the condition of an INSTRUCTION field <qpy_instructions>`.

.. _qpy_schedule_channel:

CHANNEL
-------

Channel block starts with channel subtype ``char`` that maps an object data to
:class:`~qiskit.pulse.channels.Channel` subclass. Mapping is defined as follows:

- ``d``: :class:`~qiskit.pulse.channels.DriveChannel`
- ``c``: :class:`~qiskit.pulse.channels.ControlChannel`
- ``m``: :class:`~qiskit.pulse.channels.MeasureChannel`
- ``a``: :class:`~qiskit.pulse.channels.AcquireChannel`
- ``e``: :class:`~qiskit.pulse.channels.MemorySlot`
- ``r``: :class:`~qiskit.pulse.channels.RegisterSlot`

The key is immediately followed by the channel index serialized as the INSTRUCTION_PARAM.

.. _qpy_schedule_waveform:

WAVEFORM
--------

Waveform block starts with WAVEFORM header:

.. code-block:: c

    struct {
        double epsilon;
        uint32_t data_size;
        _bool amp_limited;
    }

which is followed by ``data_size`` bytes of complex ``ndarray`` binary generated by numpy.save_.
This represents the complex IQ data points played on a quantum device.
:attr:`~qiskit.pulse.library.Waveform.name` is saved after the samples in the
INSTRUCTION_PARAM pack struct, which can be string or ``None``.

.. _numpy.save: https://numpy.org/doc/stable/reference/generated/numpy.save.html

.. _qpy_schedule_symbolic_pulse:

SYMBOLIC_PULSE
--------------

SymbolicPulse block starts with SYMBOLIC_PULSE header:

.. code-block:: c

    struct {
        uint16_t type_size;
        uint16_t envelope_size;
        uint16_t constraints_size;
        uint16_t valid_amp_conditions_size;
        _bool amp_limited;
    }

which is followed by ``type_size`` UTF-8 bytes of :attr:`.SymbolicPulse.pulse_type` string
that represents a class of waveform, such as "Gaussian" or "GaussianSquare".
Then, ``envelope_size``, ``constraints_size``, ``valid_amp_conditions_size`` UTF-8 bytes of
serialized symbolic expressions are generated for :attr:`.SymbolicPulse.envelope`,
:attr:`.SymbolicPulse.constraints`, and :attr:`.SymbolicPulse.valid_amp_conditions`, respectively.
Since string representation of these expressions are usually lengthy,
the expression binary is generated by the python zlib_ module with data compression.

To uniquely specify a pulse instance, we also need to store the associated parameters,
which consist of ``duration`` and the rest of parameters as a dictionary.
Dictionary parameters are first dumped in the :ref:`qpy_mapping` form, and then ``duration``
is dumped with the INSTRUCTION_PARAM pack struct.
Lastly, :attr:`~qiskit.pulse.library.SymbolicPulse.name` is saved also with the
INSTRUCTION_PARAM pack struct, which can be string or ``None``.

.. _zlib: https://docs.python.org/3/library/zlib.html

QPY Version 6 added support for class :class:`.~ScalableSymbolicPulse` through a new
`SymbolicPulse` object with an updated header:

SYMBOLIC_PULSE_V2
-----------------

.. code-block:: c

    # after QPY v.6

    struct {
        uint16_t class_name_size;
        uint16_t type_size;
        uint16_t envelope_size;
        uint16_t constraints_size;
        uint16_t valid_amp_conditions_size;
        _bool amp_limited;
    }

The header is then immediately followed by ``class_name_size`` UTF-8 bytes with the name of the class.
Currently, either `SymbolicPulse` or `ScalableSymbolicPulse` are supported.


.. _qpy_mapping:

MAPPING
-------

The MAPPING is a representation for arbitrary mapping object. This is a fixed length
:ref:`qpy_sequence` of key-value pair represented by the MAP_ITEM payload.

A MAP_ITEM starts with a header defined as:

.. code-block:: c

    struct {
        uint16_t key_size;
        char type;
        uint16_t size;
    }

which is immediately followed by the ``key_size`` UTF-8 bytes representing
the dictionary key in string and ``size`` UTF-8 bytes of arbitrary object data of
QPY serializable ``type``.

.. [#f1] https://tools.ietf.org/html/rfc1700
.. [#f2] https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
.. [#f3] https://docs.python.org/3/c-api/complex.html#c.Py_complex
"""

from .interface import dump, load

# For backward compatibility. Provide, Runtime, Experiment call these private functions.
from .binary_io import (
    _write_instruction,
    _read_instruction,
    _write_parameter_expression,
    _read_parameter_expression,
    _read_parameter_expression_v3,
)
