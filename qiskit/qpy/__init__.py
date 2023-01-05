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
Qiskit code). This differs from other serialization formats like
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

The QPY serialization format is a portable cross-platform binary
serialization format for :class:`~qiskit.circuit.QuantumCircuit` objects in Qiskit. The basic
file format is as follows:

A QPY file (or memory object) always starts with the following 7
byte UTF8 string: ``QISKIT`` which is immediately followed by the overall
file header. The contents of the file header as defined as a C struct are:

.. code-block:: c

    struct {
        uint8_t qpy_version;
        uint8_t qiskit_major_version;
        uint8_t qiskit_minor_version;
        uint8_t qiskit_patch_version;
        uint64_t num_circuits;
    }

All values use network byte order [#f1]_ (big endian) for cross platform
compatibility.

The file header is immediately followed by the circuit payloads.
Each individual circuit is composed of the following parts:

``HEADER | METADATA | REGISTERS | CUSTOM_DEFINITIONS | INSTRUCTIONS``

There is a circuit payload for each circuit (where the total number is dictated
by ``num_circuits`` in the file header). There is no padding between the
circuits in the data.

.. _qpy_version_5:

Version 5
=========

Version 5 changes from :ref:`qpy_version_4` by adding support for :class:`.~ScheduleBlock`
and changing two payloads the INSTRUCTION metadata payload and the CUSTOM_INSTRUCTION block.
These now have new fields to better account for :class:`~.ControlledGate` objects in a circuit.
In addition, new payload MAP_ITEM is defined to implement the :ref:`qpy_mapping` block.

With the support of :class:`.~ScheduleBlock`, now :class:`~.QuantumCircuit` can be
serialized together with :attr:`~.QuantumCircuit.calibrations`, or
`Pulse Gates <https://qiskit.org/documentation/tutorials/circuits_advanced/05_pulse_gates.html>`_.
In QPY version 5 and above, :ref:`qpy_circuit_calibrations` payload is
packed after the :ref:`qpy_instructions` block.

In QPY version 5 and above,

.. code-block:: c

    struct {
        char type;
    }

immediately follows the file header block to represent the program type stored in the file.

- When ``type==c``, :class:`~.QuantumCircuit` payload follows
- When ``type==s``, :class:`~.ScheduleBlock` payload follows

.. note::

    Different programs cannot be packed together in the same file.
    You must create different files for different program types.
    Multiple objects with the same type can be saved in a single file.

.. _qpy_schedule_block:

SCHEDULE_BLOCK
--------------

:class:`~.ScheduleBlock` is first supported in QPY Version 5. This allows
users to save pulse programs in the QPY binary format as follows:

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

SCHEDULE_BLOCK_HEADER
---------------------

:class:`~.ScheduleBlock` block starts with the following header:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint64_t metadata_size;
        uint16_t num_element;
    }

which is immediately followed by ``name_size`` utf8 bytes of schedule name and
``metadata_size`` utf8 bytes of the JSON serialized metadata dictionary
attached to the schedule.

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

The operands of these instances can be serialized through the standard QPY value serialization
mechanism, however there are special object types that only appear in the schedule operands.
Since the operands are serialized as :ref:`qpy_sequence`, each element must be packed with the
INSTRUCTION_PARAM pack struct, where each payload starts with a header block consists of
the char ``type`` and uint64_t ``size``.
Special objects start with the following type key:

- ``c``: :class:`~qiskit.pulse.channels.Channel`
- ``w``: :class:`~qiskit.pulse.library.Waveform`
- ``s``: :class:`~qiskit.pulse.library.SymbolicPulse`

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

Waveform
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

SymbolicPulse
-------------

SymbolicPulse block starts with SYMBOLIC_PULSE header:

.. code-block:: c

    struct {
        uint16_t type_size;
        uint16_t envelope_size;
        uint16_t constraints_size;
        uint16_t valid_amp_conditions_size;
        _bool amp_limited;
    }

which is followed by ``type_size`` utf8 bytes of :attr:`.SymbolicPulse.pulse_type` string
that represents a class of waveform, such as "Gaussian" or "GaussianSquare".
Then, ``envelope_size``, ``constraints_size``, ``valid_amp_conditions_size`` utf8 bytes of
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

which is immediately followed by the ``key_size`` utf8 bytes representing
the dictionary key in string and ``size`` utf8 bytes of arbitrary object data of
QPY serializable ``type``.

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

The calibration definition header is then followed by ``name_size`` utf8 bytes of
the gate name, ``num_qubits`` length of integers representing a sequence of qubits,
and ``num_params`` length of INSTRUCTION_PARAM payload for parameters
associated to the custom instruction.
The ``type`` indicates the class of pulse program which is either, in pricinple,
:class:`~.ScheduleBlock` or :class:`~.Schedule`. As of QPY Version 5,
only :class:`~.ScheduleBlock` payload is supported.
Finally, :ref:`qpy_schedule_block` payload is packed for each CALIBRATION_DEF entry.

INSTRUCTION
-----------

The INSTRUCTION block was modified to add two new fields ``num_ctrl_qubits`` and ``ctrl_state``
which are used to model the :attr:`.ControlledGate.num_ctrl_qubits` and
:attr:`.ControlledGate.ctrl_state` attributes. The new payload packed struct
format is:

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
        uint32_t num_ctrl_qubits;
        uint32_t ctrl_state;
    }

The rest of the instruction payload is the same. You can refer to
:ref:`qpy_instructions` for the details of the full payload.

CUSTOM_INSTRUCTION
------------------

The CUSTOM_INSTRUCTION block in QPY version 5 adds a new field
``base_gate_size`` which is used to define the size of the
:class:`qiskit.circuit.Instruction` object stored in the
:attr:`.ControlledGate.base_gate` attribute for a custom
:class:`~.ControlledGate` object. With this change the CUSTOM_INSTRUCTION
metadata block becomes:

.. code-block:: c

    struct {
        uint16_t name_size;
        char type;
        uint32_t num_qubits;
        uint32_t num_clbits;
        _Bool custom_definition;
        uint64_t size;
        uint32_t num_ctrl_qubits;
        uint32_t ctrl_state;
        uint64_t base_gate_size
    }

Immediately following the CUSTOM_INSTRUCTION struct is the utf8 encoded name
of size ``name_size``.

If ``custom_definition`` is ``True`` that means that the immediately following
``size`` bytes contains a QPY circuit data which can be used for the custom
definition of that gate. If ``custom_definition`` is ``False`` then the
instruction can be considered opaque (ie no definition). The ``type`` field
determines what type of object will get created with the custom definition.
If it's ``'g'`` it will be a :class:`~qiskit.circuit.Gate` object, ``'i'``
it will be a :class:`~qiskit.circuit.Instruction` object.

Following this the next ``base_gate_size`` bytes contain the ``INSTRUCTION``
payload for the :attr:`.ControlledGate.base_gate`.

Additionally an addition value for ``type`` is added ``'c'`` which is used to
indicate the custom instruction is a custom :class:`~.ControlledGate`.

.. _qpy_version_4:

Version 4
=========

Version 4 is identical to :ref:`qpy_version_3` except that it adds 2 new type strings
to the INSTRUCTION_PARAM struct, ``z`` to represent ``None`` (which is encoded as
no data), ``q`` to represent a :class:`.QuantumCircuit` (which is encoded as
a QPY circuit), ``r`` to represent a ``range`` of integers (which is encoded as
a :ref:`qpy_range_pack`), and ``t`` to represent a ``sequence`` (which is encoded as
defined by :ref:`qpy_sequence`). Additionally, version 4 changes the type of register
index mapping array from ``uint32_t`` to ``int64_t``. If the values of any of the
array elements are negative they represent a register bit that is not present in the
circuit.

The :ref:`qpy_registers` header format has also been updated to

.. code-block:: c

    struct {
        char type;
        _Bool standalone;
        uint32_t size;
        uint16_t name_size;
        _bool in_circuit;
    }

which just adds the ``in_circuit`` field which represents whether the register is
part of the circuit or not.

.. _qpy_range_pack:

RANGE
-----

A RANGE is a representation of a ``range`` object. It is defined as:

.. code-block:: c

    struct {
        int64_t start;
        int64_t stop;
        int64_t step;
    }

.. _qpy_sequence:

SEQUENCE
--------

A SEQUENCE is a reprentation of a arbitrary sequence object. As sequence are just fixed length
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

.. _qpy_version_3:

Version 3
=========

Version 3 of the QPY format is identical to :ref:`qpy_version_2` except that it defines
a struct format to represent a :class:`~qiskit.circuit.library.PauliEvolutionGate`
natively in QPY. To accomplish this the :ref:`qpy_custom_definition` struct now supports
a new type value ``'p'`` to represent a :class:`~qiskit.circuit.library.PauliEvolutionGate`.
Enties in the custom instructions tables have unique name generated that start with the
string ``"###PauliEvolutionGate_"`` followed by a uuid string. This gate name is reservered
in QPY and if you have a custom :class:`~qiskit.circuit.Instruction` object with a definition
set and that name prefix it will error. If it's of type ``'p'`` the data payload is defined
as follows:

.. _pauli_evo_qpy:

PAULI_EVOLUTION
---------------

This represents the high level :class:`~qiskit.circuit.library.PauliEvolutionGate`

.. code-block:: c

    struct {
        uint64_t operator_count;
        _Bool standalone_op;
        char time_type;
        uint64_t time_size;
        uint64_t synthesis_size;
    }

This is immediately followed by ``operator_count`` elements defined by the :ref:`qpy_pauli_sum_op`
payload.  Following that we have ``time_size`` bytes representing the ``time`` attribute. If
``standalone_op`` is ``True`` then there must only be a single operator. The
encoding of these bytes is determined by the value of ``time_type``. Possible values of
``time_type`` are ``'f'``, ``'p'``, and ``'e'``. If ``time_type`` is ``'f'`` it's a double,
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is represented by a
:ref:`qpy_param_struct`, ``e`` defines a :class:`~qiskit.circuit.ParameterExpression` object
(that's not a :class:`~qiskit.circuit.Parameter`) which is represented by a :ref:`qpy_param_expr`.
Following that is ``synthesis_size`` bytes which is a utf8 encoded json payload representing
the :class:`.EvolutionSynthesis` class used by the gate.

.. _qpy_pauli_sum_op:

SPARSE_PAULI_OP_LIST_ELEM
-------------------------

This represents an instance of :class:`.PauliSumOp`.


.. code-block:: c

    struct {
        uint32_t pauli_op_size;
    }

which is immediately followed by ``pauli_op_size`` bytes which are .npy format [#f2]_
data which represents the :class:`~qiskit.quantum_info.SparsePauliOp`.

Version 3 of the QPY format also defines a struct format to represent a
:class:`~qiskit.circuit.ParameterVectorElement` as a distinct subclass from
a :class:`~qiskit.circuit.Parameter`. This adds a new parameter type char ``'v'``
to represent a :class:`~qiskit.circuit.ParameterVectorElement` which is now
supported as a type string value for an INSTRUCTION_PARAM. The payload for these
parameters are defined below as :ref:`qpy_param_vector`.

.. _qpy_param_vector:

PARAMETER_VECTOR_ELEMENT
------------------------

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

which is immediately followed by ``vector_name_size`` utf8 bytes representing
the parameter's vector name.

.. _qpy_param_expr_v3:


PARAMETER_EXPR
--------------

Additionally, since QPY format version v3 distinguishes between a
:class:`~qiskit.circuit.Parameter` and :class:`~qiskit.circuit.ParameterVectorElement`
the payload for a :class:`~qiskit.circuit.ParameterExpression` needs to be updated
to distinguish between the types. The following is the modified payload format
which is mostly identical to the format in Version 1 and :ref:`qpy_version_2` but just
modifies the ``map_elements`` struct to include a symbol type field.

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The contents of a PARAMETER_EXPR
are defined as:

.. code-block:: c

    struct {
        uint64_t map_elements;
        uint64_t expr_size;
    }

Immediately following the header is ``expr_size`` bytes of utf8 data containing
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
object (both the struct and utf8 name bytes) and if ``symbol_type`` is ``v``
then the struct is imediately followed by :ref:`qpy_param_vector` (both the struct
and utf8 name bytes). That is followed by ``size`` bytes for the
data of the symbol. The data format is dependent on the value of ``type``. If
``type`` is ``p`` then it represents a :class:`~qiskit.circuit.Parameter` and
size will be 0, the value will just be the same as the key. Similarly if the
``type`` is ``v`` then it represents a :class:`~qiskit.circuit.ParameterVectorElement`
and size will be 0 as the value will just be the same as the key. If
``type`` is ``f`` then it represents a double precision float. If ``type`` is
``c`` it represents a double precision complex, which is represented by the
:ref:`qpy_complex`. Finally, if type is ``i`` it represents an integer which is an
``int64_t``.

.. _qpy_version_2:

Version 2
=========

Version 2 of the QPY format is identical to version 1 except for the HEADER
section is slightly different. You can refer to the :ref:`qpy_version_1` section
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

.. _qpy_version_1:

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
you then read the ``metadata_size`` number of bytes and parse the JSON to get
the metadata for the circuit.

.. _qpy_registers:

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
int64_t values of size ``size`` that contains a map of the register's index to
the circuit's qubit index. For example, array element 0's value is the index
of the ``register[0]``'s position in the containing circuit's qubits list.

.. note::

    Prior to QPY :ref:`qpy_version_4` the type of array elements was uint32_t. This was changed
    to enable negative values which represent bits in the array not present in the
    circuit


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


.. _qpy_custom_definition:

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
        uint32_t num_qubits;
        uint32_t num_clbits;
        _Bool custom_definition;
        uint64_t size;
    }

Immediately following the CUSTOM_INSTRUCTION struct is the utf8 encoded name
of size ``name_size``.

If ``custom_definition`` is ``True`` that means that the immediately following
``size`` bytes contains a QPY circuit data which can be used for the custom
definition of that gate. If ``custom_definition`` is ``False`` then the
instruction can be considered opaque (ie no definition). The ``type`` field
determines what type of object will get created with the custom definition.
If it's ``'g'`` it will be a :class:`~qiskit.circuit.Gate` object, ``'i'``
it will be a :class:`~qiskit.circuit.Instruction` object.

.. _qpy_instructions:

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
the data is represented by the struct format in the :ref:`qpy_param_expr` section.
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is
represented by a :ref:`qpy_param_struct` struct, ``e`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (that's not a
:class:`~qiskit.circuit.Parameter`) which is represented by a :ref:`qpy_param_expr`
struct (on QPY format :ref:`qpy_version_3` the format is tweak slightly see:
:ref:`qpy_param_expr_v3`), ``'n'`` represents an object from numpy (either an
``ndarray`` or a numpy type) which means the data is .npy format [#f2]_ data,
and in QPY :ref:`qpy_version_3` ``'v'`` represents a
:class:`~qiskit.circuit.ParameterVectorElement` which is represented by a
:ref:`qpy_param_vector` struct.

.. _qpy_param_struct:

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

.. _qpy_param_expr:

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
``c`` it represents a double precision complex, which is represented by :ref:`qpy_complex`.
Finally, if type is ``i`` it represents an integer which is an ``int64_t``.

.. _qpy_complex:

COMPLEX
-------

When representing a double precision complex value in QPY the following
struct is used:


.. code-block:: c

    struct {
        double real;
        double imag;
    }

this matches the internal C representation of Python's complex type. [#f3]_


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
