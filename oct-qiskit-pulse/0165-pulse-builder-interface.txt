# Pulse Builder Interface

| **Status**     | **Proposed**                                                                                                                                     |
|:-------------- |:------------------------------------------------------------------------------------------------------------------------------------------------ |
| **RFC #**      | 165                                                                                                                                              |
| **Authors**    | Thomas Alexander (talexander@ibm.com), Blake Johnson, Frank Haverkamp, Jiri Sdehlik, Soolu Thomas, Will Shanks, Naoki Kanazawa, Lauren Capelluto |
| **Deprecates** | NA                                                                                                                                               |
| **Submitted**  | 2020-03-25                                                                                                                                       |
| **Updated**    | 2020-05-08                                                                                                                                       |

## Summary
This document proposes a framework for easily constructing the pulse programming
intermediate representation (IR), ie., the `pulse.Schedule` interface in a Pythonic way. This will be a layer
existing on top of the current schedule construction API and will provide a
suite of utilities for easy pulse alignment and pseudoinstructions. In the
near-term there will be a strong push to couple real-time compute with quantum
programs as we begin paving the path forward to allow easy changes to the
underlying IR such as augmenting the
instruction set with classical instructions
while maintaining the forward facing user API.

## Motivation
The `pulse.Schedule` was always envisioned as the IR for the purely pulse
component of a quantum program, containing instructions that are scheduled to
execute with guaranteed timing constraints. Consequently it has focused on
providing a pulse program structure, the `pulse.Schedule` and an accompanying
minimal instruction set such as `play`, `set_phase` and
`acquire` that operate on `pulse.Channel`s. Each of these instructions is
designed to execute a single atomic operation that may be easily translated to
native hardware instructions. The current `pulse.Schedule` may be
viewed as a pulse-level basic block with hard real-time deadlines on instruction
execution that allows an entire pulse schedule execution to be pre-calculated.

By providing instructions to make tasks such as instruction alignment across
channels, measurements, defining amalgamated operations on qubits and
blending circuit-level program descriptions with pulse-level programs we
enable a rich set of scheduling routines and experimental programs to be
constructed on top of the Qiskit Pulse IR. This allows the pulse
programmer to program at a higher abstraction level than raw pulse
instructions.

Furthermore, there is a desire to decouple the IR from the user-level program
description. This will allow both to evolve in parallel, opening up the door
to exposing a combination of classical and pulse real-time compute in Qiskit
without breaking user program interfaces.

Within this document we propose an embedded DSL for pulse programming within
Python that will dynamically construct the pulse IR behind the while providing
an imperative assembly like user interface.

## User Benefit
The primary benefactors will be physicists who will be provided with a more
consistent and powerful programming interface, which will allow more power
routines to be constructed with this interface as the foundation.

## Design Proposal
The pulse builder interface will make use of contexts, context variables and
the preexisting `pulse.Schedule`.

We introduce a new private class, the `_PulseBuilder` which will be responsible
for maintaining an internal `pulse.Schedule` and exposing builder methods for
constructing schedules (which may be thought of as a parallel,
pulse coprocessor basic-block operating on channels). The `_PulseBuilder` will
be used to expose a suite of context-enabled builder functions, allowing users
to construct both simple and straightforward pulse programs that mix Python and
pulse programming.

When considering this design we make the following assumptions:
1. We must support a variety of backend qubit technologies and
  control electronics platforms that are evolving over time
1. We wish to provide an easy to use programming interface in Python as Qiskit's
  user-base is primarily based in Python. It is preferable that this is as close
  to native Python as possible.
1. This transition must occur gradually overtime so as to continue to rollout
  features to new users while still providing a relatively stable user
  interface.
1. We must support device-dependent transformations, optimizations and analysis
  passes at the Python level to allow easy experimentation.
## Pulse Builder Interface
The pulse builder interface is intended to expose a pulse API with semantics
that are as close to native Python as possible. The aim is to keep future
extensibility open. Behind the scenes a `_PulseBuilder` will construct pulse
programs, with possibly backend dependent functionality.

### Phase 1 - Context based Pulse DSL builder interface
Below we demonstrate a high level example of how pulse programming will work in
the first phase DSL. We utilize a context to output and construct a pulse
schedule

```python
import math

from qiskit import QuantumCircuit
import qiskit.pulse as pulse
import qiskit.pulse.pulse_lib as pulse_lib
from qiskit.test.mock import FakeOpenPulse2Q

backend = FakeOpenPulse2Q()

with pulse.build(backend) as pulse_prog:
    # Create a pulse.
    gaussian_pulse = pulse_lib.gaussian(10, 1.0, 2)
    # Get the qubit's corresponding drive channel from the backend.
    d0 = pulse.drive_channel(0)
    d1 = pulse.drive_channel(1)
    # Play a pulse at t=0.
    pulse.play(gaussian_pulse, d0)
    # Play another pulse directly after the previous pulse at t=10.
    pulse.play(gaussian_pulse, d0)
    # The default scheduling behavior is to schedule pulses in parallel
    # across channels. For example, the statement below
    # plays the same pulse on a different channel at t=0.
    pulse.play(gaussian_pulse, d1)

    # We also provide pulse scheduling alignment contexts.
    # The default alignment context is align_left.

    # The sequential context schedules pulse instructions sequentially in time.
    # This context starts at t=10 due to earlier pulses above.
    with pulse.align_sequential():
        pulse.play(gaussian_pulse, d0)
        # Play another pulse after at t=20.
        pulse.play(gaussian_pulse, d1)

        # We can also nest contexts as each instruction is
        # contained in its local scheduling context.
        # The output of a child context is a scheduled block
        # with the internal instructions timing fixed relative to
        # one another. This is block is then called in the parent context.

        # Context starts at t=30.
        with pulse.align_left():
            # Start at t=30.
            pulse.play(gaussian_pulse, d0)
            # Start at t=30.
            pulse.play(gaussian_pulse, d1)
        # Context ends at t=40.

        # Alignment context where all pulse instructions are
        # aligned to the right, ie., as late as possible.
        with pulse.align_right():
            # Shift the phase of a pulse channel.
            pulse.shift_phase(math.pi, d1)
            # Starts at t=40.
            pulse.delay(100, d0)
            # Ends at t=140.

            # Starts at t=130.
            pulse.play(gaussian_pulse, d1)
            # Ends at t=140.

        # Acquire data for a qubit and store in a memory slot.
        pulse.acquire(100, 0, pulse.MemorySlot(0))

        # We also support a variety of macros for common operations.

        # Measure all qubits.
        pulse.measure_all()

        # Delay on some qubits.
        # This requires knowledge of which channels belong to which qubits.
        pulse.delay_qubits(100, 0, 1)

        # Call a quantum circuit. The pulse builder lazily constructs a quantum
        # circuit which is then transpiled and scheduled before inserting into
        # a pulse schedule.
        # NOTE: Quantum register indices correspond to physical qubit indices.
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        pulse.call(qc)
        # Calling a small set of standard gates and decomposing to pulses is
        # also supported with more natural syntax.
        pulse.u3(0, math.pi, 0, 0)
        pulse.cx(0, 1)


        # It is also be possible to call a preexisting schedule
        tmp_sched = pulse.Schedule()
        tmp_sched += pulse.Play(gaussian_pulse, d0)
        pulse.call(tmp_sched)

        # We also support:

        # frequency instructions
        pulse.set_frequency(5.0e9, d0)

        # phase instructions
        pulse.shift_phase(0.1, d0)

        # offset contexts
        with pulse.phase_offset(math.pi, d0):
            pulse.play(gaussian_pulse, d0)


# Decorator syntax for defining a pulse function.
@pulse.function(backend=backend)
def delay_function(duration) -> List[pulse.MemorySlot]:
  """Define a function which wraps the function in an active builder
  context and replaces the function with a pulse.Function object which
  has a schedule attribute.
  """
  pulse.delay(pulse.DriveChannel(0), duration)
  pulse.delay(pulse.DriveChannel(1), duration)
  return pulse.measure_all()

with pulse.builder(backend):
    registers = delay_function(dc0, 10)


# making use of pulse.function to define a calibration for qubits on a backend
# name inherited from signature but is also settable as an argument
@backend.cal(0, 1)
def cx():
  pulse.play(pulse.Constant(100, 1.), pulse.ControlChannel(0))

# or alternatively
@pulse.cal(backend, (0, 1), gate=circuit.library.CXGate)
def inherited_cx():
  cx()

# use calibrated gate in circuit
qc = QuantumCircuit(2)
qc.cx(0, 1)
sched = schedule(qc, backend)
```

Note that it is expected that the ability to call gates directly from the pulse
interface will be deprecated with the advent of a circuit builder interface.
This will be described in a following design document. However, pulses will be
used to defined gate calibrations as described below.

Calibrations may be dynamically tagged with a `cal` decorator as shown above.
This enables a form of function overloading where a game of a given name can be
defined for separate sets of qubits. This enables using pulses to build up
calibrated gates and incrementally combine them to build up a sophisticated
library of calibrated routines.

### Phase 2 - PulseProgram
Phase 2 will implement a `PulseFunctionIR`, with the aim of building a
fully-fledged coprocessor IR. The current `Schedule` will essentially be a
pulse basic block within the `PulseFunctionIR`. The initial `PulseFunctionIR` will
aim to support the full suite of instructions of the current `Schedule`, plus
additional scheduling directives supported natively within the IR such as:
- `Barrier`
- `LeftAlign`
- `RightAlign`

We would also support a rudimentary `compiler` which would operate with
a `PulsePassManager`, that would be used to write scheduling passes to compile
from a `PulseFunctionIR` with alignment directives to an optimized and
scheduled `PulseFunctionIR` that is decomposed into device supported
instructions. This would be roughly the equivalent of the current `Schedule`,
an absolutely timed series of instructions, scheduled with delays.

From the perspective of users of the `builder` context interface, nothing
should change in phase 2. Having a stable user-facing interface is important
and simultaneously allows us to deprecate the `Schedule` interface,
hiding it from all but expert users and developers.

```python
# define IR for module
pulse_ir = PulseFunctionIR()
# define channels and pulses
dc0 = DriveChannel(0)
dc1 = DriveChannel(1)
dc2 = DriveChannel(2)
pulse1 = SamplePulse([0.])
pulse2 = SamplePulse([0., 1.])

# define an IR block, ie., the equivalent of a schedule
block = Block()
# Add an align right statement, aligning all future pulses in block to the right
block.append(AlignRight())
# add pulse t=0
block.append(Play(dc0, pulse1))
# add pulse, ends at t=2, force pulses above to start at t=1 as well
block.append(Play(dc1, pulse2))
# no we barrier these two channels
block.append(Barrier([dc0, dc1, dc2]))
# due to barrier pulse starts at t=2
block.append(Play(dc2, pulse1))

# add block to IR
# this will be extended for control flow and
# multiple functions in the future
pulse_ir.append(block)

# optimization and scheduling pass
# consuming alignments and barriers
# resulting in a schedule with only instructions
# and delays
scheduled_ir = compile(pulse_ir, passmanager=custom_pass_manager)

# these passes could be controlled by the provider to make sure it can execute
# the output ir
```

## Detailed Design

### Phase 1 implementation details
The above API may be implemented with Python [contextvars](https://docs.python.org/3/library/contextvars.html). Note that as contextvars are Python 3.7 for Python 3.6 compatibility we may use the [contextvars backport](https://github.com/MagicStack/contextvars) which is licensed under Apache2.0. This must be conditionally installed.


#### Builder context methods
```python

# Builder Contexts #############################################################

class _PulseBuilder(AbstractContextManager)
  def __init__(self,
               backend: Optional[BaseBackend] = None,
               sched: Optional[Schedule] = None,
               ):
    ...

  def __enter__(self):
    """
    Yields:
      Schedule
    """
    ...

  def __exit__(self):
    ...

  def compile(self):
    ...

# decorator for entering building context
# will use contextvars
def build(backend: Optional[BaseBackend] = None,
          sched = None: Schedule
          ) -> PulseBuilderContext:
  return PulseBuilderContext(sched, backend=backend)

# Alignment Contexts ###########################################################
@contextmanager
def align_left():
  """Align instructions in this context to the left in time."""
  ...

@contextmanager
def align_right():
  """Align instructions in this context to the right in time."""
  ...

@contextmanager
def barrier(*channels):
  """Barrier these channels on either side of the instructions within thins context."""
  ...

@contextmanager
def align_sequential():
  """Align the pulse instructions within this context sequentially across channels."""
  ...

@contextmanager
def group():
  """Group the pulse instructions within this context together and fix their relative timing."""
  ...

# Context Instructions #########################################################
def play(pulse: Pulse, channel: PulseChannel):
  ...

def acquire(duration: int,
            qubit: Union[int, AcquireChannel],
            register: MemorySlot):
  ...

def delay(duration: int,
          qubit: Union[Channel]):
  ...

def shift_phase(phase: float, channel: PulseChannel):
  ...

def set_phase(phase: float, channel: PulseChannel):
  ...

def shift_frequency(frequency: float, channel: PulseChannel):
  ...

def set_frequency(frequency: float, channel: PulseChannel):
  ...


def call(argument: Union[QuantumCircuit, Schedule]):
  """Will call out to `scheduler` using backend to output a given circuit
  instruction as a pulse schedule.

  The scheduling method must be determined by the alignment type or provided to
  call as kwargs.

  Question: Should this be a separate method for a schedule like ```inline``?
  """
  ...

# Context PseudoInstructions ###################################################
def measure(qubit: int, register: MemorySlot) -> pulse.MemorySlot:
  ...

def delay_qubit(duration: int, qubit: int):
  ...

def id(qubit: int):
  ...

def u1(qubit: int, phase: int):
  ...

def u2(qubit: int, theta: float, phi: float):
  ...

def u3(qubit: int, theta, phi, lambd: float):
  ...

def cx(control: int, target: int):
  ...

def x(qubit: int):
  ...

# PseudoInstruction Contexts ###################################################
@contextmanager
def phase_offset(channel: PulseChannel, phase: float):
  ...

@contextmanager
def frequency_offset(channel: PulseChannel, frequency: float):
  ...
```

### Phase 2 implementation details
While undertaking the implementation of the internal IR it must be kept in mind
that the eventual goal may be to wrap a subset of native Python as will be
defined in a later circuit builder RFC. As a consequence the IR description
implemented in phase 2 must be left sufficiently extensible to enable the
description of semantics such as a Python context. This
IR is not expected to be extensible optimizable as this is expected to be
occupied by the role of the target language in Phase 4 such as a LLVM language
extension for pulse programming.

*Note*: This section will be extended in a future IR focused RFC.

## Alternative Approaches

### Optional Approach to Phase 1 - Building the IR manually
This is the approach that currently exists within Qiskit where the user builds
up a `Schedule` from its `Instruction` components. This quickly runs into issues
managing complexity as in a more sophisticated IR there are often many options
and components such as modules, functions, globals, basic blocks, transformation
and validations. Users attempting to program within such an interface would be
quickly overwhelmed. Therefore we believe that this approach to pulse
programming is unsustainable and instead IR should be managed by the context
builder interface as proposed above.

### Optional Approach to Phase 2 - PulseProgramBuilder
Phase 2 will implement an underlying `_PulseBuilder` object. This would be
very similar to the current QuantumCircuit interface for building, eg.,
```python
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
```

The optional approach to Phase 2 would be to rewrite the internals of the
Phase 1 API such that it is simply syntactic sugar for calls to the underlying
`_PulseBuilder`.

The downside to this approach is that it is layering a builder interface on
a builder interface. As the `_PulseBuilder` would be entirely hidden,
it is likely an unnecessary piece of code to maintain and the context builder
can play this role. However, if an object contextmanager is used to implement
the programming DSL this may be obtained for free.

For example the phase-1 demo above could be equivalently constructed with:

```python

import math

from qiskit.pulse import (PulseProgramBuilder, Schedule, pulse_lib,
                          DriveChannel)

sched = Schedule()
builder = PulseBuilder(sched)

# create a pulse
gaussian_pulse = pulse_lib.gaussian(10, 1.0, 2)
# create a channel type
d0 = DriveChannel(0)
d1 = DriveChannel(1)
# play a pulse at time=0
builder.play(gaussian_pulse, d0)
# play another pulse directly after at t=10
builder.play(gaussian_pulse, d0)
# The default scheduling behavior is to schedule pulse in parallel
# across independent resources, for example
# play the same pulse on a different channel at t=0
builder.play(gaussian_pulse, d1)

# Can create new scheduling block
squential_block = builder.sequential()
builder.play(gaussian_pulse, d0)
  # play another pulse after at t=20
builder.play(gaussian_pulse, d1)

# enter nested scheduling block
# sets builder to start of nested parallel context
parallel_block = builder.parallel()
# start at t=20
builder.play(gaussian_pulse, d0)
# start at t=20
builder.play(gaussian_pulse, d1)
# resumes mainline execution
# could have been passed block if necessary
builder.resume()
# ends at t=30

# all pulse instructions occur as late as possible
right_block = builder.right()
builder.set_phase(math.pi, d1)
# starts at t=30
builder.delay(100, d0)
builder.resume()
# ends at t=130

# starts at t=120
builder.play(gaussian_pulse, d1)
  # ends at t=130

# acquire a qubit
builder.acquire(100, 0, ClassicalRegister(0))

builder.measure([i for i in range(n_qubits], [ClassicalRegister(i) for i in range(n_qubits)])

# delay on a qubit
# this requires knowledge of which channels belong to which qubits
builder.delay(100, 0)

# insert a quantum circuit, assumes quantum registers correspond to physical
# qubit indices
qc = QuantumCircuit(2, 2)
qc.cx(0, 1)
builder.call(qc)

# It is also be possible to call a preexisting
# schedule constructed with another
# NOTE: once internals are fleshed out, Schedule may not be the default class
tmp_sched = Schedule()
# order of play IR should be changed internally after builder interface goes live
tmpsched += Play(gaussian_pulse, dc0)
builder.append(tmp_sched)
```

## Questions
- What timeline can we expect to implement these development phases?
  - Phases 1 and 2 should each take approximately 1.5 months of focused
  development
- What is the difference between a `PulseProgram` and a `Schedule`?
  - A schedule will just be an instruction node in a quantum circuit. It defines
  a microcode implementation of its quantum circuit instruction and should be
  agnostic of what time it is scheduled. This will have implications for how
  circuit instructions can be scheduled on hardware and directives/metadata will
  likely be required to convey this info.
- Should the core compilation program be written in Python?
  - The heavy lifting for pulse target code generation will be offloaded to code
  that is not in Qiskit. This is unlikely to be Python and we are looking at
  LLVM currently.
- What does the output V-ISA look like?
  - Still being determined as part of OpenQASM3
- How should we implement the V-ISA, compiler?
  - Some parts in Qiskit, maybe output to a lower-level V-ISA like an LLVM based
  extension.

## Future Extensions
See the proposed phase 2 which involves extending the QuantumCircuit and
Schedule IR. A future RFC will follow with details on creating a circuit builder
and integrating it with the pulse builder.
