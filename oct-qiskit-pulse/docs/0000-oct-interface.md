[![](https://mermaid.ink/img/eyJjb2RlIjoiY2xhc3NEaWFncmFtXG4gICAgICBBbmltYWwgPHwtLSBEdWNrXG4gICAgICBBbmltYWwgPHwtLSBGaXNoXG4gICAgICBBbmltYWwgPHwtLSBaZWJyYVxuICAgICAgQW5pbWFsIDogK2ludCBhZ2VcbiAgICAgIEFuaW1hbCA6ICtTdHJpbmcgZ2VuZGVyXG4gICAgICBBbmltYWw6ICtpc01hbW1hbCgpXG4gICAgICBBbmltYWw6ICttYXRlKClcbiAgICAgIGNsYXNzIER1Y2t7XG4gICAgICAgICAgK1N0cmluZyBiZWFrQ29sb3JcbiAgICAgICAgICArc3dpbSgpXG4gICAgICAgICAgK3F1YWNrKClcbiAgICAgIH1cbiAgICAgIGNsYXNzIEZpc2h7XG4gICAgICAgICAgLWludCBzaXplSW5GZWV0XG4gICAgICAgICAgLWNhbkVhdCgpXG4gICAgICB9XG4gICAgICBjbGFzcyBaZWJyYXtcbiAgICAgICAgICArYm9vbCBpc193aWxkXG4gICAgICAgICAgK3J1bigpXG4gICAgICB9XG4iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiY2xhc3NEaWFncmFtXG4gICAgICBBbmltYWwgPHwtLSBEdWNrXG4gICAgICBBbmltYWwgPHwtLSBGaXNoXG4gICAgICBBbmltYWwgPHwtLSBaZWJyYVxuICAgICAgQW5pbWFsIDogK2ludCBhZ2VcbiAgICAgIEFuaW1hbCA6ICtTdHJpbmcgZ2VuZGVyXG4gICAgICBBbmltYWw6ICtpc01hbW1hbCgpXG4gICAgICBBbmltYWw6ICttYXRlKClcbiAgICAgIGNsYXNzIER1Y2t7XG4gICAgICAgICAgK1N0cmluZyBiZWFrQ29sb3JcbiAgICAgICAgICArc3dpbSgpXG4gICAgICAgICAgK3F1YWNrKClcbiAgICAgIH1cbiAgICAgIGNsYXNzIEZpc2h7XG4gICAgICAgICAgLWludCBzaXplSW5GZWV0XG4gICAgICAgICAgLWNhbkVhdCgpXG4gICAgICB9XG4gICAgICBjbGFzcyBaZWJyYXtcbiAgICAgICAgICArYm9vbCBpc193aWxkXG4gICAgICAgICAgK3J1bigpXG4gICAgICB9XG4iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)
# Quantum control instruction map interface

| **Status**        | **Proposed/Accepted/Deprecated** |
|:------------------|:---------------------------------------------|
| **RFC #**         | 0000                                         |
| **Authors**       | Ben Rosand (benjamin.rosand@ibm.com)         |
| **Deprecates**    | RFC that this RFC deprecates                 |
| **Submitted**     | YYYY-MM-DD                                   |
| **Updated**       | 2020-07-23                                   |

## Summary
This document proposes a framework for automatically building gates using
quantum optimal control. <!--The pipeline will sit on top of the
`InstructionScheduleMap`and create `QOCInstructionScheduleMap` which modifies
the `InstructionScheduleMap.get` function to design gates on the fly.
- qoc faster
- need for testing qoc
- starting steps to be useful in real time -->

Quantum optimal control (QOC) is the process of achieving a desired unitary
evolution of a system using a number of control fields. In this case we target a
transmon (the quantum system) and through microwave pulses (the control fields),
achieve a unitary evolution. Currently pulses are designed manually for each
target gate, and stored in a library for each device and calibrated regularly.
We propose an optimal control framework which will allow gates to be built on
the fly for an arbitrary unitary to the specifications of the given system.
Specifically, this framework will allow for various optimizers from various
sources, such as QuTiP's GRAPE(gradient ascent pulse engineering)
implementation, to be used to automatically generate pulses from gates. This
pipeline overloads one function in `InstructionScheduleMap`, thus allowing for
the qoc optimizers to be used in place of the default pulse library with two
lines of additional code to create the new instruction map.



## Motivation
<!-- - Why are we doing this? --> <!-- - enable qoc automatically --> We want to
  allow users to run arbitrary qoc optimizers on ibm systems, from various
  sources, without forcing users to engage with hardware specs or optimizer
  specs. <!-- - design custom gates on the fly --> This QOC framework will allow
  for arbitrary unitary gates to be performed on any system using OpenPulse, without transpiling
  these gates to basis gates. In addition, the framework will allow for a
  speedup and potential increase in gate fidelity across circuits, which we
  expect to increase in porportion with the size of the systems and the
  circuits. This work will be especially helpful for gate aggregation, as it
  will allow the automatic calibration of any aggregated gates on the fly, which
  will compound increases in fidelity and speed[1]. We expect this work to be
  impactful for everyone, as researchers will be able to test QOC methods, and
  industry customers will enjoy the gate speedups as well as the automatic
  compilation of gates for gate aggregations. <!-- - potential for large
  increase in accuracy and speed of larger circuits --> <!-- - What will this
  enable? --> <!-- - qoc designed gates (especially with blocking) --> <!-- -
  What will be the outcome? --> <!-- - gates built on the fly --> <!-- -
  aggregated unitary gates which can be automatically built --> <!-- - Who will
  benefit? --> <!-- - in the near term just researchers --> <!-- - researchers
  who want to test qoc --> <!-- - coders who want to run larger programs faster
  and more accurately -->

## User Benefit
  Researchers will enjoy an easy pipeline for testing QOC implementations, as
  well as a pipeline that enables easier compilation of arbitrary unitary gates.
  We expect industry users to benefit from an increase in gate fidelity and
  speed. This will increase the capabilities of the IBM hardware and allow for
  industry users to run more complex circuits with greater fidelity. Finally, in
  addition to all the above benefits, IBM systems groups will have an easier
  method for testing gate aggregation, and more flexibility with methodology, as
  aggregation passes can be compiled into arbitrary unitary gates.
  
  In terms of arbitrary unitary gates, anyone who wants to run their own gates
  will benefit.

## Design Proposal

At a high level, we overload the `InstructionScheduleMap` class to allow for on
the fly gate generation, which is done with the `QocOptimizer` interface.

We introduce a new class `QOCInstructionScheduleMap`, which inherits
`InstructionScheduleMap` and modifies the `InstructionScheduleMap.get` function.
The new class `QOCInstructionScheduleMap` will allow users to automatically
calibrate pulses based on desired unitary evolution. The
`QOCInstructionScheduleMap` will enable users to substitute it in wherever they
would nuse `InstructionScheduleMap`. Thus instead of using the builtin
pulse library we use pulses built on the fly for specific
hardware using quantum optimal control.

We also introduce a new abstract class `QocOptimizer`, which is an abstract
definition which enables us to plug in various optimizers, while maintining
proper dependency inversion. These optimizers will overload the
`get_pulse_schedule` method. An example optimizer is demonstrated in the class
`qutipOptimizer` illustrated later in this document.

Finally, we make several small changes to `InstructionScheduleMap` and `qiskit.scheduler.methods.basic.py`, to allow for the instruction schedule map to map from gates as well as names.


When considering this design we make the following assumptions:

  1. We must support a variety of backends, with varying qubits numbers,
     qubit-qubit couplings, and hamiltonian specifications.
  2. We must be able to support different optimizers, as described in the
     introduction to the `QocOptimizer` class. Moreover, the framework should be easily
     extensible to new optimizer: externally and internally.
  3. It should be easy for users to generate pulses for gates with arbitrary
     unitaries, not just gates which are already defined.


### QOC Instruction Schedule Map Class
The `QOCInstructionScheduleMap` is designed to allow users to plug in the qoc
module in any place where the instruction schedule map would normally be used.
This allows for quantum control to be used across the whole stack, and in
concert with simulators and various programs, as long as the underlying backend
supports `OpenPulse`.


### Optimal Control Optimizer Interface
The OCT optimizer interface (`QocOptimizer`) allows various qoc optimizers[2] to
be used without requiring the optimizers. This abstraction allows us to not be
dependent on any optimizer package. In addition, this interface sets up a
framework to easily add in new optimizers, such as those proposed by the Shuster
lab [3].


### Implementation
The current `InstructionScheduleMap` is often used in the `qiskit.schedule`
function. By overloading one method from `InstructionScheduleMap` we can simply
build a circuit as normal and use the builtin qiskit functions with just a new
inst_map parameter.

Below we illustrate the basic pipeline for Quantum Optimal Control in qiskit. This example uses the `QutipOptimizer` class which is the qutip implementation of the `qoc_optimizer` abstract class. However it is important to note this optimizer is easily replacable.
```python
# Instantiate grape optimizer with armonk backend
grape_optimizer = QutipOptimizer(backend, number_time_steps)

# Create new QocInstructionScheduleMap with this optimizer
builtin_instructions = backend.defaults().instruction_schedule_map
grape_inst_map = QocInstructionScheduleMap.from_inst_map(
                                            grape_optimizer,
                                            builtin_instructions)

#Creates a new pulse schedule using the new instruction schedule map
oct_schedule = schedule(circuits, backend=backend, inst_map = grape_inst_map)
```
#### Diagram
BELOW SHOULD BE DIAGRAM of classes, inheritance, etc (not current one but need to figure out how to)
![image](/Users/benrosand/oct-qiskit-pulse/design_doc_flowchart.pdf "Scheduling pipeline")

There is an interface and one class proposed here: `oct_optimizer` and
`QOCInstructionScheduleMap`. The class `QOCInstructionScheduleMap` is
implemented by inheriting the `InstructionScheduleMap` class. In order to
minimize any conflictss, `QOCInstructionScheduleMap` only modifies the
`self.get()` function. `grape_optimizer` is an example of an optimizer class
that implements the abstract class `QocOptimizer`


This is the new get method, in `QOCInstructionScheduleMap`
```python

def get(self,
  instruction: Union[str, qiskit.circuit.Gate]
  qubits: Union[int, Iterable[int]],
  *params: Union[int, float, complex],
  **kwparams: Union[int, float, complex]) -> Schedule:
    ...
```

In addition to writing a new `get` method for `QOCInstructionScheduleMap`, we
also modify the `get` method for the original `InstructionScheduleMap` to have
the same type hints, and simply convert the gate which is passed in the get
method to it's name for mapping to pulse sequences.

In addition to modyfying the get method we modify the constructor and define a
new class method `.from_backend`, which instantiates a new
`QOCInstructionScheduleMap` from an optimizer and an existing
`InstructionScheduleMap`, which is useful for keeping some default gates such as
the measurement gate.

```python
class QOCInstructionScheduleMap(InstructionScheduleMap):
    def __init__(self, qoc_optimizer):
        ...

    @classmethod
    def from_inst_map(cls, grape_optimizer, instruction_schedule_map, default_inst=['measure']):
        ...
```

### Additional changes to Qiskit stack

Currently, the existing infrastructure (see `qiskit/scheduler/methods/basic.py`)already converts the gate to a string to
pass it into the `InstructionScheduleMap` in `basic.py`. However, we need the
full unitary in order to perform QOC. Thus, we make a slight change to
`InstructionScheduleMap` and `basic.py` as shown below, in
`InstructionScheduleMap` we make two additions and no subtractions, and in
`qiskit.scheduler.methods.basic.py` we make one subtraction.

In `basic.py` we replace the call to `InstructionScheduleMap.get()`as shown below
```python
CircuitPulseDef(schedule=inst_map.get(inst.name, inst_qubits, *inst.params),
    qubits=inst_qubits))
#New:
CircuitPulseDef(schedule=inst_map.get(inst, inst_qubits, *inst.params),
    qubits=inst_qubits))
```

This allows us to replace the string name with the gate instance. In the old
`InstructionScheduleMap` we just check if the instruction is a string, and if so
we use it otherwise call `inst.name` to convert from gate to name.

## Alternative Approaches

One other potential approach to solving this problem would be to write a method
which takes a circuit and converts it into a pulse schedule, which is then run
normally. The problem with this method is it is less easily applicable, it
requires the oct interface to take in a full circuit, when the parsing to go
gate-by-gate is already done in the `InstructionScheduleMap`, so inheriting that
class enables us both to plug into any existing infrastructure and to perform
gate-by-gate optimization smoothly.

Another question that is open is the method of passing the gate into the
`QOCInstructionScheduleMap`, we chose to make the slight modifications to the
existing qiskit code. However, to avoid making any changes, it would be possible
to utilize the `params` field of the `instruction`, and place the gate itself
there to pass the unitary through the process. However the method we have chosen
to use allows us to avoid using the `params` field for a regular use case, as
well as increasing the flexibility of the `InstructionScheduleMap`.
## Questions
Open questions for discussion and an opening for feedback.
- Does this pipeline implement multiple qubit gates
  - This pipeline as written leaves a clear method for implementing multiple
    qubit gates, and we will leave `notImplemented` functions as placeholders
- Does this pipeline allow for gate aggregation?
  - Assuming the gate aggregation takes place before
    `QOCInstructionScheduleMap.get()`is called, then yes as long as the gate has
    a defined unitary. However this pipeline does not assist in the aggregation
    of gates, just in the implementation of the aggregations.
- What is the timeline for this implementation?
  - We anticipate the bulk of any implementation and coding work to take place
    during the week of July 27. We hope to be testing the full implementation
    during that week as well. The converter from qiskit to qutip hamiltonians
    may take slightly longer (make sure to mention this earlier), but for the
    first implementation we will use a single Hamiltonian hard coded into the
    software.
- What could cause the timeline to slow down?
  - We expect the implementation to go smoothly, as a lot of the structure is
    already in place and much of the new code already written. However, the
    success of grape has proven to be highly tempermental, and it is possible
    that we might have to devote significant time to testing the more
    generalized form of our qutip GRAPE wrappers, especially with regards to the
    variable hamiltonians.


## Future Extensions
  - We intend to modify this pipeline so that it will function for a system of
    any number of qubits. This will require the setup of a specific pipeline to
    differentiate control hamiltonians for multi-qubit gates, in addition to testing
    of GRAPE on multi-qubit hamiltonians. Gate aggregation
  - We intend to setup a simple system for gate aggregation. By setting up a
    pass before running grape to aggregate gates together, we can take advantage
    of the qoc on-the-fly gate design to run these new arbitrary gates. Much of
    the infrastructure for the aggregation already exists, this framework simply
    allows arbitrary gates to be run easily. Gate time optimization
  - Currently the time of the gate to be used is set by the user, however we
    intend to extend that to a simple heuristic, and later on use some sort of
    optimizer to find a (relatively) ideal time for the gate. Alternative optimizers
  - Currently we have only implemented qutip optimizers, however the
    infrastructure that we put in place makes it very easy to add a new
    optimizer. A future extension would involve adding optimizers from other
    researchers. Finally, creating a base class on top of the abstract class to
    make implementing one's own optimizer even easier.

## References

[1] Shi, Yunong, Nelson Leung, Pranav Gokhale, Zane Rossi, David I. Schuster,
Henry Hoffman, and Fred T. Chong. “Optimized Compilation of Aggregated
Instructions for Realistic Quantum Computers.” Proceedings of the Twenty-Fourth
International Conference on Architectural Support for Programming Languages and
Operating Systems, April 4, 2019, 1031–44.
https://doi.org/10.1145/3297858.3304018. [2] Johansson, J. R., P. D. Nation, and
Franco Nori. “QuTiP: An Open-Source Python Framework for the Dynamics of Open
Quantum Systems.” Computer Physics Communications 183, no. 8 (August 2012):
1760–72. https://doi.org/10.1016/j.cpc.2012.02.021. [3] Leung, Nelson, Mohamed
Abdelhafez, Jens Koch, and David Schuster. “Speedup for Quantum Optimal Control
from Automatic Differentiation Based on Graphics Processing Units.” Physical
Review A 95, no. 4 (April 13, 2017): 042318.
https://doi.org/10.1103/PhysRevA.95.042318.
