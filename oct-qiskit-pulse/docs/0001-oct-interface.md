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
- qoc faster - need for testing qoc
- starting steps to be useful in real time -->

Quantum optimal control (QOC) is the process of achieving a desired unitary evolution of a system using a number of control fields.
In this case we target a transmon (the quantum system) and through microwave pulses (the control fields), achieve a unitary evolution.
Currently pulses are designed manually for each target gate, and stored in a library for each device and calibrated regularly. 
We propose an optimal control framework which will allow gates to be built on the fly for an arbitrary unitary to the specifications
of the given system. Specifically, this framework will allow for various optimizers from various sources, such as QuTiP's GRAPE(gradient ascent pulse engineering) implementation,
to be used to automatically generate pulses from gates. This pipeline overloads one function in `InstructionScheduleMap`, thus allowing for the
qoc optimizers to be used in place of the default pulse library with two lines of additional code to create the new instruction map.



## Motivation
<!-- - Why are we doing this? -->
  <!-- - enable qoc automatically -->
We want to allow users to run arbitrary qoc optimizers on ibm systems, from various sources, without forcing users to engage with
hardware specs or optimizer specs.
  <!-- - design custom gates on the fly -->
This QOC framework will allow for arbitrary unitary gates to be performed on any system, without transpiling these gates to basis gates.
In addition, the framework will allow for a speedup and potential increase in gate fidelity across circuits, which we expect to increase in porportion with
the size of the systems and the circuits. This work will be especially helpful for gate aggregation, as it will allow the automatic calibration of 
any aggregated gates on the fly, which will compound increases in fidelity and speed[1].
We expect this work to be impactful for everyone, as researchers will be able to test QOC methods, and industry customers will
enjoy the gate speedups as well as the automatic compilation of gates for gate aggregations.
<!--   - potential for large increase in accuracy and speed of larger circuits  -->
<!-- - What will this enable? -->
<!--   - qoc designed gates (especially with blocking) -->
<!-- - What will be the outcome? -->
<!--   - gates built on the fly -->
<!--   - aggregated unitary gates which can be automatically built -->
<!-- - Who will benefit? -->
<!--   - in the near term just researchers -->
<!--   - researchers who want to test qoc -->
<!--   - coders who want to run larger programs faster and more accurately -->

## User Benefit
- Who are the target users of this work?
  - researchers who want to test qoc
  - researchers who want to run circuits with blocking faster and with higher accuracy
  - industry workers who want to build larger and longer circuits
- How will users or contributors benefit from the work proposed?
  - see above I guess
  Researchers will enjoy an easy pipeline for testing QOC implementations, as well as a pipeline that enables easier compilation of arbitrary unitary gates.
  We expect industry users to benefit from an increase in gate fidelity and speed. This will increase the capabilities of the IBM hardware and allow for industry
  users to run more complex circuits with greater fidelity. Finally, in addition to all the above benefits, IBM systems groups will have an easier method for testing gate aggregation, and more flexibility with methodology, as aggregation passes can be compiled into arbitrary unitary gates.
(doesn't this copy from motivation?)

## Design Proposal
This is the focus of the document. Explain the proposal from the perspective of
educating another user on the proposed features.

This generally means:

- Introducing new concepts and nomenclature
- Using examples to introduce new features
- Implementation and Migration path with associated concerns
- Communication of features and changes to users

Focus on giving an overview of impact of the proposed changes to the target
audience.

Factors to consider:

- Performance
- Dependencies
- Maintenance
- Compatibility

The quantum optimal control interface will make us of contexts, context variables, and the preexisting `InstructionScheduleMap` <- blatantly stolen from Thomas

We introduce a new class `QOCInstructionScheduleMap`, which inherits `InstructionScheduleMap` and modifies the `InstructionScheduleMap.get` function. The new class `QOCInstructionScheduleMap` will allow users to automatically calibrate pulses based on desired unitary evolution. The `QOCInstructionScheduleMap` will enable users to substitute it in wherever they would nuse `InstructionScheduleMap`, and thus instead of using the builtin pulses as determined by gate names use pulses built on the fly for specific hardware and gates.

We also introduce a new abstract class `QocOptimizer`, which is an abstract definition which enables us to plug in various optimizers, while maintining proper dependency inversion. These optimizers will overload the `get_pulse_schedule` method.
An example optimizer is demonstrated in the class `qutipOptimizer` illustrated later in this document.

Finally, we make several small changes to `InstructionScheduleMap` and `qiskit.scheduler.methods.basic.py`, to allow for the instruction schedule map to map from gates as well as names.


When considering this design we make the following assumptions:

  1. We must support a variety of backends, with varying numbers of qubits, connections, and hamiltonian specifications.
  2. We must be able to support different optimizers, as described in the introduction to the `QocOptimizer` class.
  3. We wish to make it easy for users to generate pulses for gates with arbitrary unitaries, not just gates which are already defined.

### Quantum Optimal Control Instruction Schedule Map Class
The `QOCInstructionScheduleMap` is designed to allow users to plug in the qoc module in any place where the instruction schedule map would normally be used. This allows for quantum control to be used across the whole stack, and in concert with simulators and various programs, as long as the underlying backend supports `OpenPulse`. 


### Optimal Control Optimizer Interface
The OCT optimizer interface allows various qutip qoc optimizers to be used without requiring the optimizers, maintaining the inversion of dependency by elimination any qiskit dependency on qutip. (Although isn't it already dependent in some ways? -- double check this)

### Implementation
The current `InstructionScheduleMap` is often used in the `qiskit.schedule` function. By overloading one method from `InstructionScheduleMap` we can simply build a circuit as normal and use the builtin qiskit functions with just a new inst_map parameter.

Below we illustrate the basic pipeline for Quantum Optimal Control in qiskit.
```python
circuits = ... # Any arbitrary circuit

#Creating a pulse Schedule using schedule
default_schedule = schedule(circuits, backend=backend, inst_map=backend.defaults().instruction_schedule_map)

#Now we create a schedule using optimal control
from qiskit.pulse import QOCInstructionScheduleMap
from qiskit.pulse.oct_optimizers import qutip_grape_calibrator

#Convert the hamiltonian from the backend to a format for the qutip grape algorithm
backend_hamiltonian = qutip_ham_converter(backend.configuration().hamiltonian)

#Create a grape optimizer to run optimal control
grape_optimizer = grape_optimizer(backend_hamiltonian)

#Create a new optimal control instruction schedule map using the new optimizer
oct_inst_map = QOCInstructionScheduleMap(optimizer=grape_optimizer)

#Creates a new pulse schedule using the new instruction schedule map
oct_schedule = schedule(circuits, backend=backend, inst_map = oct_inst_map)
```

There is an interface and one class proposed here: `oct_optimizer` and `QOCInstructionScheduleMap`. 
The class `QOCInstructionScheduleMap` is implemented by inheriting the `InstructionScheduleMap` class. In order to minimize any conflictss, `QOCInstructionScheduleMap` only modifies the `self.get()` function.
`grape_optimizer` is an example of an optimizer class that implements the abstract class `QocOptimizer`

This is the original get function from `InstructionScheduleMap` (Does this need to be here?)
```python

def get(self,
  instruction: str,
  qubits: Union[int, Iterable[int]],
  *params: Union[int, float, complex],
  **kwparams: Union[int, float, complex]) -> Schedule:
  """Return the defined :py:class:`~qiskit.pulse.Schedule` for the given instruction on
  the given qubits.
    
  Args:
  instruction: Name of the instruction.
  qubits: The qubits for the instruction.
  *params: Command parameters for generating the output schedule.
  **kwparams: Keyworded command parameters for generating the schedule.
    
  Returns:
  The Schedule defined for the input.
  """
  self.assert_has(instruction, qubits)
  schedule_generator = self._map[instruction].get(_to_tuple(qubits))
  
  if callable(schedule_generator):
      return schedule_generator(*params, **kwparams)
  # otherwise this is just a Schedule
  return schedule_generator

```

This is the new get method, in `QOCInstructionScheduleMap
```python

def get(self,
  #The first change that QOCInstructionScheduleMap has is that instruction can be either a string or a gate, if it is a string
  #get will function as if it is an InstructionScheduleMap but with a gate it will run quantum optimal control
  instruction: Union[str, qiskit.circuit.Gate]
  qubits: Union[int, Iterable[int]],
  *params: Union[int, float, complex],
  **kwparams: Union[int, float, complex]) -> Schedule:
  """Return the defined :py:class:`~qiskit.pulse.Schedule` for the given instruction on
  the given qubits.
    
  Args:
  instruction: Name of the instruction.
  qubits: The qubits for the instruction.
  *params: Command parameters for generating the output schedule.
  **kwparams: Keyworded command parameters for generating the schedule.
    
  Returns:
  The Schedule defined for the input.
  """
  self.assert_has(instruction, qubits)
  schedule_generator = self._map[instruction].get(_to_tuple(qubits))
  
  if type(instruction) == 'gate':
      #This is the line where we run qoc, get_pulse_schedule is the main function
      schedule_generator = self.qoc_optimizer.get_pulse_schedule(instruction)
  else:
      self.assert_has(instruction, qubits)
      schedule_generator = self._map[instruction].get(_to_tuple(qubits))
        # don't forget in here to use _gate.to_matrix

  if callable(schedule_generator):
      return schedule_generator(*params, **kwparams)
  # otherwise this is just a Schedule
  return schedule_generator
```

TODO: We also change the constructor if we want to construct with an optimizer? Do we need to show all of that?

### Changes to existing qiskit stack

Currently, `InstructionScheduleMap.get` recieves a gate name and matches it to an existing pulse library. Given that we want to do this with arbitrary pulses, we need to be able to take  in any gate, and based on that input determine a pulse sequence. To do this we overload the `get()` method in `QOCInstructionScheduleMap`.
However, the existing infrastructure already converts the gate to as tring to pass it into the `InstructionScheduleMap` in `basic.py`.
Thus, we make a slight change to `InstructionScheduleMap` and `basic.py` as shown below, in `InstructionScheduleMap` we make two additions and no subtractions, and in `qiskit.scheduler.methods.basic.py`
we make one subtraction.

In `basic.py` we replace the call to `InstructionScheduleMap.get()`as shown below
```python
CircuitPulseDef(schedule=inst_map.get(inst.name, inst_qubits, *inst.params),
    qubits=inst_qubits))
#New:
CircuitPulseDef(schedule=inst_map.get(inst, inst_qubits, *inst.params),
    qubits=inst_qubits))
```

This allows us to replace the string name with the gate instance. In the old `InstructionScheduleMap` we just check if the instruction is a string, and if so we use it otherwise call `inst.name` to convert from gate to name.

## Alternative Approaches
Discuss other approaches to solving this problem and why these were not
selected.

One other potential approach to solving this problem would be to write a method which takes a circuit and converts it into a pulse schedule, which is then run normally. The problem with this method is it is less easily applicable, it requires the oct interface to take in a full circuit, when the parsing to go gate-by-gate is already done in the `InstructionScheduleMap`, so inheriting that class enables us both to plug into any existing infrastructure and to perform gate-by-gate optimization smoothly.

Another question that is open is the method of passing the gate into the `QOCInstructionScheduleMap`, we chose to make the slight modifications to the existing qiskit code.
However, to avoid making any changes, it would be possible to utilize the `params` field of the `instruction`, and place the 
gate itself there to pass the unitary through the process. However the method we have chosen to use allows
us to avoid using the `params` field for a regular use case, as well as increasing the flexibility of the `InstructionScheduleMap`.
## Questions
Open questions for discussion and an opening for feedback.
- Does this pipeline implement multiple qubit gates
  - This pipeline as written leaves a clear method for implementing multiple qubit gates, and we will leave `notImplemented` functions as placeholders
- Does this pipeline allow for gate aggregation?
  - Assuming the gate aggregation takes place before `QOCInstructionScheduleMap.get()`is called, then yes as long as the gate has a defined unitary. However this pipeline does not assist in the aggregation of gates, just in the implementation of the aggregations.
- What is the timeline for this implementation?
  - We anticipate the bulk of any implementation and coding work to take place during the week of July 27. We hope to be testing the full implementation during that week as well. The converter from qiskit to qutip hamiltonians may take slightly longer (make sure to mention this earlier), but for the first implementation we will use a single hailtonian hard coded into the software.
- What could cause the timeline to slow down?
  - We expect the implementation to go smoothly, as a lot of the structure is already in place and much of the new code already written. However, the success of grape has proven to be highly tempermental, and it is possible that we might have to devote significant time to testing the more generalized form of our qutip GRAPE wrappers, especially with regards to the variable hamiltonians.


## Future Extensions
Consider what extensions might spawn from this RFC. Discuss the roadmap of
related projects and how these might interact. This section is also an opening
for discussions and a great place to dump ideas.

If you do not have any future extensions in mind, state that you cannot think
of anything. This section should not be left blank.

- N-qubits
  - We intend to modify this pipeline so that it will function for a system of any number of qubits. This will require the setup of a specific pipeline to differentiate control hamiltonians for multi-qubit gates, in addition to testing of GRAPE on multi-qubit hamiltonians.
- Gate aggregation
  - We intend to setup a simple system for gate aggregation. By setting up a pass before running grape to aggregate gates together, we can take advantage of the qoc on-the-fly gate design to run these new arbitrary gates.
    Much of the infrastructure for the aggregation already exists, this framework simply allows arbitrary gates to be run easily.
- Alternative optimizers
  - Currently we only support qutip optimizers, however the infrastructure that we put in place makes it very easy to add a new optimizer.
    A future extension would involve adding optimizers from other researchers.
    Finally, creating a base class on top of the abstract class to make implementing one's own optimizer even easier.

## References

[1] Shi, Yunong, Nelson Leung, Pranav Gokhale, Zane Rossi, David I. Schuster, Henry Hoffman, and Fred T. Chong. “Optimized Compilation of Aggregated Instructions for Realistic Quantum Computers.” Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, April 4, 2019, 1031–44. https://doi.org/10.1145/3297858.3304018.
