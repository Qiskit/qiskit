# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Example showing how to create a custom representation using a bell circuit
  # Print a small circuit
  python qiskit/examples/python/customer_repr.py --object=cond
  # Print a larger circuit with wrap at 180
  python qiskit/examples/python/customer_repr.py --object=bell
"""
import sys
from getopt import getopt, GetoptError
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuitdata import CircuitInstruction

from qiskit.utils.reprparse import ReprParser
from qiskit.utils.reprbuild import is_valid_repr, build_repr

_usage = "No program usage available"


def Instruction_rebuild(class_repr=None):
    """Return an Instruction based on the input dictionary

    Args:
        class_repr(string): string representation of the repr dictionary

    Returns:
        Instruction: The Instruction equivalent to the circuit generating the repr

    Raises:

    Additional Information:
        Should be embedded as a class method if rebuilding from representations is a desired feature set

        @classmethod
        def rebuild(cls, class_repr=None):
            from qiskit.utils.reprparse import ReprParser
            ...
    """

    new_instruction = None
    parser = ReprParser(class_repr, attr_rebuilders())
    defn_parser = parser.get_parser("_definition")
    if defn_parser is not None:
        new_instruction = defn_parser.rebuild()

    return new_instruction


def CircuitInstruction_rebuild(class_repr=None):
    """Return a QuantumCircuit based on the input dictionary

    Args:
        class_repr(string): string representation of the repr dictionary

    Returns:
        CircuitInstruction: The circuit equivalent to the circuit generating the repr

    Raises:
        QiskitError: if the dictionary is not valid
    Additional Information:
        Should be embedded as a class method if rebuilding from representations is a desired feature set

        @classmethod
        def rebuild(cls, class_repr=None):
            from qiskit.utils.reprparse import ReprParser
            from qiskit.utils.reprbuild import is_valid_repr
            ...
    """
    from qiskit.circuit import Instruction

    parser = ReprParser(class_repr, attr_rebuilders())

    qubits = []
    clbits = []
    new_instruction = None

    bits_parser = parser.get_parser("qubits")
    if bits_parser is not None:
        bits_tuple = bits_parser.rebuild()
        for curbit in bits_tuple:
            if is_valid_repr(curbit):
                new_bit = bits_parser.get_parser(curbit).rebuild()
            else:
                new_bit = int(curbit)
            qubits.append(new_bit)

    bits_parser = parser.get_parser("clbits")
    if bits_parser is not None:
        bits_tuple = bits_parser.rebuild()
        for curbit in bits_tuple:
            if is_valid_repr(curbit):
                new_bit = bits_parser.get_parser(curbit).rebuild()
            else:
                new_bit = int(curbit)
            clbits.append(new_bit)

    operation = parser.get_parser("operation")
    if operation is not None:
        new_op = operation.rebuild()
    if isinstance(new_op, Instruction):
        new_instruction = CircuitInstruction(new_op, qubits, clbits)
    elif new_op is not None:
        new_instruction = (new_op, qubits, clbits)

    return new_instruction


def QuantumCircuit_rebuild(class_repr=None):
    """Return a QuantumCircuit based on the input representation

    Args:
        class_repr(string): string representation of the circuit to be rebuilt

    Returns:
        QuantumCircuit: The circuit equivalent to the representation input

    Raises:
        ReprError: if the dictionary is not valid

    Additional Information:
        This can be embedded as a class method in QuantumCircuit if desired

        @classmethod
        def rebuild(cls,class_repr=None):
            from qiskit.utils.reprparse import ReprParser
            ...

    """
    from collections import defaultdict

    parser = ReprParser(class_repr, attr_rebuilders())

    global_phase = parser.get_parser("_global_phase")
    if global_phase is not None:
        global_phase = global_phase.rebuild()

    qc = QuantumCircuit(
        parser.get_int("num_qubits"),
        parser.get_int("num_clbits"),
        name=parser.get_string("_base_name"),
        global_phase=global_phase,
    )

    calibrations = parser.get_dict("_calibrations", None)
    if calibrations is not None and isinstance(calibrations, (defaultdict, dict)):
        qc.calibrations = calibrations
    metadata = parser.get_dict("_metadata", None)
    if metadata is not None and isinstance(metadata, (defaultdict, dict)):
        qc.metadata = metadata

    data_parser = parser.get_parser("_data")
    data_list = data_parser.rebuild()
    if isinstance(data_list, list) and len(data_list) > 0:
        for data_item in data_list:
            item_parser = parser.get_parser(data_item)
            if item_parser is not None:
                new_item = item_parser.rebuild()
                if isinstance(new_item, tuple):
                    qc.append(*new_item)
                elif new_item is not None:
                    qc.append(new_item)

    return qc


def Measure_rebuild(cls, class_repr=None):  # pylint: disable=unused-argument
    """Return an Instruction based on the input dictionary

    Args:
        class_repr(string): string representation of the repr dictionary

    Returns:
        Instruction: The Measure Instruction equivalent to the circuit generating the repr

    Raises:

    Additional Information:
        This can be embedded as a class method in Measure if rebuilding is a desired feature

        @classmethod
        def rebuild(cls,class_repr=None):  # pylint: disable=unused-argument

    """
    from qiskit.circuit import Measure

    return Measure()


def Reset_rebuild(class_repr=None):  # pylint: disable=unused-argument
    """Return a Reset Instruction

    Args:
        class_repr(string): string representation of the repr dictionary

    Returns:
        Instruction: The Measure Instruction equivalent to the circuit generating the repr

    Raises:

    Additional Information:
        This can be embedded as a class method in Reset if rebuilding is a desired feature

        @classmethod
        def rebuild(cls,class_repr=None):  # pylint: disable=unused-argument

    """
    from qiskit.circuit import Reset

    return Reset()


def Clbit_rebuild(obj_repr):
    """Return the integer bit index

    Args:
        obj_repr(string): string representation of the repr dictionary

    Returns:
        int: The bit index

    Raises:
    """

    parser = ReprParser(obj_repr)
    index = parser.rebuild("_index")
    return index


def Qubit_rebuild(obj_repr):
    """Return the integer bit index

    Args:
        obj_repr(string): string representation of the repr dictionary

    Returns:
        int: The bit index

    Raises:
    """
    from qiskit.circuit import Qubit

    parser = ReprParser(obj_repr)
    index = parser.rebuild("_index")
    if index is not None:
        return index
    return Qubit()


def _map_barrier(gate_repr):
    parser = ReprParser(gate_repr)
    if parser is not None and parser.class_name == "Barrier":
        from qiskit.circuit import Barrier

        label = parser.get_string("_label")
        num_qubits = parser.get_int("_num_qubits")
        return Barrier(num_qubits, label=label)
    else:
        return None


def _map_h_gate(gate_repr):
    parser = ReprParser(gate_repr)
    if parser is None:
        return None

    if parser.class_name == "HGate":
        from qiskit.circuit.library.standard_gates.h import HGate

        return HGate()
    elif parser.class_name == "CHGate":
        from qiskit.circuit.library.standard_gates.h import CHGate

        label = parser.get_string("_label")
        ctrl_state = parser.get_int("ctrl_state", 1)
        return CHGate(label=label, ctrl_state=ctrl_state)
    else:
        return None


def _map_x_gate(gate_repr):
    parser = ReprParser(gate_repr)
    # TODO: Finish the rest of the classes from x.py
    if parser is None:
        return_val = None
    elif parser.class_name == "XGate":
        from qiskit.circuit.library.standard_gates.x import XGate

        label = parser.get_string("_label")
        return_val = XGate(label=label)
    elif parser.class_name == "CXGate":
        from qiskit.circuit.library.standard_gates.x import CXGate

        label = parser.get_string("_label")
        ctrl_state = parser.get_int("_ctrl_state", 1)
        return_val = CXGate(label=label, ctrl_state=ctrl_state)
    elif parser.class_name == "CCXGate":
        from qiskit.circuit.library.standard_gates.x import CCXGate

        label = parser.get_string("_label")
        ctrl_state = parser.get_int("_ctrl_state", 3)
        return_val = CCXGate(label=label, ctrl_state=ctrl_state)
    elif parser.class_name == "RCCXGate":
        from qiskit.circuit.library.standard_gates.x import RCCXGate

        label = parser.get_string("_label")
        return_val = RCCXGate(label=label)
    elif parser.class_name == "RC3XGate":
        from qiskit.circuit.library.standard_gates.x import RC3XGate

        label = parser.get_string("_label")
        return_val = RC3XGate(label=label)
    elif parser.class_name == "RXGate":
        from qiskit.circuit.library.standard_gates.rx import RXGate

        theta = None
        label = parser.get_string("_label")
        params_parser = parser.get_parser("_params")
        if params_parser is not None:
            params = params_parser.rebuild()
        if params is not None:
            theta = params[0]
        return_val = RXGate(theta, label=label)
    else:
        return_val = None
    return return_val


_standard_gate_map = {
    "HGate": _map_h_gate,
    "CHGate": _map_h_gate,
    "XGate": _map_x_gate,
    "CXGate": _map_x_gate,
    "CCXGate": _map_x_gate,
    "RXGate": _map_x_gate,
    "RCCXGate": _map_x_gate,
    "C3SXGate": _map_x_gate,
    "C3XGate": _map_x_gate,
    "RC3SXGate": _map_x_gate,
    "C4XGate": _map_x_gate,
    "MCXGate": _map_x_gate,
    "MCXGrayCode": _map_x_gate,
    "MCXVChain": _map_x_gate,
    "Barrier": _map_barrier,
    "None": None,
    "NoneType": None,
}


def attr_rebuilders():
    """Name to method mapping for all the classes utilized by a quantum circuit.  The mappings
        allow the parser(s) to rebuild attributes recursively.
    Args:
    Returns:
        dict: Name to rebuilder method mapping

    Raises:
    Additional Information:
        Should be embedded as a class method in
            quantumcircuitdata.CircuitInstruction
            quantumcircuit.QuantumCircuit
            instruction.Instruction
        if rebuilding from representations is a desired feature set

    @class_method(cls):
        from qiskit.circuit.library.standard_gates.reprbuild import get_standard_gate_map
        from qiskit.circuit import QuantumCircuit, Instruction, Measure, Reset
        rebuilders = {
            "QuantumCircuit": QuantumCircuit.rebuild,
            "CircuitInstruction": CircuitInstruction.rebuild,
            "Instruction": Instruction.rebuild,
            "Qubit": Qubit.rebuild,
            "Clbit": Clbit.rebuild,
            "Measure": Measure.rebuild,
            "Reset": Reset.rebuild,
        }
        ...

    """
    rebuilders = {
        "QuantumCircuit": QuantumCircuit_rebuild,
        "CircuitInstruction": CircuitInstruction_rebuild,
        "Instruction": Instruction_rebuild,
        "Qubit": Qubit_rebuild,
        "Clbit": Clbit_rebuild,
        "Measure": Measure_rebuild,
        "Reset": Reset_rebuild,
    }
    rebuilders.update(_standard_gate_map)
    return rebuilders


class myClass:
    """class template compatible with recursive repr() generation and
        instantiation

    Args:
        name (str): myClass instance name (if it was output by repr())
    Raises:

    Additional Information:
        self._repr_attrs being defined is all that is reuired to allow
        the recursive algorithm include myClass in any class in
        which a myClass object is listed in the repr() without requiring
        changes to any existing __repr__ methods
        if repr_dict is passed in to __init__ it can be used to instiated
        a new instance of myClass equivialent to the instance which generated
        the representation
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self._name = name
        self._repr_attrs = ["_name"]

    def __repr__(self) -> str:
        """to create a recursive repr() for myClass invoke build_repr
        as shown"""

        return build_repr(self, attr_list=self._repr_attrs, depth=-1, deepdive=False)

    def __str__(self):
        # from qiskit.utils.reprbuild import build_repr
        from qiskit.utils.reprparse import format_repr

        return format_repr(build_repr(self, attr_list=self._repr_attrs))

    def __eq__(self, other) -> bool:
        if not isinstance(other, myClass):
            return False
        else:
            return self._name == other._name

    @classmethod
    def rebuilder(cls, class_repr=None):
        """Return a QuantumCircuit based on the input dictionary

        Args:
            class_repr(string): string representation of the repr dictionary

        Returns:
            myClass: The object equivalent to the instance generating the repr

        Raises:

        Additional Information:
            Create an instance as defined by the dictionary
            if needed to create instances of an attribute
        """

        parser = ReprParser(class_repr)

        return myClass(name=parser.get_string("_name", "None"))


_custom_repr_attrs = ["_global_phase", "_base_name", "_data"]


def _get_calibrated_circuit():
    from qiskit import pulse
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate
    import numpy as np

    theta = Parameter("theta")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(np.pi, 0)
    qc.rx(theta, 1)
    qc = qc.assign_parameters({theta: np.pi})

    with pulse.build() as h_sched:
        pulse.play(pulse.library.Drag(1, 0.15, 4, 2), pulse.DriveChannel(0))

    with pulse.build() as x180:
        pulse.play(pulse.library.Gaussian(1, 0.2, 5), pulse.DriveChannel(0))

    qc.add_calibration("h", [0], h_sched)
    qc.add_calibration(RXGate(np.pi), [0], x180)
    return qc


def _get_unbound_circuit():
    from qiskit.circuit import Parameter

    # create the parameter
    phi = Parameter("phi")
    ro = Parameter("ro")
    qc = QuantumCircuit(1, global_phase=ro, name="Unbound")

    # parameterize the rotation
    qc.rx(phi, 0)
    return qc


def _get_bound_circuit():
    from qiskit.circuit import Parameter
    from math import sqrt

    # create the parameter
    phi = Parameter("phi")
    ro = Parameter("ro")
    qc = QuantumCircuit(1, global_phase=ro, name="Bound")

    # parameterize the rotation
    qc.rx(phi, 0)

    # bind the parameters after circuit to create a bound circuit
    bc = qc.bind_parameters({phi: 3.14, ro: sqrt(2)})
    return bc


def _get_ccx_circuit():
    qc = QuantumCircuit(3, 3, name="CCX Circuit")
    qc.h([0, 1])
    qc.ccx(0, 1, 2)
    return qc


def _get_bell_circuit():
    """Returns a circuit putting 2 qubits in the Bell state."""
    qc = QuantumCircuit(2, 2, name="Hadamard")
    qc.h(0)
    qc.cx(0, 1, label="cx label")
    return qc


def _get_inner_circuit():
    qc = QuantumCircuit(2, 2, name="Inner")
    qc.h(0)
    qc.cx(0, 1)
    inner = QuantumCircuit(1, 1)
    qc.barrier()
    inner.h(0)
    inner.measure(0, 0)
    qc.append(inner, [0], [0])
    return qc


def _get_cond_object():
    from qiskit.circuit import Qubit, Clbit

    bits = [Qubit(), Clbit()]
    cond = (bits[1], 0)
    test = QuantumCircuit(bits, name="Cond")
    test.if_test(cond)
    test.h(0)
    return test


def _get_cx_gate():
    from qiskit.circuit.library.standard_gates.x import CXGate

    return CXGate(label="cx gate", ctrl_state=1)


def _get_cx_instruction():
    new_op = _get_cx_gate()
    return CircuitInstruction(new_op, [0, 1], [])


def _get_cmdline(argv):

    try:
        opts, args = getopt(  # pylint: disable=unused-variable
            argv, "hcew:o:", ["width=", "custom", "object=", "eval"]
        )
    except GetoptError:
        print(_usage)
        sys.exit(2)
    # Set default values
    _options = {}
    _options["width"] = 120
    _options["object"] = "all"
    _options["eval"] = False
    for opt, arg in opts:
        if opt == "-h":
            print(_usage)
            sys.exit()
        elif opt in ("-o", "--object"):
            _options["object"] = arg
        elif opt in ("-c", "--custom"):
            _options["object"] = "custom"
        elif opt in ("-w", "--width"):
            _options["width"] = arg
        elif opt in ("-e", "--eval"):
            _options["eval"] = True
    return _options


circuit_dict = {
    "cxgate": _get_cx_gate,
    "cxinstruction": _get_cx_instruction,
    "bell": _get_bell_circuit,
    "inner": _get_inner_circuit,
    "ccx": _get_ccx_circuit,
    "cond": _get_cond_object,
    "bound": _get_bound_circuit,
    "unbound": _get_unbound_circuit,
    "calibrated": _get_calibrated_circuit,
}

_usage = """custom_repr.p -w <width> -e <eval> -o <object>
         eval   : If set, evaluate returned repr and compareot original
         object : csv list of objects to test. Default is all
              all           : Run all the objects except custom
              cxgate        : CXGate(0,1)
              cxinstruction : CXGate(0,1) as Instruction
              bell          : Bell Quantum Circuit
              bound         : Circuit with bound parameter
              calibrated    : Circuit with calibrations
              ccx           : Circuit with ccx gate
              cond          : Quantum circuit with conditional test
              inner         : Circuit built from qc.append
              unbound       : Circuit with unbound parameters
              custom        : Custom repr on the bell Quantum Circuit
                      ["_global_phase","_base_name",'_data']
     """


def main():
    """Create quantum circuit and print out the representations for it
    Args:

    Returns:

    Raises:

    Additional Information:
    """

    options = _get_cmdline(sys.argv[1:])

    disp_obj = options["object"]
    if disp_obj == "all":
        disp_list = circuit_dict
        for circ in disp_list:
            source_obj = circuit_dict[circ]()
            obj_repr = repr(source_obj)
            if not is_valid_repr(obj_repr) and hasattr(source_obj, "_repr_attrs"):
                obj_repr = build_repr(source_obj, attr_list=source_obj._repr_attrs)
            parser = ReprParser(obj_repr, attr_rebuilders())
            print(f"------------- print( {circ} )---------------")
            print(source_obj)
            print(f"------------- parser({circ}).print ---------------")
            parser.print()
            if options["eval"] and circ != "custom":
                try:
                    new_obj = parser.rebuild(parser.class_name, obj_repr)
                    if new_obj == source_obj:
                        print(f"New Object {circ} is equivalent(==) to Original")
                    else:
                        print(
                            f"---------- New Object {circ} fails equivalence(==) test -------------"
                        )
                    print(f"------------- print( New {circ} )---------------")
                    print(new_obj)
                    new_repr = repr(new_obj)
                    if not is_valid_repr(new_repr) and hasattr(new_obj, "_repr_attrs"):
                        new_repr = build_repr(new_obj, attr_list=new_obj._repr_attrs)
                    print(f"------------- parser( New {circ}).print ---------------")
                    ReprParser(new_repr).print()
                except Exception as error:  # pylint: disable=broad-except
                    print(
                        f"----- Unable to rebuild {circ} to perform equivalence(==) test ---------"
                    )
                    print(f"Returned Exception{error}")
    else:
        disp_list = disp_obj.split(",")
        for circ in disp_list:
            source_obj = circuit_dict[circ]()
            obj_repr = repr(source_obj)
            if not is_valid_repr(obj_repr) and hasattr(source_obj, "_repr_attrs"):
                obj_repr = build_repr(source_obj, attr_list=source_obj._repr_attrs)
            parser = ReprParser(obj_repr, attr_rebuilders())
            print(f"------------- print( {circ} )---------------")
            print(source_obj)
            print(f"------------- parser({circ}).print ---------------")
            parser.print()
            if options["eval"] and circ != "custom":
                try:
                    new_obj = parser.rebuild(parser.class_name, obj_repr)
                    if new_obj == source_obj:
                        print(f"New Object {circ} is equivalent(==) to Original")
                    else:
                        print(
                            f"---------- New Object {circ} fails equivalence(==) test -------------"
                        )
                    print(f"------------- print( New {circ} )---------------")
                    print(new_obj)
                    new_repr = repr(new_obj)
                    if not is_valid_repr(new_repr) and hasattr(new_obj, "_repr_attrs"):
                        new_repr = build_repr(new_obj, attr_list=new_obj._repr_attrs)
                    print(f"------------- parser( New {circ}).print ---------------")
                    ReprParser(new_repr).print()
                except Exception as error:  # pylint: disable=broad-except
                    print(
                        f"----- Unable to rebuild {disp_obj} to perform equivalence(==) test ---------"
                    )
                    print(f"Returned Exception{error}")


if __name__ == "__main__":
    main()
