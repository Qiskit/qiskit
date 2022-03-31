# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Randomized tests of transpiler circuit equivalence."""

import os
from math import pi

from distutils.util import strtobool
from hypothesis import assume, settings, HealthCheck
from hypothesis.stateful import multiple, rule, precondition, invariant
from hypothesis.stateful import Bundle, RuleBasedStateMachine

import hypothesis.strategies as st

from hypothesis import settings

default_profile = "transpiler_equivalence"
settings.register_profile(
    default_profile,
    report_multiple_bugs=False,
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", default_profile))

from qiskit import execute, transpile, Aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Measure, Reset, Gate, Barrier
from qiskit.test.mock import (
    FakeProvider,
    FakeOpenPulse2Q,
    FakeOpenPulse3Q,
    FakeYorktown,
    FakeTenerife,
    FakeOurense,
    FakeVigo,
    FakeMelbourne,
    FakeRueschlikon,
    FakeTokyo,
    FakePoughkeepsie,
    FakeAlmaden,
    FakeSingapore,
    FakeJohannesburg,
    FakeBoeblingen,
    FakeRochester,
    FakeBurlington,
    FakeCambridge,
    FakeCambridgeAlternativeBasis,
    FakeEssex,
    FakeLondon,
    FakeQasmSimulator,
    FakeArmonk,
    FakeRome,
    FakeSantiago,
    FakeSydney,
    FakeToronto,
    FakeValencia,
)
from qiskit.test.base import dicts_almost_equal


# pylint: disable=wildcard-import,unused-wildcard-import
from qiskit.circuit.library.standard_gates import *

oneQ_gates = [HGate, IGate, SGate, SdgGate, TGate, TdgGate, XGate, YGate, ZGate, Reset]
twoQ_gates = [CXGate, CYGate, CZGate, SwapGate, CHGate]
threeQ_gates = [CCXGate, CSwapGate]

oneQ_oneP_gates = [U1Gate, RXGate, RYGate, RZGate]
oneQ_twoP_gates = [U2Gate]
oneQ_threeP_gates = [U3Gate]

twoQ_oneP_gates = [CRZGate, RZZGate, CU1Gate]
twoQ_threeP_gates = [CU3Gate]

oneQ_oneC_gates = [Measure]
variadic_gates = [Barrier]

if not strtobool(os.getenv("QISKIT_RANDOMIZED_TEST_ALLOW_BARRIERS", "True")):
    variadic_gates.remove(Barrier)


def getenv_list(var_name):
    value = os.getenv(var_name)
    return None if value is None else value.split()


# Note: a value of `None` for any of the following methods means that
# the selected pass manager gets to choose. However, to avoid complexity,
# its not possible to specify `None` when overriding these with environment
# variables. Really, `None` is useful only for testing Terra's pass managers,
# and if you're overriding these, your goal is probably to test a specific
# pass or set of passes instead.
layout_methods = getenv_list("QISKIT_RANDOMIZED_TEST_LAYOUT_METHODS") or [
    None,
    "trivial",
    "dense",
    "noise_adaptive",
    "sabre",
]
routing_methods = getenv_list("QISKIT_RANDOMIZED_TEST_ROUTING_METHODS") or [
    None,
    "basic",
    "stochastic",
    "lookahead",
    "sabre",
]
scheduling_methods = getenv_list("QISKIT_RANDOMIZED_TEST_SCHEDULING_METHODS") or [
    None,
    "alap",
    "asap",
]

backend_needs_durations = strtobool(
    os.getenv("QISKIT_RANDOMIZED_TEST_BACKEND_NEEDS_DURATIONS", "False")
)


def fully_supports_scheduling(backend):
    """Checks if backend is not in the set of backends known not to have specified gate durations."""
    return not isinstance(
        backend,
        (
            # no coupling map
            FakeArmonk,
            # no measure durations
            FakeAlmaden,
            FakeBurlington,
            FakeCambridge,
            FakeCambridgeAlternativeBasis,
            FakeEssex,
            FakeJohannesburg,
            FakeLondon,
            FakeOpenPulse2Q,
            FakeOpenPulse3Q,
            FakePoughkeepsie,
            FakeQasmSimulator,
            FakeRochester,
            FakeRueschlikon,
            FakeSingapore,
            FakeTenerife,
            FakeTokyo,
            # No reset duration
            FakeAlmaden,
            FakeArmonk,
            FakeBoeblingen,
            FakeBurlington,
            FakeCambridge,
            FakeCambridgeAlternativeBasis,
            FakeEssex,
            FakeJohannesburg,
            FakeLondon,
            FakeMelbourne,
            FakeOpenPulse2Q,
            FakeOpenPulse3Q,
            FakeOurense,
            FakePoughkeepsie,
            FakeQasmSimulator,
            FakeRochester,
            FakeRome,
            FakeRueschlikon,
            FakeSantiago,
            FakeSingapore,
            FakeSydney,
            FakeTenerife,
            FakeTokyo,
            FakeToronto,
            FakeValencia,
            FakeVigo,
            FakeYorktown,
        ),
    )


fake_provider = FakeProvider()
mock_backends = fake_provider.backends()
mock_backends_with_scheduling = [b for b in mock_backends if fully_supports_scheduling(b)]


class QCircuitMachine(RuleBasedStateMachine):
    """Build a Hypothesis rule based state machine for constructing, transpiling
    and simulating a series of random QuantumCircuits.

    Build circuits with up to QISKIT_RANDOM_QUBITS qubits, apply a random
    selection of gates from qiskit.circuit.library with randomly selected
    qargs, cargs, and parameters. At random intervals, transpile the circuit for
    a random backend with a random optimization level and simulate both the
    initial and the transpiled circuits to verify that their counts are the
    same.

    """

    qubits = Bundle("qubits")
    clbits = Bundle("clbits")

    backend = Aer.get_backend("qasm_simulator")
    max_qubits = int(backend.configuration().n_qubits / 2)

    def __init__(self):
        super().__init__()
        self.qc = QuantumCircuit()
        self.enable_variadic = bool(variadic_gates)

    @precondition(lambda self: len(self.qc.qubits) < self.max_qubits)
    @rule(target=qubits, n=st.integers(min_value=1, max_value=max_qubits))
    def add_qreg(self, n):
        """Adds a new variable sized qreg to the circuit, up to max_qubits."""
        n = min(n, self.max_qubits - len(self.qc.qubits))
        qreg = QuantumRegister(n)
        self.qc.add_register(qreg)
        return multiple(*list(qreg))

    @rule(target=clbits, n=st.integers(1, 5))
    def add_creg(self, n):
        """Add a new variable sized creg to the circuit."""
        creg = ClassicalRegister(n)
        self.qc.add_register(creg)
        return multiple(*list(creg))

    # Gates of various shapes

    @rule(gate=st.sampled_from(oneQ_gates), qarg=qubits)
    def add_1q_gate(self, gate, qarg):
        """Append a random 1q gate on a random qubit."""
        self.qc.append(gate(), [qarg], [])

    @rule(
        gate=st.sampled_from(twoQ_gates),
        qargs=st.lists(qubits, max_size=2, min_size=2, unique=True),
    )
    def add_2q_gate(self, gate, qargs):
        """Append a random 2q gate across two random qubits."""
        self.qc.append(gate(), qargs)

    @rule(
        gate=st.sampled_from(threeQ_gates),
        qargs=st.lists(qubits, max_size=3, min_size=3, unique=True),
    )
    def add_3q_gate(self, gate, qargs):
        """Append a random 3q gate across three random qubits."""
        self.qc.append(gate(), qargs)

    @rule(
        gate=st.sampled_from(oneQ_oneP_gates),
        qarg=qubits,
        param=st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-10 * pi,
            max_value=10 * pi,
            allow_subnormal=False,
        ),
    )
    def add_1q1p_gate(self, gate, qarg, param):
        """Append a random 1q gate with 1 random float parameter."""
        self.qc.append(gate(param), [qarg])

    @rule(
        gate=st.sampled_from(oneQ_twoP_gates),
        qarg=qubits,
        params=st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-10 * pi,
                max_value=10 * pi,
                allow_subnormal=False,
            ),
            min_size=2,
            max_size=2,
        ),
    )
    def add_1q2p_gate(self, gate, qarg, params):
        """Append a random 1q gate with 2 random float parameters."""
        self.qc.append(gate(*params), [qarg])

    @rule(
        gate=st.sampled_from(oneQ_threeP_gates),
        qarg=qubits,
        params=st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-10 * pi,
                max_value=10 * pi,
                allow_subnormal=False,
            ),
            min_size=3,
            max_size=3,
        ),
    )
    def add_1q3p_gate(self, gate, qarg, params):
        """Append a random 1q gate with 3 random float parameters."""
        self.qc.append(gate(*params), [qarg])

    @rule(
        gate=st.sampled_from(twoQ_oneP_gates),
        qargs=st.lists(qubits, max_size=2, min_size=2, unique=True),
        param=st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-10 * pi,
            max_value=10 * pi,
            allow_subnormal=False,
        ),
    )
    def add_2q1p_gate(self, gate, qargs, param):
        """Append a random 2q gate with 1 random float parameter."""
        self.qc.append(gate(param), qargs)

    @rule(
        gate=st.sampled_from(twoQ_threeP_gates),
        qargs=st.lists(qubits, max_size=2, min_size=2, unique=True),
        params=st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-10 * pi,
                max_value=10 * pi,
                allow_subnormal=False,
            ),
            min_size=3,
            max_size=3,
        ),
    )
    def add_2q3p_gate(self, gate, qargs, params):
        """Append a random 2q gate with 3 random float parameters."""
        self.qc.append(gate(*params), qargs)

    @rule(gate=st.sampled_from(oneQ_oneC_gates), qarg=qubits, carg=clbits)
    def add_1q1c_gate(self, gate, qarg, carg):
        """Append a random 1q, 1c gate."""
        self.qc.append(gate(), [qarg], [carg])

    @precondition(lambda self: self.enable_variadic)
    @rule(gate=st.sampled_from(variadic_gates), qargs=st.lists(qubits, min_size=1, unique=True))
    def add_variQ_gate(self, gate, qargs):
        """Append a gate with a variable number of qargs."""
        self.qc.append(gate(len(qargs)), qargs)

    @precondition(lambda self: len(self.qc.data) > 0)
    @rule(carg=clbits, data=st.data())
    def add_c_if_last_gate(self, carg, data):
        """Modify the last gate to be conditional on a classical register."""
        creg = carg.register
        val = data.draw(st.integers(min_value=0, max_value=2 ** len(creg) - 1))

        last_gate = self.qc.data[-1]

        # Conditional instructions are not supported
        assume(isinstance(last_gate[0], Gate))

        last_gate[0].c_if(creg, val)

    # Properties to check

    @invariant()
    def qasm(self):
        """After each circuit operation, it should be possible to build QASM."""
        self.qc.qasm()

    @st.composite
    def transpiler_conf(draw):
        opt_level = draw(st.integers(min_value=0, max_value=3))
        layout_method = draw(st.sampled_from(layout_methods))
        routing_method = draw(st.sampled_from(routing_methods))
        scheduling_method = draw(st.sampled_from(scheduling_methods))

        compatible_backends = st.one_of(st.none(), st.sampled_from(mock_backends))
        if scheduling_method is not None or backend_needs_durations:
            compatible_backends = st.sampled_from(mock_backends_with_scheduling)

        backend = draw(st.one_of(compatible_backends))

        return (backend, opt_level, layout_method, routing_method, scheduling_method)

    @precondition(lambda self: any(isinstance(d[0], Measure) for d in self.qc.data))
    @rule(conf=transpiler_conf())
    def equivalent_transpile(self, conf):
        """Simulate, transpile and simulate the present circuit. Verify that the
        counts are not significantly different before and after transpilation.

        """
        backend, opt_level, layout_method, routing_method, scheduling_method = conf

        assume(backend is None or backend.configuration().n_qubits >= len(self.qc.qubits))

        print(
            f"Evaluating circuit at level {opt_level} on {backend} "
            f"using layout_method={layout_method} routing_method={routing_method} "
            f"and scheduling_method={scheduling_method}:\n{self.qc.qasm()}"
        )

        shots = 4096

        aer_counts = execute(self.qc, backend=self.backend, shots=shots).result().get_counts()

        try:
            xpiled_qc = transpile(
                self.qc,
                backend=backend,
                optimization_level=opt_level,
                layout_method=layout_method,
                routing_method=routing_method,
                scheduling_method=scheduling_method,
            )
        except Exception as e:
            failed_qasm = "Exception caught during transpilation of circuit: \n{}".format(
                self.qc.qasm()
            )
            raise RuntimeError(failed_qasm) from e

        xpiled_aer_counts = (
            execute(xpiled_qc, backend=self.backend, shots=shots).result().get_counts()
        )

        count_differences = dicts_almost_equal(aer_counts, xpiled_aer_counts, 0.05 * shots)

        assert (
            count_differences == ""
        ), "Counts not equivalent: {}\nFailing QASM Input:\n{}\n\nFailing QASM Output:\n{}".format(
            count_differences, self.qc.qasm(), xpiled_qc.qasm()
        )


TestQuantumCircuit = QCircuitMachine.TestCase
