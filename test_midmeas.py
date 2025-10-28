if __name__ == "__main__":
    from math import pi
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RXGate, XGate, UGate
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
            ASAPScheduleAnalysis,
            ALAPScheduleAnalysis,
            PadDynamicalDecoupling,
        )
    from qiskit.transpiler.passes.scheduling.padding.dynamical_decoupling import (
        PadDynamicalDecoupling,
    )
    from qiskit.transpiler.instruction_durations import InstructionDurations
    import traceback
    try:
        durations = InstructionDurations(
                [
                    ("h", 0, 50),
                    ("cx", [0, 1], 700),
                    ("cx", [1, 2], 200),
                    ("cx", [2, 3], 300),
                    ("x", None, 50),
                    ("y", None, 50),
                    ("u", None, 100),
                    ("rx", None, 100),
                    ("measure", None, 1000),
                    ("reset", None, 1500),
                ],
                dt=1e-9,
            )
        midmeas = QuantumCircuit(3, 1)
        midmeas.cx(0, 1)
        midmeas.cx(1, 2)
        midmeas.u(pi, 0, pi, 0)
        midmeas.measure(2, 0)
        midmeas.cx(1, 2)
        midmeas.cx(0, 1)
        dd_sequence = [RXGate(pi / 4)]
        pm = PassManager(
            [
                ASAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(durations, dd_sequence),
            ]
        )

        midmeas_dd = pm.run(midmeas)

        combined_u = UGate(3 * pi / 4, -pi / 2, pi / 2)

        expected = QuantumCircuit(3, 1)
        expected.cx(0, 1)
        expected.compose(combined_u, [0], inplace=True)
        expected.delay(600, 0)
        expected.rx(pi / 4, 0)
        expected.delay(600, 0)
        expected.delay(700, 2)
        expected.cx(1, 2)
        expected.delay(1000, 1)
        expected.measure(2, 0)
        expected.cx(1, 2)
        expected.cx(0, 1)
        expected.delay(700, 2)

        assert midmeas_dd == expected
        # check the absorption into U was done correctly
        assert (Operator(XGate()).equiv(Operator(UGate(3 * pi / 4, -pi / 2, pi / 2)) & Operator(RXGate(pi / 4)))) == True

    except AssertionError as e:
        print("An error occurred during the test execution:")
        print(str(e))
        traceback.print_exc()