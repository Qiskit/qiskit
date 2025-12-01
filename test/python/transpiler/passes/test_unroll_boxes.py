"""
Tests for UnrollBoxes transpiler pass.
"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnrollBoxes


@pytest.fixture
def basic_box_circuit():
    qc = QuantumCircuit(1)
    box = QuantumCircuit(1)
    box.h(0)
    qc.append(BoxOp(box), [0])
    return qc


def test_basic_unroll(basic_box_circuit):
    pass_ = UnrollBoxes()
    pm = PassManager([pass_])
    result = pm.run(basic_box_circuit)
    assert any(op.name == "h" for op in result.data)
    assert not any(isinstance(op.operation, BoxOp) for op in result.data)


def test_recursive_unroll():
    inner = QuantumCircuit(1)
    inner.h(0)
    mid = QuantumCircuit(1)
    mid.append(BoxOp(inner), [0])
    outer = QuantumCircuit(1)
    outer.append(BoxOp(mid), [0])

    pass_ = UnrollBoxes()
    pm = PassManager([pass_])
    result = pm.run(outer)
    assert any(op.name == "h" for op in result.data)
    assert not any(isinstance(op.operation, BoxOp) for op in result.data)


def test_annotation_passes():
    qc = QuantumCircuit(1)
    box = QuantumCircuit(1)
    box.h(0)
    box.annotations = [{"safe": True}]
    qc.append(BoxOp(box), [0])

    def safe_only(ann):
        return "safe" in ann

    pass_ = UnrollBoxes(known_annotations=safe_only)
    pm = PassManager([pass_])
    result = pm.run(qc)
    assert any(op.name == "h" for op in result.data)


def test_empty_annotations():
    qc = QuantumCircuit(1)
    box = QuantumCircuit(1)
    box.h(0)
    qc.append(BoxOp(box), [0])

    pass_ = UnrollBoxes()
    pm = PassManager([pass_])
    result = pm.run(qc)
    assert any(op.name == "h" for op in result.data)
