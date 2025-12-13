# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


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


def test_unknown_annotation_keeps_box():
    qc = QuantumCircuit(1)
    box = QuantumCircuit(1)
    box.h(0)
    box.annotations = [{"unknown": True}]
    qc.append(BoxOp(box), [0])

    def safe_only(ann):
        return "safe" in ann

    pass_ = UnrollBoxes(known_annotations=safe_only)
    pm = PassManager([pass_])
    result = pm.run(qc)

    assert any(isinstance(op.operation, BoxOp) for op in result.data)
