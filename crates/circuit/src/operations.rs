// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::PI;

use crate::circuit_data::CircuitData;
use crate::imports::{PARAMETER_EXPRESSION, QUANTUM_CIRCUIT};
use crate::{gate_matrix, Qubit};

use ndarray::{aview2, Array2};
use num_complex::Complex64;
use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::{intern, IntoPy, Python};
use smallvec::smallvec;

/// Valid types for an operation field in a CircuitInstruction
///
/// These are basically the types allowed in a QuantumCircuit
#[derive(FromPyObject, Clone, Debug)]
pub enum OperationType {
    Standard(StandardGate),
    Instruction(PyInstruction),
    Gate(PyGate),
    Operation(PyOperation),
}

impl Operation for OperationType {
    fn name(&self) -> &str {
        match self {
            Self::Standard(op) => op.name(),
            Self::Gate(op) => op.name(),
            Self::Instruction(op) => op.name(),
            Self::Operation(op) => op.name(),
        }
    }

    fn num_qubits(&self) -> u32 {
        match self {
            Self::Standard(op) => op.num_qubits(),
            Self::Gate(op) => op.num_qubits(),
            Self::Instruction(op) => op.num_qubits(),
            Self::Operation(op) => op.num_qubits(),
        }
    }
    fn num_clbits(&self) -> u32 {
        match self {
            Self::Standard(op) => op.num_clbits(),
            Self::Gate(op) => op.num_clbits(),
            Self::Instruction(op) => op.num_clbits(),
            Self::Operation(op) => op.num_clbits(),
        }
    }

    fn num_params(&self) -> u32 {
        match self {
            Self::Standard(op) => op.num_params(),
            Self::Gate(op) => op.num_params(),
            Self::Instruction(op) => op.num_params(),
            Self::Operation(op) => op.num_params(),
        }
    }
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self {
            Self::Standard(op) => op.matrix(params),
            Self::Gate(op) => op.matrix(params),
            Self::Instruction(op) => op.matrix(params),
            Self::Operation(op) => op.matrix(params),
        }
    }

    fn control_flow(&self) -> bool {
        match self {
            Self::Standard(op) => op.control_flow(),
            Self::Gate(op) => op.control_flow(),
            Self::Instruction(op) => op.control_flow(),
            Self::Operation(op) => op.control_flow(),
        }
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            Self::Standard(op) => op.definition(params),
            Self::Gate(op) => op.definition(params),
            Self::Instruction(op) => op.definition(params),
            Self::Operation(op) => op.definition(params),
        }
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        match self {
            Self::Standard(op) => op.standard_gate(),
            Self::Gate(op) => op.standard_gate(),
            Self::Instruction(op) => op.standard_gate(),
            Self::Operation(op) => op.standard_gate(),
        }
    }

    fn directive(&self) -> bool {
        match self {
            Self::Standard(op) => op.directive(),
            Self::Gate(op) => op.directive(),
            Self::Instruction(op) => op.directive(),
            Self::Operation(op) => op.directive(),
        }
    }
}

/// Trait for generic circuit operations these define the common attributes
/// needed for something to be addable to the circuit struct
pub trait Operation {
    fn name(&self) -> &str;
    fn num_qubits(&self) -> u32;
    fn num_clbits(&self) -> u32;
    fn num_params(&self) -> u32;
    fn control_flow(&self) -> bool;
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>>;
    fn definition(&self, params: &[Param]) -> Option<CircuitData>;
    fn standard_gate(&self) -> Option<StandardGate>;
    fn directive(&self) -> bool;
}

#[derive(Clone, Debug)]
pub enum Param {
    ParameterExpression(PyObject),
    Float(f64),
    Obj(PyObject),
}

impl<'py> FromPyObject<'py> for Param {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        Ok(
            if b.is_instance(PARAMETER_EXPRESSION.get_bound(b.py()))?
                || b.is_instance(QUANTUM_CIRCUIT.get_bound(b.py()))?
            {
                Param::ParameterExpression(b.clone().unbind())
            } else if let Ok(val) = b.extract::<f64>() {
                Param::Float(val)
            } else {
                Param::Obj(b.clone().unbind())
            },
        )
    }
}

impl IntoPy<PyObject> for Param {
    fn into_py(self, py: Python) -> PyObject {
        match &self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.clone_ref(py),
            Self::Obj(val) => val.clone_ref(py),
        }
    }
}

impl ToPyObject for Param {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.clone_ref(py),
            Self::Obj(val) => val.clone_ref(py),
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[pyclass(module = "qiskit._accelerate.circuit")]
pub enum StandardGate {
    ZGate = 0,
    YGate = 1,
    XGate = 2,
    CZGate = 3,
    CYGate = 4,
    CXGate = 5,
    CCXGate = 6,
    RXGate = 7,
    RYGate = 8,
    RZGate = 9,
    ECRGate = 10,
    SwapGate = 11,
    SXGate = 12,
    GlobalPhaseGate = 13,
    IGate = 14,
    HGate = 15,
    PhaseGate = 16,
    UGate = 17,
    SGate = 18,
    SdgGate = 19,
}

static STANDARD_GATE_NUM_QUBITS: [u32; STANDARD_GATE_SIZE] =
    [1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 2, 2, 1, 0, 1, 1, 1, 1, 1, 1];

static STANDARD_GATE_NUM_PARAMS: [u32; STANDARD_GATE_SIZE] =
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 3, 0, 0];

static STANDARD_GATE_NAME: [&str; STANDARD_GATE_SIZE] = [
    "z",
    "y",
    "x",
    "cz",
    "cy",
    "cx",
    "ccx",
    "rx",
    "ry",
    "rz",
    "ecr",
    "swap",
    "sx",
    "global_phase",
    "id",
    "h",
    "p",
    "u",
    "s",
    "sdg",
];

#[pymethods]
impl StandardGate {
    pub fn copy(&self) -> Self {
        *self
    }

    // These pymethods are for testing:
    pub fn _to_matrix(&self, py: Python, params: Vec<Param>) -> Option<PyObject> {
        self.matrix(&params)
            .map(|x| x.into_pyarray_bound(py).into())
    }

    pub fn _num_params(&self) -> u32 {
        self.num_params()
    }

    pub fn _get_definition(&self, params: Vec<Param>) -> Option<CircuitData> {
        self.definition(&params)
    }

    #[getter]
    pub fn get_num_qubits(&self) -> u32 {
        self.num_qubits()
    }

    #[getter]
    pub fn get_num_clbits(&self) -> u32 {
        self.num_clbits()
    }

    #[getter]
    pub fn get_num_params(&self) -> u32 {
        self.num_params()
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        self.name()
    }
}

// This must be kept up-to-date with `StandardGate` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_GATE_SIZE: usize = 20;

impl Operation for StandardGate {
    fn name(&self) -> &str {
        STANDARD_GATE_NAME[*self as usize]
    }

    fn num_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_QUBITS[*self as usize]
    }

    fn num_params(&self) -> u32 {
        STANDARD_GATE_NUM_PARAMS[*self as usize]
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn control_flow(&self) -> bool {
        false
    }

    fn directive(&self) -> bool {
        false
    }

    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self {
            Self::ZGate => match params {
                [] => Some(aview2(&gate_matrix::Z_GATE).to_owned()),
                _ => None,
            },
            Self::YGate => match params {
                [] => Some(aview2(&gate_matrix::Y_GATE).to_owned()),
                _ => None,
            },
            Self::XGate => match params {
                [] => Some(aview2(&gate_matrix::X_GATE).to_owned()),
                _ => None,
            },
            Self::CZGate => match params {
                [] => Some(aview2(&gate_matrix::CZ_GATE).to_owned()),
                _ => None,
            },
            Self::CYGate => match params {
                [] => Some(aview2(&gate_matrix::CY_GATE).to_owned()),
                _ => None,
            },
            Self::CXGate => match params {
                [] => Some(aview2(&gate_matrix::CX_GATE).to_owned()),
                _ => None,
            },
            Self::CCXGate => match params {
                [] => Some(aview2(&gate_matrix::CCX_GATE).to_owned()),
                _ => None,
            },
            Self::RXGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RYGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::ry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RZGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::ECRGate => match params {
                [] => Some(aview2(&gate_matrix::ECR_GATE).to_owned()),
                _ => None,
            },
            Self::SwapGate => match params {
                [] => Some(aview2(&gate_matrix::SWAP_GATE).to_owned()),
                _ => None,
            },
            Self::SXGate => match params {
                [] => Some(aview2(&gate_matrix::SX_GATE).to_owned()),
                _ => None,
            },
            Self::GlobalPhaseGate => match params {
                [Param::Float(theta)] => {
                    Some(aview2(&gate_matrix::global_phase_gate(*theta)).to_owned())
                }
                _ => None,
            },
            Self::IGate => match params {
                [] => Some(aview2(&gate_matrix::ONE_QUBIT_IDENTITY).to_owned()),
                _ => None,
            },
            Self::HGate => match params {
                [] => Some(aview2(&gate_matrix::H_GATE).to_owned()),
                _ => None,
            },
            Self::PhaseGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::phase_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::UGate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::SGate => match params {
                [] => Some(aview2(&gate_matrix::S_GATE).to_owned()),
                _ => None,
            },
            Self::SdgGate => match params {
                [] => Some(aview2(&gate_matrix::SDG_GATE).to_owned()),
                _ => None,
            },
        }
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            Self::ZGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::YGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![
                                Param::Float(PI),
                                Param::Float(PI / 2.),
                                Param::Float(PI / 2.),
                            ],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI), Param::Float(0.), Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::HGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CYGate => todo!("Add when we have S and S dagger"),
            Self::CXGate => None,
            Self::CCXGate => todo!("Add when we have T and TDagger"),
            Self::RXGate => todo!("Add when we have R"),
            Self::RYGate => todo!("Add when we have R"),
            Self::RZGate => Python::with_gil(|py| -> Option<CircuitData> {
                match &params[0] {
                    Param::Float(theta) => Some(
                        CircuitData::from_standard_gates(
                            py,
                            1,
                            [(
                                Self::PhaseGate,
                                smallvec![Param::Float(*theta)],
                                smallvec![Qubit(0)],
                            )],
                            Param::Float(-0.5 * theta),
                        )
                        .expect("Unexpected Qiskit python bug"),
                    ),
                    Param::ParameterExpression(theta) => Some(
                        CircuitData::from_standard_gates(
                            py,
                            1,
                            [(
                                Self::PhaseGate,
                                smallvec![Param::ParameterExpression(theta.clone_ref(py))],
                                smallvec![Qubit(0)],
                            )],
                            Param::ParameterExpression(
                                theta
                                    .call_method1(py, intern!(py, "__rmul__"), (-0.5,))
                                    .expect("Parameter expression for global phase failed"),
                            ),
                        )
                        .expect("Unexpected Qiskit python bug"),
                    ),
                    Param::Obj(_) => unreachable!(),
                }
            }),
            Self::ECRGate => todo!("Add when we have RZX"),
            Self::SwapGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SXGate => todo!("Add when we have S dagger"),
            Self::GlobalPhaseGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(py, 0, [], params[0].clone())
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::IGate => None,
            Self::HGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI / 2.), Param::Float(0.), Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::PhaseGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(0.), Param::Float(0.), params[0].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::UGate => None,
            Self::SGate => None,
            Self::SdgGate => None,
        }
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        Some(*self)
    }
}

const FLOAT_ZERO: Param = Param::Float(0.0);

/// This class is used to wrap a Python side Instruction that is not in the standard library
#[derive(Clone, Debug)]
#[pyclass(freelist = 20, module = "qiskit._accelerate.circuit")]
pub struct PyInstruction {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub instruction: PyObject,
}

#[pymethods]
impl PyInstruction {
    #[new]
    fn new(op_name: String, qubits: u32, clbits: u32, params: u32, instruction: PyObject) -> Self {
        PyInstruction {
            qubits,
            clbits,
            params,
            op_name,
            instruction,
        }
    }
}

impl Operation for PyInstruction {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        None
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
            match self.instruction.getattr(py, intern!(py, "definition")) {
                Ok(definition) => {
                    let res: Option<PyObject> = definition.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let out: CircuitData =
                                x.getattr(py, intern!(py, "data")).ok()?.extract(py).ok()?;
                            Some(out)
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
            match self.instruction.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

/// This class is used to wrap a Python side Gate that is not in the standard library
#[derive(Clone, Debug)]
#[pyclass(freelist = 20, module = "qiskit._accelerate.circuit")]
pub struct PyGate {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub gate: PyObject,
}

#[pymethods]
impl PyGate {
    #[new]
    fn new(op_name: String, qubits: u32, clbits: u32, params: u32, gate: PyObject) -> Self {
        PyGate {
            qubits,
            clbits,
            params,
            op_name,
            gate,
        }
    }
}

impl Operation for PyGate {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        Python::with_gil(|py| -> Option<Array2<Complex64>> {
            match self.gate.getattr(py, intern!(py, "to_matrix")) {
                Ok(to_matrix) => {
                    let res: Option<PyObject> = to_matrix.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let array: PyReadonlyArray2<Complex64> = x.extract(py).ok()?;
                            Some(array.as_array().to_owned())
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
            match self.gate.getattr(py, intern!(py, "definition")) {
                Ok(definition) => {
                    let res: Option<PyObject> = definition.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let out: CircuitData =
                                x.getattr(py, intern!(py, "data")).ok()?.extract(py).ok()?;
                            Some(out)
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        Python::with_gil(|py| -> Option<StandardGate> {
            match self.gate.getattr(py, intern!(py, "_standard_gate")) {
                Ok(stdgate) => match stdgate.extract(py) {
                    Ok(out_gate) => out_gate,
                    Err(_) => None,
                },
                Err(_) => None,
            }
        })
    }
    fn directive(&self) -> bool {
        false
    }
}

/// This class is used to wrap a Python side Operation that is not in the standard library
#[derive(Clone, Debug)]
#[pyclass(freelist = 20, module = "qiskit._accelerate.circuit")]
pub struct PyOperation {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub operation: PyObject,
}

#[pymethods]
impl PyOperation {
    #[new]
    fn new(op_name: String, qubits: u32, clbits: u32, params: u32, operation: PyObject) -> Self {
        PyOperation {
            qubits,
            clbits,
            params,
            op_name,
            operation,
        }
    }
}

impl Operation for PyOperation {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        None
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        None
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
            match self.operation.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}
