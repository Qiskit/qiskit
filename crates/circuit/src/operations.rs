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
use crate::imports::{DEEPCOPY, PARAMETER_EXPRESSION, QUANTUM_CIRCUIT};
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

impl IntoPy<PyObject> for OperationType {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Self::Standard(gate) => gate.into_py(py),
            Self::Instruction(inst) => inst.into_py(py),
            Self::Gate(gate) => gate.into_py(py),
            Self::Operation(op) => op.into_py(py),
        }
    }
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
    TGate = 20,
    TdgGate = 21,
    SXdgGate = 22,
    ISwapGate = 23,
    XXMinusYYGate = 24,
    XXPlusYYGate = 25,
    U1Gate = 26,
    U2Gate = 27,
    U3Gate = 28,
    CRXGate = 29,
    CRYGate = 30,
    CRZGate = 31,
    RGate = 32,
    CHGate = 33,
    CPhaseGate = 34,
    CSGate = 35,
    CSdgGate = 36,
    CSXGate = 37,
    CSwapGate = 38,
    CUGate = 39,
    CU1Gate = 40,
    CU3Gate = 41,
    C3XGate = 42,
    C3SXGate = 43,
    C4XGate = 44,
    DCXGate = 45,
    CCZGate = 46,
    RCCXGate = 47,
    RC3XGate = 48,
    RXXGate = 49,
    RYYGate = 50,
    RZZGate = 51,
    RZXGate = 52,
}

// TODO: replace all 34s (placeholders) with actual number
static STANDARD_GATE_NUM_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    1, 1, 1, 2, 2, 2, 3, 1, 1, 1, // 0-9
    2, 2, 1, 0, 1, 1, 1, 1, 1, 1, // 10-19
    1, 1, 1, 2, 2, 2, 1, 1, 1, 2, // 20-29
    2, 2, 1, 2, 2, 2, 2, 2, 3, 2, // 30-39
    2, 2, 34, 34, 34, 2, 34, 34, 34, 34, // 40-49
    34, 34, 34, // 50-52
];

// TODO: replace all 34s (placeholders) with actual number
static STANDARD_GATE_NUM_PARAMS: [u32; STANDARD_GATE_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, // 0-9
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, // 10-19
    0, 0, 0, 0, 2, 2, 1, 2, 3, 1, // 20-29
    1, 1, 2, 0, 1, 0, 0, 0, 0, 3, // 30-39
    1, 3, 34, 34, 34, 0, 34, 34, 34, 34, // 40-49
    34, 34, 34, // 50-52
];

static STANDARD_GATE_NAME: [&str; STANDARD_GATE_SIZE] = [
    "z",            // 0
    "y",            // 1
    "x",            // 2
    "cz",           // 3
    "cy",           // 4
    "cx",           // 5
    "ccx",          // 6
    "rx",           // 7
    "ry",           // 8
    "rz",           // 9
    "ecr",          // 10
    "swap",         // 11
    "sx",           // 12
    "global_phase", // 13
    "id",           // 14
    "h",            // 15
    "p",            // 16
    "u",            // 17
    "s",            // 18
    "sdg",          // 19
    "t",            // 20
    "tdg",          // 21
    "sxdg",         // 22
    "iswap",        // 23
    "xx_minus_yy",  // 24
    "xx_plus_yy",   // 25
    "u1",           // 26
    "u2",           // 27
    "u3",           // 28
    "crx",          // 29
    "cry",          // 30
    "crz",          // 31
    "r",            // 32
    "ch",           // 33
    "cp",           // 34
    "cs",           // 35
    "csdg",         // 36
    "csx",          // 37
    "cswap",        // 38
    "cu",           // 39
    "cu1",          // 40
    "cu3",          // 41
    "c3x",          // 42
    "c3sx",         // 43
    "c4x",          // 44
    "dcx",          // 45
    "ccz",          // 46
    "rccx",         // 47
    "rc3x",         // 48
    "rxx",          // 49
    "ryy",          // 50
    "rzz",          // 51
    "rzx",          // 52
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
pub const STANDARD_GATE_SIZE: usize = 53;

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
            Self::CRXGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRYGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::cry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRZGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crz_gate(*theta)).to_owned()),
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
            Self::SXdgGate => match params {
                [] => Some(aview2(&gate_matrix::SXDG_GATE).to_owned()),
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
            Self::TGate => match params {
                [] => Some(aview2(&gate_matrix::T_GATE).to_owned()),
                _ => None,
            },
            Self::TdgGate => match params {
                [] => Some(aview2(&gate_matrix::TDG_GATE).to_owned()),
                _ => None,
            },
            Self::ISwapGate => match params {
                [] => Some(aview2(&gate_matrix::ISWAP_GATE).to_owned()),
                _ => None,
            },
            Self::XXMinusYYGate => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_minus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::XXPlusYYGate => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_plus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::U1Gate => match params[0] {
                Param::Float(val) => Some(aview2(&gate_matrix::u1_gate(val)).to_owned()),
                _ => None,
            },
            Self::U2Gate => match params {
                [Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u2_gate(*phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::U3Gate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::RGate => match params {
                [Param::Float(theta), Param::Float(phi)] => {
                    Some(aview2(&gate_matrix::r_gate(*theta, *phi)).to_owned())
                }
                _ => None,
            },
            Self::CHGate => todo!(),
            Self::CPhaseGate => todo!(),
            Self::CSGate => todo!(),
            Self::CSdgGate => todo!(),
            Self::CSXGate => todo!(),
            Self::CSwapGate => todo!(),
            Self::CUGate | Self::CU1Gate | Self::CU3Gate => todo!(),
            Self::C3XGate | Self::C3SXGate | Self::C4XGate => todo!(),
            Self::DCXGate => match params {
                [] => Some(aview2(&gate_matrix::DCX_GATE).to_owned()),
                _ => None,
            },
            Self::CCZGate => todo!(),
            Self::RCCXGate | Self::RC3XGate => todo!(),
            Self::RXXGate | Self::RYYGate | Self::RZZGate => todo!(),
            Self::RZXGate => todo!(),
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
            Self::CYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::SdgGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::SGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CXGate => None,
            Self::CCXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q2 = smallvec![Qubit(2)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (Self::HGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2.clone()),
                            (Self::TdgGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q0_2.clone()),
                            (Self::TGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2),
                            (Self::TdgGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q0_2),
                            (Self::TGate, smallvec![], q1.clone()),
                            (Self::TGate, smallvec![], q2.clone()),
                            (Self::HGate, smallvec![], q2),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (Self::TGate, smallvec![], q0),
                            (Self::TdgGate, smallvec![], q1),
                            (Self::CXGate, smallvec![], q0_1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::RGate,
                            smallvec![theta.clone(), FLOAT_ZERO],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::RGate,
                            smallvec![theta.clone(), Param::Float(PI / 2.0)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![theta.clone()],
                            smallvec![Qubit(0)],
                        )],
                        multiply_param(theta, -0.5, py),
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CRXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 2.)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(theta, -0.5, py),
                                    Param::Float(0.0),
                                    Param::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(theta, 0.5, py),
                                    Param::Float(-PI / 2.),
                                    Param::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CRYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CRZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RZGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
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
            Self::SXGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (Self::SdgGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SdgGate, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SXdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
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
            Self::SGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U1Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(-PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U2Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI / 2.), params[0].clone(), params[1].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::TGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U3Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::TdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(-PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::ISwapGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SGate, smallvec![], smallvec![Qubit(1)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XXMinusYYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZGate,
                                smallvec![multiply_param(beta, -1.0, py)],
                                q1.clone(),
                            ),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q0.clone()),
                            (Self::SXGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q0.clone()),
                            (Self::SGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                q0.clone(),
                            ),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::SdgGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q0.clone()),
                            (Self::SXdgGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q0),
                            (Self::RZGate, smallvec![beta.clone()], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XXPlusYYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q1_0 = smallvec![Qubit(1), Qubit(0)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::RZGate, smallvec![beta.clone()], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q1.clone()),
                            (Self::SXGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q1.clone()),
                            (Self::SGate, smallvec![], q0.clone()),
                            (Self::CXGate, smallvec![], q1_0.clone()),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q0.clone(),
                            ),
                            (Self::CXGate, smallvec![], q1_0),
                            (Self::SdgGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q1.clone()),
                            (Self::SXdgGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q1),
                            (Self::RZGate, smallvec![multiply_param(beta, -1.0, py)], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta_expr = clone_param(&params[0], py);
                let phi_expr1 = add_param(&params[1], -PI / 2., py);
                let phi_expr2 = multiply_param(&phi_expr1, -1.0, py);
                let defparams = smallvec![theta_expr, phi_expr1, phi_expr2];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(Self::UGate, defparams, smallvec![Qubit(0)])],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CHGate => todo!(),
            Self::CPhaseGate => todo!(),
            Self::CSGate => todo!(),
            Self::CSdgGate => todo!(),
            Self::CSXGate => todo!(),
            Self::CSwapGate => todo!(),
            Self::CUGate => todo!(),
            Self::CU1Gate => todo!(),
            Self::CU3Gate => todo!(),
            Self::C3XGate | Self::C3SXGate | Self::C4XGate => todo!(),
            Self::DCXGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),

            Self::CCZGate => todo!(),
            Self::RCCXGate | Self::RC3XGate => todo!(),
            Self::RXXGate | Self::RYYGate | Self::RZZGate => todo!(),
            Self::RZXGate => todo!(),
        }
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        Some(*self)
    }
}

const FLOAT_ZERO: Param = Param::Float(0.0);

// Return explictly requested copy of `param`, handling
// each variant separately.
fn clone_param(param: &Param, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta),
        Param::ParameterExpression(theta) => Param::ParameterExpression(theta.clone_ref(py)),
        Param::Obj(_) => unreachable!(),
    }
}

fn multiply_param(param: &Param, mult: f64, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta * mult),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__rmul__"), (mult,))
                .expect("Multiplication of Parameter expression by float failed."),
        ),
        Param::Obj(_) => unreachable!(),
    }
}

fn add_param(param: &Param, summand: f64, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta + summand),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__add__"), (summand,))
                .expect("Sum of Parameter expression and float failed."),
        ),
        Param::Obj(_) => unreachable!(),
    }
}

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

    fn __deepcopy__(&self, py: Python, _memo: PyObject) -> PyResult<Self> {
        Ok(PyInstruction {
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
            instruction: DEEPCOPY.get_bound(py).call1((&self.instruction,))?.unbind(),
        })
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

    fn __deepcopy__(&self, py: Python, _memo: PyObject) -> PyResult<Self> {
        Ok(PyGate {
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
            gate: DEEPCOPY.get_bound(py).call1((&self.gate,))?.unbind(),
        })
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

    fn __deepcopy__(&self, py: Python, _memo: PyObject) -> PyResult<Self> {
        Ok(PyOperation {
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
            operation: DEEPCOPY.get_bound(py).call1((&self.operation,))?.unbind(),
        })
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
