// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::PI;
use std::sync::Arc;

use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::bit::QuantumRegister;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;

use crate::equivalence::{EquivalenceError, EquivalenceLibrary};

/// Decomposition of CNOT gate.
///
/// NOTE: this differs to CNOT by a global phase.
/// The matrix returned is given by exp(1j * pi/4) * CNOT
///
/// // Arguments:
/// * `plus_ry` - positive initial RY rotation
/// * `plus_rxx` - positive RXX rotation.
///
/// // Returns
///
/// * The decomposed circuit for CNOT gate (up to global phase).
fn cnot_rxx_decompose(plus_ry: bool, plus_rxx: bool) -> PyResult<CircuitData> {
    let sgn_ry: f64 = match plus_ry {
        true => 1.0,
        false => -1.0,
    };
    let sgn_rxx: f64 = match plus_rxx {
        true => 1.0,
        false => -1.0,
    };
    let mut circuit =
        CircuitData::with_capacity(2, 0, 5, (-sgn_ry * sgn_rxx * PI / 4.0).into()).unwrap();

    circuit.push_standard_gate(
        StandardGate::RY,
        &[Param::Float(sgn_ry * PI / 2.0)],
        &[Qubit(0)],
    )?;
    circuit.push_standard_gate(
        StandardGate::RXX,
        &[Param::Float(sgn_rxx * PI / 2.0)],
        &[Qubit(0), Qubit(1)],
    )?;
    circuit.push_standard_gate(
        StandardGate::RX,
        &[Param::Float(-sgn_rxx * PI / 2.0)],
        &[Qubit(0)],
    )?;
    circuit.push_standard_gate(
        StandardGate::RX,
        &[Param::Float(-sgn_rxx * sgn_ry * PI / 2.0)],
        &[Qubit(1)],
    )?;
    circuit.push_standard_gate(
        StandardGate::RY,
        &[Param::Float(-sgn_ry * PI / 2.0)],
        &[Qubit(0)],
    )?;
    Ok(circuit)
}

fn create_standard_equivalence<P>(
    gate: StandardGate,
    params: &[Param],
    eq_instructions: &[(StandardGate, &[Qubit], &[Param])],
    eq_global_phase: P,
    eq_library: &mut EquivalenceLibrary,
) -> Result<(), EquivalenceError>
where
    P: Into<Param>,
{
    let mut circuit =
        CircuitData::with_capacity(0, 0, eq_instructions.len(), eq_global_phase.into()).unwrap();
    let qreg = QuantumRegister::new_owning("q", gate.num_qubits());
    circuit.add_qreg(qreg, true).unwrap();
    for (operation, qargs, eq_params) in eq_instructions {
        circuit
            .push_standard_gate(*operation, eq_params, qargs)
            .map_err(|e| EquivalenceError::new_err(format!("{}", e)))?;
    }
    eq_library.add_equivalence(&gate.into(), params, circuit)
}

pub fn generate_standard_equivalence_library() -> EquivalenceLibrary {
    let mut equiv = EquivalenceLibrary::new(None);

    // Import existing gate definitions
    //
    // HGate
    //
    //    ┌───┐        ┌────────────┐
    // q: ┤ H ├  ≡  q: ┤ U(π/2,0,π) ├
    //    └───┘        └────────────┘
    create_standard_equivalence(
        StandardGate::H,
        &[],
        &[(
            StandardGate::U,
            &[Qubit(0)],
            &[(PI / 2.0).into(), 0.0.into(), PI.into()],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding H gate equivalence");

    // HGate
    //
    //    ┌───┐        ┌─────────┐
    // q: ┤ H ├  ≡  q: ┤ U2(0,π) ├
    //    └───┘        └─────────┘
    create_standard_equivalence(
        StandardGate::H,
        &[],
        &[(StandardGate::U2, &[Qubit(0)], &[0.0.into(), PI.into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding H gate equivalence");

    // CHGate
    // q_0: ──■──     q_0: ─────────────────■─────────────────────
    //      ┌─┴─┐  ≡       ┌───┐┌───┐┌───┐┌─┴─┐┌─────┐┌───┐┌─────┐
    // q_1: ┤ H ├     q_1: ┤ S ├┤ H ├┤ T ├┤ X ├┤ Tdg ├┤ H ├┤ Sdg ├
    //      └───┘          └───┘└───┘└───┘└───┘└─────┘└───┘└─────┘
    create_standard_equivalence(
        StandardGate::CH,
        &[],
        &[
            (StandardGate::S, &[Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::T, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::Tdg, &[Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::Sdg, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CH gate equivalence");

    // PhaseGate
    //
    //    ┌──────┐        ┌───────┐
    // q: ┤ P(ϴ) ├  ≡  q: ┤ U1(ϴ) ├
    //    └──────┘        └───────┘
    let theta = Arc::new(ParameterExpression::from_symbol(Symbol::new(
        "theta", None, None,
    )));
    create_standard_equivalence(
        StandardGate::Phase,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::U1,
            &[Qubit(0)],
            &[Param::ParameterExpression(theta.clone())],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Phase gate equivalence");

    // PhaseGate
    //
    //    ┌──────┐        ┌──────────┐
    // q: ┤ P(ϴ) ├  ≡  q: ┤ U(0,ϴ,0) ├
    //    └──────┘        └──────────┘
    create_standard_equivalence(
        StandardGate::Phase,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::U,
            &[Qubit(0)],
            &[
                0.0.into(),
                0.0.into(),
                Param::ParameterExpression(theta.clone()),
            ],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Phase gate equivalence");

    // CPhaseGate
    //                      ┌────────┐
    // q_0: ─■────     q_0: ┤ P(ϴ/2) ├──■───────────────■────────────
    //       │P(ϴ)  ≡       └────────┘┌─┴─┐┌─────────┐┌─┴─┐┌────────┐
    // q_1: ─■────     q_1: ──────────┤ X ├┤ P(-ϴ/2) ├┤ X ├┤ P(ϴ/2) ├
    //                                └───┘└─────────┘└───┘└────────┘
    let theta_div_2 = Arc::new(theta.div(&ParameterExpression::from_f64(2.0)).unwrap());
    let neg_theta_div_2 = Arc::new(
        theta_div_2
            .mul(&ParameterExpression::from_f64(-1.0))
            .unwrap(),
    );
    create_standard_equivalence(
        StandardGate::CPhase,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (
                StandardGate::Phase,
                &[Qubit(0)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::Phase,
                &[Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::Phase,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CPhase gate equivalence");

    // CPhaseGate
    //
    // q_0: ─■────     q_0: ─■────
    //       │P(ϴ)  ≡        │U1(ϴ)
    // q_1: ─■────     q_1: ─■────
    create_standard_equivalence(
        StandardGate::CPhase,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::CU1,
            &[Qubit(0), Qubit(1)],
            &[Param::ParameterExpression(theta.clone())],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CPhase gate equivalence");

    // CPhaseGate
    //
    //                  global phase: ϴ/4
    //                                  ┌─────────┐
    //  q_0: ─■────     q_0: ─■─────────┤ Rz(ϴ/2) ├
    //        │P(ϴ)  ≡        │ZZ(-ϴ/2) ├─────────┤
    //  q_1: ─■────     q_1: ─■─────────┤ Rz(ϴ/2) ├
    //                                  └─────────┘
    let theta_div_4 = Arc::new(theta.div(&ParameterExpression::from_f64(4.0)).unwrap());
    create_standard_equivalence(
        StandardGate::CPhase,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (
                StandardGate::RZZ,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
        ],
        Param::ParameterExpression(theta_div_4.clone()),
        &mut equiv,
    )
    .expect("Error while adding CPhase gate equivalence");

    // RGate
    //
    //    ┌────────┐        ┌──────────────────────┐
    // q: ┤ R(ϴ,φ) ├  ≡  q: ┤ U(ϴ,φ - π/2,π/2 - φ) ├
    //    └────────┘        └──────────────────────┘
    let phi = Arc::new(ParameterExpression::from_symbol(Symbol::new(
        "phi", None, None,
    )));
    // π/2
    let pi_div_2 = Arc::new(ParameterExpression::from_f64(PI / 2.0));
    // PHI - π/2
    let phi_sub_pi_2 = Arc::new(phi.sub(&pi_div_2).unwrap());
    // π/2 - PHI
    let pi_2_sub_phi = Arc::new(pi_div_2.sub(&phi).unwrap());
    create_standard_equivalence(
        StandardGate::R,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(phi.clone()),
        ],
        &[(
            StandardGate::U,
            &[Qubit(0)],
            &[
                Param::ParameterExpression(theta.clone()),
                Param::ParameterExpression(phi_sub_pi_2),
                Param::ParameterExpression(pi_2_sub_phi),
            ],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding R gate equivalence");

    // IGate
    //    ┌───┐        ┌──────────┐
    // q: ┤ I ├  ≡  q: ┤ U(0,0,0) ├
    //    └───┘        └──────────┘
    create_standard_equivalence(
        StandardGate::I,
        &[],
        &[(
            StandardGate::U,
            &[Qubit(0)],
            &[0.0.into(), 0.0.into(), 0.0.into()],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding I gate equivalence");

    // IGate
    //    ┌───┐        ┌───────┐
    // q: ┤ I ├  ≡  q: ┤ Rx(0) ├
    //    └───┘        └───────┘
    create_standard_equivalence(
        StandardGate::I,
        &[],
        &[(StandardGate::RX, &[Qubit(0)], &[0.0.into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding I gate equivalence");

    // IGate
    //    ┌───┐        ┌───────┐
    // q: ┤ I ├  ≡  q: ┤ Ry(0) ├
    //    └───┘        └───────┘
    create_standard_equivalence(
        StandardGate::I,
        &[],
        &[(StandardGate::RY, &[Qubit(0)], &[0.0.into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding I gate equivalence");

    // IGate
    //    ┌───┐        ┌───────┐
    // q: ┤ I ├  ≡  q: ┤ Rz(0) ├
    //    └───┘        └───────┘
    create_standard_equivalence(
        StandardGate::I,
        &[],
        &[(StandardGate::RZ, &[Qubit(0)], &[0.0.into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding I gate equivalence");

    // RCCXGate
    //
    //      ┌───────┐
    // q_0: ┤0      ├     q_0: ────────────────────────■────────────────────────
    //      │       │                                  │
    // q_1: ┤1 Rccx ├  ≡  q_1: ────────────■───────────┼─────────■──────────────
    //      │       │          ┌───┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌───┐
    // q_2: ┤2      ├     q_2: ┤ H ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ H ├
    //      └───────┘          └───┘└───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘
    create_standard_equivalence(
        StandardGate::RCCX,
        &[],
        &[
            (StandardGate::H, &[Qubit(2)], &[]),
            (StandardGate::T, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(1), Qubit(2)], &[]),
            (StandardGate::Tdg, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(2)], &[]),
            (StandardGate::T, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(1), Qubit(2)], &[]),
            (StandardGate::Tdg, &[Qubit(2)], &[]),
            (StandardGate::H, &[Qubit(2)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RCCX gate equivalence");

    // RXGate
    //
    //    ┌───────┐        ┌───┐┌───────┐┌───┐
    // q: ┤ Rx(ϴ) ├  ≡  q: ┤ H ├┤ Rz(ϴ) ├┤ H ├
    //    └───────┘        └───┘└───────┘└───┘
    create_standard_equivalence(
        StandardGate::RX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RX gate equivalence");

    // RXGate
    //
    //    ┌───────┐        ┌────────┐
    // q: ┤ Rx(ϴ) ├  ≡  q: ┤ R(ϴ,0) ├
    //    └───────┘        └────────┘
    create_standard_equivalence(
        StandardGate::RX,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::R,
            &[Qubit(0)],
            &[Param::ParameterExpression(theta.clone()), 0.0.into()],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RX gate equivalence");

    // CRXGate
    //
    // q_0: ────■────     q_0: ───────■────────────────────■────────────────────
    //      ┌───┴───┐  ≡       ┌───┐┌─┴─┐┌──────────────┐┌─┴─┐┌────────────────┐
    // q_1: ┤ Rx(ϴ) ├     q_1: ┤ S ├┤ X ├┤ U(-ϴ/2,0,0)  ├┤ X ├┤ U(ϴ/2,-π/2,0)  ├
    //      └───────┘          └───┘└───┘└──────────────┘└───┘└────────────────┘
    create_standard_equivalence(
        StandardGate::CRX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::S, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::U,
                &[Qubit(1)],
                &[
                    Param::ParameterExpression(neg_theta_div_2.clone()),
                    0.0.into(),
                    0.0.into(),
                ],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::U,
                &[Qubit(1)],
                &[
                    Param::ParameterExpression(theta_div_2.clone()),
                    (-PI / 2.0).into(),
                    0.0.into(),
                ],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRX gate equivalence");

    // CRXGate
    //
    // q_0: ────■────     q_0: ───────■────────────────■────────────────────
    //      ┌───┴───┐  ≡       ┌───┐┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐┌─────┐
    // q_1: ┤ Rx(ϴ) ├     q_1: ┤ S ├┤ X ├┤ Ry(-ϴ/2) ├┤ X ├┤ Ry(ϴ/2) ├┤ Sdg ├
    //      └───────┘          └───┘└───┘└──────────┘└───┘└─────────┘└─────┘
    create_standard_equivalence(
        StandardGate::CRX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::S, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RY,
                &[Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RY,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (StandardGate::Sdg, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRX gate equivalence");

    // CRX in terms of one RXX
    //                          ┌───┐   ┌────────────┐┌───┐
    // q_0: ────■────   q_0: ───┤ H ├───┤0           ├┤ H ├
    //      ┌───┴───┐ ≡      ┌──┴───┴──┐│  Rxx(-ϴ/2) │└───┘
    // q_1: ┤ Rx(ϴ) ├   q_1: ┤ Rx(ϴ/2) ├┤1           ├─────
    //      └───────┘        └─────────┘└────────────┘
    create_standard_equivalence(
        StandardGate::CRX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (
                StandardGate::RX,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (
                StandardGate::RXX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::H, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRX gate equivalence");

    // CRX to CRZ
    //
    // q_0: ────■────     q_0: ─────────■─────────
    //      ┌───┴───┐  ≡       ┌───┐┌───┴───┐┌───┐
    // q_1: ┤ Rx(ϴ) ├     q_1: ┤ H ├┤ Rz(ϴ) ├┤ H ├
    //      └───────┘          └───┘└───────┘└───┘
    create_standard_equivalence(
        StandardGate::CRX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (
                StandardGate::CRZ,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRX gate equivalence");

    // RXXGate
    //
    //      ┌─────────┐          ┌───┐                   ┌───┐
    // q_0: ┤0        ├     q_0: ┤ H ├──■─────────────■──┤ H ├
    //      │  Rxx(ϴ) │  ≡       ├───┤┌─┴─┐┌───────┐┌─┴─┐├───┤
    // q_1: ┤1        ├     q_1: ┤ H ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ H ├
    //      └─────────┘          └───┘└───┘└───────┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::RXX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RXX gate equivalence");

    // RXX to RZX
    //      ┌─────────┐        ┌───┐┌─────────┐┌───┐
    // q_0: ┤0        ├   q_0: ┤ H ├┤0        ├┤ H ├
    //      │  Rxx(ϴ) │ ≡      └───┘│  Rzx(ϴ) │└───┘
    // q_1: ┤1        ├   q_1: ─────┤1        ├─────
    //      └─────────┘             └─────────┘
    create_standard_equivalence(
        StandardGate::RXX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (
                StandardGate::RZX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RXX gate equivalence");

    // RXX to RZX
    //      ┌─────────┐        ┌───┐            ┌───┐
    // q_0: ┤0        ├   q_0: ┤ H ├─■──────────┤ H ├
    //      │  Rxx(ϴ) │ ≡      ├───┤ │ZZ(theta) ├───┤
    // q_1: ┤1        ├   q_1: ┤ H ├─■──────────┤ H ├
    //      └─────────┘        └───┘            └───┘
    create_standard_equivalence(
        StandardGate::RXX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
            (
                StandardGate::RZZ,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RXX gate equivalence");

    // RZXGate
    //
    //      ┌─────────┐
    // q_0: ┤0        ├     q_0: ───────■─────────────■───────
    //      │  Rzx(ϴ) │  ≡       ┌───┐┌─┴─┐┌───────┐┌─┴─┐┌───┐
    // q_1: ┤1        ├     q_1: ┤ H ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ H ├
    //      └─────────┘          └───┘└───┘└───────┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::RZX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZX gate equivalence");

    // RZXGate to RZZGate
    //      ┌─────────┐
    // q_0: ┤0        ├     q_0: ──────■───────────
    //      │  Rzx(ϴ) │  ≡       ┌───┐ │ZZ(ϴ) ┌───┐
    // q_1: ┤1        ├     q_1: ┤ H ├─■──────┤ H ├
    //      └─────────┘          └───┘        └───┘
    create_standard_equivalence(
        StandardGate::RZX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (
                StandardGate::RZZ,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZX gate equivalence");

    // RYGate
    //
    //    ┌───────┐        ┌──────────┐
    // q: ┤ Ry(ϴ) ├  ≡  q: ┤ R(ϴ,π/2) ├
    //    └───────┘        └──────────┘
    create_standard_equivalence(
        StandardGate::RY,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::R,
            &[Qubit(0)],
            &[Param::ParameterExpression(theta.clone()), (PI / 2.0).into()],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RY gate equivalence");

    // RYGate
    //
    //    ┌───────┐        ┌─────┐┌───────────┐┌───┐
    // q: ┤ Ry(ϴ) ├  ≡  q: ┤ Sdg ├┤ Rx(theta) ├┤ S ├
    //    └───────┘        └─────┘└───────────┘└───┘
    create_standard_equivalence(
        StandardGate::RY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (
                StandardGate::RX,
                &[Qubit(0)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::S, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RY gate equivalence");

    // CRYGate
    //
    // q_0: ────■────      q_0: ─────────────■────────────────■──
    //      ┌───┴───┐   ≡       ┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐
    // q_1: ┤ Ry(ϴ) ├      q_1: ┤ Ry(ϴ/2) ├┤ X ├┤ Ry(-ϴ/2) ├┤ X ├
    //      └───────┘           └─────────┘└───┘└──────────┘└───┘
    create_standard_equivalence(
        StandardGate::CRY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (
                StandardGate::RY,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RY,
                &[Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRY gate equivalence");

    // CRY to CRZ
    //
    // q_0: ────■────     q_0: ───────────────■────────────────
    //      ┌───┴───┐  ≡       ┌─────────┐┌───┴───┐┌──────────┐
    // q_1: ┤ Ry(ϴ) ├     q_1: ┤ Rx(π/2) ├┤ Rz(ϴ) ├┤ Rx(-π/2) ├
    //      └───────┘          └─────────┘└───────┘└──────────┘
    create_standard_equivalence(
        StandardGate::CRY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::RX, &[Qubit(1)], &[(PI / 2.0).into()]),
            (
                StandardGate::CRZ,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::RX, &[Qubit(1)], &[(-PI / 2.0).into()]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRY gate equivalence");

    // CRY to CRZ
    //
    // q_0: ────■────     q_0: ────────────────────■─────────────────────
    //      ┌───┴───┐  ≡       ┌───┐┌─────────┐┌───┴───┐┌──────────┐┌───┐
    // q_1: ┤ Ry(ϴ) ├     q_1: ┤ H ├┤ Rz(π/2) ├┤ Rx(ϴ) ├┤ Rz(-π/2) ├┤ H ├
    //      └───────┘          └───┘└─────────┘└───────┘└──────────┘└───┘
    create_standard_equivalence(
        StandardGate::CRY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::RZ, &[Qubit(1)], &[(PI / 2.0).into()]),
            (
                StandardGate::CRX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::RZ, &[Qubit(1)], &[(-PI / 2.0).into()]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRY gate equivalence");

    // CRY to RZZ
    //
    // q_0: ────■────    q_0: ────────────────────────■───────────────────
    //      ┌───┴───┐  ≡      ┌─────┐┌─────────┐┌───┐ │ZZ(-ϴ/2) ┌───┐┌───┐
    // q_1: ┤ Ry(ϴ) ├    q_1: ┤ Sdg ├┤ Rx(ϴ/2) ├┤ H ├─■─────────┤ H ├┤ S ├
    //      └───────┘         └─────┘└─────────┘└───┘           └───┘└───┘
    create_standard_equivalence(
        StandardGate::CRY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::Sdg, &[Qubit(1)], &[]),
            (
                StandardGate::RX,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (StandardGate::H, &[Qubit(1)], &[]),
            (
                StandardGate::RZZ,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::S, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRY gate equivalence");

    // RYYGate
    //
    //      ┌─────────┐          ┌─────────┐                   ┌──────────┐
    // q_0: ┤0        ├     q_0: ┤ Rx(π/2) ├──■─────────────■──┤ Rx(-π/2) ├
    //      │  Ryy(ϴ) │  ≡       ├─────────┤┌─┴─┐┌───────┐┌─┴─┐├──────────┤
    // q_1: ┤1        ├     q_1: ┤ Rx(π/2) ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ Rx(-π/2) ├
    //      └─────────┘          └─────────┘└───┘└───────┘└───┘└──────────┘
    create_standard_equivalence(
        StandardGate::RYY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::RX, &[Qubit(0)], &[(PI / 2.0).into()]),
            (StandardGate::RX, &[Qubit(1)], &[(PI / 2.0).into()]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::RX, &[Qubit(1)], &[(-PI / 2.0).into()]),
            (StandardGate::RX, &[Qubit(0)], &[(-PI / 2.0).into()]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RYY gate equivalence");

    // RYYGate
    //
    //      ┌─────────┐          ┌──────┐                   ┌────┐
    // q_0: ┤0        ├     q_0: ┤ √Xdg ├──■─────────────■──┤ √X ├
    //      │  Ryy(ϴ) │  ≡       ├──────┤┌─┴─┐┌───────┐┌─┴─┐├────┤
    // q_1: ┤1        ├     q_1: ┤ √Xdg ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ √X ├
    //      └─────────┘          └──────┘└───┘└───────┘└───┘└────┘
    create_standard_equivalence(
        StandardGate::RYY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::SXdg, &[Qubit(0)], &[]),
            (StandardGate::SXdg, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::SX, &[Qubit(1)], &[]),
            (StandardGate::SX, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RYY gate equivalence");

    // RYY to RZZ
    //
    //      ┌─────────┐          ┌─────────┐            ┌──────────┐
    // q_0: ┤0        ├     q_0: ┤ Rx(π/2) ├─■──────────┤ Rx(-π/2) ├
    //      │  Ryy(ϴ) │  ≡       ├─────────┤ │ZZ(theta) ├──────────┤
    // q_1: ┤1        ├     q_1: ┤ Rx(π/2) ├─■──────────┤ Rx(-π/2) ├
    //      └─────────┘          └─────────┘            └──────────┘
    create_standard_equivalence(
        StandardGate::RYY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::RX, &[Qubit(0)], &[(PI / 2.0).into()]),
            (StandardGate::RX, &[Qubit(1)], &[(PI / 2.0).into()]),
            (
                StandardGate::RZZ,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::RX, &[Qubit(1)], &[(-PI / 2.0).into()]),
            (StandardGate::RX, &[Qubit(0)], &[(-PI / 2.0).into()]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RYY gate equivalence");

    // RYY to RXX
    //
    //      ┌─────────┐          ┌─────┐┌─────────────┐┌───┐
    // q_0: ┤0        ├     q_0: ┤ Sdg ├┤0            ├┤ S ├
    //      │  Ryy(ϴ) │  ≡       ├─────┤│  Rxx(theta) │├───┤
    // q_1: ┤1        ├     q_1: ┤ Sdg ├┤1            ├┤ S ├
    //      └─────────┘          └─────┘└─────────────┘└───┘
    create_standard_equivalence(
        StandardGate::RYY,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::Sdg, &[Qubit(1)], &[]),
            (
                StandardGate::RXX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::S, &[Qubit(1)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RYY gate equivalence");

    // RZGate
    //                  global phase: -ϴ/2
    //    ┌───────┐        ┌──────┐
    // q: ┤ Rz(ϴ) ├  ≡  q: ┤ P(ϴ) ├
    //    └───────┘        └──────┘
    create_standard_equivalence(
        StandardGate::RZ,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::Phase,
            &[Qubit(0)],
            &[Param::ParameterExpression(theta.clone())],
        )],
        Param::ParameterExpression(neg_theta_div_2.clone()),
        &mut equiv,
    )
    .expect("Error while adding RZ gate equivalence");

    // RZGate to RY
    //
    //    ┌───────┐        ┌────┐┌────────┐┌──────┐
    // q: ┤ Rz(ϴ) ├  ≡  q: ┤ √X ├┤ Ry(-ϴ) ├┤ √Xdg ├
    //    └───────┘        └────┘└────────┘└──────┘
    let neg_theta = Arc::new(theta.mul(&ParameterExpression::from_f64(-1.0)).unwrap());
    create_standard_equivalence(
        StandardGate::RZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::SX, &[Qubit(0)], &[]),
            (
                StandardGate::RY,
                &[Qubit(0)],
                &[Param::ParameterExpression(neg_theta.clone())],
            ),
            (StandardGate::SXdg, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZ gate equivalence");

    // RZGate to RX
    //
    //    ┌───────┐        ┌───┐┌───────┐┌───┐
    // q: ┤ Rz(ϴ) ├  ≡  q: ┤ H ├┤ Rx(ϴ) ├┤ H ├
    //    └───────┘        └───┘└───────┘└───┘
    create_standard_equivalence(
        StandardGate::RZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (
                StandardGate::RX,
                &[Qubit(0)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZ gate equivalence");

    // CRZGate
    //
    // q_0: ────■────     q_0: ─────────────■────────────────■──
    //      ┌───┴───┐  ≡       ┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐
    // q_1: ┤ Rz(ϴ) ├     q_1: ┤ Rz(ϴ/2) ├┤ X ├┤ Rz(-ϴ/2) ├┤ X ├
    //      └───────┘          └─────────┘└───┘└──────────┘└───┘
    create_standard_equivalence(
        StandardGate::CRZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRZ gate equivalence");

    // CRZ to CRY
    //
    // q_0: ────■────     q_0: ────────────────■───────────────
    //      ┌───┴───┐  ≡       ┌──────────┐┌───┴───┐┌─────────┐
    // q_1: ┤ Rz(ϴ) ├     q_1: ┤ Rx(-π/2) ├┤ Ry(ϴ) ├┤ Rx(π/2) ├
    //      └───────┘          └──────────┘└───────┘└─────────┘
    create_standard_equivalence(
        StandardGate::CRZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::RX, &[Qubit(1)], &[(-PI / 2.0).into()]),
            (
                StandardGate::CRY,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::RX, &[Qubit(1)], &[(PI / 2.0).into()]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRZ gate equivalence");

    // CRZ to CRX
    //
    // q_0: ────■────     q_0: ─────────■─────────
    //      ┌───┴───┐  ≡       ┌───┐┌───┴───┐┌───┐
    // q_1: ┤ Rz(ϴ) ├     q_1: ┤ H ├┤ Rx(ϴ) ├┤ H ├
    //      └───────┘          └───┘└───────┘└───┘
    create_standard_equivalence(
        StandardGate::CRZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (
                StandardGate::CRX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRZ gate equivalence");

    // CRZ to RZZ
    //
    // q_0: ────■────    q_0: ────────────■────────
    //      ┌───┴───┐  ≡      ┌─────────┐ │ZZ(-ϴ/2)
    // q_1: ┤ Rz(ϴ) ├    q_1: ┤ Rz(ϴ/2) ├─■────────
    //      └───────┘         └─────────┘
    create_standard_equivalence(
        StandardGate::CRZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (
                StandardGate::RZZ,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CRZ gate equivalence");

    // RZZGate
    //
    // q_0: ─■─────     q_0: ──■─────────────■──
    //       │ZZ(ϴ)  ≡       ┌─┴─┐┌───────┐┌─┴─┐
    // q_1: ─■─────     q_1: ┤ X ├┤ Rz(ϴ) ├┤ X ├
    //                       └───┘└───────┘└───┘
    create_standard_equivalence(
        StandardGate::RZZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZZ gate equivalence");

    // RZZ to RXX
    //                      ┌───┐┌─────────────┐┌───┐
    // q_0: ─■─────    q_0: ┤ H ├┤0            ├┤ H ├
    //       │ZZ(ϴ)  ≡      ├───┤│  Rxx(theta) │├───┤
    // q_1: ─■─────    q_1: ┤ H ├┤1            ├┤ H ├
    //                      └───┘└─────────────┘└───┘
    create_standard_equivalence(
        StandardGate::RZZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
            (
                StandardGate::RXX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZZ gate equivalence");

    // RZZ to RZX
    //                          ┌─────────┐
    // q_0: ─■─────   q_0: ─────┤0        ├─────
    //       │ZZ(ϴ) ≡      ┌───┐│  Rzx(ϴ) │┌───┐
    // q_1: ─■─────   q_1: ┤ H ├┤1        ├┤ H ├
    //                     └───┘└─────────┘└───┘
    create_standard_equivalence(
        StandardGate::RZZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (
                StandardGate::RZX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZZ gate equivalence");

    // RZZ to CPhase
    //
    //                 global phase: ϴ/2
    //                                ┌───────┐
    //  q_0: ─■─────   q_0: ─■────────┤ Rz(ϴ) ├
    //        │ZZ(ϴ) ≡       │P(-2*ϴ) ├───────┤
    //  q_1: ─■─────   q_1: ─■────────┤ Rz(ϴ) ├
    //                                └───────┘
    let neg_two_theta = Arc::new(theta.mul(&ParameterExpression::from_f64(-2.0)).unwrap());
    create_standard_equivalence(
        StandardGate::RZZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (
                StandardGate::CPhase,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(neg_two_theta.clone())],
            ),
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
        ],
        Param::ParameterExpression(theta_div_2.clone()),
        &mut equiv,
    )
    .expect("Error while adding RZZ gate equivalence");

    // RZZ to RYY
    //
    //                      ┌──────────┐┌─────────────┐┌─────────┐
    //  q_0: ─■─────   q_0: ┤ Rx(-π/2) ├┤0            ├┤ Rx(π/2) ├
    //        │ZZ(ϴ) ≡      ├──────────┤│  Ryy(theta) │├─────────┤
    //  q_1: ─■─────   q_1: ┤ Rx(-π/2) ├┤1            ├┤ Rx(π/2) ├
    //                      └──────────┘└─────────────┘└─────────┘
    create_standard_equivalence(
        StandardGate::RZZ,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::RX, &[Qubit(0)], &[(-PI / 2.0).into()]),
            (StandardGate::RX, &[Qubit(1)], &[(-PI / 2.0).into()]),
            (
                StandardGate::RYY,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::RX, &[Qubit(0)], &[(PI / 2.0).into()]),
            (StandardGate::RX, &[Qubit(1)], &[(PI / 2.0).into()]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZZ gate equivalence");

    // RZXGate
    //
    //      ┌─────────┐
    // q_0: ┤0        ├     q_0: ───────■─────────────■───────
    //      │  Rzx(ϴ) │  ≡       ┌───┐┌─┴─┐┌───────┐┌─┴─┐┌───┐
    // q_1: ┤1        ├     q_1: ┤ H ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ H ├
    //      └─────────┘          └───┘└───┘└───────┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::RZX,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding RZX gate equivalence");

    // ECRGate
    //
    //      ┌──────┐          ┌───────────┐┌───┐┌────────────┐
    // q_0: ┤0     ├     q_0: ┤0          ├┤ X ├┤0           ├
    //      │  Ecr │  ≡       │  Rzx(π/4) │└───┘│  Rzx(-π/4) │
    // q_1: ┤1     ├     q_1: ┤1          ├─────┤1           ├
    //      └──────┘          └───────────┘     └────────────┘
    create_standard_equivalence(
        StandardGate::ECR,
        &[],
        &[
            (
                StandardGate::RZX,
                &[Qubit(0), Qubit(1)],
                &[(PI / 4.0).into()],
            ),
            (StandardGate::X, &[Qubit(0)], &[]),
            (
                StandardGate::RZX,
                &[Qubit(0), Qubit(1)],
                &[(-PI / 4.0).into()],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding ECR gate equivalence");

    // ECRGate decomposed to Clifford gates (up to a global phase)
    //
    //                  global phase: 7π/4
    //      ┌──────┐         ┌───┐      ┌───┐
    // q_0: ┤0     ├    q_0: ┤ S ├───■──┤ X ├
    //      │  Ecr │  ≡      ├───┴┐┌─┴─┐└───┘
    // q_1: ┤1     ├    q_1: ┤ √X ├┤ X ├─────
    //      └──────┘         └────┘└───┘
    create_standard_equivalence(
        StandardGate::ECR,
        &[],
        &[
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::SX, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::X, &[Qubit(0)], &[]),
        ],
        -PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding ECR gate equivalence");

    // CXGate decomposed using an ECRGate and Clifford 1-qubit gates
    //                global phase: π/4
    // q_0: ──■──          ┌─────┐ ┌──────┐┌───┐
    //      ┌─┴─┐  ≡  q_0: ┤ Sdg ├─┤0     ├┤ X ├
    // q_1: ┤ X ├          ├─────┴┐│  Ecr │└───┘
    //      └───┘     q_1: ┤ √Xdg ├┤1     ├─────
    //                     └──────┘└──────┘
    create_standard_equivalence(
        StandardGate::CX,
        &[],
        &[
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::SXdg, &[Qubit(1)], &[]),
            (StandardGate::ECR, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::X, &[Qubit(0)], &[]),
        ],
        PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding CX gate equivalence");

    // SGate
    //
    //    ┌───┐        ┌────────┐
    // q: ┤ S ├  ≡  q: ┤ P(π/2) ├
    //    └───┘        └────────┘
    create_standard_equivalence(
        StandardGate::S,
        &[],
        &[(StandardGate::Phase, &[Qubit(0)], &[(PI / 2.0).into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding S gate equivalence");

    // SGate
    //
    //    ┌───┐        ┌───┐┌───┐
    // q: ┤ S ├  ≡  q: ┤ T ├┤ T ├
    //    └───┘        └───┘└───┘
    create_standard_equivalence(
        StandardGate::S,
        &[],
        &[
            (StandardGate::T, &[Qubit(0)], &[]),
            (StandardGate::T, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding S gate equivalence");

    // SdgGate
    //
    //    ┌─────┐        ┌─────────┐
    // q: ┤ Sdg ├  ≡  q: ┤ P(-π/2) ├
    //    └─────┘        └─────────┘
    create_standard_equivalence(
        StandardGate::Sdg,
        &[],
        &[(StandardGate::Phase, &[Qubit(0)], &[(-PI / 2.0).into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Sdg gate equivalence");

    // SdgGate
    //
    //    ┌─────┐        ┌───┐┌───┐
    // q: ┤ Sdg ├  ≡  q: ┤ S ├┤ Z ├
    //    └─────┘        └───┘└───┘
    create_standard_equivalence(
        StandardGate::Sdg,
        &[],
        &[
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::Z, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Sdg gate equivalence");

    // SdgGate
    //
    //    ┌─────┐        ┌───┐┌───┐
    // q: ┤ Sdg ├  ≡  q: ┤ Z ├┤ S ├
    //    └─────┘        └───┘└───┘
    create_standard_equivalence(
        StandardGate::Sdg,
        &[],
        &[
            (StandardGate::Z, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Sdg gate equivalence");

    // SdgGate
    //
    //    ┌─────┐        ┌───┐┌───┐┌───┐
    // q: ┤ Sdg ├  ≡  q: ┤ S ├┤ S ├┤ S ├
    //    └─────┘        └───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::Sdg,
        &[],
        &[
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Sdg gate equivalence");

    // SdgGate
    //
    //    ┌─────┐        ┌─────┐┌─────┐
    // q: ┤ Sdg ├  ≡  q: ┤ Tdg ├┤ Tdg ├
    //    └─────┘        └─────┘└─────┘
    create_standard_equivalence(
        StandardGate::Sdg,
        &[],
        &[
            (StandardGate::Tdg, &[Qubit(0)], &[]),
            (StandardGate::Tdg, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding SDG gate equivalence");

    // CSGate
    //
    //                    ┌───┐
    // q_0: ──■──    q_0: ┤ T ├──■───────────■──
    //      ┌─┴─┐         ├───┤┌─┴─┐┌─────┐┌─┴─┐
    // q_1: ┤ S ├ =  q_1: ┤ T ├┤ X ├┤ Tdg ├┤ X ├
    //      └───┘         └───┘└───┘└─────┘└───┘
    create_standard_equivalence(
        StandardGate::CS,
        &[],
        &[
            (StandardGate::T, &[Qubit(0)], &[]),
            (StandardGate::T, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::Tdg, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CS gate equivalence");

    // CSGate
    //
    // q_0: ──■──   q_0: ───────■────────
    //      ┌─┴─┐        ┌───┐┌─┴──┐┌───┐
    // q_1: ┤ S ├ = q_1: ┤ H ├┤ Sx ├┤ H ├
    //      └───┘        └───┘└────┘└───┘
    create_standard_equivalence(
        StandardGate::CS,
        &[],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::CSX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CS gate equivalence");

    // CSdgGate
    //
    //                                     ┌─────┐
    // q_0: ───■───    q_0: ──■─────────■──┤ Tdg ├
    //      ┌──┴──┐         ┌─┴─┐┌───┐┌─┴─┐├─────┤
    // q_1: ┤ Sdg ├ =  q_1: ┤ X ├┤ T ├┤ X ├┤ Tdg ├
    //      └─────┘         └───┘└───┘└───┘└─────┘
    create_standard_equivalence(
        StandardGate::CSdg,
        &[],
        &[
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::T, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::Tdg, &[Qubit(0)], &[]),
            (StandardGate::Tdg, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CSDG gate equivalence");

    // CSdgGate
    //
    // q_0: ───■───   q_0: ───────■────■────────
    //      ┌──┴──┐        ┌───┐┌─┴─┐┌─┴──┐┌───┐
    // q_1: ┤ Sdg ├ = q_1: ┤ H ├┤ X ├┤ Sx ├┤ H ├
    //      └─────┘        └───┘└───┘└────┘└───┘
    create_standard_equivalence(
        StandardGate::CSdg,
        &[],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::CSX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CSDG gate equivalence");

    // SwapGate
    //                        ┌───┐
    // q_0: ─X─     q_0: ──■──┤ X ├──■──
    //       │   ≡       ┌─┴─┐└─┬─┘┌─┴─┐
    // q_1: ─X─     q_1: ┤ X ├──■──┤ X ├
    //                   └───┘     └───┘
    create_standard_equivalence(
        StandardGate::Swap,
        &[],
        &[
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(1), Qubit(0)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding SWAP gate equivalence");

    // SwapGate
    //
    // q_0: ─X─
    //       │   ≡
    // q_1: ─X─
    //
    //      ┌──────────┐┌──────┐   ┌────┐   ┌──────┐┌──────────┐┌──────┐
    // q_0: ┤ Rz(-π/2) ├┤0     ├───┤ √X ├───┤1     ├┤ Rz(-π/2) ├┤0     ├
    //      └──┬────┬──┘│  Ecr │┌──┴────┴──┐│  Ecr │└──┬────┬──┘│  Ecr │
    // q_1: ───┤ √X ├───┤1     ├┤ Rz(-π/2) ├┤0     ├───┤ √X ├───┤1     ├
    //         └────┘   └──────┘└──────────┘└──────┘   └────┘   └──────┘
    //
    create_standard_equivalence(
        StandardGate::Swap,
        &[],
        &[
            (StandardGate::RZ, &[Qubit(0)], &[(-PI / 2.0).into()]),
            (StandardGate::SX, &[Qubit(1)], &[]),
            (StandardGate::ECR, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::RZ, &[Qubit(1)], &[(-PI / 2.0).into()]),
            (StandardGate::SX, &[Qubit(0)], &[]),
            (StandardGate::ECR, &[Qubit(1), Qubit(0)], &[]),
            (StandardGate::RZ, &[Qubit(0)], &[(-PI / 2.0).into()]),
            (StandardGate::SX, &[Qubit(1)], &[]),
            (StandardGate::ECR, &[Qubit(0), Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding SWAP gate equivalence");

    // SwapGate
    //
    // q_0: ─X─
    //       │   ≡
    // q_1: ─X─
    //
    // global phase: 3π/2
    //      ┌────┐   ┌────┐   ┌────┐
    // q_0: ┤ √X ├─■─┤ √X ├─■─┤ √X ├─■─
    //      ├────┤ │ ├────┤ │ ├────┤ │
    // q_1: ┤ √X ├─■─┤ √X ├─■─┤ √X ├─■─
    //      └────┘   └────┘   └────┘
    create_standard_equivalence(
        StandardGate::Swap,
        &[],
        &[
            (StandardGate::SX, &[Qubit(0)], &[]),
            (StandardGate::SX, &[Qubit(1)], &[]),
            (StandardGate::CZ, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::SX, &[Qubit(0)], &[]),
            (StandardGate::SX, &[Qubit(1)], &[]),
            (StandardGate::CZ, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::SX, &[Qubit(0)], &[]),
            (StandardGate::SX, &[Qubit(1)], &[]),
            (StandardGate::CZ, &[Qubit(0), Qubit(1)], &[]),
        ],
        -PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding Swap gate equivalence");

    // iSwapGate
    //
    //      ┌────────┐          ┌───┐┌───┐     ┌───┐
    // q_0: ┤0       ├     q_0: ┤ S ├┤ H ├──■──┤ X ├─────
    //      │  Iswap │  ≡       ├───┤└───┘┌─┴─┐└─┬─┘┌───┐
    // q_1: ┤1       ├     q_1: ┤ S ├─────┤ X ├──■──┤ H ├
    //      └────────┘          └───┘     └───┘     └───┘
    create_standard_equivalence(
        StandardGate::ISwap,
        &[],
        &[
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(1), Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding ISWAP gate equivalence");

    // SXGate
    //               global phase: π/4
    //    ┌────┐        ┌─────┐┌───┐┌─────┐
    // q: ┤ √X ├  ≡  q: ┤ Sdg ├┤ H ├┤ Sdg ├
    //    └────┘        └─────┘└───┘└─────┘
    create_standard_equivalence(
        StandardGate::SX,
        &[],
        &[
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::Sdg, &[Qubit(0)], &[]),
        ],
        PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding SX gate equivalence");

    // HGate decomposed into SXGate and SGate
    //              global phase: -π/4
    //    ┌───┐        ┌───┐┌────┐┌───┐
    // q: ┤ H ├  ≡  q: ┤ S ├┤ √X ├┤ S ├
    //    └───┘        └───┘└────┘└───┘
    create_standard_equivalence(
        StandardGate::H,
        &[],
        &[
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::SX, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
        ],
        -PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding H gate equivalence");

    // SXGate
    //               global phase: π/4
    //    ┌────┐        ┌─────────┐
    // q: ┤ √X ├  ≡  q: ┤ Rx(π/2) ├
    //    └────┘        └─────────┘
    create_standard_equivalence(
        StandardGate::SX,
        &[],
        &[(StandardGate::RX, &[Qubit(0)], &[(PI / 2.0).into()])],
        PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding SX gate equivalence");

    // SXdgGate
    //                 global phase: 7π/4
    //    ┌──────┐        ┌───┐┌───┐┌───┐
    // q: ┤ √Xdg ├  ≡  q: ┤ S ├┤ H ├┤ S ├
    //    └──────┘        └───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::SXdg,
        &[],
        &[
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
        ],
        -PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding SXDG gate equivalence");

    // HGate decomposed into SXdgGate and SdgGate
    //              global phase: π/4
    //    ┌───┐        ┌─────┐┌──────┐┌─────┐
    // q: ┤ H ├  ≡  q: ┤ Sdg ├┤ √Xdg ├┤ Sdg ├
    //    └───┘        └─────┘└──────┘└─────┘
    create_standard_equivalence(
        StandardGate::H,
        &[],
        &[
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::SXdg, &[Qubit(0)], &[]),
            (StandardGate::Sdg, &[Qubit(0)], &[]),
        ],
        PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding H gate equivalence");

    // SXdgGate
    //                 global phase: 7π/4
    //    ┌──────┐        ┌──────────┐
    // q: ┤ √Xdg ├  ≡  q: ┤ Rx(-π/2) ├
    //    └──────┘        └──────────┘
    create_standard_equivalence(
        StandardGate::SXdg,
        &[],
        &[(StandardGate::RX, &[Qubit(0)], &[(-PI / 2.0).into()])],
        -PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding SXDG gate equivalence");

    // CSXGate
    //
    // q_0: ──■───     q_0: ───────■───────
    //      ┌─┴──┐  ≡       ┌───┐┌─┴─┐┌───┐
    // q_1: ┤ Sx ├     q_1: ┤ H ├┤ S ├┤ H ├
    //      └────┘          └───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::CSX,
        &[],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::CS, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CSX gate equivalence");

    // CSXGate
    //                 global phase: π/4
    //                      ┌───┐┌───────────┐  ┌─────┐  ┌───┐
    // q_0: ──■───     q_0: ┤ X ├┤0          ├──┤ Tdg ├──┤ X ├
    //      ┌─┴──┐  ≡       └───┘│  Rzx(π/4) │┌─┴─────┴─┐└───┘
    // q_1: ┤ Sx ├     q_1: ─────┤1          ├┤ Rx(π/4) ├─────
    //      └────┘               └───────────┘└─────────┘
    create_standard_equivalence(
        StandardGate::CSX,
        &[],
        &[
            (StandardGate::X, &[Qubit(0)], &[]),
            (
                StandardGate::RZX,
                &[Qubit(0), Qubit(1)],
                &[(PI / 4.0).into()],
            ),
            (StandardGate::Tdg, &[Qubit(0)], &[]),
            (StandardGate::X, &[Qubit(0)], &[]),
            (StandardGate::RX, &[Qubit(1)], &[(PI / 4.0).into()]),
        ],
        PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding CSX gate equivalence");

    // DCXGate
    //
    //      ┌──────┐               ┌───┐
    // q_0: ┤0     ├     q_0: ──■──┤ X ├
    //      │  Dcx │  ≡       ┌─┴─┐└─┬─┘
    // q_1: ┤1     ├     q_1: ┤ X ├──■──
    //      └──────┘          └───┘
    create_standard_equivalence(
        StandardGate::DCX,
        &[],
        &[
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(1), Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding DCX gate equivalence");

    // DCXGate
    //
    //      ┌──────┐           ┌───┐ ┌─────┐┌────────┐
    // q_0: ┤0     ├     q_0: ─┤ H ├─┤ Sdg ├┤0       ├─────
    //      │  Dcx │  ≡       ┌┴───┴┐└─────┘│  Iswap │┌───┐
    // q_1: ┤1     ├     q_1: ┤ Sdg ├───────┤1       ├┤ H ├
    //      └──────┘          └─────┘       └────────┘└───┘
    create_standard_equivalence(
        StandardGate::DCX,
        &[],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::Sdg, &[Qubit(1)], &[]),
            (StandardGate::ISwap, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding DCX gate equivalence");

    // CSwapGate
    //
    // q_0: ─■─     q_0: ───────■───────
    //       │           ┌───┐  │  ┌───┐
    // q_1: ─X─  ≡  q_1: ┤ X ├──■──┤ X ├
    //       │           └─┬─┘┌─┴─┐└─┬─┘
    // q_2: ─X─     q_2: ──■──┤ X ├──■──
    //                        └───┘
    create_standard_equivalence(
        StandardGate::CSwap,
        &[],
        &[
            (StandardGate::CX, &[Qubit(2), Qubit(1)], &[]),
            (StandardGate::CCX, &[Qubit(0), Qubit(1), Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(2), Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CSWAP gate equivalence");

    // TGate
    //
    //    ┌───┐        ┌────────┐
    // q: ┤ T ├  ≡  q: ┤ P(π/4) ├
    //    └───┘        └────────┘
    create_standard_equivalence(
        StandardGate::T,
        &[],
        &[(StandardGate::Phase, &[Qubit(0)], &[(PI / 4.0).into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding T gate equivalence");

    // TGate
    //
    //    ┌─────┐┌─────┐┌───┐
    // q: ┤ Tdg ├┤ Sdg ├┤ Z ├
    //    └─────┘└─────┘└───┘
    create_standard_equivalence(
        StandardGate::T,
        &[],
        &[
            (StandardGate::Tdg, &[Qubit(0)], &[]),
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::Z, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding T gate equivalence");

    // TdgGate
    //
    //    ┌─────┐        ┌─────────┐
    // q: ┤ Tdg ├  ≡  q: ┤ P(-π/4) ├
    //    └─────┘        └─────────┘
    create_standard_equivalence(
        StandardGate::Tdg,
        &[],
        &[(StandardGate::Phase, &[Qubit(0)], &[(-PI / 4.0).into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding TDG gate equivalence");

    // TdgGate
    //
    //    ┌───┐┌───┐┌───┐
    // q: ┤ T ├┤ S ├┤ Z ├
    //    └───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::Tdg,
        &[],
        &[
            (StandardGate::T, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::Z, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding TDG gate equivalence");

    // UGate
    //
    //    ┌──────────┐        ┌───────────┐
    // q: ┤ U(θ,ϕ,λ) ├  ≡  q: ┤ U3(θ,ϕ,λ) ├
    //    └──────────┘        └───────────┘
    let lam = Arc::new(ParameterExpression::from_symbol(Symbol::new(
        "lam", None, None,
    )));
    create_standard_equivalence(
        StandardGate::U,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
        ],
        &[(
            StandardGate::U3,
            &[Qubit(0)],
            &[
                Param::ParameterExpression(theta.clone()),
                Param::ParameterExpression(phi.clone()),
                Param::ParameterExpression(lam.clone()),
            ],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding U gate equivalence");

    // CUGate
    //                                  ┌──────┐    ┌──────────────┐     »
    // q_0: ──────■───────     q_0: ────┤ P(γ) ├────┤ P(λ/2 + ϕ/2) ├──■──»
    //      ┌─────┴──────┐  ≡       ┌───┴──────┴───┐└──────────────┘┌─┴─┐»
    // q_1: ┤ U(θ,ϕ,λ,γ) ├     q_1: ┤ P(λ/2 - ϕ/2) ├────────────────┤ X ├»
    //      └────────────┘          └──────────────┘                └───┘»
    // «
    // «q_0: ──────────────────────────■────────────────
    // «     ┌──────────────────────┐┌─┴─┐┌────────────┐
    // «q_1: ┤ U(-θ/2,0,-λ/2 - ϕ/2) ├┤ X ├┤ U(θ/2,ϕ,0) ├
    // «     └──────────────────────┘└───┘└────────────┘
    let gamma = Arc::new(ParameterExpression::from_symbol(Symbol::new(
        "gamma", None, None,
    )));
    let lam_plus_phi = Arc::new(lam.add(&phi).unwrap());
    let lam_plus_phi_div_2 = Arc::new(
        lam_plus_phi
            .div(&ParameterExpression::from_f64(2.0))
            .unwrap(),
    );
    let neg_lam_plus_phi_div_2 = Arc::new(
        lam_plus_phi_div_2
            .mul(&ParameterExpression::from_f64(-1.0))
            .unwrap(),
    );
    let lam_sub_phi = Arc::new(lam.sub(&phi).unwrap());
    let lam_sub_phi_div_2 = Arc::new(
        lam_sub_phi
            .div(&ParameterExpression::from_f64(2.0))
            .unwrap(),
    );

    create_standard_equivalence(
        StandardGate::CU,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
            Param::ParameterExpression(gamma.clone()),
        ],
        &[
            (
                StandardGate::Phase,
                &[Qubit(0)],
                &[Param::ParameterExpression(gamma.clone())],
            ),
            (
                StandardGate::Phase,
                &[Qubit(0)],
                &[Param::ParameterExpression(lam_plus_phi_div_2.clone())],
            ),
            (
                StandardGate::Phase,
                &[Qubit(1)],
                &[Param::ParameterExpression(lam_sub_phi_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::U,
                &[Qubit(1)],
                &[
                    Param::ParameterExpression(neg_theta_div_2.clone()),
                    0.0.into(),
                    Param::ParameterExpression(neg_lam_plus_phi_div_2.clone()),
                ],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::U,
                &[Qubit(1)],
                &[
                    Param::ParameterExpression(theta_div_2.clone()),
                    Param::ParameterExpression(phi.clone()),
                    0.0.into(),
                ],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CU gate equivalence");

    // CUGate
    //                              ┌──────┐
    // q_0: ──────■───────     q_0: ┤ P(γ) ├──────■──────
    //      ┌─────┴──────┐  ≡       └──────┘┌─────┴─────┐
    // q_1: ┤ U(θ,ϕ,λ,γ) ├     q_1: ────────┤ U3(θ,ϕ,λ) ├
    //      └────────────┘                  └───────────┘
    create_standard_equivalence(
        StandardGate::CU,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
            Param::ParameterExpression(gamma.clone()),
        ],
        &[
            (
                StandardGate::Phase,
                &[Qubit(0)],
                &[Param::ParameterExpression(gamma.clone())],
            ),
            (
                StandardGate::CU3,
                &[Qubit(0), Qubit(1)],
                &[
                    Param::ParameterExpression(theta.clone()),
                    Param::ParameterExpression(phi.clone()),
                    Param::ParameterExpression(lam.clone()),
                ],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CU gate equivalence");

    // U1Gate
    //
    //    ┌───────┐        ┌───────────┐
    // q: ┤ U1(θ) ├  ≡  q: ┤ U3(0,0,θ) ├
    //    └───────┘        └───────────┘
    create_standard_equivalence(
        StandardGate::U1,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::U3,
            &[Qubit(0)],
            &[
                0.0.into(),
                0.0.into(),
                Param::ParameterExpression(theta.clone()),
            ],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding U1 gate equivalence");

    // U1Gate
    //
    //    ┌───────┐        ┌──────┐
    // q: ┤ U1(θ) ├  ≡  q: ┤ P(0) ├
    //    └───────┘        └──────┘
    create_standard_equivalence(
        StandardGate::U1,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::Phase,
            &[Qubit(0)],
            &[Param::ParameterExpression(theta.clone())],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding U1 gate equivalence");

    // U1Gate
    //                  global phase: θ/2
    //    ┌───────┐        ┌───────┐
    // q: ┤ U1(θ) ├  ≡  q: ┤ Rz(θ) ├
    //    └───────┘        └───────┘
    create_standard_equivalence(
        StandardGate::U1,
        &[Param::ParameterExpression(theta.clone())],
        &[(
            StandardGate::RZ,
            &[Qubit(0)],
            &[Param::ParameterExpression(theta.clone())],
        )],
        Param::ParameterExpression(theta_div_2.clone()),
        &mut equiv,
    )
    .expect("Error while adding U1 gate equivalence");

    // CU1Gate
    //                       ┌────────┐
    // q_0: ─■─────     q_0: ┤ P(θ/2) ├──■───────────────■────────────
    //       │U1(θ)  ≡       └────────┘┌─┴─┐┌─────────┐┌─┴─┐┌────────┐
    // q_1: ─■─────     q_1: ──────────┤ X ├┤ P(-θ/2) ├┤ X ├┤ P(θ/2) ├
    //                                 └───┘└─────────┘└───┘└────────┘
    create_standard_equivalence(
        StandardGate::CU1,
        &[Param::ParameterExpression(theta.clone())],
        &[
            (
                StandardGate::Phase,
                &[Qubit(0)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::Phase,
                &[Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::Phase,
                &[Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CU1 gate equivalence");

    // U2Gate
    //
    //    ┌─────────┐        ┌────────────┐
    // q: ┤ U2(ϕ,λ) ├  ≡  q: ┤ U(π/2,ϕ,λ) ├
    //    └─────────┘        └────────────┘
    create_standard_equivalence(
        StandardGate::U2,
        &[
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
        ],
        &[(
            StandardGate::U,
            &[Qubit(0)],
            &[
                (PI / 2.0).into(),
                Param::ParameterExpression(phi.clone()),
                Param::ParameterExpression(lam.clone()),
            ],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding U2 gate equivalence");

    // U2Gate
    //                    global phase: 7π/4
    //    ┌─────────┐        ┌─────────────┐┌────┐┌─────────────┐
    // q: ┤ U2(ϕ,λ) ├  ≡  q: ┤ U1(λ - π/2) ├┤ √X ├┤ U1(ϕ + π/2) ├
    //    └─────────┘        └─────────────┘└────┘└─────────────┘
    let lam_sub_pi_div_2 = Arc::new(lam.sub(&ParameterExpression::from_f64(PI / 2.0)).unwrap());
    let pi_div_2_plus_phi = Arc::new(ParameterExpression::from_f64(PI / 2.0).add(&phi).unwrap());
    create_standard_equivalence(
        StandardGate::U2,
        &[
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
        ],
        &[
            (
                StandardGate::U1,
                &[Qubit(0)],
                &[Param::ParameterExpression(lam_sub_pi_div_2)],
            ),
            (StandardGate::SX, &[Qubit(0)], &[]),
            (
                StandardGate::U1,
                &[Qubit(0)],
                &[Param::ParameterExpression(pi_div_2_plus_phi)],
            ),
        ],
        7.0 * PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding U2 gate equivalence");

    // U3Gate
    //                         global phase: λ/2 + ϕ/2 - π/2
    //    ┌───────────┐        ┌───────┐┌────┐┌───────────┐┌────┐┌────────────┐
    // q: ┤ U3(θ,ϕ,λ) ├  ≡  q: ┤ Rz(λ) ├┤ √X ├┤ Rz(θ + π) ├┤ √X ├┤ Rz(ϕ + 3π) ├
    //    └───────────┘        └───────┘└────┘└───────────┘└────┘└────────────┘
    let tetha_plus_pi = Arc::new(theta.add(&ParameterExpression::from_f64(PI)).unwrap());
    let phi_plus_3pi = Arc::new(ParameterExpression::from_f64(3.0 * PI).add(&phi).unwrap());
    let lam_plus_phi_minus_pi = Arc::new(
        lam_plus_phi
            .sub(&ParameterExpression::from_f64(PI))
            .unwrap(),
    );
    let lam_plus_phi_minus_pi_div_2 = Arc::new(
        lam_plus_phi_minus_pi
            .div(&ParameterExpression::from_f64(2.0))
            .unwrap(),
    );
    create_standard_equivalence(
        StandardGate::U3,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
        ],
        &[
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(lam.clone())],
            ),
            (StandardGate::SX, &[Qubit(0)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(tetha_plus_pi)],
            ),
            (StandardGate::SX, &[Qubit(0)], &[]),
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(phi_plus_3pi)],
            ),
        ],
        Param::ParameterExpression(lam_plus_phi_minus_pi_div_2.clone()),
        &mut equiv,
    )
    .expect("Error while adding U3 gate equivalence");

    // U3Gate
    //
    //    ┌───────────┐        ┌──────────┐
    // q: ┤ U3(θ,ϕ,λ) ├  ≡  q: ┤ U(θ,ϕ,λ) ├
    //    └───────────┘        └──────────┘
    create_standard_equivalence(
        StandardGate::U3,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
        ],
        &[(
            StandardGate::U,
            &[Qubit(0)],
            &[
                Param::ParameterExpression(theta.clone()),
                Param::ParameterExpression(phi.clone()),
                Param::ParameterExpression(lam.clone()),
            ],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding U3 gate equivalence");

    // CU3Gate
    //                             ┌──────────────┐                                  »
    // q_0: ──────■──────     q_0: ┤ P(λ/2 + ϕ/2) ├──■────────────────────────────■──»
    //      ┌─────┴─────┐  ≡       ├──────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐»
    // q_1: ┤ U3(θ,ϕ,λ) ├     q_1: ┤ P(λ/2 - ϕ/2) ├┤ X ├┤ U(-θ/2,0,-λ/2 - ϕ/2) ├┤ X ├»
    //      └───────────┘          └──────────────┘└───┘└──────────────────────┘└───┘»
    // «
    // «q_0: ──────────────
    // «     ┌────────────┐
    // «q_1: ┤ P(θ/2,ϕ,0) ├
    // «     └────────────┘
    let neg_lam_plus_phi_div_2 = Arc::new(
        lam_plus_phi_div_2
            .mul(&ParameterExpression::from_f64(-1.0))
            .unwrap(),
    );
    create_standard_equivalence(
        StandardGate::CU3,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
        ],
        &[
            (
                StandardGate::Phase,
                &[Qubit(0)],
                &[Param::ParameterExpression(lam_plus_phi_div_2.clone())],
            ),
            (
                StandardGate::Phase,
                &[Qubit(1)],
                &[Param::ParameterExpression(lam_sub_phi_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::U,
                &[Qubit(1)],
                &[
                    Param::ParameterExpression(neg_theta_div_2.clone()),
                    0.0.into(),
                    Param::ParameterExpression(neg_lam_plus_phi_div_2.clone()),
                ],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::U,
                &[Qubit(1)],
                &[
                    Param::ParameterExpression(theta_div_2.clone()),
                    Param::ParameterExpression(phi.clone()),
                    0.0.into(),
                ],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CU3 gate equivalence");

    // CU3Gate
    //
    // q_0: ──────■──────     q_0: ──────────■───────────
    //      ┌─────┴─────┐  ≡       ┌─────────┴──────────┐
    // q_1: ┤ U3(θ,ϕ,λ) ├     q_1: ┤ U(theta,phi,lam,0) ├
    //      └───────────┘          └────────────────────┘
    create_standard_equivalence(
        StandardGate::CU3,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(phi.clone()),
            Param::ParameterExpression(lam.clone()),
        ],
        &[(
            StandardGate::CU,
            &[Qubit(0), Qubit(1)],
            &[
                Param::ParameterExpression(theta.clone()),
                Param::ParameterExpression(phi.clone()),
                Param::ParameterExpression(lam.clone()),
                0.0.into(),
            ],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CU3 gate equivalence");

    // XGate
    //
    //    ┌───┐        ┌──────────┐
    // q: ┤ X ├  ≡  q: ┤ U(π,0,π) ├
    //    └───┘        └──────────┘
    create_standard_equivalence(
        StandardGate::X,
        &[],
        &[(
            StandardGate::U,
            &[Qubit(0)],
            &[PI.into(), 0.0.into(), PI.into()],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding X gate equivalence");

    // XGate
    //
    //    ┌───┐        ┌───┐┌───┐┌───┐┌───┐
    // q: ┤ X ├  ≡  q: ┤ H ├┤ S ├┤ S ├┤ H ├
    //    └───┘        └───┘└───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::X,
        &[],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding X gate equivalence");

    // XGate
    //                 global phase: π/2
    //    ┌───┐        ┌───┐┌───┐
    // q: ┤ X ├  ≡  q: ┤ Y ├┤ Z ├
    //    └───┘        └───┘└───┘
    create_standard_equivalence(
        StandardGate::X,
        &[],
        &[
            (StandardGate::Y, &[Qubit(0)], &[]),
            (StandardGate::Z, &[Qubit(0)], &[]),
        ],
        PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding X gate equivalence");

    // CXGate
    //
    // q_0: ──■──
    //      ┌─┴─┐
    // q_1: ┤ X ├  ≡
    //      └───┘
    //
    // global phase: 7π/4
    //      ┌──────────┐┌────────────┐┌─────────┐ ┌─────────┐
    // q_0: ┤ Ry(-π/2) ├┤0           ├┤ Rx(π/2) ├─┤ Ry(π/2) ├
    //      └──────────┘│  Rxx(-π/2) │├─────────┴┐└─────────┘
    // q_1: ────────────┤1           ├┤ Rx(-π/2) ├───────────  ≡
    //                  └────────────┘└──────────┘
    //
    // global phase: π/4
    //      ┌──────────┐┌───────────┐┌──────────┐┌─────────┐
    // q_0: ┤ Ry(-π/2) ├┤0          ├┤ Rx(-π/2) ├┤ Ry(π/2) ├
    //      └──────────┘│  Rxx(π/2) │├─────────┬┘└─────────┘
    // q_1: ────────────┤1          ├┤ Rx(π/2) ├────────────  ≡
    //                  └───────────┘└─────────┘
    //
    // global phase: π/4
    //      ┌─────────┐┌────────────┐┌─────────┐┌──────────┐
    // q_0: ┤ Ry(π/2) ├┤0           ├┤ Rx(π/2) ├┤ Ry(-π/2) ├
    //      └─────────┘│  Rxx(-π/2) │├─────────┤└──────────┘
    // q_1: ───────────┤1           ├┤ Rx(π/2) ├────────────  ≡
    //                 └────────────┘└─────────┘
    //
    // global phase: 7π/4
    //      ┌─────────┐┌───────────┐┌──────────┐┌──────────┐
    // q_0: ┤ Ry(π/2) ├┤0          ├┤ Rx(-π/2) ├┤ Ry(-π/2) ├
    //      └─────────┘│  Rxx(π/2) │├──────────┤└──────────┘
    // q_1: ───────────┤1          ├┤ Rx(-π/2) ├────────────  ≡
    //                 └───────────┘└──────────┘
    for pos_ry in [false, true] {
        for pos_rxx in [false, true] {
            let cx_to_rxx = cnot_rxx_decompose(pos_ry, pos_rxx).unwrap();
            equiv
                .add_equivalence(&StandardGate::CX.into(), &[], cx_to_rxx)
                .expect("Error while adding CX gate equivalence")
        }
    }

    // CXGate
    //
    // q_0: ──■──     q_0: ──────■──────
    //      ┌─┴─┐  ≡       ┌───┐ │ ┌───┐
    // q_1: ┤ X ├     q_1: ┤ H ├─■─┤ H ├
    //      └───┘          └───┘   └───┘
    create_standard_equivalence(
        StandardGate::CX,
        &[],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::CZ, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CX gate equivalence");

    // CXGate
    //                global phase: 3π/4
    //                     ┌───┐     ┌────────┐┌───┐     ┌────────┐┌───┐┌───┐
    // q_0: ──■──     q_0: ┤ H ├─────┤0       ├┤ X ├─────┤0       ├┤ H ├┤ S ├─────
    //      ┌─┴─┐  ≡       ├───┤┌───┐│  Iswap │├───┤┌───┐│  Iswap │├───┤├───┤┌───┐
    // q_1: ┤ X ├     q_1: ┤ X ├┤ H ├┤1       ├┤ X ├┤ H ├┤1       ├┤ S ├┤ X ├┤ H ├
    //      └───┘          └───┘└───┘└────────┘└───┘└───┘└────────┘└───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::CX,
        &[],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::X, &[Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::ISwap, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::X, &[Qubit(0)], &[]),
            (StandardGate::X, &[Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::ISwap, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(1)], &[]),
            (StandardGate::X, &[Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        3.0 * PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding CX gate equivalence");

    // CXGate
    //                global phase: 7π/4
    //                     ┌──────────┐┌───────┐┌──────┐
    // q_0: ──■──     q_0: ┤ Rz(-π/2) ├┤ Ry(π) ├┤0     ├
    //      ┌─┴─┐  ≡       ├─────────┬┘└───────┘│  Ecr │
    // q_1: ┤ X ├     q_1: ┤ Rx(π/2) ├──────────┤1     ├
    //      └───┘          └─────────┘          └──────┘
    create_standard_equivalence(
        StandardGate::CX,
        &[],
        &[
            (StandardGate::RZ, &[Qubit(0)], &[(-PI / 2.0).into()]),
            (StandardGate::RY, &[Qubit(0)], &[PI.into()]),
            (StandardGate::RX, &[Qubit(1)], &[(PI / 2.0).into()]),
            (StandardGate::ECR, &[Qubit(0), Qubit(1)], &[]),
        ],
        -PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding CX gate equivalence");

    // CXGate
    // q_0: ──■──     q_0: ───────────────■───────────────────
    //      ┌─┴─┐  ≡       ┌────────────┐ │P(π) ┌────────────┐
    // q_1: ┤ X ├     q_1: ┤ U(π/2,0,π) ├─■─────┤ U(π/2,0,π) ├
    //      └───┘          └────────────┘       └────────────┘
    create_standard_equivalence(
        StandardGate::CX,
        &[],
        &[
            (
                StandardGate::U,
                &[Qubit(1)],
                &[(PI / 2.0).into(), 0.0.into(), PI.into()],
            ),
            (StandardGate::CPhase, &[Qubit(0), Qubit(1)], &[PI.into()]),
            (
                StandardGate::U,
                &[Qubit(1)],
                &[(PI / 2.0).into(), 0.0.into(), PI.into()],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CX gate equivalence");

    // CXGate
    //                     ┌────────────┐
    // q_0: ──■──     q_0: ┤ U(0,0,π/2) ├────■──────────────────
    //      ┌─┴─┐  ≡       ├────────────┤┌───┴───┐┌────────────┐
    // q_1: ┤ X ├     q_1: ┤ U(π/2,0,π) ├┤ Rz(π) ├┤ U(π/2,0,π) ├
    //      └───┘          └────────────┘└───────┘└────────────┘
    create_standard_equivalence(
        StandardGate::CX,
        &[],
        &[
            (
                StandardGate::U,
                &[Qubit(1)],
                &[(PI / 2.0).into(), 0.0.into(), PI.into()],
            ),
            (
                StandardGate::U,
                &[Qubit(0)],
                &[0.0.into(), 0.0.into(), (PI / 2.0).into()],
            ),
            (StandardGate::CRZ, &[Qubit(0), Qubit(1)], &[PI.into()]),
            (
                StandardGate::U,
                &[Qubit(1)],
                &[(PI / 2.0).into(), 0.0.into(), PI.into()],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CX gate equivalence");

    // CXGate
    //                global phase: π/4
    //                     ┌───────────┐┌─────┐
    // q_0: ──■──     q_0: ┤0          ├┤ Sdg ├─
    //      ┌─┴─┐  ≡       │  Rzx(π/2) │├─────┴┐
    // q_1: ┤ X ├     q_1: ┤1          ├┤ √Xdg ├
    //      └───┘          └───────────┘└──────┘
    create_standard_equivalence(
        StandardGate::CX,
        &[],
        &[
            (
                StandardGate::RZX,
                &[Qubit(0), Qubit(1)],
                &[(PI / 2.0).into()],
            ),
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::SXdg, &[Qubit(1)], &[]),
        ],
        PI / 4.0,
        &mut equiv,
    )
    .expect("Error while adding CX gate equivalence");

    // CCXGate
    //                                                                       ┌───┐
    // q_0: ──■──     q_0: ───────────────────■─────────────────────■────■───┤ T ├───■──
    //        │                               │             ┌───┐   │  ┌─┴─┐┌┴───┴┐┌─┴─┐
    // q_1: ──■──  ≡  q_1: ───────■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├┤ X ├
    //      ┌─┴─┐          ┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐├───┤└┬───┬┘└───┘
    // q_2: ┤ X ├     q_2: ┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├──────
    //      └───┘          └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘ └───┘
    create_standard_equivalence(
        StandardGate::CCX,
        &[],
        &[
            (StandardGate::H, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(1), Qubit(2)], &[]),
            (StandardGate::Tdg, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(2)], &[]),
            (StandardGate::T, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(1), Qubit(2)], &[]),
            (StandardGate::Tdg, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(2)], &[]),
            (StandardGate::T, &[Qubit(1)], &[]),
            (StandardGate::T, &[Qubit(2)], &[]),
            (StandardGate::H, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::T, &[Qubit(0)], &[]),
            (StandardGate::Tdg, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CCX gate equivalence");

    // CCXGate
    //
    // q_0: ──■──     q_0: ────────■─────────────────■────■───
    //        │                  ┌─┴─┐┌─────┐      ┌─┴─┐  │
    // q_1: ──■──  ≡  q_1: ──■───┤ X ├┤ Sdg ├──■───┤ X ├──┼───
    //      ┌─┴─┐          ┌─┴──┐├───┤└─────┘┌─┴──┐├───┤┌─┴──┐
    // q_2: ┤ X ├     q_2: ┤ Sx ├┤ Z ├───────┤ Sx ├┤ Z ├┤ Sx ├
    //      └───┘          └────┘└───┘       └────┘└───┘└────┘
    create_standard_equivalence(
        StandardGate::CCX,
        &[],
        &[
            (StandardGate::CSX, &[Qubit(1), Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::Z, &[Qubit(2)], &[]),
            (StandardGate::Sdg, &[Qubit(1)], &[]),
            (StandardGate::CSX, &[Qubit(1), Qubit(2)], &[]),
            (StandardGate::Z, &[Qubit(2)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::CSX, &[Qubit(0), Qubit(2)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CCX gate equivalence");

    // YGate
    //
    //    ┌───┐        ┌──────────────┐
    // q: ┤ Y ├  ≡  q: ┤ U(π,π/2,π/2) ├
    //    └───┘        └──────────────┘
    create_standard_equivalence(
        StandardGate::Y,
        &[],
        &[(
            StandardGate::U,
            &[Qubit(0)],
            &[PI.into(), (PI / 2.0).into(), (PI / 2.0).into()],
        )],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Y gate equivalence");

    // YGate
    //              global phase: 3π/2
    //    ┌───┐        ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
    // q: ┤ Y ├  ≡  q: ┤ H ├┤ S ├┤ S ├┤ H ├┤ S ├┤ S ├
    //    └───┘        └───┘└───┘└───┘└───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::Y,
        &[],
        &[
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
        ],
        3.0 * PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding Y gate equivalence");

    // YGate
    //              global phase: π/2
    //    ┌───┐        ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
    // q: ┤ Y ├  ≡  q: ┤ S ├┤ S ├┤ H ├┤ S ├┤ S ├┤ H ├
    //    └───┘        └───┘└───┘└───┘└───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::Y,
        &[],
        &[
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::H, &[Qubit(0)], &[]),
        ],
        PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding Y gate equivalence");

    // YGate
    //                 global phase: π/2
    //    ┌───┐        ┌───┐┌───┐
    // q: ┤ Y ├  ≡  q: ┤ Z ├┤ X ├
    //    └───┘        └───┘└───┘
    create_standard_equivalence(
        StandardGate::Y,
        &[],
        &[
            (StandardGate::Z, &[Qubit(0)], &[]),
            (StandardGate::X, &[Qubit(0)], &[]),
        ],
        PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding Y gate equivalence");

    // CYGate
    //
    // q_0: ──■──     q_0: ─────────■───────
    //      ┌─┴─┐  ≡       ┌─────┐┌─┴─┐┌───┐
    // q_1: ┤ Y ├     q_1: ┤ Sdg ├┤ X ├┤ S ├
    //      └───┘          └─────┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::CY,
        &[],
        &[
            (StandardGate::Sdg, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::S, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CY gate equivalence");

    // ZGate
    //
    //    ┌───┐        ┌──────┐
    // q: ┤ Z ├  ≡  q: ┤ P(π) ├
    //    └───┘        └──────┘
    create_standard_equivalence(
        StandardGate::Z,
        &[],
        &[(StandardGate::Phase, &[Qubit(0)], &[PI.into()])],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Z gate equivalence");

    // ZGate
    //
    //    ┌───┐        ┌───┐┌───┐
    // q: ┤ Z ├  ≡  q: ┤ S ├┤ S ├
    //    └───┘        └───┘└───┘
    create_standard_equivalence(
        StandardGate::Z,
        &[],
        &[
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::S, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Z gate equivalence");

    // ZGate
    //
    //    ┌───┐        ┌─────┐┌─────┐
    // q: ┤ Z ├  ≡  q: ┤ Sdg ├┤ Sdg ├
    //    └───┘        └─────┘└─────┘
    create_standard_equivalence(
        StandardGate::Z,
        &[],
        &[
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::Sdg, &[Qubit(0)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding Z gate equivalence");

    // ZGate
    //                 global phase: π/2
    //    ┌───┐        ┌───┐┌───┐
    // q: ┤ Z ├  ≡  q: ┤ X ├┤ Y ├
    //    └───┘        └───┘└───┘
    create_standard_equivalence(
        StandardGate::Z,
        &[],
        &[
            (StandardGate::X, &[Qubit(0)], &[]),
            (StandardGate::Y, &[Qubit(0)], &[]),
        ],
        PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding Z gate equivalence");

    // CZGate
    //
    // q_0: ─■─     q_0: ───────■───────
    //       │   ≡       ┌───┐┌─┴─┐┌───┐
    // q_1: ─■─     q_1: ┤ H ├┤ X ├┤ H ├
    //                   └───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::CZ,
        &[],
        &[
            (StandardGate::H, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::H, &[Qubit(1)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CZ gate equivalence");

    // CCZGate
    //
    // q_0: ─■─   q_0: ───────■───────
    //       │                │
    // q_1: ─■─ = q_1: ───────■───────
    //       │         ┌───┐┌─┴─┐┌───┐
    // q_2: ─■─   q_2: ┤ H ├┤ X ├┤ H ├
    //                 └───┘└───┘└───┘
    create_standard_equivalence(
        StandardGate::CCZ,
        &[],
        &[
            (StandardGate::H, &[Qubit(2)], &[]),
            (StandardGate::CCX, &[Qubit(0), Qubit(1), Qubit(2)], &[]),
            (StandardGate::H, &[Qubit(2)], &[]),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding CCZ gate equivalence");

    // XGate
    //              global phase: π/2
    //    ┌───┐        ┌───────┐
    // q: ┤ X ├  ≡  q: ┤ Rx(π) ├
    //    └───┘        └───────┘
    create_standard_equivalence(
        StandardGate::X,
        &[],
        &[(StandardGate::RX, &[Qubit(0)], &[PI.into()])],
        PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding X gate equivalence");

    // YGate
    //              global phase: π/2
    //    ┌───┐        ┌───────┐
    // q: ┤ Y ├  ≡  q: ┤ Ry(π) ├
    //    └───┘        └───────┘
    create_standard_equivalence(
        StandardGate::Y,
        &[],
        &[(StandardGate::RY, &[Qubit(0)], &[PI.into()])],
        PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding Y gate equivalence");

    // HGate
    //              global phase: π/2
    //    ┌───┐        ┌─────────┐┌───────┐
    // q: ┤ H ├  ≡  q: ┤ Ry(π/2) ├┤ Rx(π) ├
    //    └───┘        └─────────┘└───────┘
    create_standard_equivalence(
        StandardGate::H,
        &[],
        &[
            (StandardGate::RY, &[Qubit(0)], &[(PI / 2.0).into()]),
            (StandardGate::RX, &[Qubit(0)], &[PI.into()]),
        ],
        PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding H gate equivalence");

    // HGate
    //              global phase: π/2
    //    ┌───┐        ┌────────────┐┌────────┐
    // q: ┤ H ├  ≡  q: ┤ R(π/2,π/2) ├┤ R(π,0) ├
    //    └───┘        └────────────┘└────────┘
    create_standard_equivalence(
        StandardGate::H,
        &[],
        &[
            (
                StandardGate::R,
                &[Qubit(0)],
                &[(PI / 2.0).into(), (PI / 2.0).into()],
            ),
            (StandardGate::R, &[Qubit(0)], &[PI.into(), 0.0.into()]),
        ],
        PI / 2.0,
        &mut equiv,
    )
    .expect("Error while adding H gate equivalence");

    // XXPlusYYGate
    // ┌───────────────┐
    // ┤0              ├
    // │  {XX+YY}(θ,β) │
    // ┤1              ├
    // └───────────────┘
    //    ┌───────┐  ┌───┐            ┌───┐┌────────────┐┌───┐  ┌─────┐   ┌────────────┐
    //   ─┤ Rz(β) ├──┤ S ├────────────┤ X ├┤ Ry(-0.5*θ) ├┤ X ├──┤ Sdg ├───┤ Rz(-1.0*β) ├───────────
    // ≡ ┌┴───────┴─┐├───┴┐┌─────────┐└─┬─┘├────────────┤└─┬─┘┌─┴─────┴──┐└──┬──────┬──┘┌─────────┐
    //   ┤ Rz(-π/2) ├┤ √X ├┤ Rz(π/2) ├──■──┤ Ry(-0.5*θ) ├──■──┤ Rz(-π/2) ├───┤ √Xdg ├───┤ Rz(π/2) ├
    //   └──────────┘└────┘└─────────┘     └────────────┘     └──────────┘   └──────┘   └─────────┘
    let beta = Arc::new(ParameterExpression::from_symbol(Symbol::new(
        "beta", None, None,
    )));
    let neg_beta = Arc::new(beta.mul(&ParameterExpression::from_f64(-1.0)).unwrap());
    create_standard_equivalence(
        StandardGate::XXPlusYY,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(beta.clone()),
        ],
        &[
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(beta.clone())],
            ),
            (StandardGate::RZ, &[Qubit(1)], &[(-PI / 2.0).into()]),
            (StandardGate::SX, &[Qubit(1)], &[]),
            (StandardGate::RZ, &[Qubit(1)], &[(PI / 2.0).into()]),
            (StandardGate::S, &[Qubit(0)], &[]),
            (StandardGate::CX, &[Qubit(1), Qubit(0)], &[]),
            (
                StandardGate::RY,
                &[Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (
                StandardGate::RY,
                &[Qubit(0)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(1), Qubit(0)], &[]),
            (StandardGate::Sdg, &[Qubit(0)], &[]),
            (StandardGate::RZ, &[Qubit(1)], &[(-PI / 2.0).into()]),
            (StandardGate::SXdg, &[Qubit(1)], &[]),
            (StandardGate::RZ, &[Qubit(1)], &[(PI / 2.0).into()]),
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(neg_beta.clone())],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding XX_PLUS_YY gate equivalence");

    // XXPlusYYGate
    // ┌───────────────┐
    // ┤0              ├
    // │  {XX+YY}(θ,β) │
    // ┤1              ├
    // └───────────────┘
    //   ┌───────┐┌─────────────┐┌─────────────┐┌────────┐
    //   ┤ Rz(β) ├┤0            ├┤0            ├┤ Rz(-β) ├
    // ≡ └───────┘│  Rxx(0.5*θ) ││  Ryy(0.5*θ) │└────────┘
    //   ─────────┤1            ├┤1            ├──────────
    //            └─────────────┘└─────────────┘
    create_standard_equivalence(
        StandardGate::XXPlusYY,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(beta.clone()),
        ],
        &[
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(beta.clone())],
            ),
            (
                StandardGate::RXX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (
                StandardGate::RYY,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(neg_beta.clone())],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding XX_PLUS_YY gate equivalence");

    // XXMinusYYGate
    // ┌───────────────┐
    // ┤0              ├
    // │  {XX-YY}(θ,β) │
    // ┤1              ├
    // └───────────────┘
    //    ┌──────────┐ ┌────┐┌─────────┐      ┌─────────┐       ┌──────────┐ ┌──────┐┌─────────┐
    //   ─┤ Rz(-π/2) ├─┤ √X ├┤ Rz(π/2) ├──■───┤ Ry(θ/2) ├────■──┤ Rz(-π/2) ├─┤ √Xdg ├┤ Rz(π/2) ├
    // ≡ ┌┴──────────┴┐├───┬┘└─────────┘┌─┴─┐┌┴─────────┴─┐┌─┴─┐└─┬─────┬──┘┌┴──────┤└─────────┘
    //   ┤ Rz(-1.0*β) ├┤ S ├────────────┤ X ├┤ Ry(-0.5*θ) ├┤ X ├──┤ Sdg ├───┤ Rz(β) ├───────────
    //   └────────────┘└───┘            └───┘└────────────┘└───┘  └─────┘   └───────┘
    create_standard_equivalence(
        StandardGate::XXMinusYY,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(beta.clone()),
        ],
        &[
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(neg_beta.clone())],
            ),
            (StandardGate::RZ, &[Qubit(0)], &[(-PI / 2.0).into()]),
            (StandardGate::SX, &[Qubit(0)], &[]),
            (StandardGate::RZ, &[Qubit(0)], &[(PI / 2.0).into()]),
            (StandardGate::S, &[Qubit(1)], &[]),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (
                StandardGate::RY,
                &[Qubit(0)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (
                StandardGate::RY,
                &[Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (StandardGate::CX, &[Qubit(0), Qubit(1)], &[]),
            (StandardGate::Sdg, &[Qubit(1)], &[]),
            (StandardGate::RZ, &[Qubit(0)], &[(-PI / 2.0).into()]),
            (StandardGate::SXdg, &[Qubit(0)], &[]),
            (StandardGate::RZ, &[Qubit(0)], &[(PI / 2.0).into()]),
            (
                StandardGate::RZ,
                &[Qubit(1)],
                &[Param::ParameterExpression(beta.clone())],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding XX_MINUS_YY gate equivalence");

    // XXMinusYYGate
    // ┌───────────────┐
    // ┤0              ├
    // │  {XX-YY}(θ,β) │
    // ┤1              ├
    // └───────────────┘
    //   ┌────────┐┌─────────────┐┌──────────────┐┌───────┐
    //   ┤ Rz(-β) ├┤0            ├┤0             ├┤ Rz(β) ├
    // ≡ └────────┘│  Rxx(0.5*θ) ││  Ryy(-0.5*θ) │└───────┘
    //   ──────────┤1            ├┤1             ├─────────
    //             └─────────────┘└──────────────┘
    create_standard_equivalence(
        StandardGate::XXMinusYY,
        &[
            Param::ParameterExpression(theta.clone()),
            Param::ParameterExpression(beta.clone()),
        ],
        &[
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(neg_beta.clone())],
            ),
            (
                StandardGate::RXX,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(theta_div_2.clone())],
            ),
            (
                StandardGate::RYY,
                &[Qubit(0), Qubit(1)],
                &[Param::ParameterExpression(neg_theta_div_2.clone())],
            ),
            (
                StandardGate::RZ,
                &[Qubit(0)],
                &[Param::ParameterExpression(beta.clone())],
            ),
        ],
        0.0,
        &mut equiv,
    )
    .expect("Error while adding XX_MINUS_YY gate equivalence");

    equiv
}
