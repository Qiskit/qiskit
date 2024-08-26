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

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3::types::PyString;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::imports;
use qiskit_circuit::operations::PyInstruction;
use qiskit_circuit::operations::{add_param, multiply_param, rmultiply_param, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;

use crate::circuit_library::entanglement;
use crate::QiskitError;

const PI2: f64 = PI / 2.;

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);
type StandardInstruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

fn pauli_evolution<'a>(
    py: Python<'a>,
    pauli: &'a str,
    indices: Vec<u32>,
    time: Param,
) -> Box<dyn Iterator<Item = StandardInstruction> + 'a> {
    let qubits = indices.iter().map(|i| Qubit(*i)).collect_vec();
    // get pairs of (pauli, qubit) that are active, i.e. that are not the identity
    let binding = pauli.to_lowercase(); // lowercase for convenience
    let active_paulis = binding
        .as_str()
        .chars()
        .rev() // reverse due to Qiskit's bit ordering convention
        .zip(qubits)
        .filter(|(p, _)| *p != 'i')
        .collect_vec();

    // if there are no paulis, return an empty iterator -- this case here is also why we use
    // a Box<Iterator>, otherwise the compiler will complain that we return empty one time and
    // a chain another time
    if active_paulis.len() == 0 {
        return Box::new(std::iter::empty::<StandardInstruction>());
    }
    // get the basis change: x -> HGate, y -> RXGate(pi/2), z -> nothing
    let basis_change = active_paulis
        .clone()
        .into_iter()
        .filter(|(p, _)| *p != 'z')
        .map(|(p, q)| match p {
            'x' => (StandardGate::HGate, smallvec![], smallvec![q]),
            'y' => (
                StandardGate::RXGate,
                smallvec![Param::Float(PI2)],
                smallvec![q],
            ),
            _ => unreachable!(),
        });
    // get the inverse basis change
    let inverse_basis_change = basis_change.clone().map(|(gate, _, qubit)| match gate {
        StandardGate::HGate => (gate, smallvec![], qubit),
        StandardGate::RXGate => (gate, smallvec![Param::Float(-PI2)], qubit),
        _ => unreachable!(),
    });
    // get the CX chain down to the target rotation qubit
    let chain_down = active_paulis
        .clone()
        .into_iter()
        .map(|(_, q)| q)
        .tuple_windows()
        .map(|(ctrl, target)| (StandardGate::CXGate, smallvec![], smallvec![ctrl, target]));
    // get the CX chain up (cannot use chain_down.rev since tuple_windows is not double ended)
    let chain_up = active_paulis
        .clone()
        .into_iter()
        .rev()
        .map(|(_, q)| q)
        .tuple_windows()
        .map(|(target, ctrl)| (StandardGate::CXGate, smallvec![], smallvec![ctrl, target]));
    // get the RZ gate on the last qubit
    let last_qubit = active_paulis.last().unwrap().1;
    let z_rotation = std::iter::once((
        StandardGate::PhaseGate,
        smallvec![multiply_param(&time, 1.0, py)],
        smallvec![last_qubit],
    ));
    // and finally chain everything together
    Box::new(
        basis_change
            .chain(chain_down)
            .chain(z_rotation)
            .chain(chain_up)
            .chain(inverse_basis_change),
    )
}

#[pyfunction]
#[pyo3(signature = (feature_dimension, parameters, *, reps=1, entanglement=None, paulis=None, alpha=2.0, insert_barriers=false, data_map_func=None))]
pub fn pauli_feature_map(
    py: Python,
    feature_dimension: u32,
    parameters: Bound<PyAny>,
    reps: usize,
    entanglement: Option<&Bound<PyAny>>,
    paulis: Option<&Bound<PySequence>>,
    alpha: f64,
    insert_barriers: bool,
    data_map_func: Option<&Bound<PyAny>>,
) -> PyResult<CircuitData> {
    // Normalize the Pauli strings to a Vec<String>. We first define the default, which is
    // ["z", "zz"], unless we only have a single qubit, in which case we default to ["z"].
    // Then, ``pauli_strings`` is either set to the default, or we try downcasting to a
    // PyString->String, followed by a check whether the feature dimension is large enough
    // for the Pauli (e.g. we cannot implement a "zzz" Pauli on a 2 qubit circuit).
    let default_pauli: Vec<String> = if feature_dimension == 1 {
        vec!["z".to_string()]
    } else {
        vec!["z".to_string(), "zz".to_string()]
    };

    let pauli_strings = paulis.map_or_else(
        || Ok(default_pauli), // use Ok() since we might raise an error in the other arm
        |v| {
            let v = PySequenceMethods::to_list(v)?; // sequence to list
            v.iter()
                .map(|el| {
                    // downcast and check whether it is valid
                    let as_string = (*el
                        .downcast::<PyString>()
                        .expect("Error unpacking the ``paulis`` argument"))
                    .to_string();
                    if as_string.len() > feature_dimension as usize {
                        Err(QiskitError::new_err(format!(
                            "feature_dimension ({}) smaller than the Pauli ({})",
                            feature_dimension, as_string
                        )))
                    } else {
                        Ok(as_string)
                    }
                })
                .collect::<Result<Vec<String>, _>>()
        },
    )?;

    // set the default value for entanglement
    let default = PyString::new_bound(py, "full");
    let entanglement = entanglement.unwrap_or(&default);

    // extract the parameters from the input variable ``parameters``
    let parameter_vector = parameters
        .iter()?
        .map(|el| Param::extract_no_coerce(&el.expect("no idea man")).unwrap())
        .collect_vec();

    let packed_evo = _pauli_feature_map(
        py,
        feature_dimension,
        &pauli_strings,
        &entanglement,
        &parameter_vector,
        reps,
        alpha,
        data_map_func,
        insert_barriers,
    );
    CircuitData::from_packed_operations(py, feature_dimension, 0, packed_evo, Param::Float(0.0))
}

fn _pauli_feature_map<'a>(
    py: Python<'a>,
    feature_dimension: u32,
    pauli_strings: &'a Vec<String>,
    entanglement: &'a Bound<PyAny>,
    parameter_vector: &'a Vec<Param>,
    reps: usize,
    alpha: f64,
    data_map_func: Option<&'a Bound<PyAny>>,
    insert_barriers: bool,
) -> impl Iterator<Item = Instruction> + 'a {
    let barrier_cls = imports::BARRIER.get_bound(py);
    let barrier = barrier_cls
        .call1((feature_dimension,))
        .expect("Could not create Barrier Python-side");
    let barrier_inst = PyInstruction {
        qubits: feature_dimension,
        clbits: 0,
        params: 0,
        op_name: "barrier".to_string(),
        control_flow: false,
        instruction: barrier.into(),
    };
    let packed_barrier = (
        barrier_inst.into(),
        smallvec![],
        (0..feature_dimension).map(|i| Qubit(i)).collect(),
        vec![] as Vec<Clbit>,
    );

    (0..reps).flat_map(move |rep| {
        let h_layer = (0..feature_dimension).map(|i| {
            (
                StandardGate::HGate.into(),
                smallvec![],
                vec![Qubit(i)],
                vec![] as Vec<Clbit>,
            )
        });
        let evo = pauli_strings.into_iter().flat_map(move |pauli| {
            let block_size = pauli.len() as u32;
            let entanglement =
                entanglement::get_entanglement(feature_dimension, block_size, entanglement, rep)
                    .unwrap();
            entanglement.flat_map(move |indices| {
                // get the parameters the evolution is acting on and the corresponding angle,
                // which is given by the data_map_func (we provide a default if not given)
                let active_parameters = indices
                    .as_ref()
                    .unwrap()
                    .into_iter()
                    .map(|i| parameter_vector[*i as usize].clone())
                    .collect();
                let angle = match data_map_func {
                    Some(fun) => fun
                        .call1((active_parameters,))
                        .expect("Failed running ``data_map_func`` Python-side.")
                        .extract()
                        .expect("Failed retrieving the Param"),
                    None => _default_reduce(py, active_parameters),
                };

                // Get the pauli evolution and map it into
                //   (PackedOperation, SmallVec<[Params; 3]>, Vec<Qubit>, Vec<Clbit>)
                // to call CircuitData::from_packed_operations. This is needed since we might
                // have to interject barriers, which are not a standard gate and prevents us
                // from using CircuitData::from_standard_gates.
                pauli_evolution(
                    py,
                    pauli,
                    indices.unwrap(),
                    multiply_param(&angle, alpha, py),
                )
                .map(|(gate, params, qargs)| {
                    (gate.into(), params, qargs.to_vec(), vec![] as Vec<Clbit>)
                })
            })
        });

        // Chain the H layer and evolution together, adding barriers around the evolutions,
        // if necessary. Note that in the last repetition, there's no barrier after the evolution.
        let mut out: Box<dyn Iterator<Item = Instruction>> = Box::new(h_layer);
        if insert_barriers {
            out = Box::new(out.chain(std::iter::once(packed_barrier.clone())));
        }
        out = Box::new(out.chain(evo));
        if insert_barriers && rep < reps - 1 {
            out = Box::new(out.chain(std::iter::once(packed_barrier.clone())));
        }
        out
    })
}

fn _default_reduce<'a>(py: Python<'a>, parameters: Vec<Param>) -> Param {
    if parameters.len() == 1 {
        parameters[0].clone()
    } else {
        parameters.iter().fold(Param::Float(1.0), |acc, param| {
            rmultiply_param(acc, add_param(&multiply_param(param, -1.0, py), PI, py), py)
        })
    }
}
