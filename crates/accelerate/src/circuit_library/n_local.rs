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

use hashbrown::HashSet;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::packed_instruction::PackedOperation;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Operation, Param, PyInstruction, StandardGate};
use qiskit_circuit::{imports, Clbit, Qubit};

use itertools::izip;

use crate::QiskitError;

use super::entanglement::get_entanglement;

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

/// Enum to determine the type of circuit layer. This is used in the ParameterLedger.
enum LayerType {
    Rotation,
    Entangle,
}

/// The ParameterLedger stores the parameter objects contained in the n-local circuit.
///
/// Internally, the parameters are stored in a 1-D vector and the ledger keeps track of
/// which indices belong to which layer. For example, a 2-qubit circuit where both the
/// rotation and entanglement layer have 1 block with 2 parameters each, we would store
///    
///     [x0 x1 x2 x3 x4 x5 x6 x7 ....]
///      ----- ----- ----- -----
///      rep0  rep0  rep1  rep2
///      rot   ent   rot   ent
///
/// This allows accessing the parameters by index of the rotation or entanglement layer by means
/// of the ``get_parameters`` method, e.g. as
///
///     let layer: usize = 4;
///     let params_in_that_layer: Vec<Vec<Vec<&Param>>> =
///         ledger.get_parameter(LayerType::Rotation, layer);
///
struct ParameterLedger {
    parameter_vector: Vec<Param>, // all parameters
    rotation_indices: Vec<usize>, // indices where rotation blocks start
    entangle_indices: Vec<usize>,
    rotation_blocks: Vec<(u32, usize)>, // (#blocks per layer, #params per block) for each block
    entangle_blocks: Vec<Vec<(u32, usize)>>, // this might additionally change per layer
}

impl ParameterLedger {
    /// Initialize the ledger n-local input data. This will call Python to create a new
    /// ``ParameterVector`` of adequate size and compute all required indices to access
    /// parameter of a specific layer.
    fn from_nlocal(
        py: Python,
        num_qubits: u32,
        reps: usize,
        entanglement: &Vec<Vec<Vec<Vec<u32>>>>,
        packed_rotations: &[PyRef<Block>],
        packend_entanglings: &[PyRef<Block>],
        skip_final_rotation_layer: bool,
        parameter_prefix: &Bound<PyString>,
    ) -> PyResult<Self> {
        // if we keep the final layer (i.e. skip=false), add parameters on the final layer
        let final_layer_rep = match skip_final_rotation_layer {
            true => 0,
            false => 1,
        };

        // compute the number of parameters used for the rotation layers
        let mut num_rotation_params_per_layer: usize = 0;
        let mut rotation_blocks: Vec<(u32, usize)> = Vec::new();

        for block in packed_rotations {
            let num_blocks = num_qubits / block.num_qubits;
            rotation_blocks.push((num_blocks, block.num_parameters));
            num_rotation_params_per_layer += (num_blocks as usize) * block.num_parameters;
        }

        // compute the number of parameters used for the entanglement layers
        let mut num_entangle_params_per_layer: Vec<usize> = Vec::with_capacity(reps);
        let mut entangle_blocks: Vec<Vec<(u32, usize)>> = Vec::with_capacity(reps);
        for this_entanglement in entanglement {
            let mut this_entangle_blocks: Vec<(u32, usize)> = Vec::new();
            let mut this_num_params: usize = 0;
            for (block, block_entanglement) in packend_entanglings.iter().zip(this_entanglement) {
                let num_blocks = block_entanglement.len();
                this_num_params += num_blocks * block.num_parameters;
                this_entangle_blocks.push((num_blocks as u32, block.num_parameters));
            }
            num_entangle_params_per_layer.push(this_num_params);
            entangle_blocks.push(this_entangle_blocks);
        }

        let num_rotation_params: usize = (reps + final_layer_rep) * num_rotation_params_per_layer;
        let num_entangle_params: usize = num_entangle_params_per_layer.iter().sum();

        // generate a ParameterVector Python-side, containing all parameters, and then
        // map it onto Rust-space parameters
        let num_parameters = num_rotation_params + num_entangle_params;
        let parameter_vector: Vec<Param> = imports::PARAMETER_VECTOR
            .get_bound(py)
            .call1((parameter_prefix, num_parameters))? // get the Python ParameterVector
            .iter()? // iterate over the elements and cast them to Rust Params
            .map(|ob| Param::extract_no_coerce(&ob?))
            .collect::<PyResult<_>>()?;

        // finally, distribute the parameters onto the repetitions and blocks for each
        // rotation layer and entanglement layer
        let mut rotation_indices: Vec<usize> = Vec::with_capacity(reps + final_layer_rep);
        let mut entangle_indices: Vec<usize> = Vec::with_capacity(reps);
        let mut index: usize = 0;
        for num_entangle in num_entangle_params_per_layer {
            rotation_indices.push(index);
            index += num_rotation_params_per_layer;
            entangle_indices.push(index);
            index += num_entangle;
        }
        if !skip_final_rotation_layer {
            rotation_indices.push(index);
        }

        Ok(ParameterLedger {
            parameter_vector,
            rotation_indices,
            entangle_indices,
            rotation_blocks,
            entangle_blocks,
        })
    }

    /// Get the parameters in the rotation or entanglement layer.
    fn get_parameters(&self, kind: LayerType, layer: usize) -> Vec<Vec<Vec<&Param>>> {
        let (mut index, blocks) = match kind {
            LayerType::Rotation => (
                *self
                    .rotation_indices
                    .get(layer)
                    .expect("Out of bounds in rotation_indices."),
                &self.rotation_blocks,
            ),
            LayerType::Entangle => (
                *self
                    .entangle_indices
                    .get(layer)
                    .expect("Out of bounds in entangle_indices."),
                &self.entangle_blocks[layer],
            ),
        };

        let mut parameters: Vec<Vec<Vec<&Param>>> = Vec::new();
        for (num_blocks, num_params) in blocks {
            let mut per_block: Vec<Vec<&Param>> = Vec::new();
            for _ in 0..*num_blocks {
                let block_params: Vec<&Param> = (index..index + num_params)
                    .map(|i| {
                        self.parameter_vector
                            .get(i)
                            .expect("Ran out of parameters!")
                    })
                    .collect();
                index += num_params;
                per_block.push(block_params);
            }
            parameters.push(per_block);
        }

        parameters
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub enum BlockOperation {
    Standard { gate: StandardGate },
    Custom { builder: Py<PyAny> },
}

impl BlockOperation {
    fn assign_parameters(
        &self,
        py: Python,
        params: &[&Param],
    ) -> PyResult<(PackedOperation, SmallVec<[Param; 3]>)> {
        match self {
            Self::Standard { gate } => Ok((
                (*gate).into(),
                SmallVec::from_iter(params.iter().map(|&p| p.clone())),
            )),
            Self::Custom { builder } => {
                println!("Using custom.");
                // the builder returns a Python operation plus the bound parameters
                let py_params =
                    PyList::new_bound(py, params.iter().map(|&p| p.clone().into_py(py))).into_any();

                let job = builder.call1(py, (py_params,))?;
                let result = job.downcast_bound::<PyTuple>(py)?;

                let operation: OperationFromPython = result.get_item(0)?.extract()?;
                let bound_params = result
                    .get_item(1)?
                    .iter()?
                    .map(|ob| Param::extract_no_coerce(&ob?))
                    .collect::<PyResult<Vec<Param>>>()?;

                Ok((
                    operation.operation,
                    SmallVec::<[Param; 3]>::from_vec(bound_params),
                ))
            }
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Block {
    operation: BlockOperation,
    num_qubits: u32,
    num_parameters: usize,
}

#[pymethods]
impl Block {
    #[staticmethod]
    #[pyo3(signature = (gate,))]
    pub fn from_standard_gate(gate: StandardGate) -> Self {
        Block {
            operation: BlockOperation::Standard { gate },
            num_qubits: gate.num_qubits(),
            num_parameters: gate.num_params() as usize,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (num_qubits, num_parameters, builder,))]
    pub fn from_callable(
        py: Python,
        num_qubits: i64,
        num_parameters: i64,
        builder: &Bound<PyAny>,
    ) -> PyResult<Self> {
        if !builder.is_callable() {
            return Err(QiskitError::new_err(
                "builder must be a callable: parameters->(bound gate, bound gate params)",
            ));
        }
        let block = Block {
            operation: BlockOperation::Custom {
                builder: builder.to_object(py),
            },
            num_qubits: num_qubits as u32,
            num_parameters: num_parameters as usize,
        };

        Ok(block)
    }
}

/// Construct a rotation layer.
///
/// Args:
///     num_qubits: The number of qubits in the circuit.
///     packed_rotations: A reference to a vector containing the instructions to insert.
///         This is a vector (sind we can have multiple rotations  operations per layer), with
///         3-tuple elements containing (packed_operation, num_qubits, num_params).
///     parameters: The set of parameter objects to use for the operations. This is a 3x nested
///         vector, organized as operation -> block -> param. That means that for operation ``i``
///         and block ``j``, the parameters are given by ``parameters[i][j]``.
///     skipped_qubits: A hash-set containing which qubits are skipped in the rotation layer.
///
/// Returns:
///     An iterator for the rotation instructions.
fn rotation_layer<'a>(
    py: Python<'a>,
    num_qubits: u32,
    packed_rotations: &'a [PyRef<Block>],
    parameters: Vec<Vec<Vec<&'a Param>>>,
    skipped_qubits: &'a HashSet<u32>,
) -> impl Iterator<Item = PyResult<Instruction>> + 'a {
    packed_rotations
        .iter()
        .zip(parameters)
        .flat_map(move |(block, block_params)| {
            (0..num_qubits)
                .step_by(block.num_qubits as usize)
                .filter(move |start_idx| {
                    skipped_qubits.is_disjoint(&HashSet::from_iter(
                        *start_idx..(*start_idx + block.num_qubits),
                    ))
                })
                .zip(block_params)
                .map(move |(start_idx, params)| {
                    let (bound_op, bound_params) = block
                        .operation
                        .assign_parameters(py, &params)
                        .expect("Failed to rebind");
                    Ok((
                        bound_op,
                        bound_params,
                        (0..block.num_qubits)
                            .map(|i| Qubit(start_idx + i))
                            .collect(),
                        vec![] as Vec<Clbit>,
                    ))
                })
        })
}

/// Construct an entanglement layer.
///
/// Args:
///     entanglement: The entanglement structure in this layer. Given as 3x nested vector, which
///         for each entanglement block contains a vector of connections, where each connection
///         is a vector of indices.
///     packed_entanglings: A reference to a vector containing the instructions to insert.
///         This is a vector (sind we can have multiple entanglement operations per layer), with
///         3-tuple elements containing (packed_operation, num_qubits, num_params).
///     parameters: The set of parameter objects to use for the operations. This is a 3x nested
///         vector, organized as operation -> block -> param. That means that for operation ``i``
///         and block ``j``, the parameters are given by ``parameters[i][j]``.
///
/// Returns:
///     An iterator for the entanglement instructions.
fn entanglement_layer<'a>(
    py: Python<'a>,
    entanglement: &'a Vec<Vec<Vec<u32>>>,
    packend_entanglings: &'a [PyRef<Block>],
    parameters: Vec<Vec<Vec<&'a Param>>>,
) -> impl Iterator<Item = PyResult<Instruction>> + 'a {
    let zipped = izip!(packend_entanglings, parameters, entanglement);
    zipped.flat_map(move |(block, block_params, block_entanglement)| {
        block_entanglement
            .iter()
            .zip(block_params)
            .map(move |(indices, params)| {
                let (bound_op, bound_params) =
                    // rebind_op(py, &packed_op.0, &params).expect("Failed to rebind.");
                    block.operation.assign_parameters(py, &params).expect("Failed to rebind");
                Ok((
                    bound_op,
                    bound_params,
                    indices.iter().map(|i| Qubit(*i)).collect(),
                    vec![] as Vec<Clbit>,
                ))
            })
    })
}

#[pyfunction]
#[pyo3(signature = (num_qubits, reps, rotation_blocks, entanglement_blocks, entanglement, insert_barriers, skip_final_rotation_layer, skip_unentangled_qubits, parameter_prefix))]
#[allow(clippy::too_many_arguments)]
pub fn n_local(
    py: Python,
    num_qubits: i64,
    reps: i64,
    rotation_blocks: Vec<PyRef<Block>>,
    entanglement_blocks: Vec<PyRef<Block>>,
    entanglement: &Bound<PyAny>,
    insert_barriers: bool,
    skip_final_rotation_layer: bool,
    skip_unentangled_qubits: bool,
    parameter_prefix: &Bound<PyString>,
) -> PyResult<CircuitData> {
    // normalize the Python input data
    let num_qubits = num_qubits as u32;
    let reps = reps as usize;

    // map the input gate blocks to Rust objects
    // maybe the deref and clone is redundant and we can just deal with PyRef<Block> throughout
    // let packed_rotations: Vec<Block> = rotation_blocks.iter().map(|b| b.deref().clone()).collect();
    // let packed_entanglings: Vec<Block> = entanglement_blocks
    //     .iter()
    //     .map(|b| b.deref().clone())
    //     .collect();
    let packed_rotations = rotation_blocks;
    let packed_entanglings = entanglement_blocks;

    // Expand the entanglement. This is done since the entanglement can change in between
    // different repetitions/layers, therefore influencing the number of total parameters.
    // To avoid querying the entanglement multiple times, we generate it once only.
    // It is stored as nested vector, being index as reps->blocks->connections, i.e.:
    //
    //     connection: Vec<usize> = entanglement[repetition][entanglement_block][i]
    //
    // TODO make entanglement a separate struct that's easier to understand than a quadruple
    // nested vector
    let entanglement: Vec<Vec<Vec<Vec<u32>>>> = (0..reps)
        .map(|layer| -> PyResult<Vec<Vec<Vec<u32>>>> {
            if entanglement.is_callable() {
                let as_any = entanglement.call1((layer,))?;
                let as_list = as_any.downcast::<PyList>()?;
                unpack_entanglement(num_qubits, layer, as_list, &packed_entanglings)
            } else {
                let as_list = entanglement.downcast::<PyList>()?;
                unpack_entanglement(num_qubits, layer, as_list, &packed_entanglings)
            }

            // let ent_layer = ent_layer.downcast::<PyList>()?;

            // packed_entanglings
            //     .iter()
            //     .zip(ent_layer.iter())
            //     .map(|(block, ent)| -> PyResult<Vec<Vec<u32>>> {
            //         get_entanglement(num_qubits, block.num_qubits, &ent, layer)?.collect()
            //     })
            //     .collect()
        })
        .collect::<PyResult<_>>()?;

    let ledger = ParameterLedger::from_nlocal(
        py,
        num_qubits,
        reps,
        &entanglement,
        &packed_rotations,
        &packed_entanglings,
        skip_final_rotation_layer,
        parameter_prefix,
    )?;

    // Compute the qubits that are skipped in the rotation layer. If this is set,
    // we skip qubits that do not appear in any of the entanglement layers.
    let skipped_qubits = if skip_unentangled_qubits {
        let active: HashSet<&u32> =
            HashSet::from_iter(entanglement.iter().flatten().flatten().flatten());
        HashSet::from_iter((0..num_qubits).filter(|i| !active.contains(i)))
    } else {
        HashSet::new()
    };

    let mut packed_insts: Box<dyn Iterator<Item = PyResult<Instruction>>> =
        Box::new((0..reps).flat_map(|layer| {
            rotation_layer(
                py,
                num_qubits,
                &packed_rotations,
                ledger.get_parameters(LayerType::Rotation, layer),
                &skipped_qubits,
            )
            .chain(maybe_barrier(py, num_qubits, insert_barriers))
            .chain(entanglement_layer(
                py,
                &entanglement[layer],
                &packed_entanglings,
                ledger.get_parameters(LayerType::Entangle, layer),
            ))
            .chain(maybe_barrier(py, num_qubits, insert_barriers))
        }));
    if !skip_final_rotation_layer {
        packed_insts = Box::new(packed_insts.chain(rotation_layer(
            py,
            num_qubits,
            &packed_rotations,
            ledger.get_parameters(LayerType::Rotation, reps),
            &skipped_qubits,
        )))
    }

    CircuitData::from_packed_operations(py, num_qubits, 0, packed_insts, Param::Float(0.0))
}

fn unpack_entanglement<'a>(
    num_qubits: u32,
    layer: usize,
    entanglement: &Bound<PyList>,
    packed_entanglings: &'a [PyRef<Block>],
) -> PyResult<Vec<Vec<Vec<u32>>>> {
    packed_entanglings
        .iter()
        .zip(entanglement.iter())
        .map(|(block, ent)| -> PyResult<Vec<Vec<u32>>> {
            get_entanglement(num_qubits, block.num_qubits, &ent, layer)?.collect()
        })
        .collect()
}

/// Compute the number of free parameters in the circuit.
fn maybe_barrier(
    py: Python,
    num_qubits: u32,
    insert_barriers: bool,
) -> Box<dyn Iterator<Item = PyResult<Instruction>>> {
    // TODO could speed this up by only defining the barrier class once
    if !insert_barriers {
        Box::new(std::iter::empty())
    } else {
        let barrier_cls = imports::BARRIER.get_bound(py);
        let barrier = barrier_cls
            .call1((num_qubits,))
            .expect("Could not create Barrier Python-side");
        let barrier_inst = PyInstruction {
            qubits: num_qubits,
            clbits: 0,
            params: 0,
            op_name: "barrier".to_string(),
            control_flow: false,
            instruction: barrier.into(),
        };
        Box::new(std::iter::once(Ok((
            barrier_inst.into(),
            smallvec![],
            (0..num_qubits).map(Qubit).collect(),
            vec![] as Vec<Clbit>,
        ))))
    }
}
