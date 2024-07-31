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

use std::iter::Chain;
use hashbrown::HashMap;
use itertools::chain;
use ndarray::Array2;
use ndarray::linalg::kron;
use num_complex::Complex64;
use smallvec::SmallVec;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyTuple};
use pyo3::types::iter::BoundTupleIterator;
use rustworkx_core::distancemap::DistanceMap;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::dag_node::DAGOpNode;
use qiskit_circuit::operations::{Operation, OperationRef, Param, PyInstruction, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation, PackedOperationType};
use qiskit_circuit::Qubit;
use crate::nlayout::PhysicalQubit;
use crate::unitary_compose::compose_unitary;
use ndarray_einsum_beta::*;
#[derive(Clone)]
pub enum CommutationLibraryEntry {
    Commutes(bool),
    QubitMapping(HashMap<SmallVec<[Option<Qubit>; 2]>, bool>),
}

impl<'py> FromPyObject<'py> for CommutationLibraryEntry {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        if let Some(b) = b.extract::<bool>().ok() {
            return Ok(CommutationLibraryEntry::Commutes(b));
        }
        let dict = b.downcast::<PyDict>()?;
        let mut ret = hashbrown::HashMap::with_capacity(dict.len());
        for (k, v) in dict {
            let raw_key: SmallVec<[Option<u32>; 2]> = k.extract()?;
            let v: bool = v.extract()?;
            let key = raw_key
                .into_iter()
                .map(|key| key.map(|x| Qubit(x)))
                .collect();
            ret.insert(key, v);
        }
        Ok(CommutationLibraryEntry::QubitMapping(ret))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct CommutationLibrary {
    pub library: HashMap<[StandardGate; 2], CommutationLibraryEntry>,
}

impl CommutationLibrary {
    fn check_commutation_entries(
        &self,
        first_op: &CircuitInstruction,
        second_op: &CircuitInstruction,
    ) -> Option<bool> {
        None
    }
}

#[pymethods]
impl CommutationLibrary {
    #[new]
    fn new(library: HashMap<[StandardGate; 2], CommutationLibraryEntry>) -> Self {
        CommutationLibrary { library }
    }
}

type CommutationCacheEntry = HashMap<
    (
        SmallVec<[Option<Qubit>; 2]>,
        [SmallVec<[ParameterKey; 3]>; 2],
    ),
    bool,
>;

#[pyclass]
struct CommutationChecker {
    library: CommutationLibrary,
    cache_max_entries: usize,
    cache: HashMap<[String; 2], CommutationCacheEntry>,
    current_cache_entries: usize,
}

#[pymethods]
impl CommutationChecker {
    #[pyo3(signature = (standard_gate_commutations=None, cache_max_entries=1_000_000))]
    #[new]
    fn py_new(
        standard_gate_commutations: Option<CommutationLibrary>,
        cache_max_entries: usize,
    ) -> Self {
        CommutationChecker {
            library: standard_gate_commutations
                .unwrap_or_else(|| CommutationLibrary::new(HashMap::new())),
            cache: HashMap::with_capacity(cache_max_entries),
            cache_max_entries,
            current_cache_entries: 0,
        }
    }
    #[pyo3(signature=(op1, op2, max_num_qubits=3))]
    fn commute_nodes(
        &self,
        py: Python,
        op1: &DAGOpNode,
        op2: &DAGOpNode,
        max_num_qubits: u32
    ) -> PyResult<bool> {

        let get_hashmap_for_bits = |chain: Chain<BoundTupleIterator, BoundTupleIterator>| -> HashMap<isize, usize> {
            let mut qqmap: HashMap<isize, usize> = HashMap::new();
            for b in chain{
                let len = qqmap.len();
                qqmap.entry(b.hash().expect("Error building bit map!")).or_insert(len);
            }
            qqmap
        };
        let mut qmap = get_hashmap_for_bits(op1.instruction.qubits.bind(py).iter().chain(op2.instruction.qubits.bind(py).iter()));
        let mut cmap = get_hashmap_for_bits(op1.instruction.clbits.bind(py).iter().chain(op2.instruction.clbits.bind(py).iter()));

        let qmapping = |op: &DAGOpNode| -> Vec<_>  {
            op.instruction.qubits.bind(py).iter().map(|q| qmap.get_item(q.hash().expect("Error building qubit map!")).unwrap().clone()).collect::<Vec<_>>()
        };
        let cmapping = |op: &DAGOpNode| -> Vec<_>  {
            op.instruction.clbits.bind(py).iter().map(|q| cmap.get_item(q.hash().expect("Error building qubit map!")).unwrap().clone()).collect::<Vec<_>>()
        };

        let qargs1 = qmapping(op1);
        let cargs1 = cmapping(op1);

        let qargs2 = qmapping(op2);
        let cargs2 = cmapping(op2);

        Ok(commute_inner(&op1.instruction, &qargs1, &cargs1,
                         &op2.instruction, &qargs2, &cargs2, max_num_qubits))
    }
}

/*
        OperationType::Standard(_) | OperationType::Gate(_) => {
            if let Some(attr) = &op.extra_attrs {
                if attr.condition.is_some() {
                    return false;
                }
            }
            true
        }
*/
fn is_commutation_supported(op: &OperationRef) -> bool {
    match op { (OperationRef::Standard(_) | OperationRef::Gate(_)) => !op.control_flow(),
        _ => false,
    }
}

const SKIPPED_NAMES: [&str; 4] = ["measure", "reset", "delay", "initialize"];

fn is_commutation_skipped(instr: &CircuitInstruction, max_qubits: u32) -> bool {
    let op = instr.op();
    op.num_qubits() > max_qubits
        || op.directive()
        || SKIPPED_NAMES.contains(&op.name())
        || instr.is_parameterized()
}


fn commutation_precheck(
    op1: &CircuitInstruction,
    qargs1: &Vec<usize>,
    cargs1: &Vec<usize>,
    op2: &CircuitInstruction,
    qargs2: &Vec<usize>,
    cargs2: &Vec<usize>,
    max_num_qubits: u32) -> Option<bool> {

    if !is_commutation_supported(&op1.op()) || !is_commutation_supported(&op2.op()) {
        return Some(false);
    }

    // assuming the number of involved qubits to be small, this might be faster than set operations
    if !qargs1.iter().any(|e| qargs2.contains(e)) &&
       !cargs1.iter().any(|e| cargs2.contains(e)) {
        return Some(true);
    }

    if is_commutation_skipped(op1, max_num_qubits) || is_commutation_skipped(op2, max_num_qubits) {
        return Some(false);
    }

    None
}


fn commute_inner(
    instr1: &CircuitInstruction,
    qargs1: &Vec<usize>,
    cargs1: &Vec<usize>,
    instr2: &CircuitInstruction,
    qargs2: &Vec<usize>,
    cargs2: &Vec<usize>,
    max_num_qubits: u32
)-> bool {
    let commutation: Option<bool> = commutation_precheck(instr1, qargs1, cargs1,
                                                         instr2, qargs2, cargs2, max_num_qubits);
    if !commutation.is_none() {
        return commutation.unwrap();
    }
    let op1 = instr1.op();
    let op2 = instr2.op();
    let reversed = if op1.num_qubits() != op2.num_qubits() {
        op1.num_qubits() > op2.num_qubits()
    } else {
        // TODO is this consistent between machines?
        op1.name() > op2.name()
    };
    let (first_instr, second_instr) = if reversed { (instr2, instr1) } else { (instr1, instr2) };
    let (first_op, second_op) = if reversed { (op2, op1) } else { (op1, op2) };
    let (first_qargs, second_qargs) = if reversed { (qargs2, qargs1) } else { (qargs1, qargs2) };
    let (first_cargs, second_cargs) = if reversed { (cargs2, cargs1) } else { (cargs1, cargs2) };

    if first_op.name() == "annotated" || second_op.name() == "annotated" {
        return commute_matmul(first_instr, first_qargs, second_instr, second_qargs);
    }

    //TODO else, look into commutation library!


    return commute_matmul(first_instr, first_qargs, second_instr, second_qargs);

    //TODO cache result
    //circ: &CircuitData, //TBD <- maybe take it also now for performance reasons
    //map qubits to indices  python objects -> indices
    // vf2_mappings error map
    // packed instruction pr
    //op1.instruction.qubits[qreg, idx]
    //qmap: &HashMap<Qubit, usize>
    //println!()
    // StandardGate -> rust space
    // Otherwise -> python
    //    match node.instruction.op() {
    //         gate @ (OperationRef::Standard(_) | OperationRef::Gate(_)) => Some(
    true
}

fn commute_matmul(
    first_instr: &CircuitInstruction,
    first_qargs: &Vec<usize>,
    second_instr: &CircuitInstruction,
    second_qargs: &Vec<usize>
) -> bool {

    // compute relative positioning in qarg
    let mut qarg: HashMap<&usize, usize> = HashMap::with_capacity(first_qargs.len() + second_qargs.len());
    for (i, q) in first_qargs.iter().enumerate() {
        qarg.entry(q).or_insert(i);
    }
    let mut num_qubits = first_qargs.len();
    for q in second_qargs {
        if !qarg.contains_key(q) {
            qarg.insert(q, num_qubits);
            num_qubits += 1;
        }
    }

    //let first_qarg: Vec<usize> = first_qargs.iter().map(|q| qarg.entry(q)).collect();
    let first_qarg: Vec<_> = first_qargs.iter().map(|q| qarg.get_item(q).unwrap().clone()).collect();
    let second_qarg: Vec<_> = second_qargs.iter().map(|q| qarg.get_item(q).unwrap().clone()).collect();
    assert_eq!(&first_qarg, first_qargs, "hm, should be ok");
    assert_eq!(&second_qarg, second_qargs, "hm, should be ok");
    //second_qarg = tuple(qarg[q] for q in second_qargs)


    assert!(first_qargs.len() <= second_qargs.len(), "first instructions must have at most as many qubits as the second instruction");

    let first_op = first_instr.op();
    let second_op = second_instr.op();
    let first_mat = match first_op.matrix(&first_instr.params) {
        Some(mat) => mat,
        None => return false,
    };
    let second_mat = match second_op.matrix(&second_instr.params) {
        Some(mat) => mat,
        None => return false,
    };
    let [op12, op21] = if first_qargs == second_qargs {
        [second_mat.dot(&first_mat), first_mat.dot(&second_mat)]
    } else {
        let first_mat = if second_op.num_qubits() > first_op.num_qubits() {
            let extra_qarg2 = num_qubits - first_qarg.len();
            println!("qdiff: {}", extra_qarg2);
            let id_op = Array2::<Complex64>::eye(usize::pow(2, extra_qarg2 as u32));
            kron(&id_op, &first_mat)
        } else {
            first_mat
        };
        println!("{} {} {:?} {:?}", first_op.num_qubits(), second_op.num_qubits(), first_qargs, second_qargs);
        println!("{:?}", first_mat);
        let op12 = compose_unitary(second_mat.view(), first_mat.view(), second_qargs);
        let op21 = compose_unitary(first_mat.view(), second_mat.view(), second_qargs);
        [op12, op21]
    };
    op12 == op21
}



#[derive(Debug, Copy, Clone)]
struct ParameterKey(f64);

impl ParameterKey {
    fn key(&self) -> u64 {
        self.0.to_bits()
    }
}

impl std::hash::Hash for ParameterKey {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.key().hash(state)
    }
}

impl PartialEq for ParameterKey {
    fn eq(&self, other: &ParameterKey) -> bool {
        self.key() == other.key()
    }
}

impl Eq for ParameterKey {}

fn hashable_params(params: &[Param]) -> SmallVec<[ParameterKey; 3]> {
    params
        .iter()
        .map(|x| {
            if let Param::Float(x) = x {
                ParameterKey(*x)
            } else {
                panic!()
            }
        })
        .collect()
}


#[pymodule]
pub fn commutation_checker(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CommutationLibrary>()?;
    m.add_class::<CommutationChecker>()?;
    Ok(())
}