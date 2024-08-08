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

use hashbrown::HashMap;
use approx::abs_diff_eq;
use hashbrown::hash_map::Iter;
use ndarray::linalg::kron;
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::SmallVec;

use crate::unitary_compose::compose;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PySequence, PyString, PyTuple};
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::dag_node::DAGOpNode;
use qiskit_circuit::operations::{Operation, OperationRef, Param};
use qiskit_circuit::{Clbit, Qubit};
use rustworkx_core::distancemap::DistanceMap;
use qiskit_circuit::bit_data::BitData;

const SKIPPED_NAMES: [&'static str; 4] = ["measure", "reset", "delay", "initialize"];
const NO_CACHE_NAMES: [&'static str; 2] = ["annotated", "linear_function"];

#[pyclass(module = "qiskit._accelerate.commutation_checker")]
struct CommutationChecker {
    library: CommutationLibrary,
    cache_max_entries: usize,
    cache: HashMap<(String, String), CommutationCacheEntry>,
    #[pyo3(get)]
    current_cache_entries: usize,
    #[pyo3(get)]
    _cache_miss: usize,
    #[pyo3(get)]
    _cache_hit: usize
}

#[pymethods]
impl CommutationChecker {
    #[pyo3(signature = (standard_gate_commutations=None, cache_max_entries=1_000_000))]
    #[new]
    fn py_new(
        py: Python,
        standard_gate_commutations: Option<Py<PyAny>>,
        cache_max_entries: usize,
    ) -> Self {
        CommutationChecker {
            library: CommutationLibrary::new(py, standard_gate_commutations),
            cache: HashMap::with_capacity(cache_max_entries),
            cache_max_entries,
            current_cache_entries: 0,
            _cache_miss: 0,
            _cache_hit: 0,
        }
    }

    #[pyo3(signature=(op1, op2, max_num_qubits=3))]
    fn commute_nodes(
        &mut self,
        py: Python,
        op1: &DAGOpNode,
        op2: &DAGOpNode,
        max_num_qubits: u32,
    ) -> PyResult<bool> {
        let mut bq: BitData<Qubit> = BitData::new(py, "qubits".to_string());
        op1.instruction.qubits.bind(py).iter().for_each(|q| bq.add(py, &q, false).unwrap());
        op2.instruction.qubits.bind(py).iter().for_each(|q| bq.add(py, &q, false).unwrap());
        let qargs1 = op1.instruction.qubits.bind(py).iter().map(|q| bq.find(&q).unwrap().0 as usize).collect::<Vec<_>>();
        let qargs2 = op2.instruction.qubits.bind(py).iter().map(|q| bq.find(&q).unwrap().0 as usize).collect::<Vec<_>>();

        let mut bc: BitData<Clbit> = BitData::new(py, "clbits".to_string());
        op1.instruction.clbits.bind(py).iter().for_each(|c| bc.add(py, &c, false).unwrap());
        op2.instruction.clbits.bind(py).iter().for_each(|c| bc.add(py, &c, false).unwrap());
        let cargs1 = op1.instruction.clbits.bind(py).iter().map(|c| bc.find(&c).unwrap().0 as usize).collect::<Vec<_>>();
        let cargs2 = op2.instruction.clbits.bind(py).iter().map(|c| bc.find(&c).unwrap().0 as usize).collect::<Vec<_>>();

        Ok(self.commute_inner(
            &op1.instruction,
            &qargs1,
            &cargs1,
            &op2.instruction,
            &qargs2,
            &cargs2,
            max_num_qubits,
        ))
    }

    #[pyo3(signature=(op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits=3))]
    fn commute(&mut self,
               py: Python,
               op1: &Bound<PyAny>,
               qargs1: Option<&Bound<PySequence>>,
               cargs1: Option<&Bound<PySequence>>,
               op2: &Bound<PyAny>,
               qargs2: Option<&Bound<PySequence>>,
               cargs2: Option<&Bound<PySequence>>,
               max_num_qubits: u32
    )
               -> PyResult<bool>
    {
        let qargs1 =
            qargs1.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        let cargs1 =
            cargs1.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        let qargs2 =
            qargs2.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        let cargs2 =
            cargs2.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        self.commute_nodes(py,             &DAGOpNode {
            instruction: CircuitInstruction::py_new(
                op1,
                Some(qargs1.into_any()),
                Some(cargs1.into_any()),
            )?,
            sort_key: PyString::new_bound(py, "do_not_use").into_any().into()

        }, &DAGOpNode {
            instruction: CircuitInstruction::py_new(
                op2,
                Some(qargs2.into_any()),
                Some(cargs2.into_any()),
            )?,
            sort_key: PyString::new_bound(py, "do_not_use").into_any().into()
        }, max_num_qubits)

    }

    #[pyo3(signature=())]
    fn num_cached_entries(&self) -> usize {
        self.current_cache_entries
    }
    #[pyo3(signature=())]
    fn clear_cached_commutations(&mut self){
        self.clear_cache()
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new_bound(py);
        out_dict.set_item("cache_max_entries", self.cache_max_entries.clone())?;
        out_dict.set_item("current_cache_entries", self.current_cache_entries.clone())?;
        out_dict.set_item("_cache_miss", self._cache_miss.clone())?;
        out_dict.set_item("_cache_hit", self._cache_hit.clone())?;
        out_dict.set_item("cache", self.cache.clone())?;
        out_dict.set_item("library", self.library.clone())?;
        Ok(out_dict.unbind())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let dict_state = state.downcast_bound::<PyDict>(py)?;
        self.cache_max_entries = dict_state.get_item("cache_max_entries")?.unwrap().extract()?;
        self.current_cache_entries = dict_state.get_item("current_cache_entries")?.unwrap().extract()?;
        self._cache_miss = dict_state.get_item("_cache_miss")?.unwrap().extract()?;
        self._cache_hit = dict_state.get_item("_cache_hit")?.unwrap().extract()?;
        self.library = CommutationLibrary {library : dict_state.get_item("library")?.unwrap().extract()?};
        self.cache = dict_state.get_item("cache")?.unwrap().extract()?;
        Ok(())
    }
}

impl CommutationChecker {
    fn commute_inner(&mut self,
                     instr1: &CircuitInstruction,
                     qargs1: &Vec<usize>,
                     cargs1: &Vec<usize>,
                     instr2: &CircuitInstruction,
                     qargs2: &Vec<usize>,
                     cargs2: &Vec<usize>,
                     max_num_qubits: u32,
    ) -> bool {
        let commutation: Option<bool> = self.commutation_precheck(
            instr1,
            qargs1,
            cargs1,
            instr2,
            qargs2,
            cargs2,
            max_num_qubits,
        );
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
        let (first_instr, second_instr) = if reversed {
            (instr2, instr1)
        } else {
            (instr1, instr2)
        };
        let (first_op, second_op) = if reversed { (op2, op1) } else { (op1, op2) };
        let (first_qargs, second_qargs) = if reversed {
            (qargs2, qargs1)
        } else {
            (qargs1, qargs2)
        };
        let (_first_cargs, _second_cargs) = if reversed {
            (cargs2, cargs1)
        } else {
            (cargs1, cargs2)
        };

        let skip_cache: bool = NO_CACHE_NAMES.contains(&first_op.name()) ||
            NO_CACHE_NAMES.contains(&second_op.name()) ||
            //skip params that do not evaluate to floats for caching and commutation library
            first_instr.params.iter().any(|p| !matches!(p, Param::Float(_))) ||
            second_instr.params.iter().any(|p| !matches!(p, Param::Float(_)));

        if skip_cache {
            return self.commute_matmul(first_instr, first_qargs, second_instr, second_qargs);
        }

        //query commutation library
        if let Some(is_commuting) =  self.library.check_commutation_entries(&first_op, first_qargs, &second_op, second_qargs){
            return is_commuting;
        }
        //query cache
        if let Some(commutation_dict) = self.cache.get(&(first_op.name().to_string(), second_op.name().to_string())){
            if let Some(commutation) = commutation_dict.get(&(Self::get_relative_placement(first_qargs, second_qargs), (hashable_params(&first_instr.params), hashable_params(&second_instr.params)))){
                self._cache_hit += 1;
                return commutation.clone();
            }
            else {
                self._cache_miss += 1;
            }
        }
        else {
            self._cache_miss += 1;
        }

        // perform matrix multiplication to determine commutation
        let is_commuting = self.commute_matmul(first_instr, first_qargs, second_instr, second_qargs);

        // TODO: implement a LRU cache for this
        if self.current_cache_entries >= self.cache_max_entries {
            self.clear_cache();
        }
        // cache results from is_commuting
        self.cache
            .entry((
                first_op.name().to_string(),
                second_op.name().to_string(),
            ))
            .and_modify(|entries| {
                let key = (Self::get_relative_placement(first_qargs, second_qargs), (hashable_params(&first_instr.params), hashable_params(&second_instr.params)));
                entries.insert(key, is_commuting);
                self.current_cache_entries += 1;
            })
            .or_insert_with(|| {
                let mut entries = HashMap::with_capacity(1);
                let key = (Self::get_relative_placement(first_qargs, second_qargs), (hashable_params(&first_instr.params), hashable_params(&second_instr.params)));
                entries.insert(key, is_commuting);
                self.current_cache_entries += 1;
                CommutationCacheEntry{ mapping: entries}
            });
        is_commuting
    }

    fn commute_matmul(
        &self,
        first_instr: &CircuitInstruction,
        first_qargs: &Vec<usize>,
        second_instr: &CircuitInstruction,
        second_qargs: &Vec<usize>,
    ) -> bool {
        // compute relative positioning of qargs of the second gate to the first gate
        let mut qarg: HashMap<&usize, usize> =
            HashMap::with_capacity(first_qargs.len() + second_qargs.len());
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

        let first_qarg: Vec<_> = first_qargs
            .iter()
            .map(|q| qarg.get_item(q).unwrap().clone())
            .collect();
        let second_qarg: Vec<_> = second_qargs
            .iter()
            .map(|q| qarg.get_item(q).unwrap().clone())
            .collect();

        assert!(
            first_qarg.len() <= second_qarg.len(),
            "first instructions must have at most as many qubits as the second instruction"
        );

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

        if first_qarg == second_qarg {
            abs_diff_eq!(
            second_mat.dot(&first_mat),
            first_mat.dot(&second_mat),
            epsilon = 1e-8
        )
        } else {
            let extra_qarg2 = num_qubits - first_qarg.len();
            let first_mat = if extra_qarg2 > 0 {
                let id_op = Array2::<Complex64>::eye(usize::pow(2, extra_qarg2 as u32));
                kron(&id_op, &first_mat)
            } else {
                first_mat
            };
            let op12 = compose(first_mat.clone(), second_mat.clone(), &second_qarg, false);
            let op21 = compose(first_mat, second_mat, &second_qarg, true);
            abs_diff_eq!(
            op12,
            op21,
            epsilon = 1e-8
        )
        }
    }

    fn commutation_precheck(&self,
                            op1: &CircuitInstruction,
                            qargs1: &Vec<usize>,
                            cargs1: &Vec<usize>,
                            op2: &CircuitInstruction,
                            qargs2: &Vec<usize>,
                            cargs2: &Vec<usize>,
                            max_num_qubits: u32,
    ) -> Option<bool> {
        if op1.op().control_flow() || op2.op().control_flow() || op1.is_conditioned() || op2.is_conditioned() {
            return Some(false);
        }

        // assuming the number of involved qubits to be small, this might be faster than set operations
        if !qargs1.iter().any(|e| qargs2.contains(e)) && !cargs1.iter().any(|e| cargs2.contains(e)) {
            return Some(true);
        }

        if self.is_commutation_skipped(op1, max_num_qubits) || self.is_commutation_skipped(op2, max_num_qubits) {
            return Some(false);
        }

        None
    }

    fn is_commutation_skipped(&self, instr: &CircuitInstruction, max_qubits: u32) -> bool {
        let op = instr.op();
        op.num_qubits() > max_qubits
            || op.directive()
            || SKIPPED_NAMES.contains(&op.name())
            || instr.is_parameterized()
    }

    fn get_relative_placement(
        first_qargs: &Vec<usize>,
        second_qargs: &Vec<usize>)
        -> SmallVec<[Option<Qubit>; 2]>
    {
        let qubits_g2: HashMap<_, _> = second_qargs.iter().enumerate().map(|(i_g1, q_g1)| (q_g1, Qubit(i_g1 as u32))).collect();

        first_qargs.iter().map(|q_g0| qubits_g2.get(q_g0).copied()).collect()
    }

    fn clear_cache(&mut self){
        self.cache.clear();
        self.current_cache_entries = 0;
        self._cache_miss = 0;
        self._cache_hit = 0;
    }
}


#[derive(Clone)]
#[pyclass]
pub struct CommutationLibrary {
    pub library: HashMap<(String, String), CommutationLibraryEntry>,
}

impl CommutationLibrary {
    fn check_commutation_entries(
        &self,
        first_op: &OperationRef,
        first_qargs: &Vec<usize>,
        second_op: &OperationRef,
        second_qargs: &Vec<usize>,
    ) -> Option<bool> {

        match self.library.get(&(first_op.name().to_string(), second_op.name().to_string())){
            Some(CommutationLibraryEntry::Commutes(b)) => { Some(*b)},
            Some(CommutationLibraryEntry::QubitMapping(qm)) => {qm.get(&CommutationChecker::get_relative_placement(first_qargs, second_qargs)).copied()},
            _ => {None}
        }
    }
}


impl ToPyObject for CommutationLibrary {
    fn to_object(&self, py: Python) -> PyObject {
        self.library.to_object(py)
    }
}


#[pymethods]
impl CommutationLibrary {
    #[new]
    fn new(py: Python, py_any: Option<Py<PyAny>>) -> Self {
        match py_any {
            Some(pyob) => {
                CommutationLibrary { library: pyob.bind(py).extract::<HashMap<(String, String), CommutationLibraryEntry>>().expect("Input parameter standard_gate_commutations to CommutationChecker was not a dict with the right format")}

            },
            None => CommutationLibrary { library: HashMap::new() }
        }
    }
}



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

impl ToPyObject for CommutationLibraryEntry {
    fn to_object(&self, py: Python) -> PyObject {

        match self{
            CommutationLibraryEntry::Commutes(b) => { PyBool::new_bound(py, *b).to_object(py)},
            CommutationLibraryEntry::QubitMapping(qm) => {
                let out_dict = PyDict::new_bound(py);

                qm.iter().for_each(|(k, v)| out_dict.set_item(PyTuple::new_bound(py, k.iter().map(|q| q.map(|t| t.0 as u32)).collect::<Vec<Option<u32>>>()).to_object(py), PyBool::new_bound(py, *v).to_object(py)).ok().unwrap());
                out_dict.to_object(py)},
        }

    }
}


//need a struct instead of a type definition because we cannot implement serialization traits otherwise
#[derive(Clone)]
struct CommutationCacheEntry{
    mapping :HashMap<
    (
        SmallVec<[Option<Qubit>; 2]>,
        (SmallVec<[ParameterKey; 3]>, SmallVec<[ParameterKey; 3]>),
    ),
    bool,
>
}
impl CommutationCacheEntry{
    fn get(
        &self,
        key: &(SmallVec<[Option<Qubit>; 2]>,
               (SmallVec<[ParameterKey; 3]>, SmallVec<[ParameterKey; 3]>)),
    ) -> Option<&bool> {
        self.mapping.get(key)
    }
    fn iter(&self) -> Iter<'_, (SmallVec<[Option<Qubit>; 2]>,
                                (SmallVec<[ParameterKey; 3]>, SmallVec<[ParameterKey; 3]>)), bool> {
        self.mapping.iter()
    }

    fn insert(&mut self, k: (SmallVec<[Option<Qubit>; 2]>,
                             (SmallVec<[ParameterKey; 3]>, SmallVec<[ParameterKey; 3]>)), v: bool) -> Option<bool>{self.mapping.insert(k, v)}
}


impl ToPyObject for CommutationCacheEntry{
    fn to_object(&self, py: Python) -> PyObject {
        let out_dict = PyDict::new_bound(py);
        for (k, v) in self.iter() {
            let qubits = PyTuple::new_bound(py, k.0.iter().map(|q| q.map(|t| t.0 as u32)).collect::<Vec<Option<u32>>>()).to_object(py);
            let params0 = PyTuple::new_bound(py, k.1.0.iter().map(|pk| pk.0).collect::<Vec<f64>>()).to_object(py);
            let params1 = PyTuple::new_bound(py, k.1.1.iter().map(|pk| pk.0).collect::<Vec<f64>>()).to_object(py);
            out_dict.set_item(PyTuple::new_bound(py, [qubits, PyTuple::new_bound(py, [params0, params1].iter().collect::<Vec<&PyObject>>()).to_object(py)].iter().collect::<Vec<&PyObject>>()), PyBool::new_bound(py, *v).to_object(py)).expect("Failed to construct commutation cache for serialization");
        }
        out_dict.to_object(py)
    }
}


impl<'py> FromPyObject<'py> for CommutationCacheEntry{
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        let dict = b.downcast::<PyDict>()?;
        let mut ret = hashbrown::HashMap::with_capacity(dict.len());
        for (k, v) in dict {
            let raw_key: (SmallVec<[Option<u32>; 2]>,
                          (SmallVec<[f64; 3]>, SmallVec<[f64; 3]>)) = k.extract()?;
            let qubits = raw_key.0.iter().map(|q| q.map(|t| Qubit(t))).collect();
            let params0: SmallVec<_> = raw_key.1.0.iter().map(|p| ParameterKey(*p)).collect();
            let params1: SmallVec<_> = raw_key.1.1.iter().map(|p| ParameterKey(*p)).collect();
            let v: bool = v.extract()?;
            ret.insert((qubits, (params0, params1)), v);
        }
        Ok(CommutationCacheEntry { mapping: ret})
    }
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
pub fn commutation_checker(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CommutationLibrary>()?;
    m.add_class::<CommutationChecker>()?;
    Ok(())
}
