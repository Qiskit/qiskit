// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Rewrite multi-arity PBC intrustructions in terms of complete data modules.

use hashbrown::HashMap;
use std::sync::Arc;
use std::{mem, slice};

use qiskit_mir::{
    mir::Mir,
    ops::MirOp,
    ops::pbc::{self, Pauli, PauliString},
    qid::{self, Qid, QidList, ResourceTable},
};
use qiskit_passmanager::{PassContext, StaticPass};
use qiskit_util::IndexMap;

/// Rewrite PBC operations on "individual" qids into operations acting on "module" qids.
///
/// # Example
///
/// If qids `p0, p1, p2, p3` form a "module" `p4` with each smaller qid of bit-length 1 in the bit
/// of `p4` with the same index, then this pass would rewrite:
///
/// * (`ZYXZ`, `[p0, p1, p2, p3]`) -> (`ZYXZ`, `[p4]`): matching Lsb0 string labelling with
///   little-endian qid conventions.
/// * (`ZYXZ`, `[p1, p0, p3, p2]`) -> (`YZZX`, `[p4]`): bits have to be rearranged to match the
///   module layout.
/// * (`ZY`, `[p0, p1]`) -> (`IIZY`, `[p4]`): string needs to be `I`-padded to match the module len
/// * (`YXZ`, `[v0, p2, v1]`) -> (`YIXIIZ`, `[v0, p4, v1]`): unmapped qids are left as-is.
pub struct ModularizePbc {
    phys_table: Arc<ResourceTable>,
    map: ModuleMap,
}

#[derive(thiserror::Error, Debug)]
enum ModularizerError {
    #[error("qid {0} is not in any resource table")]
    AbsentQid(Qid),
    #[error("qid {0} does not have a length")]
    Widthless(Qid),
    #[error("modularization introduced qid overlap violations")]
    NewOverlap,
}

/// State object for re-using allocations between multiple modularisation calls.
///
/// This state object contains an overlap checker, but the nature of it having access to the
/// resource table from which the module map is drawn from is making an assumption: that the only
/// _new_ overlaps that can have been introduced are caused by modules in the `QidSpace` of the
/// table held by `ModularizePbc`.  In turn, that's (approximately) assuming that:
///
/// * `ModularizePbc` is holding a physical resource table
/// * there can be no overlap between resource tables in different spaces, because each table is
///   associated with only one space, and a table defines the overlaps.
///
/// The interfaces of `ResourceTable` imply that overlap can be decided with just a single table,
/// but it's possible that in the future we may allow a table to have more than one space.  That
/// wasn't in the original vision of the Qid system, but we can't necessarily predict how our
/// modelling will evolve.
#[derive(Clone, Debug)]
struct ModularizerState<'a> {
    check: qid::OverlapChecker<'a>,
    map: &'a ModuleMap,
    /// Ordered map of each qid that will be in the new list, and which bit index in the output
    /// Pauli string it starts at.
    new_qid_start: IndexMap<Qid, u32>,
    /// Mapping of which qids are in the source but don't undergo modularisation, to its bitwidth.
    passthrough_qid_len: HashMap<Qid, u32>,
}
impl<'a> ModularizerState<'a> {
    fn new(pass: &'a ModularizePbc) -> Self {
        Self {
            check: qid::OverlapChecker::new(&pass.phys_table),
            map: &pass.map,
            new_qid_start: Default::default(),
            passthrough_qid_len: Default::default(),
        }
    }
}
impl ModularizerState<'_> {
    fn reset(&mut self) {
        self.new_qid_start.clear();
        self.passthrough_qid_len.clear();
    }

    /// Rewrite a Pauli string and qid list in terms of the modules we need.
    ///
    /// Returns `None` if there's nothing to do for this qid list.
    pub fn modularize(
        &mut self,
        mir_table: &ResourceTable,
        old_paulis: &PauliString,
        old_qids: &[Qid],
    ) -> Result<Option<(PauliString, QidList)>, ModularizerError> {
        self.reset();
        let mut changed = false;
        let mut len = 0u32;
        let info = |qid: Qid| match qid.space() {
            qid::QidSpace::Virtual => mir_table.info(qid),
            qid::QidSpace::Physical => self.check.table().info(qid),
        };
        for &qid in old_qids {
            match self.map.rewrite.get(&qid) {
                Some((module, _)) => {
                    changed = true;
                    let inserted = *self.new_qid_start.entry(*module).or_insert(len);
                    if inserted == len {
                        // We fully control `ModuleMap`, so we can trust it to be consistent.
                        len += self.map.module_len[module];
                    }
                }
                None => {
                    self.new_qid_start.insert(qid, len);
                    let this_len = info(qid)
                        .ok_or(ModularizerError::AbsentQid(qid))?
                        .width()
                        .ok_or(ModularizerError::Widthless(qid))?;
                    self.passthrough_qid_len.insert(qid, this_len);
                    len += this_len;
                }
            }
        }
        if !changed {
            return Ok(None);
        }
        let our_space = self.check.table().space();
        let new_qids = self
            .new_qid_start
            .keys()
            .copied()
            .filter(|q| q.space() == our_space);
        if self
            .check
            .try_overlap(new_qids)
            .expect("filtered to only matching qids")
        {
            return Err(ModularizerError::NewOverlap);
        }
        // TODO: we're in the process of changing the Pauli representation, so the details here will
        // change significantly.  This pass _mostly_ shouldn't need `Vec` allocations.
        let mut new_paulis = vec![Pauli::I; len as usize];
        let mut old_paulis = old_paulis.as_slice();
        for &qid in old_qids {
            let len = match self.map.rewrite.get(&qid) {
                Some((module, loc)) => {
                    let offset = self.new_qid_start[module] as usize;
                    for (&old, &idx) in old_paulis.iter().zip(loc.as_slice()) {
                        new_paulis[offset + idx as usize] = old;
                    }
                    loc.len() as usize
                }
                None => {
                    let offset = self.new_qid_start[&qid] as usize;
                    let this_len = self.passthrough_qid_len[&qid] as usize;
                    new_paulis[offset..offset + this_len].copy_from_slice(&old_paulis[..this_len]);
                    this_len
                }
            };
            old_paulis = &old_paulis[len..];
        }
        Ok(Some((
            PauliString::new(new_paulis),
            QidList::new(self.new_qid_start.drain(..).map(|(q, _)| q).collect()),
        )))
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ModuleMapError {
    #[error("qid not present in table: {0}")]
    AbsentQid(Qid),
    #[error("qid is not a module containing other qids: {0}")]
    NotModule(Qid),
    #[error("leaf qid {0} is in more than one module")]
    AmbiguousMap(Qid),
}

/// Which bit(s) in the module the leaf qid corresponds to.
#[derive(Clone, Debug)]
enum BitLoc {
    Single(u32),
    Multi(Box<[u32]>),
}
impl BitLoc {
    fn len(&self) -> u32 {
        match self {
            Self::Single(_) => 1,
            Self::Multi(x) => x.len() as u32,
        }
    }
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Single(x) => slice::from_ref(x),
            Self::Multi(x) => x,
        }
    }
}

/// A mapping from "leaf" qid to its containing module and which bits in that module it corresponds
/// to for use in "modularization".
///
/// For an arbitrary [`ResourceTable`], this is not necessarily a unique mapping; it's valid to have
/// multiple qids that each have a bit layout, and for the same leaf layout to be contained in
/// multiple places.  However, for modularisation, we require an unambiguous mapping of
/// which module to put each qid in.  You filter to only the modules you care about as you construct
/// this object.
///
/// A `ModuleMap` doesn't need to remap every leaf qubit that might appear in the MIR: those not
/// named are simply left as-is.
///
/// This [`ModuleMap`](Self) is effectively the reverse mapping of the
/// [`BitLayout`](qid::BitLayout)s stored within a subset of a [`ResourceTable`].
#[derive(Clone, Debug)]
pub struct ModuleMap {
    module_len: HashMap<Qid, u32>,
    rewrite: HashMap<Qid, (Qid, BitLoc)>,
}
impl ModuleMap {
    /// Create a modularisation mapping for the given modules that maps the bits named in its
    /// corresponding [`BitLayout`](qid::BitLayout).
    ///
    /// The modules given here are allowed to alias via abstract qids, but each qid in a
    /// [`BitLayout`](qid::BitLayout) must be unambiguously assigned to only a single module.
    pub fn for_modules(
        table: &ResourceTable,
        modules: impl IntoIterator<Item = Qid>,
    ) -> Result<Self, ModuleMapError> {
        // TODO: this currently doesn't have a mechanism of doing "transitive" modularisation.
        // For example, consider single-qubit qids `p0..=p11`, then:
        //
        // * p12 := (p6 p0)
        // * p13 := (p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0)
        //
        // It's clear that there is a well-defined modularisation that lifts `p12` to `p13` and that
        // `ModuleMap` can represent it, but we don't currently provide a constructor that can do
        // it.
        //
        // We can add that if it becomes necessary.
        let mut rewrite = HashMap::new();
        let mut module_len = HashMap::new();
        // The layout of a module may include "base" qids with a width >1.  The bits there aren't
        // required to be mapped to a contiguous ascending set of bits in the module, but they _are_
        // required (by consistency contracts of `ResourceTable`) to all be used without duplicates.
        let mut partials = HashMap::<Qid, Box<[u32]>>::new();
        for module in modules {
            let info = table
                .info(module)
                .ok_or(ModuleMapError::AbsentQid(module))?;
            let layout: &[_] = match info.layout() {
                qid::BitLayout::Abstract | qid::BitLayout::Base(_) => {
                    return Err(ModuleMapError::NotModule(module));
                }
                qid::BitLayout::Alias(layout) => layout,
            };
            module_len.insert(module, layout.len() as u32);
            for (i, bit) in layout
                .iter()
                .enumerate()
                .filter_map(|(i, x)| x.map(|q| (i as u32, q)))
            {
                // The consistency of the table requires that it is present.
                let qid = bit.qid();
                let width = table[qid].width().expect("layout entries must be sized");
                if width == 1 {
                    if rewrite.insert(qid, (module, BitLoc::Single(i))).is_some() {
                        return Err(ModuleMapError::AmbiguousMap(qid));
                    }
                    continue;
                }
                let slice = partials
                    .entry(qid)
                    .or_insert_with(|| vec![0; width as usize].into_boxed_slice());
                slice[bit.bit() as usize] = i;
            }
            for (qid, layout) in partials.drain() {
                if rewrite
                    .insert(qid, (module, BitLoc::Multi(layout)))
                    .is_some()
                {
                    return Err(ModuleMapError::AmbiguousMap(qid));
                }
            }
        }
        Ok(Self {
            module_len,
            rewrite,
        })
    }
}

impl StaticPass<Mir> for ModularizePbc {
    fn run(&self, mut mir: Box<Mir>, context: &mut PassContext) -> anyhow::Result<Box<Mir>> {
        let mut state = ModularizerState::new(self);
        // TODO: ideally the encapsulation wouldn't require us to partially deconstruct the `Mir`;
        // doing this `take`/`replace` dance isn't any different to direct `pub` access, really.
        let mut instructions = mem::take(mir.instructions_mut());
        for node in instructions.iter_unordered_mut() {
            let inst = node.weight;
            let pauli = match inst.op() {
                MirOp::PauliRotation(rot) => rot.paulis(),
                MirOp::PauliMeasure(ppm) => ppm.paulis(),
                _ => continue,
            };
            let Some((pauli, qids)) =
                state.modularize(mir.resource_table(), pauli, inst.qids().inner())?
            else {
                continue;
            };
            match inst.op_mut() {
                MirOp::PauliRotation(rot) => *rot = pbc::PauliRotation::new(pauli, rot.angle()),
                MirOp::PauliMeasure(ppm) => *ppm = pbc::PauliMeasure::new(pauli),
                _ => unreachable!("we checked the op type while extracting the Pauli string"),
            }
            *inst.qids_mut() = qids;
            context.ir_modified = true;
        }
        *mir.instructions_mut() = instructions;
        Ok(mir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter;

    use qiskit_mir::instruction::MirInstruction;
    use qiskit_mir::mir::Mir;
    use qiskit_mir::ops::pbc;
    use qiskit_mir::qid::{BitLayout, QidInfo, QidSpace};
    use qiskit_mir::symbol_table::{Symbol, SymbolIdList, SymbolType};

    fn pauli_from_ascii(ascii: &[u8]) -> Option<pbc::PauliString> {
        ascii
            .iter()
            .rev()
            .map(|p| match p {
                b'I' => Some(pbc::Pauli::I),
                b'X' => Some(pbc::Pauli::X),
                b'Y' => Some(pbc::Pauli::Y),
                b'Z' => Some(pbc::Pauli::Z),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(pbc::PauliString::new)
    }
    fn pauli_to_ascii(p: &[Pauli]) -> String {
        p.iter()
            .rev()
            .map(|p| match p {
                pbc::Pauli::I => 'I',
                pbc::Pauli::X => 'X',
                pbc::Pauli::Y => 'Y',
                pbc::Pauli::Z => 'Z',
            })
            .collect()
    }
    type Case = (String, Vec<Qid>);
    fn fmt_failures(bad: &[(Case, Case)]) -> String {
        let fmt_single = |(pauli, qids): &Case| {
            format!(
                "({}, [{}])",
                pauli,
                qids.iter()
                    .map(|q| q.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        bad.iter()
            .map(|(left, right)| {
                format!(
                    "* left : {}\n  right: {}",
                    fmt_single(left),
                    fmt_single(right)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn single_simple_module() {
        let mut ptable = ResourceTable::new(QidSpace::Physical);
        let phys = (0..4)
            .map(|_| ptable.add_qid(QidInfo::base(1)).unwrap())
            .collect::<Vec<_>>();
        let layout = phys.iter().map(|&q| Some(q.bit(0))).collect::<Box<[_]>>();
        let module = ptable
            .add_qid(QidInfo::new(BitLayout::Alias(layout)))
            .unwrap();
        let extra_phys = ptable.add_qid(QidInfo::new(BitLayout::Base(1))).unwrap();
        let map = ModuleMap::for_modules(&ptable, [module]).unwrap();
        let pass = ModularizePbc {
            phys_table: Arc::new(ptable),
            map,
        };

        let mut mir = Mir::new();
        let virts = (0..2)
            .map(|_| mir.resource_table_mut().add_qid(QidInfo::base(1)).unwrap())
            .collect::<Vec<_>>();

        let cases = [
            [("ZYXZ", phys.clone()), ("ZYXZ", vec![module])],
            [
                ("ZYXZ", vec![phys[1], phys[0], phys[3], phys[2]]),
                ("YZZX", vec![module]),
            ],
            [("ZX", vec![phys[0], phys[1]]), ("IIZX", vec![module])],
            [
                ("ZYXZ", vec![virts[0], phys[2], virts[1], extra_phys]),
                ("ZYIXIIZ", vec![virts[0], module, virts[1], extra_phys]),
            ],
            [
                // Explicit `I` in incoming Pauli string.
                ("XIZ", vec![phys[3], virts[0], phys[2]]),
                ("IZXII", vec![module, virts[0]]),
            ],
        ];

        for [(pauli, qids), _] in cases.iter() {
            let rot = pbc::PauliRotation::new(
                pauli_from_ascii(pauli.as_bytes()).unwrap(),
                pbc::AngleType::new(0.25),
            );
            mir.instructions_mut().push_back(MirInstruction::new(
                MirOp::PauliRotation(rot),
                QidList::new(qids.clone()),
                Default::default(),
                Default::default(),
            ));
        }

        let expected = cases
            .into_iter()
            .map(|[_, (s, q)]| (s.to_owned(), q))
            .collect::<Vec<_>>();

        let (out, _) = pass.into_passmanager().run::<_, Mir>(mir).unwrap();
        let actual = out
            .instructions()
            .iter_main()
            .map(|n| {
                let inst = n.weight();
                let MirOp::PauliRotation(rot) = inst.op() else {
                    panic!("modularise produced a non-rotation")
                };
                (
                    pauli_to_ascii(rot.paulis().as_slice()),
                    inst.qids().inner().to_owned(),
                )
            })
            .collect::<Vec<_>>();

        let mismatches = iter::zip(actual, expected)
            .filter(|(a, b)| a != b)
            .collect::<Vec<_>>();
        assert!(mismatches.is_empty(), "{}", fmt_failures(&mismatches))
    }

    #[test]
    fn multi_modules() {
        let mut table = ResourceTable::new(QidSpace::Physical);
        // p12 is loose - not assigned to a module.
        let p = (0..13)
            .map(|_| table.add_qid(QidInfo::base(1)).unwrap())
            .collect::<Vec<_>>();
        let modules = p
            .chunks_exact(4)
            .map(|qids| {
                let layout = qids.iter().map(|q| Some(q.bit(0))).collect();
                table
                    .add_qid(QidInfo::new(BitLayout::Alias(layout)))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let map = ModuleMap::for_modules(&table, modules.iter().cloned()).unwrap();
        let pass = ModularizePbc {
            phys_table: Arc::new(table),
            map,
        };

        let mut mir = Mir::new();
        let v = (0..2)
            .map(|_| mir.resource_table_mut().add_qid(QidInfo::base(1)).unwrap())
            .collect::<Vec<_>>();

        let cases = [
            [
                ("ZXYIIYZZXYYZ", p[..12].to_owned()),
                ("ZXYIIYZZXYYZ", modules.clone()),
            ],
            [
                ("XYZ", vec![p[0], p[5], p[10]]),
                ("IXIIIIYIIIIZ", modules.clone()),
            ],
            [
                ("ZXZYX", vec![v[0], p[0], v[1], p[2], p[12]]),
                ("ZZIXIYX", vec![v[0], modules[0], v[1], p[12]]),
            ],
            [
                // Each group of four reverses internally.
                ("XIXYXZYIXXYZ", p[..12].iter().rev().copied().collect()),
                ("YXIXIYZXZYXX", modules.iter().rev().copied().collect()),
            ],
        ];

        let sym = mir
            .symbol_table_mut()
            .add(Symbol::new("c".into(), SymbolType::Bool))
            .unwrap();
        for [(pauli, qids), _] in cases.iter() {
            let ppm = pbc::PauliMeasure::new(pauli_from_ascii(pauli.as_bytes()).unwrap());
            mir.instructions_mut().push_back(MirInstruction::new(
                MirOp::PauliMeasure(ppm),
                QidList::new(qids.clone()),
                Default::default(),
                SymbolIdList::new(vec![sym]),
            ));
        }
        let expected = cases
            .into_iter()
            .map(|[_, (s, q)]| (s.to_owned(), q))
            .collect::<Vec<_>>();

        let (out, _) = pass.into_passmanager().run::<_, Mir>(mir).unwrap();
        let actual = out
            .instructions()
            .iter_main()
            .map(|n| {
                let inst = n.weight();
                let MirOp::PauliMeasure(ppm) = inst.op() else {
                    panic!("modularise produced a non-ppm")
                };
                (
                    pauli_to_ascii(ppm.paulis().as_slice()),
                    inst.qids().inner().to_owned(),
                )
            })
            .collect::<Vec<_>>();

        let mismatches = iter::zip(actual, expected)
            .filter(|(a, b)| a != b)
            .collect::<Vec<_>>();
        assert!(mismatches.is_empty(), "{}", fmt_failures(&mismatches))
    }

    #[test]
    fn nontrivial_bases_single() {
        let mut table = ResourceTable::new(QidSpace::Physical);
        let p = (0..4)
            .map(|len| table.add_qid(QidInfo::base(len)).unwrap())
            .collect::<Vec<_>>();
        let layouts = [
            vec![Some(p[2].bit(0)), Some(p[2].bit(1))],
            vec![Some(p[1].bit(0)), Some(p[2].bit(0)), Some(p[2].bit(1))],
            vec![Some(p[2].bit(1)), None, Some(p[2].bit(0))],
            vec![
                Some(p[3].bit(0)),
                Some(p[2].bit(0)),
                Some(p[3].bit(1)),
                Some(p[2].bit(1)),
                Some(p[3].bit(2)),
            ],
        ];
        let modules = layouts
            .into_iter()
            .map(|layout| {
                table
                    .add_qid(QidInfo::new(BitLayout::Alias(layout.into())))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let cases = [
            [("XY", vec![p[2]]), ("XY", vec![modules[0]])],
            [("XYZ", vec![p[1], p[2]]), ("XYZ", vec![modules[1]])],
            [("ZX", vec![p[2]]), ("XIZ", vec![modules[2]])],
            [("XYZXY", vec![p[2], p[3]]), ("XXYYZ", vec![modules[3]])],
        ];

        let table = Arc::new(table);
        let pm = |modules| {
            ModularizePbc {
                phys_table: table.clone(),
                map: ModuleMap::for_modules(&table, modules).unwrap(),
            }
            .into_passmanager()
        };
        let expected = cases
            .iter()
            .map(|[_, (s, q)]| (s.to_string(), q.clone()))
            .collect::<Vec<_>>();

        let to_mir = |pauli: &str, qids: Vec<Qid>| {
            let mut mir = Mir::new();
            let rot = pbc::PauliRotation::new(
                pauli_from_ascii(pauli.as_bytes()).unwrap(),
                pbc::AngleType::new(0.25),
            );
            mir.instructions_mut().push_back(MirInstruction::new(
                MirOp::PauliRotation(rot),
                QidList::new(qids.clone()),
                Default::default(),
                Default::default(),
            ));
            mir
        };
        let actual = cases
            .into_iter()
            .map(|[_, (pauli, qids)]| {
                let (mir, _) = pm(qids.clone()).run::<_, Mir>(to_mir(pauli, qids)).unwrap();
                let node = mir.instructions().iter_main().next().unwrap();
                let inst = node.weight();
                let MirOp::PauliRotation(rot) = inst.op() else {
                    panic!("modularise produced a non-rotation")
                };
                (
                    pauli_to_ascii(rot.paulis().as_slice()),
                    inst.qids().inner().to_owned(),
                )
            })
            .collect::<Vec<_>>();

        let mismatches = iter::zip(actual, expected)
            .filter(|(a, b)| a != b)
            .collect::<Vec<_>>();
        assert!(mismatches.is_empty(), "{}", fmt_failures(&mismatches))
    }

    #[test]
    fn new_overlap_detection() {
        let mut table = ResourceTable::new(QidSpace::Physical);
        let p = (0..2)
            .map(|_| table.add_qid(QidInfo::base(1)).unwrap())
            .collect::<Vec<_>>();
        let module = table
            .add_qid(QidInfo::new(BitLayout::Alias(
                vec![Some(p[0].bit(0)), Some(p[1].bit(0))].into(),
            )))
            .unwrap();
        let exclusive = table.add_qid(QidInfo::base(1)).unwrap();
        // `module` and `exclusive` are mutually exclusive, but `exclusive` isn't literally a bit in
        // the module layout.
        table.mark_contains(module, exclusive).unwrap();

        let pass = ModularizePbc {
            map: ModuleMap::for_modules(&table, [module]).unwrap(),
            phys_table: Arc::new(table),
        };
        let err = ModularizerState::new(&pass)
            .modularize(
                &ResourceTable::new(QidSpace::Virtual),
                &pauli_from_ascii(b"XYZ").unwrap(),
                &[p[0], p[1], exclusive],
            )
            .unwrap_err();
        assert!(matches!(err, ModularizerError::NewOverlap));
    }
}
