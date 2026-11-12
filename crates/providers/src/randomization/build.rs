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

//! Lower an annotated circuit into emits and collects. 

use std::sync::Arc;

use hashbrown::HashSet;
use thiserror::Error;
use uuid::Uuid;

use qiskit_circuit::annotation::Annotation;
use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::dag_circuit::{
    DAGCircuit, DAGCircuitBuilder, DAGError, DuplicateWireError, NodeIndex,
};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    BoxDuration, ControlFlow, ControlFlowInstruction, ControlFlowView, Operation, OperationRef,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Block, BlocksMode, Clbit, Qubit, VarsMode};

use super::annotations::{
    Dressing, InjectNoise, InjectionSite, InvalidAnnotationError, RandomizationAnnotations, Twirl,
    annotation_kind,
};
use super::emissions::{Collect, Distribution, Draw, Emit};
use super::{Direction, SynthesizerType};

/// Why a circuit could not be lowered.
#[derive(Debug, Error)]
pub enum BuildError {
    #[error("a distribution {width} qubits is not compatible with a box {box_width} qubits wide")]
    DistributionDoesNotDivideBox { width: u32, box_width: usize },
    #[error("only `box` control flow can be randomized, not `{0}`")]
    UnsupportedControlFlow(String),
    #[error("a box instruction has no body")]
    BoxMissingBody,
    #[error("qubit {0} is out of its scope")]
    QubitOutOfScope(usize),
    #[error("clbit {0} is out of its scope")]
    ClbitOutOfScope(usize),

    #[error(transparent)]
    InvalidAnnotation(#[from] InvalidAnnotationError),
    #[error(transparent)]
    Dag(#[from] DAGError),
    #[error(transparent)]
    DuplicateWire(#[from] DuplicateWireError),
}

pub type Result<T> = std::result::Result<T, BuildError>;

/// Lower an annotated circuit into emissions and collectors.
pub fn build(dag: &DAGCircuit) -> Result<DAGCircuit> {
    let identity_q: Vec<usize> = (0..dag.num_qubits()).collect();
    let identity_c: Vec<usize> = (0..dag.num_clbits()).collect();

    let mut out = dag
        .copy_empty_like_with_capacity(dag.num_ops(), 0, VarsMode::Alike, BlocksMode::Drop)
        .into_builder();
    let mut build = Build;
    let scope = Scope {
        qubits: &identity_q,
        clbits: &identity_c,
    };
    build.walk(dag, &mut out, &scope)?;
    Ok(out.build())
}

/// Locations within a box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Location {
    Start,
    Between,
    End,
}

impl Location {
    /// Locations written in chronological order.
    const ALL: [Location; 3] = [Location::Start, Location::Between, Location::End];

    /// The direction to the nearer collector.
    fn outward(self, dressing: Dressing) -> Direction {
        match (self, dressing) {
            (Location::Start, _) => Direction::Backward,
            (Location::End, _) => Direction::Forward,
            (Location::Between, Dressing::Left) => Direction::Backward,
            (Location::Between, Dressing::Right) => Direction::Forward,
        }
    }

    /// The direction to the hard content.
    fn inward(self, dressing: Dressing) -> Direction {
        self.outward(dressing).reverse()
    }
}

/// An emission to be inserted onto the given wires.
struct WiredEmit {
    emit: Emit,
    /// The wires to place the emit on as indices into its box's qargs.
    wires: Vec<u32>,
}

/// A collector to be inserted onto the given wires.
struct WiredCollect {
    collect: Collect,
    /// The wires to place the collect on as indices into its box's qargs.
    wires: Vec<u32>,
}

/// A box's emissions ordered by location.
struct EmissionLayout {
    /// Which side of the hard content a collector could absorb into.
    dressing: Dressing,
    runs: [Vec<WiredEmit>; 3],
}

impl EmissionLayout {
    /// An empty layout for a box with this dressing.
    fn new(dressing: Dressing) -> Self {
        EmissionLayout {
            dressing,
            runs: Default::default(),
        }
    }

    fn len(&self) -> usize {
        self.runs.iter().map(Vec::len).sum()
    }

    /// Take the emissions at `location`.
    fn take(&mut self, location: Location) -> Vec<WiredEmit> {
        std::mem::take(&mut self.runs[location as usize])
    }

    /// Add an emission at `location` inside everything already there.
    ///
    /// A location's emissions are one unbroken run so this separates nothing. It is a determinism
    /// choice and what it fixes is the sequence a collector composes them in.
    fn push_inward(&mut self, location: Location, emit: Emit, wires: Vec<u32>) {
        let dressing = self.dressing;
        let placed = WiredEmit { emit, wires };
        let run = &mut self.runs[location as usize];
        if location.outward(dressing) == Direction::Backward {
            run.push(placed);
        } else {
            run.insert(0, placed);
        }
    }
}

/// How a walked scope's qubits map into the output being written.
struct Scope<'a> {
    qubits: &'a [usize],
    clbits: &'a [usize],
}

impl Scope<'_> {
    fn out_qubits(&self, locals: &[Qubit]) -> Result<Vec<Qubit>> {
        locals
            .iter()
            .map(|q| {
                self.qubits
                    .get(q.index())
                    .map(|&i| Qubit(i as u32))
                    .ok_or(BuildError::QubitOutOfScope(q.index()))
            })
            .collect()
    }

    fn out_clbits(&self, locals: &[Clbit]) -> Result<Vec<Clbit>> {
        locals
            .iter()
            .map(|c| {
                self.clbits
                    .get(c.index())
                    .map(|&i| Clbit(i as u32))
                    .ok_or(BuildError::ClbitOutOfScope(c.index()))
            })
            .collect()
    }
}

/// State carried across the whole walk.
struct Build;

impl Build {
    /// Lower every operation in `dag` into `out`.
    fn walk(&mut self, dag: &DAGCircuit, out: &mut DAGCircuitBuilder, scope: &Scope) -> Result<()> {
        for node in dag.topological_op_nodes(false) {
            let inst = dag.dag()[node].unwrap_operation();
            match inst.op.view() {
                OperationRef::ControlFlow(cf) => {
                    if !matches!(cf.control_flow, ControlFlow::Box { .. }) {
                        return Err(BuildError::UnsupportedControlFlow(cf.name().to_string()));
                    }
                    self.walk_box(dag, inst, out, scope)?;
                }
                _ => copy_instruction(dag, inst, out, scope)?,
            }
        }
        Ok(())
    }

    /// Lower one annotated box or flatten it if it declares no randomization.
    fn walk_box(
        &mut self,
        dag: &DAGCircuit,
        inst: &PackedInstruction,
        out: &mut DAGCircuitBuilder,
        scope: &Scope,
    ) -> Result<()> {
        let Declaration {
            randomization,
            foreign,
            duration,
        } = declaration(inst)?;

        let locals = dag.qargs_interner().get(inst.qubits);
        let out_qargs = scope.out_qubits(locals)?;
        let out_cargs = scope.out_clbits(dag.cargs_interner().get(inst.clbits))?;
        let body = match dag.try_view_control_flow(inst) {
            Some(ControlFlowView::Box { body, .. }) => body,
            _ => return Err(BuildError::BoxMissingBody),
        };
        let num_clbits = dag.cargs_interner().get(inst.clbits).len();
        let width = locals.len();

        if !randomization.emits() {
            if foreign.is_empty() && duration.is_none() {
                let flat_q: Vec<usize> = out_qargs.iter().map(|q| q.index()).collect();
                let flat_c: Vec<usize> = out_cargs.iter().map(|c| c.index()).collect();
                let inner = Scope {
                    qubits: &flat_q,
                    clbits: &flat_c,
                };
                return self.walk(body, out, &inner);
            }
            let content = self.copied_body(body, width, num_clbits)?;
            return append_box(out, content, foreign, duration, &out_qargs, &out_cargs);
        }

        let dressing = randomization.dressing().unwrap_or(Dressing::Left);
        let synthesizer = randomization.synthesizer().unwrap_or(SynthesizerType::RzSx);
        let layout = self.build_emissions(&randomization, width, dressing)?;
        let content = self.content_body(body, width, num_clbits, layout)?;

        // A box gets a collector on each side and both start empty, so the second set is built
        // rather than cloned from the first.
        append_collectors(out, collector_wires(width, synthesizer), &out_qargs)?;
        append_box(out, content, foreign, duration, &out_qargs, &out_cargs)?;
        append_collectors(out, collector_wires(width, synthesizer), &out_qargs)?;
        Ok(())
    }

    /// Walk a box's body into a fresh body holding its emissions at the locations they belong to.
    fn copied_body(
        &mut self,
        body: &DAGCircuit,
        width: usize,
        num_clbits: usize,
    ) -> Result<DAGCircuit> {
        let identity_q: Vec<usize> = (0..width).collect();
        let identity_c: Vec<usize> = (0..num_clbits).collect();
        let inner = Scope {
            qubits: &identity_q,
            clbits: &identity_c,
        };
        let mut builder = new_dag_body(width, num_clbits, body.num_ops())?.into_builder();
        self.walk(body, &mut builder, &inner)?;
        Ok(builder.build())
    }

    /// Walk a box's body into a fresh body holding its emissions at the locations they belong to.
    fn content_body(
        &mut self,
        body: &DAGCircuit,
        width: usize,
        num_clbits: usize,
        mut layout: EmissionLayout,
    ) -> Result<DAGCircuit> {
        let identity_q: Vec<usize> = (0..width).collect();
        let identity_c: Vec<usize> = (0..num_clbits).collect();
        let inner = Scope {
            qubits: &identity_q,
            clbits: &identity_c,
        };
        let (easy_nodes, hard_nodes) = classify_body(body, layout.dressing);
        let mut builder =
            new_dag_body(width, num_clbits, body.num_ops() + layout.len())?.into_builder();

        let (first, second) = match layout.dressing {
            Dressing::Right => (&hard_nodes, &easy_nodes),
            Dressing::Left => (&easy_nodes, &hard_nodes),
        };
        for location in Location::ALL {
            if location == Location::Between {
                self.copy_nodes(body, first, &mut builder, &inner)?;
            }
            append_emissions(&mut builder, layout.take(location))?;
            if location == Location::Between {
                self.copy_nodes(body, second, &mut builder, &inner)?;
            }
        }
        Ok(builder.build())
    }

    /// Copy nodes from body while lowering any boxes in place.
    fn copy_nodes(
        &mut self,
        body: &DAGCircuit,
        nodes: &[NodeIndex],
        out: &mut DAGCircuitBuilder,
        inner: &Scope,
    ) -> Result<()> {
        for node in nodes {
            let inst = body.dag()[*node].unwrap_operation();
            match inst.op.view() {
                OperationRef::ControlFlow(cf) => {
                    if !matches!(cf.control_flow, ControlFlow::Box { .. }) {
                        return Err(BuildError::UnsupportedControlFlow(cf.name().to_string()));
                    }
                    self.walk_box(body, inst, out, inner)?;
                }
                _ => copy_instruction(body, inst, out, inner)?,
            }
        }
        Ok(())
    }

    /// Turn a box's annotations into emissions.
    fn build_emissions(
        &mut self,
        annotations: &RandomizationAnnotations,
        box_width: usize,
        dressing: Dressing,
    ) -> Result<EmissionLayout> {
        let mut layout = EmissionLayout::new(dressing);
        if let Some(injection) = annotations.inject_noise {
            self.noise_emissions(injection, box_width, dressing, &mut layout);
        }
        if let Some(twirl) = annotations.twirl {
            self.twirl_emissions(twirl, box_width, dressing, &mut layout)?;
        }
        Ok(layout)
    }

    /// The inverse pair a twirl declares for each subsystem the distribution covers.
    fn twirl_emissions(
        &mut self,
        twirl: &Twirl,
        box_width: usize,
        dressing: Dressing,
        layout: &mut EmissionLayout,
    ) -> Result<()> {
        let factorization = twirl.distribution.factorization();
        let width = factorization.num_elements() as u32;
        let location = Location::Between;
        for sample in subsystems(width, box_width)? {
            let draw = Arc::new(Draw {
                uuid: Uuid::new_v4(),
                distribution: Distribution::Sample(twirl.distribution),
            });
            for (factor, offsets) in factorization.iter().enumerate() {
                let wires: Vec<u32> = offsets
                    .iter()
                    .map(|offset| sample[*offset as usize])
                    .collect();
                let half = |direction, adjoint| Emit {
                    direction: Some(direction),
                    draw: Arc::clone(&draw),
                    factor: factor as u32,
                    adjoint,
                };
                layout.push_inward(
                    location,
                    half(location.outward(dressing), false),
                    wires.clone(),
                );
                layout.push_inward(location, half(location.inward(dressing), true), wires);
            }
        }
        Ok(())
    }

    /// The emissions a noise injection declares: one per qubit and all of one sample.
    fn noise_emissions(
        &mut self,
        injection: &InjectNoise,
        box_width: usize,
        dressing: Dressing,
        layout: &mut EmissionLayout,
    ) {
        let draw = Arc::new(Draw {
            uuid: Uuid::new_v4(),
            distribution: Distribution::Noise {
                reference: injection.reference.as_str().into(),
                modifier: injection.modifier.as_deref().map(Box::from),
                num_qubits: box_width as u32,
            },
        });
        // A site names a side of the hard content. One of those sides is the twirl point and the
        // other is the box end beyond the content, which is what the dressing decides.
        let location = match (injection.site, dressing) {
            (InjectionSite::Between, _) => Location::Between,
            (InjectionSite::Before, Dressing::Left) => Location::Between,
            (InjectionSite::Before, Dressing::Right) => Location::Start,
            (InjectionSite::After, Dressing::Left) => Location::End,
            (InjectionSite::After, Dressing::Right) => Location::Between,
        };
        for factor in 0..box_width {
            let emit = Emit {
                direction: Some(location.outward(dressing)),
                draw: Arc::clone(&draw),
                factor: factor as u32,
                adjoint: false,
            };
            layout.push_inward(location, emit, vec![slot as u32]);
        }
    }
}

/// Split a box's qubits into the subsystems a distribution of this width covers.
fn subsystems(width: u32, box_width: usize) -> Result<Vec<Vec<u32>>> {
    if width == 0 || !box_width.is_multiple_of(width as usize) {
        return Err(BuildError::DistributionDoesNotDivideBox { width, box_width });
    }
    Ok((0..box_width as u32)
        .step_by(width as usize)
        .map(|start| (start..start + width).collect())
        .collect())
}

/// One collector per subsystem a synthesizer covers with the wires each sits on.
fn collector_wires(box_width: usize, synthesizer: SynthesizerType) -> Vec<WiredCollect> {
    let width = synthesizer.num_qubits().max(1) as usize;
    (0..box_width)
        .step_by(width)
        .map(|start| {
            let wires = (start..(start + width).min(box_width))
                .map(|q| q as u32)
                .collect();
            WiredCollect {
                collect: Collect::new(synthesizer),
                wires,
            }
        })
        .collect()
}

/// Sweep a box body from the dressing edge to split the absorbable run from the rest.
fn classify_body(body: &DAGCircuit, dressing: Dressing) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
    let nodes: Vec<_> = body.topological_op_nodes(false).collect();
    let right = matches!(dressing, Dressing::Right);
    let order: Vec<_> = if right {
        nodes.into_iter().rev().collect()
    } else {
        nodes
    };

    let mut poisoned: HashSet<usize> = HashSet::new();
    let mut easy_nodes = Vec::new();
    let mut hard_nodes = Vec::new();
    for node in order {
        let inst = body.dag()[node].unwrap_operation();
        let qargs = body.qargs_interner().get(inst.qubits);
        let absorbable = is_absorbable(body, inst) && !poisoned.contains(&qargs[0].index());
        if absorbable {
            easy_nodes.push(node);
        } else {
            poisoned.extend(qargs.iter().map(|q| q.index()));
            hard_nodes.push(node);
        }
    }
    if right {
        easy_nodes.reverse();
        hard_nodes.reverse();
    }
    (easy_nodes, hard_nodes)
}

/// Whether a dressing could absorb this instruction.
fn is_absorbable(dag: &DAGCircuit, inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::StandardGate(_))
        && dag.qargs_interner().get(inst.qubits).len() == 1
}

/// What a box declares and what it carries that is not ours.
struct Declaration<'a> {
    /// The randomization the box asks for as resolved from its annotations.
    randomization: RandomizationAnnotations<'a>,
    /// The annotations we do not interpret.
    foreign: Vec<Arc<dyn Annotation>>,
    /// Carried onto the content box for the same reason as a foreign annotation.
    duration: Option<BoxDuration>,
}

/// Read what a box declares.
fn declaration(inst: &PackedInstruction) -> Result<Declaration<'_>> {
    let OperationRef::ControlFlow(cf) = inst.op.view() else {
        return Err(BuildError::BoxMissingBody);
    };
    let ControlFlow::Box {
        annotations,
        duration,
    } = &cf.control_flow
    else {
        return Err(BuildError::BoxMissingBody);
    };
    Ok(Declaration {
        randomization: RandomizationAnnotations::new(annotations)?,
        foreign: annotations
            .iter()
            .filter(|a| annotation_kind(a.as_ref()).is_none())
            .cloned()
            .collect(),
        duration: duration.clone(),
    })
}

/// Create an empty body with the given dimensions and anonymous wires.
fn new_dag_body(num_qubits: usize, num_clbits: usize, capacity: usize) -> Result<DAGCircuit> {
    let mut body =
        DAGCircuit::with_capacity(num_qubits, num_clbits, None, Some(capacity), None, None);
    for _ in 0..num_qubits {
        body.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
    }
    for _ in 0..num_clbits {
        body.add_clbit_unchecked(ShareableClbit::new_anonymous())?;
    }
    Ok(body)
}

/// Append an operation to the back of a DAG under construction.
fn append(
    out: &mut DAGCircuitBuilder,
    op: PackedOperation,
    params: Option<Parameters<Block>>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> Result<()> {
    out.apply_operation_back(op, qargs, cargs, params, None)?;
    Ok(())
}

/// Copy an instruction into `out` while remapping its bits through `scope`.
fn copy_instruction(
    dag: &DAGCircuit,
    inst: &PackedInstruction,
    out: &mut DAGCircuitBuilder,
    scope: &Scope,
) -> Result<()> {
    let qargs = scope.out_qubits(dag.qargs_interner().get(inst.qubits))?;
    let cargs = scope.out_clbits(dag.cargs_interner().get(inst.clbits))?;
    let params = (!inst.params_view().is_empty())
        .then(|| Parameters::Params(inst.params_view().iter().cloned().collect()));
    append(out, inst.op.clone(), params, &qargs, &cargs)
}

/// Append emissions in the given order.
fn append_emissions(out: &mut DAGCircuitBuilder, emissions: Vec<WiredEmit>) -> Result<()> {
    for wired in emissions {
        let qargs: Vec<Qubit> = wired.wires.iter().map(|w| Qubit(*w)).collect();
        let op = PackedOperation::from_custom_operation(Box::new(wired.emit));
        append(out, op, None, &qargs, &[])?;
    }
    Ok(())
}

/// Append collectors in the given order.
fn append_collectors(
    out: &mut DAGCircuitBuilder,
    collectors: Vec<WiredCollect>,
    qargs: &[Qubit],
) -> Result<()> {
    for wired in collectors {
        let on: Vec<Qubit> = wired.wires.iter().map(|w| qargs[*w as usize]).collect();
        let op = PackedOperation::from_custom_operation(Box::new(wired.collect));
        append(out, op, None, &on, &[])?;
    }
    Ok(())
}

/// Append a `box` with this body and these annotations.
fn append_box(
    out: &mut DAGCircuitBuilder,
    body: DAGCircuit,
    annotations: Vec<Arc<dyn Annotation>>,
    duration: Option<BoxDuration>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> Result<()> {
    let op = PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration,
            annotations,
        },
        num_qubits: qargs.len() as u32,
        num_clbits: cargs.len() as u32,
    }));
    let block = out.add_block(body);
    append(out, op, Some(Parameters::Blocks(vec![block])), qargs, cargs)
}

#[cfg(test)]
mod tests {
    use super::*;

    use qiskit_circuit::bit::QuantumRegister;
    use qiskit_circuit::operations::StandardGate;

    use crate::randomization::DistributionType;
    use crate::randomization::annotations::InjectionSite;
    use crate::randomization::emissions::{Collect, Emit};

    /// A DAG of `width` qubits holding the operations `fill` appends.
    fn circuit(width: u32, fill: impl FnOnce(&mut DAGCircuitBuilder)) -> DAGCircuit {
        let mut dag = DAGCircuit::new();
        dag.add_qreg(QuantumRegister::new_owning("q", width))
            .expect("a fresh register");
        let mut builder = dag.into_builder();
        fill(&mut builder);
        builder.build()
    }

    /// Append a single-qubit gate.
    fn entangler(builder: &mut DAGCircuitBuilder, control: u32, target: u32) {
        builder
            .apply_operation_back(
                PackedOperation::from_standard_gate(StandardGate::CX),
                &[Qubit(control), Qubit(target)],
                &[],
                None,
                None,
            )
            .expect("an entangler on qubits the circuit has");
    }

    fn gate(builder: &mut DAGCircuitBuilder, gate: StandardGate, qubit: u32) {
        builder
            .apply_operation_back(
                PackedOperation::from_standard_gate(gate),
                &[Qubit(qubit)],
                &[],
                None,
                None,
            )
            .expect("a gate on a qubit the circuit has");
    }

    /// Append a box over `qubits` carrying `annotations` and holding `body`.
    fn annotated_box(
        builder: &mut DAGCircuitBuilder,
        qubits: &[u32],
        annotations: Vec<Arc<dyn Annotation>>,
        body: DAGCircuit,
    ) {
        let qargs: Vec<Qubit> = qubits.iter().copied().map(Qubit).collect();
        append_box(builder, body, annotations, None, &qargs, &[]).expect("a box on real wires");
    }

    fn twirl(distribution: DistributionType, dressing: Dressing) -> Arc<dyn Annotation> {
        Arc::new(Twirl {
            distribution,
            dressing,
            decomposition: SynthesizerType::RzSx,
        })
    }

    fn noise(reference: &str, site: InjectionSite) -> Arc<dyn Annotation> {
        Arc::new(InjectNoise {
            reference: reference.to_string(),
            modifier: None,
            site,
        })
    }

    /// Every operation of a circuit flattened into `(name, qubits)` in topological order.
    fn ops_in_order(dag: &DAGCircuit) -> Vec<&PackedInstruction> {
        dag.topological_op_nodes(false)
            .map(|node| dag.dag()[node].unwrap_operation())
            .collect()
    }

    fn shape(dag: &DAGCircuit) -> Vec<(String, Vec<u32>)> {
        let mut out = Vec::new();
        for inst in ops_in_order(dag) {
            let qubits: Vec<u32> = dag
                .qargs_interner()
                .get(inst.qubits)
                .iter()
                .map(|q| q.0)
                .collect();
            out.push((inst.op.view().name().to_string(), qubits));
            if let [block] = inst.blocks_view() {
                out.extend(shape(&dag.blocks()[*block]));
            }
        }
        out
    }

    /// The wired emissions of a circuit.
    fn emissions(dag: &DAGCircuit) -> Vec<WiredEmit> {
        let mut out = Vec::new();
        for inst in ops_in_order(dag) {
            if let OperationRef::CustomOperation(op) = inst.op.view()
                && let Some(emit) = op.downcast_ref::<Emit>()
            {
                let qubits = dag
                    .qargs_interner()
                    .get(inst.qubits)
                    .iter()
                    .map(|q| q.0)
                    .collect();
                out.push(WiredEmit {
                    emit: emit.clone(),
                    wires: qubits,
                });
            }
            if let [block] = inst.blocks_view() {
                out.extend(emissions(&dag.blocks()[*block]));
            }
        }
        out
    }

    /// The wired collectors of a circuit.
    fn collectors(dag: &DAGCircuit) -> Vec<WiredCollect> {
        let mut out = Vec::new();
        for inst in ops_in_order(dag) {
            if let OperationRef::CustomOperation(op) = inst.op.view()
                && let Some(collect) = op.downcast_ref::<Collect>()
            {
                let qubits = dag
                    .qargs_interner()
                    .get(inst.qubits)
                    .iter()
                    .map(|q| q.0)
                    .collect();
                out.push(WiredCollect {
                    collect: collect.clone(),
                    wires: qubits,
                });
            }
            if let [block] = inst.blocks_view() {
                out.extend(collectors(&dag.blocks()[*block]));
            }
        }
        out
    }

    #[test]
    fn test_a_twirled_box_gets_collectors_and_a_pair() {
        let body = circuit(1, |b| gate(b, StandardGate::H, 0));
        let lowered = build(&circuit(1, |b| {
            annotated_box(
                b,
                &[0],
                vec![twirl(DistributionType::UniformPauli, Dressing::Left)],
                body,
            )
        }))
        .expect("a twirled box lowers");

        assert_eq!(collectors(&lowered).len(), 2, "one collector on each side");
        let pair = emissions(&lowered);
        assert_eq!(pair.len(), 2, "a twirl is one pair");
        assert_eq!(pair[0].emit.draw.uuid, pair[1].emit.draw.uuid);
        assert_ne!(pair[0].emit.adjoint, pair[1].emit.adjoint);
        assert_ne!(pair[0].emit.direction, pair[1].emit.direction);
    }

    #[test]
    fn test_a_distribution_divides_its_box() {
        let lowered = build(&circuit(4, |b| {
            annotated_box(
                b,
                &[0, 1, 2, 3],
                vec![twirl(DistributionType::UniformPauli, Dressing::Left)],
                circuit(4, |_| {}),
            )
        }))
        .expect("a one-qubit distribution divides any box");
        let pair = emissions(&lowered);
        assert_eq!(pair.len(), 8, "four subsystems, a pair each");
        for WiredEmit { emit, wires } in &pair {
            assert_eq!(wires.len(), 1, "one qubit per sample");
            assert_eq!(emit.num_qubits(), 1);
        }
        let uuids: HashSet<Uuid> = pair.iter().map(|wired| wired.emit.draw.uuid).collect();
        assert_eq!(uuids.len(), 4);
    }

    #[test]
    fn test_a_correlated_distribution_is_placed_one_slot_at_a_time() {
        let lowered = build(&circuit(4, |b| {
            annotated_box(
                b,
                &[0, 1, 2, 3],
                vec![twirl(DistributionType::UniformLocalC1, Dressing::Left)],
                circuit(4, |_| {}),
            )
        }))
        .expect("two divides four");
        let pair = emissions(&lowered);
        assert_eq!(
            pair.len(),
            8,
            "two samples, two slots each, a pair per slot"
        );
        for WiredEmit { emit, wires } in &pair {
            assert_eq!(wires.len(), 1, "a local-C1 slot is one qubit");
            assert_eq!(emit.num_qubits(), 1);
        }
        let uuids: HashSet<Uuid> = pair.iter().map(|wired| wired.emit.draw.uuid).collect();
        assert_eq!(uuids.len(), 2);
        let first = pair[0].emit.draw.uuid;
        let wires: HashSet<u32> = pair
            .iter()
            .filter(|wired| wired.emit.draw.uuid == first)
            .flat_map(|wired| wired.wires.iter().copied())
            .collect();
        assert_eq!(
            wires.len(),
            2,
            "one sample is connected to the two qubits it correlates"
        );
    }

    #[test]
    fn test_a_distribution_that_does_not_divide_its_box_is_refused() {
        let error = build(&circuit(3, |b| {
            annotated_box(
                b,
                &[0, 1, 2],
                vec![twirl(DistributionType::UniformLocalC1, Dressing::Left)],
                circuit(3, |_| {}),
            )
        }))
        .expect_err("two does not divide three");
        assert!(matches!(
            error,
            BuildError::DistributionDoesNotDivideBox {
                width: 2,
                box_width: 3
            }
        ));
    }

    #[test]
    fn test_noise_factorizes_into_one_emission_per_qubit() {
        let lowered = build(&circuit(3, |b| {
            annotated_box(
                b,
                &[0, 1, 2],
                vec![
                    twirl(DistributionType::UniformPauli, Dressing::Left),
                    noise("layer_1", InjectionSite::Before),
                ],
                circuit(3, |_| {}),
            )
        }))
        .expect("noise beside a twirl lowers");

        let noise_emissions: Vec<_> = emissions(&lowered)
            .into_iter()
            .filter(|wired| matches!(wired.emit.draw.distribution, Distribution::Noise { .. }))
            .collect();
        assert_eq!(noise_emissions.len(), 3, "one per qubit of the box");
        let uuids: HashSet<Uuid> = noise_emissions
            .iter()
            .map(|wired| wired.emit.draw.uuid)
            .collect();
        assert_eq!(uuids.len(), 1);
        let slots: Vec<u32> = noise_emissions
            .iter()
            .map(|wired| wired.emit.slot)
            .collect();
        assert_eq!(slots, vec![0, 1, 2]);
    }

    #[test]
    fn test_every_slot_of_one_injection_shares_one_description() {
        let lowered = build(&circuit(5, |b| {
            annotated_box(
                b,
                &[0, 1, 2, 3, 4],
                vec![
                    twirl(DistributionType::UniformPauli, Dressing::Left),
                    noise("layer_1", InjectionSite::Before),
                ],
                circuit(5, |_| {}),
            )
        }))
        .expect("a noise injection lowers");

        let draws: Vec<Arc<Draw>> = emissions(&lowered)
            .into_iter()
            .filter(|wired| matches!(wired.emit.draw.distribution, Distribution::Noise { .. }))
            .map(|wired| wired.emit.draw)
            .collect();
        assert_eq!(draws.len(), 5, "one slot per qubit of the box");
        for draw in &draws[1..] {
            assert_eq!(
                &draws[0], draw,
                "one draw whatever allocation it is read from"
            );
        }
        assert_eq!(draws[0].distribution.num_slots(), 5);
    }

    #[test]
    fn test_an_emission_travels_towards_the_collector_it_resolves_at() {
        for (dressing, outward) in [
            (Dressing::Left, Direction::Backward),
            (Dressing::Right, Direction::Forward),
        ] {
            let lowered = build(&circuit(1, |b| {
                annotated_box(
                    b,
                    &[0],
                    vec![twirl(DistributionType::UniformPauli, dressing)],
                    circuit(1, |b| gate(b, StandardGate::H, 0)),
                )
            }))
            .expect("a twirled box lowers");

            let pair = emissions(&lowered);
            assert_eq!(pair.len(), 2, "{dressing:?}");
            let near = pair
                .iter()
                .find(|wired| !wired.emit.adjoint)
                .expect("the pair has a near half");
            assert_eq!(near.emit.direction, Some(outward), "{dressing:?}");
            let far = pair
                .iter()
                .find(|wired| wired.emit.adjoint)
                .expect("the pair has a far half");
            assert_eq!(far.emit.direction, Some(outward.reverse()), "{dressing:?}");
        }
    }

    #[test]
    fn test_an_injection_sits_where_its_site_says_and_faces_away_from_the_content() {
        for (site, outward) in [
            (InjectionSite::Before, Direction::Backward),
            (InjectionSite::After, Direction::Forward),
        ] {
            let lowered = build(&circuit(1, |b| {
                annotated_box(
                    b,
                    &[0],
                    vec![
                        twirl(DistributionType::UniformPauli, Dressing::Left),
                        noise("layer_1", site),
                    ],
                    circuit(1, |b| gate(b, StandardGate::H, 0)),
                )
            }))
            .expect("a noise injection lowers");

            let injected: Vec<Emit> = emissions(&lowered)
                .into_iter()
                .filter(|wired| matches!(wired.emit.draw.distribution, Distribution::Noise { .. }))
                .map(|wired| wired.emit)
                .collect();
            assert_eq!(injected.len(), 1, "{site:?}");
            assert_eq!(injected[0].direction, Some(outward), "{site:?}");
            assert!(!injected[0].adjoint, "an injection is never undone");
        }
    }

    #[test]
    fn test_a_boxs_emissions_sit_together_at_the_twirl_point() {
        let lowered = build(&circuit(2, |b| {
            annotated_box(
                b,
                &[0, 1],
                vec![
                    twirl(DistributionType::UniformPauli, Dressing::Left),
                    noise("layer_1", InjectionSite::Before),
                ],
                circuit(2, |b| entangler(b, 0, 1)),
            )
        }))
        .expect("a twirled box with an injection lowers");

        let block = ops_in_order(&lowered)
            .iter()
            .find_map(|inst| match inst.blocks_view() {
                [block] => Some(*block),
                _ => None,
            })
            .expect("the content box");
        let names: Vec<String> = ops_in_order(&lowered.blocks()[block])
            .iter()
            .map(|inst| inst.op.view().name().to_string())
            .collect();

        let hard = names
            .iter()
            .position(|name| name == "cx")
            .expect("the entangler stays hard");
        assert!(
            names[..hard].iter().all(|name| name == "emit"),
            "every emission is at the twirl point: {names:?}"
        );
        assert_eq!(hard, names.len() - 1, "and the content follows them all");

        assert_eq!(hard, 6);
        let kinds = emissions(&lowered);
        assert_eq!(
            kinds
                .iter()
                .filter(|wired| matches!(wired.emit.draw.distribution, Distribution::Noise { .. }))
                .count(),
            2
        );

        let twirls: Vec<&Emit> = kinds
            .iter()
            .map(|wired| &wired.emit)
            .filter(|emit| matches!(emit.draw.distribution, Distribution::Sample(_)))
            .collect();
        assert_eq!(twirls.len(), 4);
        for wire in [0, 1] {
            let halves: Vec<&&Emit> = twirls
                .iter()
                .filter(|emit| emit.draw.uuid == twirls[wire * 2].draw.uuid)
                .collect();
            assert_eq!(halves.len(), 2, "one pair per wire");
            assert_ne!(halves[0].adjoint, halves[1].adjoint);
            assert_eq!(
                halves[0].direction,
                halves[1].direction.map(Direction::reverse)
            );
        }
    }

    #[test]
    fn test_the_absorbable_run_sits_on_the_dressing_side_of_the_twirl_point() {
        // The twirl point is the easy/hard boundary rather than the box boundary. Whichever side
        // the dressing names carries the absorbable run and the emissions sit against it so those
        // gates multiply into the dressing. Writing them at the box boundary instead would leave
        // every gate in the body to be crossed and a non-Clifford cannot be crossed by a Pauli at
        // all.
        //
        // Absorbability is DAG ancestry so which gate is absorbable depends on the dressing: a
        // leading single-qubit gate can sweep forward to a left dressing while a trailing one can
        // sweep back to a right dressing. Each body here is ordered to give its dressing a run.
        for (dressing, body, expected) in [
            (
                Dressing::Left,
                circuit(2, |b| {
                    gate(b, StandardGate::H, 0);
                    entangler(b, 0, 1);
                }),
                vec!["h", "emit", "emit", "cx"],
            ),
            (
                Dressing::Right,
                circuit(2, |b| {
                    entangler(b, 0, 1);
                    gate(b, StandardGate::H, 0);
                }),
                vec!["cx", "emit", "emit", "h"],
            ),
        ] {
            let lowered = build(&circuit(2, |b| {
                annotated_box(
                    b,
                    &[0, 1],
                    vec![twirl(DistributionType::UniformPauli, dressing)],
                    body,
                )
            }))
            .expect("a twirled box lowers");

            let block = ops_in_order(&lowered)
                .iter()
                .find_map(|inst| match inst.blocks_view() {
                    [block] => Some(*block),
                    _ => None,
                })
                .expect("the content box");
            // One wire's view, because a topological read of the whole body interleaves wires that
            // do not depend on each other and so says nothing about where anything sits.
            let body = &lowered.blocks()[block];
            let on_wire: Vec<String> = ops_in_order(body)
                .iter()
                .filter(|inst| body.qargs_interner().get(inst.qubits).contains(&Qubit(0)))
                .map(|inst| inst.op.view().name().to_string())
                .collect();
            assert_eq!(on_wire, expected, "{dressing:?}");
        }
    }

    #[test]
    fn test_a_box_declaring_nothing_is_dissolved() {
        let body = circuit(1, |b| gate(b, StandardGate::H, 0));
        let lowered = build(&circuit(1, |b| annotated_box(b, &[0], Vec::new(), body)))
            .expect("an unannotated box lowers");
        assert_eq!(shape(&lowered), vec![("h".to_string(), vec![0])]);
        assert!(emissions(&lowered).is_empty() && collectors(&lowered).is_empty());
    }

    /// An annotation from outside this vocabulary.
    #[derive(Debug, Clone, PartialEq)]
    struct Foreign;

    impl Annotation for Foreign {
        fn namespace(&self) -> &str {
            "someone.else"
        }
    }

    #[test]
    fn test_a_box_we_do_not_randomize_is_kept_with_its_body_copied() {
        // A box carrying something we do not interpret stays a box because dissolving it would
        // discard that. There is no twirl point in it so its body is copied rather than split
        // about one and a nested annotated box inside it still lowers.
        let inner = circuit(1, |b| gate(b, StandardGate::H, 0));
        let body = circuit(1, |b| {
            gate(b, StandardGate::S, 0);
            annotated_box(
                b,
                &[0],
                vec![twirl(DistributionType::UniformPauli, Dressing::Left)],
                inner,
            );
            gate(b, StandardGate::T, 0);
        });
        let lowered = build(&circuit(1, |b| {
            annotated_box(
                b,
                &[0],
                vec![Arc::new(Foreign) as Arc<dyn Annotation>],
                body,
            )
        }))
        .expect("a box with a foreign annotation lowers");

        // The outer box survives and its contents keep the order they were written in.
        assert_eq!(
            shape(&lowered),
            vec![
                ("box".to_string(), vec![0]),
                ("s".to_string(), vec![0]),
                ("collect".to_string(), vec![0]),
                ("box".to_string(), vec![0]),
                ("h".to_string(), vec![0]),
                ("emit".to_string(), vec![0]),
                ("emit".to_string(), vec![0]),
                ("collect".to_string(), vec![0]),
                ("t".to_string(), vec![0]),
            ]
        );
        // Nothing was absorbed into the outer box: a box we do not randomize has no dressing, so
        // the `s` and `t` stay where they were rather than sweeping to an edge.
        assert_eq!(emissions(&lowered).len(), 2);
        assert_eq!(collectors(&lowered).len(), 2);
    }

    #[test]
    fn test_control_flow_that_is_not_a_box_is_refused() {
        let dag = circuit(1, |builder| {
            builder
                .apply_operation_back(
                    PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
                        control_flow: ControlFlow::BreakLoop,
                        num_qubits: 1,
                        num_clbits: 0,
                    })),
                    &[Qubit(0)],
                    &[],
                    None,
                    None,
                )
                .expect("a break on a real wire");
        });
        let error = build(&dag).expect_err("only a box can be randomized");
        assert!(matches!(error, BuildError::UnsupportedControlFlow(_)));
    }
}
