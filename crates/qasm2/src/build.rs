// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Build a native [CircuitData] directly from the parser's bytecode stream, without going
//! through Python.

use std::fmt;
use std::sync::Arc;

use num_bigint::BigUint;

use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Condition, ControlFlow, ControlFlowInstruction, CustomOperation, Operation, Param,
    StandardInstruction,
};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::standard_gate::StandardGate;
use qiskit_circuit::{Clbit, Qubit};

use crate::bytecode::InternalBytecode;
use crate::error::{ParseError, message_generic};
use crate::expr::{Expr, evaluate};
use crate::ext::ClassicalEvaluator;
use crate::parse::{ClbitId, CregId, GateId, QELIB1_STANDARD_GATES, QubitId};

/// Either a native gate, or a gate declared by an OQ2 `gate`/`opaque` statement.
#[derive(Clone)]
enum GateEntry {
    Standard(StandardGate),
    Defined(Arc<DefinedGateTemplate>),
}

impl GateEntry {
    fn num_qubits(&self) -> u32 {
        match self {
            GateEntry::Standard(gate) => gate.num_qubits(),
            GateEntry::Defined(template) => template.num_qubits,
        }
    }
}

/// A single instruction recorded from inside a `gate`/`opaque` body; see [DefinedGateTemplate].
enum BodyInstruction {
    Gate {
        entry: GateEntry,
        arguments: Vec<Expr>,
        qubits: Vec<QubitId>,
    },
    Barrier {
        qubits: Vec<QubitId>,
    },
}

/// The shared "recipe" for an OQ2 `gate`/`opaque` statement, declared once and reused by every
/// usage.  `body` is `None` for `opaque` gates, whose definition is unknown.  Body arguments are
/// `Expr` trees rather than numbers, since they can reference this gate's *own* symbolic
/// parameters (e.g. `gate rz(theta) q { u1(theta) q; }`); they're only evaluated to numbers once
/// a specific usage's concrete parameters are known, in `build_definition`.
struct DefinedGateTemplate {
    name: String,
    num_qubits: u32,
    body: Option<Vec<BodyInstruction>>,
}

impl DefinedGateTemplate {
    /// Body parameters are evaluated with a detached [ClassicalEvaluator]:
    /// [CustomOperation::definition] hands us no interpreter token, and a C caller may not have
    /// an initialised interpreter at all.  A Python custom classical function reaching here is
    /// therefore reported as an error rather than panicking.
    ///
    /// Returns [None] rather than an error for anything that cannot be evaluated -- a
    /// non-[Param::Float] parameter, an out-of-range parameter index, or a classical function that
    /// fails -- because [CustomOperation::definition] gives us no error channel.  Python's
    /// `_DefinedGate` raises `QASM2ParseError` for the same inputs, so a gate is left without a
    /// definition here where Python would have complained; giving this an error channel is a
    /// follow-up.
    fn build_definition(&self, params: &[Param]) -> Option<CircuitData> {
        let body = self.body.as_ref()?;
        let float_params: Vec<f64> = params
            .iter()
            .map(|p| match p {
                Param::Float(v) => Some(*v),
                _ => None,
            })
            .collect::<Option<_>>()?;

        let mut circuit = CircuitData::new(None, None, Param::Float(0.0)).ok()?;
        circuit.add_anonymous_qubits(self.num_qubits).ok()?;

        for instruction in body {
            match instruction {
                BodyInstruction::Gate {
                    entry,
                    arguments,
                    qubits,
                } => {
                    let arguments: Vec<f64> = arguments
                        .iter()
                        .map(|expr| {
                            evaluate(expr, &float_params, ClassicalEvaluator::detached()).ok()
                        })
                        .collect::<Option<_>>()?;
                    push_gate(&mut circuit, entry, &arguments, &to_qubits(qubits)).ok()?;
                }
                BodyInstruction::Barrier { qubits } => {
                    push_standard_instruction_local(
                        &mut circuit,
                        StandardInstruction::Barrier(qubits.len() as u32),
                        &to_qubits(qubits),
                        &[],
                    )
                    .ok()?;
                }
            }
        }

        Some(circuit)
    }
}

/// One usage of an OQ2-defined gate: shares its recipe ([DefinedGateTemplate]) but gets its own
/// params, matching Python's `_DefinedGate` (see `_gate_builder` in `parse.py`).
///
/// Deliberately doesn't cache its built definition: `CircuitData::assign_parameters_inner` rebinds
/// a `CustomOperation`'s params via `PackedInstruction::params_mut` without touching the operation
/// itself, so there's no hook here to invalidate a cache -- a stale one would be a silent,
/// hard-to-find bug.
#[derive(Clone)]
struct DefinedGate {
    template: Arc<DefinedGateTemplate>,
    num_params: u32,
}

impl fmt::Debug for DefinedGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefinedGate")
            .field("name", &self.template.name)
            .field("num_params", &self.num_params)
            .finish()
    }
}

/// Compares by template identity, not structurally (`Expr` trees aren't `PartialEq`) -- two
/// structurally-identical `DefinedGate`s from separate `build_circuit` calls won't compare equal.
impl PartialEq for DefinedGate {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.template, &other.template) && self.num_params == other.num_params
    }
}

impl Operation for DefinedGate {
    fn name(&self) -> &str {
        &self.template.name
    }
    fn num_qubits(&self) -> u32 {
        self.template.num_qubits
    }
    fn num_clbits(&self) -> u32 {
        0
    }
    fn num_params(&self) -> u32 {
        self.num_params
    }
    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for DefinedGate {
    fn is_unitary(&self) -> bool {
        true
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        self.template.build_definition(params)
    }
}

/// Maps each `GateId` to its native gate (see [GateEntry]).  Starts with `U`=0, `CX`=1, matching
/// `State::new` in `parse.rs` -- which only holds because no caller yet supplies its own
/// `CustomInstruction`s to the parser (unrelated to OQ2's `gate`/`opaque` statements, despite the
/// similar name).  Adding that support later must revisit this, or gate ids will silently resolve
/// to the wrong gate.
struct GateRegistry {
    gates: Vec<GateEntry>,
}

impl GateRegistry {
    fn new() -> Self {
        Self {
            gates: vec![
                GateEntry::Standard(StandardGate::U),
                GateEntry::Standard(StandardGate::CX),
            ],
        }
    }

    fn extend_from_qelib1(&mut self, indices: &[usize]) {
        for &index in indices {
            self.gates
                .push(GateEntry::Standard(QELIB1_STANDARD_GATES[index]));
        }
    }

    fn declare(&mut self, name: String, num_qubits: u32, body: Option<Vec<BodyInstruction>>) {
        self.gates
            .push(GateEntry::Defined(Arc::new(DefinedGateTemplate {
                name,
                num_qubits,
                body,
            })));
    }

    fn get(&self, id: GateId) -> Result<GateEntry, ParseError> {
        self.gates.get(id.index()).cloned().ok_or_else(|| {
            ParseError::new(message_generic(
                None,
                &format!("gate id {} was not declared", id.index()),
            ))
        })
    }
}

/// Build a [CircuitData] from a bytecode stream.  See [GateRegistry] for a caveat about custom
/// instructions.
///
/// Takes an already-materialised slice rather than the crate's usual lazily-streamed iterator
/// (see `bytecode_from_string`/`bytecode_from_file` in `lib.rs`).  Interleaving the parse and the
/// build, so that neither side needs the whole program resident, would mean hoisting the loop state
/// below into a struct that survives across calls; that is left for a follow-up.
pub(crate) fn build_circuit(bytecode: &[InternalBytecode]) -> Result<CircuitData, ParseError> {
    let mut circuit = CircuitData::new(None, None, Param::Float(0.0))
        .map_err(|err| ParseError::new(format!("failed to create circuit: {err}")))?;
    let mut registry = GateRegistry::new();
    let mut cregs: Vec<ClassicalRegister> = Vec::new();
    // Set while recording a `gate`/`opaque` body, between `DeclareGate` and `EndDeclareGate`.
    let mut current_body: Option<(String, u32, Vec<BodyInstruction>)> = None;

    for instruction in bytecode {
        match instruction {
            InternalBytecode::Gate {
                id,
                arguments,
                qubits,
            } => {
                let entry = registry.get(*id)?;
                push_gate(&mut circuit, &entry, arguments, &to_qubits(qubits))?;
            }
            InternalBytecode::Measure { qubit, clbit } => {
                push_standard_instruction(
                    &mut circuit,
                    StandardInstruction::Measure,
                    &[*qubit],
                    &[*clbit],
                )?;
            }
            InternalBytecode::Reset { qubit } => {
                push_standard_instruction(
                    &mut circuit,
                    StandardInstruction::Reset,
                    &[*qubit],
                    &[],
                )?;
            }
            InternalBytecode::Barrier { qubits } => {
                if let Some((_, _, body)) = current_body.as_mut() {
                    body.push(BodyInstruction::Barrier {
                        qubits: qubits.clone(),
                    });
                } else {
                    push_standard_instruction(
                        &mut circuit,
                        StandardInstruction::Barrier(qubits.len() as u32),
                        qubits,
                        &[],
                    )?;
                }
            }
            InternalBytecode::DeclareQreg { name, size } => {
                let register = QuantumRegister::new_owning(name.clone(), *size as u32);
                circuit.add_qreg(register, true).map_err(|err| {
                    ParseError::new(format!("failed to declare qreg '{name}': {err}"))
                })?;
            }
            InternalBytecode::DeclareCreg { name, size } => {
                let register = ClassicalRegister::new_owning(name.clone(), *size as u32);
                cregs.push(register.clone());
                circuit.add_creg(register, true).map_err(|err| {
                    ParseError::new(format!("failed to declare creg '{name}': {err}"))
                })?;
            }
            InternalBytecode::SpecialInclude { indices } => {
                registry.extend_from_qelib1(indices);
            }
            InternalBytecode::ConditionedGate {
                id,
                arguments,
                qubits,
                creg,
                value,
            } => {
                let entry = registry.get(*id)?;
                let num_qubits = qubits.len() as u32;
                push_conditioned(
                    &mut circuit,
                    &cregs,
                    *creg,
                    value,
                    num_qubits,
                    0,
                    &to_qubits(qubits),
                    &[],
                    |block| {
                        let local_qargs: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();
                        push_gate(block, &entry, arguments, &local_qargs)
                    },
                )?;
            }
            InternalBytecode::ConditionedMeasure {
                qubit,
                clbit,
                creg,
                value,
            } => {
                push_conditioned(
                    &mut circuit,
                    &cregs,
                    *creg,
                    value,
                    1,
                    1,
                    &to_qubits(&[*qubit]),
                    &to_clbits(&[*clbit]),
                    |block| {
                        push_standard_instruction_local(
                            block,
                            StandardInstruction::Measure,
                            &[Qubit(0)],
                            &[Clbit(0)],
                        )
                    },
                )?;
            }
            InternalBytecode::ConditionedReset { qubit, creg, value } => {
                push_conditioned(
                    &mut circuit,
                    &cregs,
                    *creg,
                    value,
                    1,
                    0,
                    &to_qubits(&[*qubit]),
                    &[],
                    |block| {
                        push_standard_instruction_local(
                            block,
                            StandardInstruction::Reset,
                            &[Qubit(0)],
                            &[],
                        )
                    },
                )?;
            }
            InternalBytecode::DeclareGate { name, num_qubits } => {
                if current_body.is_some() {
                    return Err(ParseError::new(message_generic(
                        None,
                        "nested gate declaration: missing an EndDeclareGate",
                    )));
                }
                current_body = Some((name.clone(), *num_qubits as u32, Vec::new()));
            }
            InternalBytecode::GateInBody {
                id,
                arguments,
                qubits,
            } => {
                let entry = registry.get(*id)?;
                let (_, _, body) = current_body.as_mut().ok_or_else(|| {
                    ParseError::new(message_generic(
                        None,
                        "gate body instruction outside of a gate declaration",
                    ))
                })?;
                body.push(BodyInstruction::Gate {
                    entry,
                    arguments: arguments.clone(),
                    qubits: qubits.clone(),
                });
            }
            InternalBytecode::EndDeclareGate {} => {
                let (name, num_qubits, body) = current_body.take().ok_or_else(|| {
                    ParseError::new(message_generic(
                        None,
                        "EndDeclareGate without a matching DeclareGate",
                    ))
                })?;
                registry.declare(name, num_qubits, Some(body));
            }
            InternalBytecode::DeclareOpaque { name, num_qubits } => {
                if current_body.is_some() {
                    return Err(ParseError::new(message_generic(
                        None,
                        "opaque declaration nested inside another gate declaration",
                    )));
                }
                registry.declare(name.clone(), *num_qubits as u32, None);
            }
        }
    }
    if current_body.is_some() {
        return Err(ParseError::new(message_generic(
            None,
            "unterminated gate declaration",
        )));
    }

    Ok(circuit)
}

fn to_qubits(qubits: &[QubitId]) -> Vec<Qubit> {
    qubits.iter().map(|q| Qubit(q.index() as u32)).collect()
}

fn to_clbits(clbits: &[ClbitId]) -> Vec<Clbit> {
    clbits.iter().map(|c| Clbit(c.index() as u32)).collect()
}

/// Applies a resolved gate (native or OQ2-defined) with concrete `arguments`, either to the
/// top-level circuit or to a block being built inside it (e.g. a conditioned or gate-body block).
fn push_gate(
    circuit: &mut CircuitData,
    entry: &GateEntry,
    arguments: &[f64],
    qargs: &[Qubit],
) -> Result<(), ParseError> {
    debug_assert_eq!(
        qargs.len(),
        entry.num_qubits() as usize,
        "gate registry desync: wrong qubit count for resolved gate"
    );
    let params: Vec<Param> = arguments.iter().map(|&v| Param::Float(v)).collect();
    match entry {
        GateEntry::Standard(gate) => {
            debug_assert_eq!(
                arguments.len(),
                gate.num_params() as usize,
                "gate registry desync: wrong param count for resolved gate"
            );
            circuit
                .push_standard_gate(*gate, &params, qargs)
                .map_err(|err| ParseError::new(format!("failed to apply gate: {err}")))
        }
        GateEntry::Defined(template) => {
            let defined = DefinedGate {
                template: template.clone(),
                num_params: arguments.len() as u32,
            };
            circuit
                .push_packed_operation(
                    PackedOperation::from_custom_operation(Box::new(defined)),
                    Some(Parameters::Params(params.into())),
                    qargs,
                    &[],
                )
                .map_err(|err| ParseError::new(format!("failed to apply gate: {err}")))
        }
    }
}

fn push_standard_instruction(
    circuit: &mut CircuitData,
    instruction: StandardInstruction,
    qubits: &[QubitId],
    clbits: &[ClbitId],
) -> Result<(), ParseError> {
    push_standard_instruction_local(circuit, instruction, &to_qubits(qubits), &to_clbits(clbits))
}

fn push_standard_instruction_local(
    circuit: &mut CircuitData,
    instruction: StandardInstruction,
    qubits: &[Qubit],
    clbits: &[Clbit],
) -> Result<(), ParseError> {
    circuit
        .push_packed_operation(PackedOperation::from(instruction), None, qubits, clbits)
        .map_err(|err| ParseError::new(format!("failed to apply instruction: {err}")))
}

/// Wraps a single instruction (built by `fill_block`, using block-local `Qubit(0..num_qubits)` /
/// `Clbit(0..num_clbits)`) in `if (cregs[creg] == value) { ... }`.
#[allow(clippy::too_many_arguments)]
fn push_conditioned(
    circuit: &mut CircuitData,
    cregs: &[ClassicalRegister],
    creg: CregId,
    value: &BigUint,
    num_qubits: u32,
    num_clbits: u32,
    qargs: &[Qubit],
    cargs: &[Clbit],
    fill_block: impl FnOnce(&mut CircuitData) -> Result<(), ParseError>,
) -> Result<(), ParseError> {
    let mut block = CircuitData::new(None, None, Param::Float(0.0))
        .map_err(|err| ParseError::new(format!("failed to create circuit: {err}")))?;
    block
        .add_anonymous_qubits(num_qubits)
        .map_err(|err| ParseError::new(format!("failed to build conditioned block: {err}")))?;
    block
        .add_anonymous_clbits(num_clbits)
        .map_err(|err| ParseError::new(format!("failed to build conditioned block: {err}")))?;
    fill_block(&mut block)?;

    let register = cregs.get(creg.index()).ok_or_else(|| {
        ParseError::new(message_generic(
            None,
            &format!("creg id {} was not declared", creg.index()),
        ))
    })?;
    let condition = Condition::Register(register.clone(), value.clone());
    let block_id = circuit.add_block(block);
    let control_flow = ControlFlowInstruction {
        control_flow: ControlFlow::IfElse { condition },
        num_qubits,
        num_clbits,
    };
    circuit
        .push_packed_operation(
            PackedOperation::from(control_flow),
            Some(Parameters::Blocks(vec![block_id])),
            qargs,
            cargs,
        )
        .map_err(|err| ParseError::new(format!("failed to apply conditioned instruction: {err}")))
}

#[cfg(test)]
mod tests {
    use super::build_circuit;
    use crate::bytecode::InternalBytecode;
    use crate::expr::Expr;
    use crate::parse::{ClbitId, CregId, GateId, ParamId, QubitId};
    use num_bigint::BigUint;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::operations::{
        Condition, ControlFlow, OperationRef, Param, StandardInstruction,
    };
    use qiskit_circuit::standard_gate::StandardGate;

    /// The hand-written bytecode in the tests below encodes assumptions about what the parser
    /// emits -- notably that `qelib1` index 8 is `h`, and that it lands at `GateId(2)` behind the
    /// builtins.  Drive the real parser once so those assumptions are checked rather than assumed,
    /// and so a parametrised `gate` body is evaluated all the way to a concrete parameter.
    #[test]
    fn parses_and_builds_a_program_end_to_end() {
        let program = concat!(
            "include \"qelib1.inc\";\n",
            "qreg q[2];\n",
            "creg c[2];\n",
            "gate my_rz(a) qq { u1(2*a) qq; }\n",
            "h q[0];\n",
            "cx q[0], q[1];\n",
            "my_rz(pi/4) q[1];\n",
            "measure q[0] -> c[0];\n",
        );
        let circuit = crate::circuit_from_string(program.to_owned(), vec![], &[], &[], false)
            .expect("the program is valid OpenQASM 2");

        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.num_clbits(), 2);
        assert_eq!(circuit.data().len(), 4);

        let ops: Vec<_> = circuit.data().iter().map(|inst| inst.op.view()).collect();
        assert!(matches!(
            ops[0],
            OperationRef::StandardGate(StandardGate::H)
        ));
        assert!(matches!(
            ops[1],
            OperationRef::StandardGate(StandardGate::CX)
        ));
        assert!(matches!(
            ops[3],
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        ));

        // `my_rz(pi/4)` should expand to `u1(2*pi/4)`, with the body expression folded in Rust.
        let OperationRef::CustomOperation(custom) = ops[2] else {
            panic!("expected the `gate` declaration to build a custom operation");
        };
        assert_eq!(custom.name(), "my_rz");
        let definition = custom
            .definition(&[Param::Float(std::f64::consts::PI / 4.0)])
            .expect("my_rz has a known definition");
        assert_eq!(definition.data().len(), 1);
        assert!(matches!(
            definition.data()[0].op.view(),
            OperationRef::StandardGate(StandardGate::U1)
        ));
        let Param::Float(angle) = definition.data()[0].params_view()[0] else {
            panic!("expected a concrete float parameter");
        };
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "got {angle}"
        );
    }

    #[test]
    fn builds_a_small_bell_pair_circuit() {
        // Equivalent to:
        //   qreg q[2]; creg c[2];
        //   include "qelib1.inc"; // only "h" registered here, for a minimal test
        //   h q[0]; CX q[0], q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];
        let bytecode = vec![
            InternalBytecode::DeclareQreg {
                name: "q".to_string(),
                size: 2,
            },
            InternalBytecode::DeclareCreg {
                name: "c".to_string(),
                size: 2,
            },
            // qelib1 index 8 is "h"; it becomes GateId(2), right after the builtins U(0), CX(1).
            InternalBytecode::SpecialInclude { indices: vec![8] },
            InternalBytecode::Gate {
                id: GateId::new(2),
                arguments: vec![],
                qubits: vec![QubitId::new(0)],
            },
            InternalBytecode::Gate {
                id: GateId::new(1),
                arguments: vec![],
                qubits: vec![QubitId::new(0), QubitId::new(1)],
            },
            InternalBytecode::Measure {
                qubit: QubitId::new(0),
                clbit: ClbitId::new(0),
            },
            InternalBytecode::Measure {
                qubit: QubitId::new(1),
                clbit: ClbitId::new(1),
            },
        ];

        let circuit = build_circuit(&bytecode).expect("all instructions are supported");
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.num_clbits(), 2);
        assert_eq!(circuit.data().len(), 4);

        let ops: Vec<_> = circuit.data().iter().map(|inst| inst.op.view()).collect();
        assert!(matches!(
            ops[0],
            OperationRef::StandardGate(StandardGate::H)
        ));
        assert!(matches!(
            ops[1],
            OperationRef::StandardGate(StandardGate::CX)
        ));
        assert!(matches!(
            ops[2],
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        ));
        assert!(matches!(
            ops[3],
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        ));

        let cx_qargs = circuit.get_qargs(circuit.data()[1].qubits);
        assert_eq!(cx_qargs.len(), 2);
        assert_eq!(cx_qargs[0].0, 0);
        assert_eq!(cx_qargs[1].0, 1);
    }

    #[test]
    fn builds_a_conditioned_gate() {
        // Equivalent to: qreg q[2]; creg c[1]; if (c==1) CX q[0], q[1];
        let bytecode = vec![
            InternalBytecode::DeclareQreg {
                name: "q".to_string(),
                size: 2,
            },
            InternalBytecode::DeclareCreg {
                name: "c".to_string(),
                size: 1,
            },
            InternalBytecode::ConditionedGate {
                id: GateId::new(1), // builtin CX
                arguments: vec![],
                qubits: vec![QubitId::new(0), QubitId::new(1)],
                creg: CregId::new(0),
                value: BigUint::from(1u32),
            },
        ];

        let circuit = build_circuit(&bytecode).expect("all instructions are supported");
        assert_eq!(circuit.data().len(), 1);

        let OperationRef::ControlFlow(control_flow) = circuit.data()[0].op.view() else {
            panic!("expected a control-flow instruction");
        };
        let ControlFlow::IfElse { condition } = &control_flow.control_flow else {
            panic!("expected an if/else");
        };
        let Condition::Register(_, value) = condition else {
            panic!("expected a register condition");
        };
        assert_eq!(*value, BigUint::from(1u32));

        let outer_qargs = circuit.get_qargs(circuit.data()[0].qubits);
        assert_eq!(outer_qargs, &[Qubit(0), Qubit(1)]);
    }

    #[test]
    fn builds_a_defined_gate_and_lazily_expands_it() {
        // Equivalent to:
        //   gate my_h q { h q; }
        //   qreg q[1];
        //   include "qelib1.inc"; // registers "h" as GateId(2)
        //   my_h q[0];            // becomes GateId(3)
        let bytecode = vec![
            InternalBytecode::SpecialInclude { indices: vec![8] }, // "h" -> GateId(2)
            InternalBytecode::DeclareGate {
                name: "my_h".to_string(),
                num_qubits: 1,
            },
            InternalBytecode::GateInBody {
                id: GateId::new(2), // "h"
                arguments: vec![],
                qubits: vec![QubitId::new(0)],
            },
            InternalBytecode::EndDeclareGate {},
            InternalBytecode::DeclareQreg {
                name: "q".to_string(),
                size: 1,
            },
            InternalBytecode::Gate {
                id: GateId::new(3), // "my_h"
                arguments: vec![],
                qubits: vec![QubitId::new(0)],
            },
        ];

        let circuit = build_circuit(&bytecode).expect("all instructions are supported");
        assert_eq!(circuit.data().len(), 1);

        let OperationRef::CustomOperation(custom) = circuit.data()[0].op.view() else {
            panic!("expected a custom operation");
        };
        assert_eq!(custom.name(), "my_h");
        assert_eq!(custom.num_qubits(), 1);

        let definition = custom.definition(&[]).expect("my_h has a known definition");
        assert_eq!(definition.data().len(), 1);
        assert!(matches!(
            definition.data()[0].op.view(),
            OperationRef::StandardGate(StandardGate::H)
        ));
    }

    #[test]
    fn substitutes_a_real_parameter_into_the_body() {
        // Equivalent to:
        //   gate my_rz(theta) q { u1(theta) q; }
        //   qreg q[1];
        //   include "qelib1.inc"; // registers "u1" as GateId(2)
        //   my_rz(1.5) q[0];      // becomes GateId(3)
        let bytecode = vec![
            InternalBytecode::SpecialInclude { indices: vec![2] }, // "u1" -> GateId(2)
            InternalBytecode::DeclareGate {
                name: "my_rz".to_string(),
                num_qubits: 1,
            },
            InternalBytecode::GateInBody {
                id: GateId::new(2), // "u1"
                arguments: vec![Expr::Parameter(ParamId::new(0))],
                qubits: vec![QubitId::new(0)],
            },
            InternalBytecode::EndDeclareGate {},
            InternalBytecode::DeclareQreg {
                name: "q".to_string(),
                size: 1,
            },
            InternalBytecode::Gate {
                id: GateId::new(3), // "my_rz"
                arguments: vec![1.5],
                qubits: vec![QubitId::new(0)],
            },
        ];

        let circuit = build_circuit(&bytecode).expect("all instructions are supported");
        let OperationRef::CustomOperation(custom) = circuit.data()[0].op.view() else {
            panic!("expected a custom operation");
        };

        let definition = custom
            .definition(&[Param::Float(1.5)])
            .expect("my_rz has a known definition");
        assert_eq!(definition.data().len(), 1);
        assert!(matches!(
            definition.data()[0].op.view(),
            OperationRef::StandardGate(StandardGate::U1)
        ));
        let Param::Float(value) = definition.data()[0].params_view()[0] else {
            panic!("expected a float param");
        };
        assert_eq!(value, 1.5);
    }

    #[test]
    fn recursive_defined_gate_stays_lazily_nested() {
        // Equivalent to:
        //   gate my_h q { h q; }
        //   gate double_h q { my_h q; my_h q; }
        //   qreg q[1];
        //   include "qelib1.inc"; // registers "h" as GateId(2)
        //   double_h q[0];        // becomes GateId(4), after my_h at GateId(3)
        let bytecode = vec![
            InternalBytecode::SpecialInclude { indices: vec![8] }, // "h" -> GateId(2)
            InternalBytecode::DeclareGate {
                name: "my_h".to_string(),
                num_qubits: 1,
            },
            InternalBytecode::GateInBody {
                id: GateId::new(2), // "h"
                arguments: vec![],
                qubits: vec![QubitId::new(0)],
            },
            InternalBytecode::EndDeclareGate {},
            InternalBytecode::DeclareGate {
                name: "double_h".to_string(),
                num_qubits: 1,
            },
            InternalBytecode::GateInBody {
                id: GateId::new(3), // "my_h"
                arguments: vec![],
                qubits: vec![QubitId::new(0)],
            },
            InternalBytecode::GateInBody {
                id: GateId::new(3), // "my_h"
                arguments: vec![],
                qubits: vec![QubitId::new(0)],
            },
            InternalBytecode::EndDeclareGate {},
            InternalBytecode::DeclareQreg {
                name: "q".to_string(),
                size: 1,
            },
            InternalBytecode::Gate {
                id: GateId::new(4), // "double_h"
                arguments: vec![],
                qubits: vec![QubitId::new(0)],
            },
        ];

        let circuit = build_circuit(&bytecode).expect("all instructions are supported");
        let OperationRef::CustomOperation(outer) = circuit.data()[0].op.view() else {
            panic!("expected a custom operation");
        };
        assert_eq!(outer.name(), "double_h");

        // The outer gate's definition should contain two *nested* `my_h` custom operations,
        // not the fully-flattened `h` gates -- expansion is lazy at each level, matching Python.
        let outer_definition = outer
            .definition(&[])
            .expect("double_h has a known definition");
        assert_eq!(outer_definition.data().len(), 2);
        for inst in outer_definition.data() {
            let OperationRef::CustomOperation(inner) = inst.op.view() else {
                panic!("expected a nested custom operation, not an already-flattened gate");
            };
            assert_eq!(inner.name(), "my_h");

            let inner_definition = inner.definition(&[]).expect("my_h has a known definition");
            assert_eq!(inner_definition.data().len(), 1);
            assert!(matches!(
                inner_definition.data()[0].op.view(),
                OperationRef::StandardGate(StandardGate::H)
            ));
        }
    }

    #[test]
    fn opaque_gates_have_no_definition() {
        let bytecode = vec![
            InternalBytecode::DeclareOpaque {
                name: "black_box".to_string(),
                num_qubits: 1,
            },
            InternalBytecode::DeclareQreg {
                name: "q".to_string(),
                size: 1,
            },
            InternalBytecode::Gate {
                id: GateId::new(2), // "black_box", after the builtins U(0), CX(1)
                arguments: vec![],
                qubits: vec![QubitId::new(0)],
            },
        ];

        let circuit = build_circuit(&bytecode).expect("all instructions are supported");
        let OperationRef::CustomOperation(custom) = circuit.data()[0].op.view() else {
            panic!("expected a custom operation");
        };
        assert!(custom.definition(&[]).is_none());
    }

    #[test]
    fn rejects_gate_body_instruction_outside_a_declaration() {
        let bytecode = vec![InternalBytecode::GateInBody {
            id: GateId::new(0),
            arguments: Vec::<Expr>::new(),
            qubits: vec![QubitId::new(0)],
        }];
        assert!(build_circuit(&bytecode).is_err());
    }
}
