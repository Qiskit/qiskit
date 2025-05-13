use crate::bit::{ClassicalRegister, Register, ShareableBit, ShareableClbit};
use crate::classical::expr;
use crate::classical::expr::{Binary, Cast, Expr, ExprRefMut, Index, Stretch, Unary, Value, Var};
use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::{Bound, FromPyObject, PyAny, PyResult};

/// A control flow operation's condition.
///
/// TODO: move this to control flow mod once that's in Rust.
pub(crate) enum Condition {
    Bit(ShareableClbit, usize),
    Register(ClassicalRegister, usize),
    Expr(Expr),
}

impl<'py> FromPyObject<'py> for Condition {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok((bit, value)) = ob.extract::<(ShareableClbit, usize)>() {
            Ok(Condition::Bit(bit, value))
        } else if let Ok((register, value)) = ob.extract::<(ClassicalRegister, usize)>() {
            Ok(Condition::Register(register, value))
        } else {
            Ok(Condition::Expr(ob.extract()?))
        }
    }
}

pub(crate) struct VariableMapper {
    target_cregs: Vec<ClassicalRegister>,
    bit_map: HashMap<ShareableClbit, ShareableClbit>,
    var_map: HashMap<Var, Var>,
    stretch_map: HashMap<Stretch, Stretch>,
}

impl VariableMapper {
    pub fn new(
        target_cregs: Vec<ClassicalRegister>,
        bit_map: HashMap<ShareableClbit, ShareableClbit>,
        var_map: HashMap<Var, Var>,
        stretch_map: HashMap<Stretch, Stretch>,
    ) -> Self {
        Self {
            target_cregs,
            bit_map,
            var_map,
            stretch_map,
        }
    }

    pub fn map_condition(&self, condition: &Condition, allow_reorder: bool) -> Condition {
        match condition {
            Condition::Bit(target, value) => {
                Condition::Bit(self.bit_map.get(target).unwrap().clone(), *value)
            }
            Condition::Expr(e) => Condition::Expr(self.map_expr(e)),
            Condition::Register(target, value) => {
                if !allow_reorder {
                    return Condition::Register(self.map_register(target), *value);
                }
                // This is maintaining the legacy behavior of `DAGCircuit.compose`.  We don't
                // attempt to speed-up this lookup with a cache, since that would just make the more
                // standard cases more annoying to deal with.

                let mapped_bits_order = target
                    .bits()
                    .map(|b| self.bit_map.get(&b).unwrap().clone())
                    .collect::<Vec<_>>();
                let mapped_bits_set: HashSet<ShareableClbit> =
                    mapped_bits_order.iter().cloned().collect();

                let found_register = self.target_cregs.iter().find(|register| {
                    let register_set: HashSet<ShareableClbit> = register.iter().collect();
                    mapped_bits_set == register_set
                });

                // TODO: handle this case
                // else:
                //     if self.add_register is None:
                //         raise self.exc_type(
                //             f"Register '{target.name}' has no counterpart in the destination."
                //         )
                //     mapped_theirs = ClassicalRegister(bits=mapped_bits_order)
                //     self.add_register(mapped_theirs)
                let mapped_theirs = found_register.cloned().unwrap();

                let new_order: HashMap<ShareableClbit, usize> = mapped_bits_order
                    .into_iter()
                    .enumerate()
                    .map(|(i, bit)| (bit, i))
                    .collect();

                // Build the little-endian bit string
                let mut value_bits: Vec<char> = format!("{:0width$b}", value, width = target.len())
                    .chars()
                    .rev()
                    .collect();

                // Reorder bits and reverse again to go back to big-endian for final conversion
                let mapped_str: String = mapped_theirs
                    .iter() // TODO: we should probably not need to collect to Vec here. Why do we?
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|bit| value_bits[*new_order.get(&bit).unwrap()])
                    .rev()
                    .collect();

                Condition::Register(mapped_theirs, usize::from_str_radix(&mapped_str, 2).unwrap())
            }
        }
    }

    pub fn map_expr(&self, expr: &Expr) -> Expr {
        let mut mapped = expr.clone();
        mapped.visit_mut(|e| match e {
            ExprRefMut::Var(var) => match var {
                Var::Standalone { .. } => {
                    if let Some(mapping) = self.var_map.get(var).cloned() {
                        *var = mapping;
                    }
                }
                Var::Bit { bit } => {
                    let bit = self.bit_map.get(bit).cloned().unwrap();
                    *var = Var::Bit { bit };
                }
                Var::Register { register, ty } => {
                    let ty = *ty;
                    let register = self.map_register(register);
                    *var = Var::Register { register, ty }
                }
            },
            ExprRefMut::Stretch(stretch) => {
                if let Some(mapping) = self.stretch_map.get(stretch).cloned() {
                    *stretch = mapping;
                }
            }
            _ => (),
        });
        mapped
    }

    fn map_register(&self, theirs: &ClassicalRegister) -> ClassicalRegister {
        todo!()
    }
}
