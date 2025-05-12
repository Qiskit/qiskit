use hashbrown::HashMap;
use crate::bit::ClassicalRegister;
use crate::{Clbit, Stretch, Var};
use crate::classical::expr;
use crate::classical::expr::{Binary, Cast, Expr, IdentifierRefMut, Index, Unary, Value};

pub(crate) struct VariableMapper {
    target_cregs: Vec<ClassicalRegister>,
    bit_map: HashMap<Clbit, Clbit>,
    var_map: HashMap<Var, Var>,
    stretch_map: HashMap<Stretch, Stretch>,
}

impl VariableMapper {
    pub fn new(target_cregs: Vec<ClassicalRegister>, bit_map: HashMap<Clbit, Clbit>, var_map: HashMap<Var, Var>, stretch_map: HashMap<Stretch, Stretch>) -> Self {
        Self {
            target_cregs,
            bit_map,
            var_map,
            stretch_map,
        }
    }

    pub fn map_expr(&self, expr: &expr::Expr) -> expr::Expr {
        let mut copy = expr.clone();
        for identifier in copy.identifiers_mut() {
            match identifier {
                IdentifierRefMut::Var(v) => {
                    todo!()
                }
                IdentifierRefMut::Stretch(s) => {
                    todo!()
                }
            }
        }
        copy
    }
}