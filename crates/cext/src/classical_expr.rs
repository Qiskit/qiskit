use std::{ptr, str::FromStr};

use qiskit_circuit::{classical::{expr::{Binary, BinaryOp, Expr, Stretch, UnaryOp, Value, Var}, types::Type}, duration::Duration};
use uuid::Uuid;
use crate::pointers::const_ptr_as_ref;
use num_bigint::BigUint;
use std::ffi::{c_char, CString};
use num_traits::ToPrimitive;

#[repr(u8)]
pub enum CExprNodeKind { 
    Unary = 0, 
    Binary = 1,
    Cast = 2,
    Value = 3,
    Var = 4, 
    Stretch = 5, 
    Index = 6,
}

impl From<&Expr> for CExprNodeKind {
    fn from(value: &Expr) -> Self {
        match value {
            Expr::Unary(_) => Self::Unary, 
            Expr::Binary(_) => Self::Binary, 
            Expr::Cast(_) => Self::Cast,
            Expr::Index(_) => Self::Index,
            Expr::Stretch(_) => Self::Stretch, 
            Expr::Value(_) => Self::Value,
            Expr::Var(_) => Self::Var,
        }
    }
}

#[repr(u8)]
pub enum CExprType {
    Bool = 0,
    Duration = 1,
    Float = 2,
    Uint = 3,
}

impl From<&Type> for CExprType {
    fn from(value: &Type) -> Self {
        match value {
            Type::Bool => Self::Bool,
            Type::Duration => Self::Duration,
            Type::Float => Self::Float,
            Type::Uint(_) => Self::Uint,
        }
    }
}

#[repr(C)]
pub struct CExprTypeInfo {
    ty: CExprType,
    width: u16,
}

impl CExprTypeInfo {
    fn to_type(&self) -> Type {
        match self.ty {
            CExprType::Bool => Type::Bool,
            CExprType::Duration => Type::Duration,
            CExprType::Float => Type::Float,
            CExprType::Uint => Type::Uint(self.width),
        }
    }
}

impl From<&Type> for CExprTypeInfo {
    fn from(ty: &Type) -> Self {
        match ty {
            Type::Bool => CExprTypeInfo{ty: CExprType::Bool, width: 0},
            Type::Duration => CExprTypeInfo{ty: CExprType::Duration, width: 0},
            Type::Float => CExprTypeInfo{ty: CExprType::Float, width: 0},
            Type::Uint(w) => CExprTypeInfo{ty: CExprType::Uint, width: *w},
        }
    }
}


#[repr(u8)]
pub enum CUnaryOpType {
    BitNot = 1, // TODO: keeping it one-based on purpose, to avoid confusion with the convention in Python
    LogicNot = 2,
    Negate = 3,
}

impl From<UnaryOp> for CUnaryOpType {
    fn from(value: UnaryOp) -> Self {
        match value { 
        UnaryOp::BitNot => Self::BitNot,
        UnaryOp::LogicNot => Self::LogicNot,
        UnaryOp::Negate => Self::Negate,
        }
    }
}

impl CUnaryOpType {
    fn to_unary_op(self) -> UnaryOp {
        match self {
            CUnaryOpType::BitNot => UnaryOp::BitNot,
            CUnaryOpType::LogicNot => UnaryOp::LogicNot,
            CUnaryOpType::Negate => UnaryOp::Negate,
        }
    }
}

#[repr(C)]
pub struct CUnaryExprInfo {
    pub op: CUnaryOpType,
    pub operand: *const Expr,
    pub ty: CExprTypeInfo,
    pub constant: bool,
}

#[repr(u8)]
pub enum CBinaryOpType {
    BitAnd = 1, // TODO: keeping it one-based on purpose, to avoid confusion with the convention in Python
    BitOr = 2,
    BitXor = 3,
    LogicAnd = 4,
    LogicOr = 5,
    Equal = 6,
    NotEqual = 7,
    Less = 8,
    LessEqual = 9,
    Greater = 10,
    GreaterEqual = 11,
    ShiftLeft = 12,
    ShiftRight = 13,
    Add = 14,
    Sub = 15,
    Mul = 16,
    Div = 17,
}

impl From<BinaryOp> for CBinaryOpType {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::BitAnd => Self::BitAnd,
            BinaryOp::BitOr => Self::BitOr,
            BinaryOp::BitXor => Self::BitXor,
            BinaryOp::LogicAnd => Self::LogicAnd,
            BinaryOp::LogicOr => Self::LogicOr,
            BinaryOp::Equal => Self::Equal,
            BinaryOp::NotEqual => Self::NotEqual,
            BinaryOp::Less => Self::Less,
            BinaryOp::LessEqual => Self::LessEqual,
            BinaryOp::Greater => Self::Greater,
            BinaryOp::GreaterEqual => Self::GreaterEqual,
            BinaryOp::ShiftLeft => Self::ShiftLeft,
            BinaryOp::ShiftRight => Self::ShiftRight,
            BinaryOp::Add => Self::Add,
            BinaryOp::Sub => Self::Sub,
            BinaryOp::Mul => Self::Mul,
            BinaryOp::Div => Self::Div,
        }
    }
}

impl CBinaryOpType {
    fn to_binary_op(self) -> BinaryOp {
        match self {
            CBinaryOpType::BitAnd => BinaryOp::BitAnd,
            CBinaryOpType::BitOr => BinaryOp::BitOr,
            CBinaryOpType::BitXor => BinaryOp::BitXor,
            CBinaryOpType::LogicAnd => BinaryOp::LogicAnd,
            CBinaryOpType::LogicOr => BinaryOp::LogicOr,
            CBinaryOpType::Equal => BinaryOp::Equal,
            CBinaryOpType::NotEqual => BinaryOp::NotEqual,
            CBinaryOpType::Less => BinaryOp::Less,
            CBinaryOpType::LessEqual => BinaryOp::LessEqual,
            CBinaryOpType::Greater => BinaryOp::Greater,
            CBinaryOpType::GreaterEqual => BinaryOp::GreaterEqual,
            CBinaryOpType::ShiftLeft => BinaryOp::ShiftLeft,
            CBinaryOpType::ShiftRight => BinaryOp::ShiftRight,
            CBinaryOpType::Add => BinaryOp::Add,
            CBinaryOpType::Sub => BinaryOp::Sub,
            CBinaryOpType::Mul => BinaryOp::Mul,
            CBinaryOpType::Div => BinaryOp::Div,
        }
    }
}



#[repr(C)]
pub struct CBinaryExprInfo {
    pub op: CBinaryOpType,
    pub left: *const Expr,
    pub right: *const Expr,
    pub ty: CExprTypeInfo,
    pub constant: bool,
}

#[repr(C)]
pub struct CCastExprInfo {
    pub operand: *const Expr,
    pub ty: CExprTypeInfo,
    pub implicit: bool,
    pub constant: bool,
}

#[repr(u8)]
pub enum CValueType {
    Duration = 0,
    Float = 1, 
    Uint = 2, 
}

impl From<&Value> for CValueType {
    fn from(value: &Value) -> Self {
        match value {
            Value::Duration(_) => Self::Duration,
            Value::Float{..} => Self::Float,
            Value::Uint{..} => Self::Uint,
        }
    }
}

#[repr(C)]
pub struct CIndexExprInfo {
    pub target: *const Expr,
    pub index: *const Expr,
    pub ty: CExprTypeInfo,
    pub constant: bool,
}

#[repr(u8)]
pub enum CDurationType {
    Dt = 0,
    Ps = 1,
    Ns = 2,
    Us = 3,
    Ms = 4,
    S = 5, 
}
impl From<&Duration> for CDurationType {
    fn from(duration: &Duration) -> Self {
        match duration {
            Duration::dt(_) => Self::Dt,
            Duration::ps(_) => Self::Ps,
            Duration::ns(_) => Self::Ns,
            Duration::us(_) => Self::Us,
            Duration::ms(_) => Self::Ms,
            Duration::s(_) => Self::S,
        }
    }
}

#[repr(C)]
pub union CDurationValue {
    dt: i64,
    time: f64,
}

#[repr(C)]
pub struct CDurationInfo {
    pub ty: CDurationType,
    pub value: CDurationValue,
}

impl From<&Duration> for CDurationInfo {
    fn from(duration: &Duration) -> Self {
        match duration {
            Duration::dt(v) => CDurationInfo {
                ty: CDurationType::Dt,
                value: CDurationValue{dt: *v},
            },
            Duration::ps(v) => CDurationInfo {
                ty: CDurationType::Ps,
                value: CDurationValue{time: *v},
            },
            Duration::ns(v) => CDurationInfo {
                ty: CDurationType::Ns,
                value: CDurationValue{time: *v},
            },
            Duration::us(v) => CDurationInfo {
                ty: CDurationType::Us,
                value: CDurationValue{time: *v},
            },
            Duration::ms(v) => CDurationInfo {
                ty: CDurationType::Ms,
                value: CDurationValue{time: *v},
            },
            Duration::s(v) => CDurationInfo {
                ty: CDurationType::S,
                value: CDurationValue{time: *v},
            },
        }
    }
}

impl CDurationInfo {
    pub fn to_duration(&self) -> Duration {
        match self.ty {
            CDurationType::Dt => Duration::dt(unsafe{self.value.dt}),
            CDurationType::Ps => Duration::ps(unsafe{self.value.time}),
            CDurationType::Ns => Duration::ns(unsafe{self.value.time}),
            CDurationType::Us => Duration::us(unsafe{self.value.time}),
            CDurationType::Ms => Duration::ms(unsafe{self.value.time}),
            CDurationType::S => Duration::s(unsafe{self.value.time}),
        }
    }
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_node_kind(expr: *const Expr) -> CExprNodeKind {
    let expr = unsafe{ const_ptr_as_ref(expr) };

    CExprNodeKind::from(expr)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_binary_info(expr: *const Expr) -> CBinaryExprInfo {
    let expr = unsafe{ const_ptr_as_ref(expr) };

    let Expr::Binary(binary) = expr else {
        panic!("TODO")
    };

    CBinaryExprInfo {
        op: CBinaryOpType::from(binary.op),
        left: &binary.left as *const Expr,
        right: &binary.right as *const Expr,
        ty: CExprTypeInfo::from(&binary.ty),
        constant: binary.constant,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_unary_info(expr: *const Expr) -> CUnaryExprInfo {
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Unary(unary) = expr else {
        panic!("TODO")
    };

    CUnaryExprInfo {
        op: CUnaryOpType::from(unary.op),
        operand: &unary.operand as *const Expr,
        ty: CExprTypeInfo::from(&unary.ty),
        constant: unary.constant,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_cast_info(expr: *const Expr) -> CCastExprInfo {
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Cast(cast) = expr else {
        panic!("TODO")
    };

    CCastExprInfo {
        operand: &cast.operand as *const Expr,
        ty: CExprTypeInfo::from(&cast.ty),
        implicit: cast.implicit,
        constant: cast.constant,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_index_info(expr: *const Expr) -> CIndexExprInfo {
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Index(index) = expr else {
        panic!("TODO")
    };

    CIndexExprInfo {
        target: &index.target as *const Expr,
        index: &index.index as *const Expr,
        ty: CExprTypeInfo::from(&index.ty),
        constant: index.constant,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_as_value(expr: *const Expr) -> *const Value {
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Value(val) = expr else {
        panic!("TODO")
    };

    val as *const Value
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_as_var(expr: *const Expr) -> *const Var {
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Var(var) = expr else {
        panic!("TODO")
    };

    ptr::from_ref(var)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_as_stretch(expr: *const Expr) -> *const Stretch {
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Stretch(stretch) = expr else {
        panic!("TODO")
    };

    ptr::from_ref(stretch)
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_type(value: *const Value) -> CValueType { 
    let value = unsafe { const_ptr_as_ref(value) };
    
    CValueType::from(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_duration_info(value: *const Value) -> CDurationInfo {
    let value = unsafe { const_ptr_as_ref(value) };
    
    let Value::Duration(duration) = value else {
        panic!("TODO")
    };
    
    CDurationInfo::from(duration)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_float(value: *const Value) -> f64{
    let value = unsafe { const_ptr_as_ref(value) };
    
    let Value::Float{raw, ..} = value else { // TODO: what should we do with ty?
        panic!("TODO")
    };

    *raw
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_uint(value: *const Value) -> u64 {
    let value = unsafe { const_ptr_as_ref(value) };
    
    let Value::Uint { raw, .. } = value else { // TODO: what should we do with ty?
        panic!("TODO")
    };

    raw.to_u64()
        .expect("TODO") // TODO: handle BitUint. Is there a better way?
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_var_name(var: *const Var) -> *mut c_char {
    let var = unsafe { const_ptr_as_ref(var) };

    let name = match var {
        Var::Standalone { name, .. } => name.as_str(),
        Var::Register { register, .. } => register.name(),
        Var::Bit { .. } => return ptr::null_mut(),
    };

    CString::new(name)
        .map_or(std::ptr::null_mut(), |name| name.into_raw()) // TODO: panic if name can't be constructed?
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_var_type_info(var: *const Var) -> CExprTypeInfo {
    let var = unsafe { const_ptr_as_ref(var) };

    let ty = match var {
        Var::Standalone { ty, .. } => ty,
        Var::Register { ty, ..} => ty,
        Var::Bit { .. } => panic!("TODO"),
    };

    let width = if let Type::Uint(width) = ty {*width} else {0u16};

    CExprTypeInfo{ty: CExprType::from(ty), width: width}
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_var_as_expr(var: *const Var) -> *mut Expr {
    let var = unsafe{ const_ptr_as_ref(var) };

    Box::into_raw(Box::new(Expr::Var(var.clone())))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_stretch_name(stretch: *const Stretch) -> *mut c_char {
    let stretch = unsafe { const_ptr_as_ref(stretch) };
    
    CString::new(stretch.name.as_str())
        .map_or(std::ptr::null_mut(), |name| name.into_raw()) // TODO: panic if name can't be constructed?
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_stretch_as_expr(stretch: *const Stretch) -> *mut Expr {
    let stretch = unsafe { const_ptr_as_ref(stretch) };

    Box::into_raw(Box::new(Expr::Stretch(stretch.clone())))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inner_build_test_expression() -> *mut Expr {
    let var = Expr::Var(Var::Standalone {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "a".to_owned(),
                    ty: Type::Uint(8),
                });

    let three = Expr::Value(Value::Uint { raw: BigUint::from_str("3").unwrap(), ty: Type::Uint(8) });
    
    let add = Expr::Binary(Box::new(Binary {
        op: BinaryOp::Add,
        left: var,
        right: three,
        ty: Type::Uint(8),
        constant: false,
    }));
    
    let five = Expr::Value(Value::Uint { raw: BigUint::from_str("5").unwrap(), ty: Type::Uint(8) });

    let lt = Expr::Binary(Box::new(Binary {
        op: BinaryOp::LessEqual,
        left: add,
        right: five,
        ty: Type::Uint(8),
        constant: false,
    }));

    Box::into_raw(Box::new(lt))
}
