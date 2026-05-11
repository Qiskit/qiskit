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

//! Definitions for a simplified version of the `cbindgen` output IR that is a little more
//! structured for output in the various stranger languages/formats we require.  This also strips
//! out extraneous detail that Qiskit doesn't use, so it's less work to write exporters.

use anyhow::bail;
use cbindgen::bindgen::ir::{self, Item};
use regex::Regex;
use std::sync::LazyLock;

/// Public re-definition of `cbindgen`'s `ReprType`, which (as of `cbindgen 0.29.2`) doesn't make
/// its members public.
///
/// Can be removed once https://github.com/mozilla/cbindgen/pull/1161 is in a release.
#[derive(Clone, Copy, Debug)]
pub struct ReprType {
    pub kind: ir::IntKind,
    pub signed: bool,
}

impl ReprType {
    pub fn try_from_cbindgen(ty: &ir::ReprType) -> anyhow::Result<Self> {
        // Ok, just bear with me here: `cbindgen` doesn't make these fields `pub`, but it _does_
        // make the struct impl `Debug`...
        static REPR_KIND_DEBUG: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r#"ReprType \{ kind: ([^,]+), signed: (\w+) \}"#).unwrap());
        let repr = format!("{ty:?}");
        let Some(captures) = REPR_KIND_DEBUG.captures(&repr) else {
            bail!("failed to parse `ReprKind` debug format: did the `cbindgen` version change?");
        };
        let kind = match captures.get(1).unwrap().as_str() {
            "Short" => ir::IntKind::Short,
            "Int" => ir::IntKind::Int,
            "Long" => ir::IntKind::Long,
            "LongLong" => ir::IntKind::LongLong,
            "SizeT" => ir::IntKind::SizeT,
            "Size" => ir::IntKind::Size,
            "B8" => ir::IntKind::B8,
            "B16" => ir::IntKind::B16,
            "B32" => ir::IntKind::B32,
            "B64" => ir::IntKind::B64,
            bad => bail!("unhandled `IntKind`: {bad}"),
        };
        let signed = match captures.get(2).unwrap().as_str() {
            "false" => false,
            "true" => true,
            bad => bail!("unhandled `bool`: {bad}"),
        };
        Ok(Self { kind, signed })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PtrKind {
    Mut,
    Const,
}
#[derive(Clone, Debug)]
pub enum TypeKind<T> {
    Builtin(T),
    Custom(String),
}
#[derive(Clone, Debug)]
pub struct Type<T> {
    /// The types of pointer that apply to the argument.  These go from "closest" to the `base` (at
    /// index 0) to furthest away (closest to the caller).
    ///
    /// For example, `*mut *const T` has `ptrs: vec![Const, Mut]`.
    pub ptrs: Vec<PtrKind>,
    pub base: TypeKind<T>,
}

#[derive(Clone, Debug)]
pub struct EnumVariant {
    pub name: String,
    pub value: String,
}

/// Representation of a Qiskit-native `enum`.
///
/// We export a corresponding `class <Name>(enum.Enum)` to Python with a named attribute for each of
/// the variants for ease of use in Python space, but in actual function interfaces we have to use
/// the raw `ctypes` backing type to get the ABI correct.
#[derive(Clone, Debug)]
pub struct Enum {
    /// The export name of the enum.
    pub name: String,
    /// The primitive-integer representaiton of the enum.
    pub repr: ReprType,
    /// Tuples of `(name, literal)` for each of the variants.
    pub variants: Vec<EnumVariant>,
}
impl Enum {
    pub fn try_from_cbindgen(val: &ir::Enum) -> anyhow::Result<Self> {
        let Some(reprtype) = val.repr.ty.as_ref() else {
            bail!("repr type of {} must be a fixed integer-like", val.name())
        };
        let repr = ReprType::try_from_cbindgen(reprtype)?;
        let variants = val
            .variants
            .iter()
            .map(|variant| -> anyhow::Result<_> {
                let Some(ir::Literal::Expr(discriminant)) = &variant.discriminant else {
                    bail!("unhandled discriminant: {:?}", &variant.discriminant);
                };
                Ok(EnumVariant {
                    name: variant.name.clone(),
                    value: discriminant.clone(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            name: val.export_name.clone(),
            repr,
            variants,
        })
    }
}

#[derive(Clone, Debug)]
pub struct StructField<T> {
    pub name: String,
    pub ty: Type<T>,
}

/// A struct to declare, either opaque or fully specified
#[derive(Clone, Debug)]
pub struct Struct<T> {
    /// The export name of the structure.
    pub name: String,
    /// The fields.  If `None`, this is an opaque `struct` that cannot be directly constructed.
    pub fields: Option<Vec<StructField<T>>>,
}
impl<T> Struct<T> {
    pub fn opaque(name: String) -> Self {
        Self { name, fields: None }
    }
}

#[derive(Clone, Debug)]
pub struct FunctionArg<T> {
    pub name: Option<String>,
    pub ty: Type<T>,
}

/// A single C API function.
#[derive(Clone, Debug)]
pub struct Function<T> {
    /// The exported name of the function.
    pub name: String,
    /// Individual argument types.
    pub args: Vec<FunctionArg<T>>,
    /// The return type (if not `void`).
    pub ret: Option<Type<T>>,
}

#[derive(Clone, Debug)]
pub struct Items<T> {
    pub enums: Vec<Enum>,
    pub structs: Vec<Struct<T>>,
    pub functions: Vec<Function<T>>,
}
