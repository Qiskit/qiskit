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

use anyhow::{anyhow, bail};
use cbindgen::bindgen::ir::{self, Item};
use hashbrown::HashMap;
use regex::Regex;
use std::sync::LazyLock;

static VOID: ir::Type = ir::Type::Primitive(ir::PrimitiveType::Void);

/// Numeric primitive types that are representable by `ctypes`.
#[derive(Clone, Copy, Debug)]
pub enum Primitive {
    PyObject,
    VoidP,
    Bool,
    Char,
    SByte,
    UByte,
    Single,
    Double,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    LongLong,
    ULongLong,
    Ssize,
    Size,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}
impl Primitive {
    /// The (module-qualified) `ctypes` constructor function of this primitive.
    pub const fn qualname(&self) -> &'static str {
        match self {
            Self::PyObject => "ctypes.py_object",
            Self::VoidP => "ctypes.c_void_p",
            Self::Bool => "ctypes.c_bool",
            Self::Char => "ctypes.c_char",
            Self::SByte => "ctypes.c_byte",
            Self::UByte => "ctypes.c_ubyte",
            Self::Single => "ctypes.c_float",
            Self::Double => "ctypes.c_double",
            Self::Short => "ctypes.c_short",
            Self::UShort => "ctypes.c_ushort",
            Self::Int => "ctypes.c_int",
            Self::UInt => "ctypes.c_uint",
            Self::Long => "ctypes.c_long",
            Self::ULong => "ctypes.c_ulong",
            Self::LongLong => "ctypes.c_longlong",
            Self::ULongLong => "ctypes.c_ulonglong",
            Self::Ssize => "ctypes.c_ssize_t",
            Self::Size => "ctypes.c_size_t",
            Self::I8 => "ctypes.c_int8",
            Self::U8 => "ctypes.c_uint8",
            Self::I16 => "ctypes.c_int16",
            Self::U16 => "ctypes.c_uin16",
            Self::I32 => "ctypes.c_int32",
            Self::U32 => "ctypes.c_uint32",
            Self::I64 => "ctypes.c_int64",
            Self::U64 => "ctypes.c_uint64",
        }
    }

    /// Convert to an equivalent `unsigned` type, if there is one in `ctypes.`
    pub const fn to_unsigned(self) -> Option<Self> {
        match self {
            Self::PyObject
            | Self::VoidP
            | Self::Bool
            | Self::Single
            | Self::Double
            | Self::Char => None,
            Self::SByte | Self::UByte => Some(Self::UByte),
            Self::Short | Self::UShort => Some(Self::UShort),
            Self::Int | Self::UInt => Some(Self::UInt),
            Self::Long | Self::ULong => Some(Self::ULong),
            Self::LongLong | Self::ULongLong => Some(Self::ULongLong),
            Self::Ssize | Self::Size => Some(Self::Size),
            Self::I8 | Self::U8 => Some(Self::U8),
            Self::I16 | Self::U16 => Some(Self::U16),
            Self::I32 | Self::U32 => Some(Self::U32),
            Self::I64 | Self::U64 => Some(Self::U64),
        }
    }

    pub const fn from_cbindgen_intkind(kind: ir::IntKind, signed: bool) -> Self {
        let signed_ty = match kind {
            ir::IntKind::Short => Self::Short,
            ir::IntKind::Int => Self::Int,
            ir::IntKind::Long => Self::Long,
            ir::IntKind::LongLong => Self::LongLong,
            ir::IntKind::SizeT => Self::Ssize,
            // We _should_ dispatch on `config.usize_is_size_t`, but we can't represent the
            // `intptr_t`/`uintptr_t` types that cbindgen uses as the alternative.
            ir::IntKind::Size => Self::Size,
            ir::IntKind::B8 => Self::I8,
            ir::IntKind::B16 => Self::I16,
            ir::IntKind::B32 => Self::I32,
            ir::IntKind::B64 => Self::I64,
        };
        if signed {
            signed_ty
        } else {
            signed_ty
                .to_unsigned()
                .expect("all integer types have unsigned variants")
        }
    }

    pub fn try_from_cbindgen_primitive(ty: &ir::PrimitiveType) -> anyhow::Result<Self> {
        match ty {
            ir::PrimitiveType::Void => Err(anyhow!("cannot represent `void` without `void *`")),
            ir::PrimitiveType::Bool => Ok(Self::Bool),
            ir::PrimitiveType::Char => Ok(Self::Char),
            ir::PrimitiveType::SChar => Ok(Self::SByte),
            ir::PrimitiveType::UChar => Ok(Self::UByte),
            // Rust's `char` has no direct equivalent, so this the best we can do.
            ir::PrimitiveType::Char32 => Ok(Self::U32),
            ir::PrimitiveType::Float => Ok(Self::Single),
            ir::PrimitiveType::Double => Ok(Self::Double),
            ir::PrimitiveType::VaList => Err(anyhow!("variadic arguments not handled")), // come on.
            // ctypes doesn't actually give us access to a `ptrdiff_t` type , which is obviously the
            // most natural conversion.  We also don't get `intptr_t`, which is the next most
            // likely, so this is a bit of a guess...
            ir::PrimitiveType::PtrDiffT => Ok(Self::Ssize),
            ir::PrimitiveType::Integer {
                zeroable: _,
                signed,
                kind,
            } => Ok(Self::from_cbindgen_intkind(*kind, *signed)),
        }
    }

    pub fn try_from_cbindgen_reprtype(ty: &ir::ReprType) -> anyhow::Result<Self> {
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
        Ok(Self::from_cbindgen_intkind(kind, signed))
    }
}

/// The base type of a potentially pointer-indirected type.
#[derive(Clone, Debug)]
pub enum TypeKind {
    Builtin(Primitive),
    Custom(String),
}
impl TypeKind {
    pub fn name(&self) -> &str {
        match self {
            Self::Builtin(primitive) => primitive.qualname(),
            Self::Custom(name) => name,
        }
    }
}

/// A single type for input/output of a `ctypes` function.
///
/// This is version of the type system that is simplified down to the subset that we are able to
/// handle in output.
#[derive(Clone, Debug)]
pub struct Type {
    /// How many levels of pointer indirection before we get to the type.
    pub num_pointers: u8,
    /// The base type.
    pub base: TypeKind,
}
impl Type {
    /// Write this type into an existing string.
    pub fn render(&self, out: &mut String) {
        for _ in 0..self.num_pointers {
            out.push_str("ctypes.POINTER(");
        }
        out.push_str(self.base.name());
        for _ in 0..self.num_pointers {
            out.push(')');
        }
    }

    pub fn try_from_cbindgen(
        mut ty: &ir::Type,
        mut override_fn: impl FnMut(&str) -> Option<Primitive>,
    ) -> anyhow::Result<Self> {
        let mut num_pointers = 0;
        let base = loop {
            match ty {
                ir::Type::Ptr { ty: inner, .. } => {
                    // ctypes special-cases void pointers to avoid needing a standalone `void`, and of
                    // course `PyObject *` is handled specially.
                    if **inner == VOID {
                        break TypeKind::Builtin(Primitive::VoidP);
                    }
                    if matches!(&**inner, ir::Type::Path(p) if p.name() == "PyObject") {
                        break TypeKind::Builtin(Primitive::PyObject);
                    }
                    num_pointers += 1;
                    ty = inner;
                }
                ir::Type::Path(p) => {
                    let name = p.export_name();
                    if let Some(primitive) = override_fn(name) {
                        break TypeKind::Builtin(primitive);
                    } else {
                        break TypeKind::Custom(name.to_owned());
                    }
                }
                ir::Type::Primitive(ty) => {
                    break TypeKind::Builtin(Primitive::try_from_cbindgen_primitive(ty)?);
                }
                ir::Type::Array(..) => bail!("array types not yet handled"),
                ir::Type::FuncPtr { .. } => bail!("funcptrs not yet handled"),
            }
        };
        Ok(Self { num_pointers, base })
    }
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
    /// The `ctypes` primitive type used for each of the enumeration's variants.
    pub ty: Primitive,
    /// Tuples of `(name, literal)` for each of the variants.
    pub variants: Vec<(String, String)>,
}
impl Enum {
    pub fn try_from_cbindgen(val: &ir::Enum) -> anyhow::Result<Self> {
        let Some(reprtype) = val.repr.ty.as_ref() else {
            bail!("repr type of {} must be a fixed integer-like", val.name())
        };
        let ty = Primitive::try_from_cbindgen_reprtype(reprtype)?;
        let variants = val
            .variants
            .iter()
            .map(|variant| -> anyhow::Result<_> {
                let Some(ir::Literal::Expr(discriminant)) = &variant.discriminant else {
                    bail!("unhandled discriminant: {:?}", &variant.discriminant);
                };
                Ok((variant.name.clone(), discriminant.clone()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            name: val.export_name.clone(),
            ty,
            variants,
        })
    }

    /// Get a string that declares this `Enum` as a Python class.
    pub fn declare(&self) -> String {
        let mut out = format!("\nclass {}(enum.Enum):", self.name);
        if self.variants.is_empty() {
            out.push_str("\n    pass");
            return out;
        }
        let constructor = self.ty.qualname();
        for (name, value) in &self.variants {
            out.push_str(&format!("\n    {} = {}({})", name, constructor, value));
        }
        out
    }
}

/// A single C API function.
#[derive(Clone, Debug)]
pub struct Function {
    /// The exported name of the function.
    pub name: String,
    /// Individual argument types.
    pub args: Vec<Type>,
    /// The return type (if not `void`).
    pub ret: Option<Type>,
}
impl Function {
    pub fn try_from_cbindgen(
        func: &ir::Function,
        mut override_fn: impl FnMut(&str) -> Option<Primitive>,
    ) -> anyhow::Result<Self> {
        let name = func.path.name().to_owned();
        let args = func
            .args
            .iter()
            .map(|arg| {
                if arg.array_length.is_some() {
                    bail!("function array arguments not handled yet");
                }
                Type::try_from_cbindgen(&arg.ty, &mut override_fn)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let ret = (func.ret != VOID)
            .then(|| Type::try_from_cbindgen(&func.ret, override_fn))
            .transpose()?;
        Ok(Self { name, args, ret })
    }

    /// Declare the argument and return types of this function.
    ///
    /// If `dllname` name is given, we assume the function is an attribute on an object with that
    /// name.
    pub fn declare(&self, dllname: &str) -> String {
        let prefix = format!("{}.{}", dllname, &self.name);
        let mut out = format!("\n{}.argtypes = [", &prefix);
        if let Some((first, rest)) = self.args.split_first() {
            first.render(&mut out);
            for arg in rest {
                out.push_str(", ");
                arg.render(&mut out);
            }
        }
        out.push_str("]\n");
        out.push_str(&prefix);
        out.push_str(".restype = ");
        match self.ret.as_ref() {
            Some(ret) => ret.render(&mut out),
            None => out.push_str("None"),
        };
        out.push('\n');
        // Re-export into main namespace.
        out.push_str(&self.name);
        out.push_str(" = ");
        out.push_str(&prefix);
        out
    }
}

/// A struct to declare, either opaque or fully specified
#[derive(Clone, Debug)]
pub struct Struct {
    /// The export name of the structure.
    pub name: String,
    /// The fields.  If `None`, this is an opaque `struct` that cannot be directly constructed.
    pub fields: Option<Vec<(String, Type)>>,
}
impl Struct {
    pub fn try_from_cbindgen(
        val: &ir::Struct,
        mut override_fn: impl FnMut(&str) -> Option<Primitive>,
    ) -> anyhow::Result<Self> {
        let fields = val
            .fields
            .iter()
            .map(|field| -> anyhow::Result<_> {
                Ok((
                    field.name.clone(),
                    Type::try_from_cbindgen(&field.ty, &mut override_fn)?,
                ))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            name: val.export_name.clone(),
            fields: Some(fields),
        })
    }

    /// Create a new opaque structure.
    pub fn opaque(name: String) -> Self {
        Self { name, fields: None }
    }

    /// Get a string representing the declaration of this `struct` as a Python `ctypes.Structure`.
    pub fn declare(&self) -> String {
        // TODO: this doesn't handle the case of (mutually) recursive `struct` definitions; we
        // assume that all the `_fields_` will refer to fully defined `ctypes` objects.
        let mut out = format!("\nclass {}(ctypes.Structure):\n", &self.name);
        let Some(fields) = self.fields.as_ref() else {
            out.push_str("    pass\n");
            return out;
        };
        out.push_str("    _fields_ = [\n");
        for (name, ty) in fields {
            out.push_str("        (\"");
            out.push_str(name);
            out.push_str("\", ");
            ty.render(&mut out);
            out.push_str("),\n");
        }
        out.push_str("    ]");
        out
    }
}

/// All of the items to export to a `ctypes` file.
#[derive(Clone, Debug, Default)]
pub struct Items {
    pub enums: Vec<Enum>,
    pub structs: Vec<Struct>,
    pub functions: Vec<Function>,
}
impl Items {
    /// Imports that are needed for our own declarations to work.
    pub const REQUIRED_IMPORTS: &[&str] = &["ctypes", "enum"];

    /// Extract all objects from a set of `cbindgen::Bindings`, adding them to ourselves.
    ///
    /// This fails if the bindings contain any unsupported constructs.
    pub fn add_from_cbindgen(&mut self, bindings: &cbindgen::Bindings) -> anyhow::Result<()> {
        let mut overrides = self
            .enums
            .iter()
            .map(|val| (val.name.clone(), val.ty))
            .collect::<HashMap<_, _>>();

        for item in bindings.items.iter() {
            match item {
                ir::ItemContainer::Enum(item) => {
                    let val = Enum::try_from_cbindgen(item)?;
                    overrides.insert(val.name.clone(), val.ty);
                    self.enums.push(val);
                }
                ir::ItemContainer::OpaqueItem(item) => {
                    self.structs.push(Struct::opaque(item.export_name.clone()))
                }
                ir::ItemContainer::Struct(item) => {
                    self.structs.push(Struct::try_from_cbindgen(item, |path| {
                        overrides.get(path).copied()
                    })?)
                }
                ir::ItemContainer::Constant(_)
                | ir::ItemContainer::Static(_)
                | ir::ItemContainer::Union(_)
                | ir::ItemContainer::Typedef(_) => {
                    bail!("unhandled item: {item:?}");
                }
            }
        }

        for func in &bindings.functions {
            self.functions
                .push(Function::try_from_cbindgen(func, |path| {
                    overrides.get(path).copied()
                })?);
        }

        Ok(())
    }

    /// Export this complete set of objects, subject to the given configuration.
    ///
    /// This includes an `__all__`, a suitable set of `import` statements, a `PyDLL`, and then all
    /// the objects to declare.
    pub fn export(&self, config: &Config, mut out: impl std::io::Write) -> anyhow::Result<()> {
        if let Some(header) = config.header.as_deref() {
            writeln!(out, "{}", header)?;
        }
        writeln!(out)?;

        writeln!(out, "__all__ = [")?;
        writeln!(out, "    \"{}\",", &config.dll.name)?;
        for val in &self.enums {
            writeln!(out, "    \"{}\",", &val.name)?;
        }
        for val in &self.structs {
            writeln!(out, "    \"{}\",", &val.name)?;
        }
        for val in &self.functions {
            writeln!(out, "    \"{}\",", &val.name)?;
        }
        writeln!(out, "]")?;
        writeln!(out)?;

        for import in Self::REQUIRED_IMPORTS {
            writeln!(out, "import {import}")?;
        }
        for import in &config.imports {
            if !Self::REQUIRED_IMPORTS.contains(&import.as_str()) {
                writeln!(out, "import {import}")?;
            }
        }

        writeln!(out)?;
        writeln!(
            out,
            "{} = ctypes.PyDLL({})",
            &config.dll.name, &config.dll.expr
        )?;

        writeln!(out)?;
        for item in &self.enums {
            writeln!(out, "{}", item.declare())?;
        }
        for item in &self.structs {
            writeln!(out, "{}", item.declare())?;
        }
        for item in &self.functions {
            writeln!(out, "{}", item.declare(&config.dll.name))?;
        }

        Ok(())
    }
}

/// Configuration of how to declare a `PyDLL` object containing the `ctypes` bindings.
#[derive(Clone, Debug)]
pub struct DllDeclaration {
    /// The name of the attribute to assign the `PyDLL` to.
    pub name: String,
    /// An expression that results in a filename that can be given to `ctypes.PyDLL`.
    pub expr: String,
}

/// Configuration of the `ctypes` export process.
#[derive(Clone, Debug)]
pub struct Config {
    /// An arbitrary string to write at the start of the file.
    pub header: Option<String>,
    /// Any additional imports to define, such as those needed for the [`DllDeclaration::expr`].
    pub imports: Vec<String>,
    pub dll: DllDeclaration,
}
