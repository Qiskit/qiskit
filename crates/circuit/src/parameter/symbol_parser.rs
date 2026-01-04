// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Parser for equation strings to generate symbolic expression
use std::sync::Arc;

use nom::branch::{alt, permutation};
use nom::bytes::complete::tag;
use nom::character::complete::{char, digit1, multispace0};
use nom::combinator::{all_consuming, opt, recognize};
use nom::multi::{many0, many0_count};
use nom::number::complete::double;
use nom::sequence::{delimited, pair};
use nom::{IResult, Parser};
use nom_language::error::{VerboseError, convert_error};
use nom_unicode::complete::{alpha1, alphanumeric1};

use num_complex::c64;

use crate::parameter::symbol_expr::{BinaryOp, SymbolExpr, UnaryOp, Value};

use super::symbol_expr::Symbol;

// parsing value as real
fn parse_value(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    double.map(|v| SymbolExpr::Value(Value::Real(v))).parse(s)
}

// parsing imaginary part of complex number as real
fn parse_imaginary_value(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    (double, char('i'))
        .map(|(v, _)| SymbolExpr::Value(Value::Complex(c64(0.0, v))))
        .parse(s)
}

fn parse_symbol_string(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
    recognize(pair(
        alt((alpha1, tag("_"), tag("\\"), tag("$"))),
        many0_count(alt((alphanumeric1, tag("_"), tag("\\"), tag("$")))),
    ))
    .parse(s)
}

// parse string as symbol
// symbol starting with alphabet and can contain numbers and '_', '\', '$', '[', ']'
fn parse_symbol(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    (
        parse_symbol_string,
        opt(delimited(char('['), digit1, char(']'))),
    )
        .map_res(|(v, array_idx)| -> Result<_, &str> {
            let index = array_idx
                .map(|i| i.parse::<u32>())
                .transpose()
                .map_err(|_| "index out of bounds")?;
            Ok(SymbolExpr::Symbol(Arc::new(Symbol::new(v, None, index))))
        })
        .parse(s)
}

// parse unary operations
fn parse_unary(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    (
        delimited(multispace0, alphanumeric1, multispace0),
        delimited(
            char('('),
            delimited(multispace0, parse_addsub, multispace0),
            char(')'),
        ),
    )
        .map_res(|(v, expr)| {
            let op = match v {
                "sin" => UnaryOp::Sin,
                "asin" => UnaryOp::Asin,
                "cos" => UnaryOp::Cos,
                "acos" => UnaryOp::Acos,
                "tan" => UnaryOp::Tan,
                "atan" => UnaryOp::Atan,
                "log" => UnaryOp::Log,
                "exp" => UnaryOp::Exp,
                "sign" => UnaryOp::Sign,
                "conjugate" => UnaryOp::Conj,
                "abs" => UnaryOp::Abs,
                &_ => return Err("unsupported unary operation found."),
            };
            Ok(SymbolExpr::Unary {
                op,
                expr: Arc::new(expr),
            })
        })
        .parse(s)
}

// sign is separetely parsed in this function
fn parse_sign(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    (
        delimited(multispace0, alt((char('-'), char('+'))), multispace0),
        alt((
            parse_imaginary_value,
            parse_value,
            parse_unary,
            parse_symbol,
            delimited(
                char('('),
                delimited(multispace0, parse_addsub, multispace0),
                char(')'),
            ),
        )),
    )
        .map(|(op, expr)| {
            if op == '+' {
                expr
            } else {
                SymbolExpr::Unary {
                    op: UnaryOp::Neg,
                    expr: Arc::new(expr),
                }
            }
        })
        .parse(s)
}

fn parse_expr(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    alt((
        parse_imaginary_value,
        parse_value,
        parse_sign,
        parse_unary,
        parse_symbol,
        delimited(
            char('('),
            delimited(multispace0, parse_addsub, multispace0),
            char(')'),
        ),
    ))
    .parse(s)
}

// parse pow
fn parse_pow(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    permutation((
        parse_expr,
        many0((multispace0, tag("**"), multispace0, parse_expr).map(|(_, _, _, rhs)| rhs)),
    ))
    .map(|(lhs, rvec)| rvec.iter().fold(lhs, |acc, x| acc.pow(x)))
    .parse(s)
}

// parse mul and div
fn parse_muldiv(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    permutation((
        parse_pow,
        many0(
            (
                multispace0,
                alt((tag("*"), tag("/"))),
                multispace0,
                parse_pow,
            )
                .map(|(_, opr, _, rhs)| {
                    if opr == "*" {
                        (BinaryOp::Mul, rhs)
                    } else {
                        (BinaryOp::Div, rhs)
                    }
                }),
        ),
    ))
    .map(|(lhs, rvec)| {
        rvec.iter().fold(lhs, |acc, x| match x.0 {
            BinaryOp::Mul => &acc * &x.1,
            BinaryOp::Div => &acc / &x.1,
            _ => acc,
        })
    })
    .parse(s)
}

// parse add and sub
fn parse_addsub(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    permutation((
        parse_muldiv,
        many0(
            (
                multispace0,
                alt((char('+'), char('-'))),
                multispace0,
                parse_muldiv,
            )
                .map(|(_, opr, _, rhs)| {
                    if opr == '+' {
                        (BinaryOp::Add, rhs)
                    } else {
                        (BinaryOp::Sub, rhs)
                    }
                }),
        ),
    ))
    .map(|(lhs, rvec)| {
        rvec.iter().fold(lhs, |acc, x| match x.0 {
            BinaryOp::Add => &acc + &x.1,
            BinaryOp::Sub => &acc - &x.1,
            _ => acc,
        })
    })
    .parse(s)
}

pub fn parse_expression(s: &str) -> Result<SymbolExpr, String> {
    match all_consuming(parse_addsub).parse(s) {
        Ok(o) => Ok(o.1),
        Err(e) => match e {
            nom::Err::Error(e) => Err(convert_error(s, e)),
            _ => Err(format!("Error while parsing '{s}': {e}")),
        },
    }
}
