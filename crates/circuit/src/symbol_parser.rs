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

/// Parser for equation strings to generate symbolic expression
use nom::branch::{alt, permutation};
use nom::bytes::complete::tag;
use nom::character::complete::{char, digit1, multispace0};
use nom::combinator::{all_consuming, map_res, opt, recognize};
use nom::error::{convert_error, VerboseError};
use nom::multi::{many0, many0_count};
use nom::number::complete::double;
use nom::sequence::{delimited, pair, tuple};
use nom::IResult;
use nom::Parser;

use num_complex::c64;

use crate::symbol_expr::{BinaryOp, SymbolExpr, UnaryOp, Value};

// parsing value as real
fn parse_value(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    map_res(double, |v| -> Result<SymbolExpr, &str> {
        Ok(SymbolExpr::Value(Value::Real(v)))
    })(s)
}

// parsing imaginary part of complex number as real
fn parse_imaginary_value(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    map_res(
        tuple((double, char('i'))),
        |(v, _)| -> Result<SymbolExpr, &str> { Ok(SymbolExpr::Value(Value::Complex(c64(0.0, v)))) },
    )(s)
}

fn alpha1(i: &str) -> IResult<&str, &str, VerboseError<&str>> {
    nom_unicode::complete::alpha1(i)
}

fn alphanumeric1(i: &str) -> IResult<&str, &str, VerboseError<&str>> {
    nom_unicode::complete::alphanumeric1(i)
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
    map_res(
        tuple((
            parse_symbol_string,
            opt(delimited(char('['), digit1, char(']'))),
        )),
        |(v, array_idx)| -> Result<SymbolExpr, &str> {
            match array_idx {
                Some(i) => {
                    // currently array index is stored as string
                    // if array indexing is required in the future
                    // add indexing in Symbol struct
                    let s = format!("{}[{}]", v, i);
                    Ok(SymbolExpr::Symbol(Box::new(s)))
                }
                None => Ok(SymbolExpr::Symbol(Box::new(v.to_string()))),
            }
        },
    )(s)
}

// parse unary operations
fn parse_unary(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    map_res(
        tuple((
            delimited(multispace0, alphanumeric1, multispace0),
            delimited(
                char('('),
                delimited(multispace0, parse_addsub, multispace0),
                char(')'),
            ),
        )),
        |(v, expr)| -> Result<SymbolExpr, &str> {
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
                expr: Box::new(expr),
            })
        },
    )(s)
}

// sign is separetely parsed in this function
fn parse_sign(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    map_res(
        tuple((
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
        )),
        |(s, expr)| -> Result<SymbolExpr, &str> {
            if s == '+' {
                Ok(expr)
            } else {
                Ok(SymbolExpr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
        },
    )(s)
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
    ))(s)
}

// parse pow
fn parse_pow(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    map_res(
        permutation((
            parse_expr,
            many0(map_res(
                tuple((multispace0, tag("**"), multispace0, parse_expr)),
                |(_, _, _, rhs)| -> Result<SymbolExpr, &str> { Ok(rhs) },
            )),
        )),
        |(lhs, rvec)| -> Result<SymbolExpr, &str> {
            let accum = rvec.iter().fold(lhs, |acc, x| acc.pow(x));
            Ok(accum)
        },
    )(s)
}

// parse mul and div
fn parse_muldiv(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    map_res(
        permutation((
            parse_pow,
            many0(map_res(
                tuple((
                    multispace0,
                    alt((tag("*"), tag("/"))),
                    multispace0,
                    parse_pow,
                )),
                |(_, opr, _, rhs)| -> Result<(BinaryOp, SymbolExpr), &str> {
                    if opr == "*" {
                        Ok((BinaryOp::Mul, rhs))
                    } else {
                        Ok((BinaryOp::Div, rhs))
                    }
                },
            )),
        )),
        |(lhs, rvec)| -> Result<SymbolExpr, &str> {
            let accum = rvec.iter().fold(lhs, |acc, x| match x.0 {
                BinaryOp::Mul => &acc * &x.1,
                BinaryOp::Div => &acc / &x.1,
                _ => acc,
            });
            Ok(accum)
        },
    )(s)
}

// parse add and sub
fn parse_addsub(s: &str) -> IResult<&str, SymbolExpr, VerboseError<&str>> {
    map_res(
        permutation((
            parse_muldiv,
            many0(map_res(
                tuple((
                    multispace0,
                    alt((char('+'), char('-'))),
                    multispace0,
                    parse_muldiv,
                )),
                |(_, opr, _, rhs)| -> Result<(BinaryOp, SymbolExpr), &str> {
                    if opr == '+' {
                        Ok((BinaryOp::Add, rhs))
                    } else {
                        Ok((BinaryOp::Sub, rhs))
                    }
                },
            )),
        )),
        |(lhs, rvec)| -> Result<SymbolExpr, &str> {
            let accum = rvec.iter().fold(lhs, |acc, x| match x.0 {
                BinaryOp::Add => &acc + &x.1,
                BinaryOp::Sub => &acc - &x.1,
                _ => acc,
            });
            Ok(accum)
        },
    )(s)
}

pub fn parse_expression(s: &str) -> Result<SymbolExpr, String> {
    let mut parser = all_consuming(parse_addsub);
    match parser(s) {
        Ok(o) => Ok(o.1),
        Err(e) => match e {
            nom::Err::Error(e) => Err(convert_error(s, e)),
            _ => Err(format!(" Error occurs while parsing expression {}.", s)),
        },
    }
}
