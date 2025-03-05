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
use nom::multi::{many0, many0_count};
use nom::number::complete::double;
use nom::sequence::{delimited, pair, tuple};
use nom::IResult;
use nom::Parser;

use num_complex::c64;

use crate::symbol_expr::{BinaryOp, Symbol, SymbolExpr, UnaryOp, Value};

// struct to contain parsed binary operation
#[derive(Clone)]
struct BinaryOpContainer {
    op: BinaryOp,
    expr: SymbolExpr,
}

impl BinaryOpContainer {
    fn accum(self, rhs: BinaryOpContainer) -> BinaryOpContainer {
        match rhs.op {
            BinaryOp::Add => BinaryOpContainer {
                op: rhs.op,
                expr: self.expr + rhs.expr,
            },
            BinaryOp::Sub => BinaryOpContainer {
                op: rhs.op,
                expr: self.expr - rhs.expr,
            },
            BinaryOp::Mul => BinaryOpContainer {
                op: rhs.op,
                expr: self.expr * rhs.expr,
            },
            BinaryOp::Div => BinaryOpContainer {
                op: rhs.op,
                expr: self.expr / rhs.expr,
            },
            BinaryOp::Pow => BinaryOpContainer {
                op: rhs.op,
                expr: self.expr.pow(&rhs.expr),
            },
        }
    }
}

// parsing value as real
fn parse_value(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(double, |v| -> Result<BinaryOpContainer, &str> {
        Ok(BinaryOpContainer {
            op: BinaryOp::Add,
            expr: SymbolExpr::Value(Value::Real(v)),
        })
    })(s)
}

// parsing imaginary part of complex number as real
fn parse_imaginary_value(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        tuple((double, char('i'))),
        |(v, _)| -> Result<BinaryOpContainer, &str> {
            Ok(BinaryOpContainer {
                op: BinaryOp::Add,
                expr: SymbolExpr::Value(Value::Complex(c64(0.0, v))),
            })
        },
    )(s)
}

fn alpha1(i: &str) -> IResult<&str, &str> {
    nom_unicode::complete::alpha1(i)
}

fn alphanumeric1(i: &str) -> IResult<&str, &str> {
    nom_unicode::complete::alphanumeric1(i)
}

fn parse_symbol_string(s: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ))
    .parse(s)
}

fn parse_special_char(s: &str) -> IResult<&str, &str> {
    recognize(tuple((tag("$\\"), alpha1, tag("$")))).parse(s)
}

// parse string as symbol
// symbol starting with alphabet and can contain numbers and '_', '[', ']'
fn parse_symbol(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        tuple((
            alt((parse_special_char, parse_symbol_string)),
            opt(delimited(char('['), digit1, char(']'))),
        )),
        |(v, array_idx)| -> Result<BinaryOpContainer, &str> {
            match array_idx {
                Some(i) => {
                    // currently array index is stored as string
                    // if array indexing is required in the future
                    // add indexing in Symbol struct
                    let s = format!("{}[{}]", v, i);
                    Ok(BinaryOpContainer {
                        op: BinaryOp::Add,
                        expr: SymbolExpr::Symbol(Symbol::new(&s)),
                    })
                }
                None => Ok(BinaryOpContainer {
                    op: BinaryOp::Add,
                    expr: SymbolExpr::Symbol(Symbol::new(v)),
                }),
            }
        },
    )(s)
}

// parse unary operations
fn parse_unary(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        tuple((
            alphanumeric1,
            delimited(
                char('('),
                delimited(multispace0, parse_addsub, multispace0),
                char(')'),
            ),
        )),
        |(v, expr)| -> Result<BinaryOpContainer, &str> {
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
                &_ => return Err("unsupported unary operation found."),
            };
            Ok(BinaryOpContainer {
                op: BinaryOp::Add,
                expr: SymbolExpr::Unary{op: op, expr: Box::new(expr.expr)},
            })
        },
    )(s)
}

// neg operation is separetely parsed in this function
fn parse_neg(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        tuple((
            char('-'),
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
        |(_, expr)| -> Result<BinaryOpContainer, &str> {
            Ok(BinaryOpContainer {
                op: BinaryOp::Add,
                expr: SymbolExpr::Unary{op: UnaryOp::Neg, expr: Box::new(expr.expr)},
            })
        },
    )(s)
}

fn parse_expr(s: &str) -> IResult<&str, BinaryOpContainer> {
    alt((
        parse_imaginary_value,
        parse_value,
        parse_neg,
        parse_unary,
        parse_symbol,
        delimited(
            char('('),
            delimited(multispace0, parse_addsub, multispace0),
            char(')'),
        ),
    ))(s)
}

// parse mul and div and pow
fn parse_muldiv(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        permutation((
            parse_expr,
            many0(map_res(
                tuple((
                    multispace0,
                    alt((tag("**"), tag("*"), tag("/"))),
                    multispace0,
                    parse_expr,
                )),
                |(_, opr, _, mut rhs)| -> Result<BinaryOpContainer, &str> {
                    if opr == "**" {
                        rhs.op = BinaryOp::Pow;
                        Ok(rhs)
                    } else if opr == "*" {
                        rhs.op = BinaryOp::Mul;
                        Ok(rhs)
                    } else {
                        rhs.op = BinaryOp::Div;
                        Ok(rhs)
                    }
                },
            )),
        )),
        |(lhs, rvec)| -> Result<BinaryOpContainer, &str> {
            Ok(rvec.iter().fold(lhs, |acc, x| acc.accum(x.clone())))
        },
    )(s)
}

// parse add and sub
fn parse_addsub(s: &str) -> IResult<&str, BinaryOpContainer> {
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
                |(_, opr, _, mut rhs)| -> Result<BinaryOpContainer, &str> {
                    if opr == '+' {
                        rhs.op = BinaryOp::Add;
                        Ok(rhs)
                    } else {
                        rhs.op = BinaryOp::Sub;
                        Ok(rhs)
                    }
                },
            )),
        )),
        |(lhs, rvec)| -> Result<BinaryOpContainer, &str> {
            Ok(rvec.iter().fold(lhs, |acc, x| acc.accum(x.clone())))
        },
    )(s)
}

pub fn parse_expression(s: &str) -> SymbolExpr {
    let mut parser = all_consuming(parse_addsub);
    parser(s).unwrap().1.expr
}
