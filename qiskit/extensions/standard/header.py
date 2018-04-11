# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Standard extension's OPENQASM header and definition update.
"""
import sympy
from qiskit import QuantumCircuit
from qiskit.qasm import _node as node


if not hasattr(QuantumCircuit, '_extension_standard'):
    QuantumCircuit._extension_standard = True
    QuantumCircuit.header = QuantumCircuit.header + "\n" \
        + "include \"qelib1.inc\";"

    # 3-parameter 2-pulse single qubit gate
    QuantumCircuit.definitions["u3"] = {
        "print": False,
        "opaque": False,
        "n_args": 3,
        "n_bits": 1,
        "args": ["theta", "phi", "lambda"],
        "bits": ["q"],
        # gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
        "body": node.GateBody([
            node.UniversalUnitary([
                node.ExpressionList([
                    node.Id("theta", 0, ""),
                    node.Id("phi", 0, ""),
                    node.Id("lambda", 0, "")
                ]),
                node.Id("q", 0, "")
            ])
        ])
    }

    # 2-parameter 1-pulse single qubit gate
    QuantumCircuit.definitions["u2"] = {
        "print": False,
        "opaque": False,
        "n_args": 2,
        "n_bits": 1,
        "args": ["phi", "lambda"],
        "bits": ["q"],
        # gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
        "body": node.GateBody([
            node.UniversalUnitary([
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Real(sympy.pi),
                        node.Int(2)
                    ]),
                    node.Id("phi", 0, ""),
                    node.Id("lambda", 0, "")
                ]),
                node.Id("q", 0, "")
            ])
        ])
    }

    # 1-parameter 0-pulse single qubit gate
    QuantumCircuit.definitions["u1"] = {
        "print": False,
        "opaque": False,
        "n_args": 1,
        "n_bits": 1,
        "args": ["lambda"],
        "bits": ["q"],
        # gate u1(lambda) q { U(0,0,lambda) q; }
        "body": node.GateBody([
            node.UniversalUnitary([
                node.ExpressionList([
                    node.Int(0),
                    node.Int(0),
                    node.Id("lambda", 0, "")
                ]),
                node.Id("q", 0, "")
            ])
        ])
    }

    # controlled-NOT
    QuantumCircuit.definitions["cx"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 2,
        "args": [],
        "bits": ["c", "t"],
        # gate cx c,t { CX c,t; }
        "body": node.GateBody([
            node.Cnot([
                node.Id("c", 0, ""),
                node.Id("t", 0, "")
            ])
        ])
    }

    # idle gate (identity)
    QuantumCircuit.definitions["id"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate id a { U(0,0,0) a; }
        "body": node.GateBody([
            node.UniversalUnitary([
                node.ExpressionList([
                    node.Int(0),
                    node.Int(0),
                    node.Int(0)
                ]),
                node.Id("a", 0, "")
            ])
        ])
    }

    # idle gate (identity) with length gamma*sqglen
    QuantumCircuit.definitions["u0"] = {
        "print": False,
        "opaque": False,
        "n_args": 1,
        "n_bits": 1,
        "args": ["gamma"],
        "bits": ["q"],
        # gate u0(gamma) q { U(0,0,0) q; }
        "body": node.GateBody([
            node.UniversalUnitary([
                node.ExpressionList([
                    node.Int(0),
                    node.Int(0),
                    node.Int(0)
                ]),
                node.Id("q", 0, "")
            ])
        ])
    }

    # Pauli gate: bit-flip
    QuantumCircuit.definitions["x"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate x a { u3(pi,0,pi) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u3", 0, ""),
                node.ExpressionList([
                    node.Real(sympy.pi),
                    node.Int(0),
                    node.Real(sympy.pi)
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # Pauli gate: bit and phase flip
    QuantumCircuit.definitions["y"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate y a { u3(pi,pi/2,pi/2) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u3", 0, ""),
                node.ExpressionList([
                    node.Real(sympy.pi),
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Real(sympy.pi),
                        node.Int(2)
                    ]),
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Real(sympy.pi),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # Pauli gate: phase flip
    QuantumCircuit.definitions["z"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate z a { u1(pi) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.Real(sympy.pi)
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # Clifford gate: Hadamard
    QuantumCircuit.definitions["h"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate h a { u2(0,pi) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u2", 0, ""),
                node.ExpressionList([
                    node.Int(0),
                    node.Real(sympy.pi)
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # Clifford gate: sqrt(Z) phase gate
    QuantumCircuit.definitions["s"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate s a { u1(pi/2) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Real(sympy.pi),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # Clifford gate: conjugate of sqrt(Z)
    QuantumCircuit.definitions["sdg"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate sdg a { u1(-pi/2) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Prefix([
                            node.UnaryOperator('-'),
                            node.Real(sympy.pi)
                        ]),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # C3 gate: sqrt(S) phase gate
    QuantumCircuit.definitions["t"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate t a { u1(pi/4) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Real(sympy.pi),
                        node.Int(4)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # C3 gate: conjugate of sqrt(S)
    QuantumCircuit.definitions["tdg"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 1,
        "args": [],
        "bits": ["a"],
        # gate tdg a { u1(-pi/4) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Prefix([
                            node.UnaryOperator('-'),
                            node.Real(sympy.pi)
                        ]),
                        node.Int(4)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # Rotation around X-axis
    QuantumCircuit.definitions["rx"] = {
        "print": False,
        "opaque": False,
        "n_args": 1,
        "n_bits": 1,
        "args": ["theta"],
        "bits": ["a"],
        # gate rx(theta) a { u3(theta, -pi/2,pi/2) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u3", 0, ""),
                node.ExpressionList([
                    node.Id("theta", 0, ""),
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Prefix([
                            node.UnaryOperator('-'),
                            node.Real(sympy.pi)
                        ]),
                        node.Int(2)
                    ]),
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Real(sympy.pi),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # Rotation around Y-axis
    QuantumCircuit.definitions["ry"] = {
        "print": False,
        "opaque": False,
        "n_args": 1,
        "n_bits": 1,
        "args": ["theta"],
        "bits": ["a"],
        # gate ry(theta) a { u3(theta,0,0) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u3", 0, ""),
                node.ExpressionList([
                    node.Id("theta", 0, ""),
                    node.Int(0),
                    node.Int(0)
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # Rotation around Z-axis
    QuantumCircuit.definitions["rz"] = {
        "print": False,
        "opaque": False,
        "n_args": 1,
        "n_bits": 1,
        "args": ["phi"],
        "bits": ["a"],
        # gate rz(phi) a { u1(phi) a; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.Id("phi", 0, "")
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # controlled-Phase
    QuantumCircuit.definitions["cz"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 2,
        "args": [],
        "bits": ["a", "b"],
        # gate cz a,b { h b; cx a,b; h b; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("h", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("h", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ])
        ])
    }

    # controlled-Y
    QuantumCircuit.definitions["cy"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 2,
        "args": [],
        "bits": ["a", "b"],
        # gate cy a,b { sdg b; cx a,b; s b; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("sdg", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("s", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ])
        ])
    }

    # swap
    QuantumCircuit.definitions["swap"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 2,
        "args": [],
        "bits": ["a", "b"],
        # gate swap a,b { cx a,b; cx b,a; cx a,b; }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, ""),
                    node.Id("a", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ])
        ])
    }

    # controlled-H
    QuantumCircuit.definitions["ch"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 2,
        "args": [],
        "bits": ["a", "b"],
        # gate ch a,b {
        # h b; sdg b;
        # cx a,b;
        # h b; t b;
        # cx a,b;
        # t b; h b; s b; x b; s a;
        # }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("h", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("sdg", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("h", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("t", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("t", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("h", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("s", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("x", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("s", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ])
        ])
    }

    # C3 gate: Toffoli
    QuantumCircuit.definitions["ccx"] = {
        "print": False,
        "opaque": False,
        "n_args": 0,
        "n_bits": 3,
        "args": [],
        "bits": ["a", "b", "c"],
        # gate ccx a,b,c
        # {
        #   h c;
        #   cx b,c; tdg c;
        #   cx a,c; t c;
        #   cx b,c; tdg c;
        #   cx a,c; t b; t c; h c;
        #   cx a,b; t a; tdg b;
        #   cx a,b;
        # }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("h", 0, ""),
                node.PrimaryList([
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, ""),
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("tdg", 0, ""),
                node.PrimaryList([
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("t", 0, ""),
                node.PrimaryList([
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, ""),
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("tdg", 0, ""),
                node.PrimaryList([
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("t", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("t", 0, ""),
                node.PrimaryList([
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("h", 0, ""),
                node.PrimaryList([
                    node.Id("c", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("t", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("tdg", 0, ""),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ])
        ])
    }

    # controlled rz rotation
    QuantumCircuit.definitions["crz"] = {
        "print": False,
        "opaque": False,
        "n_args": 1,
        "n_bits": 2,
        "args": ["lambda"],
        "bits": ["a", "b"],
        # gate crz(lambda) a,b
        # {
        # u1(lambda/2) b;
        # cx a,b;
        # u1(-lambda/2) b;
        # cx a,b;
        # }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Id("lambda", 0, ""),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Prefix([
                            node.UnaryOperator('-'),
                            node.Id("lambda", 0, "")
                        ]),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ])
        ])
    }

    # controlled phase rotation
    QuantumCircuit.definitions["cu1"] = {
        "print": False,
        "opaque": False,
        "n_args": 1,
        "n_bits": 2,
        "args": ["lambda"],
        "bits": ["a", "b"],
        # gate cu1(lambda) a,b
        # {
        # u1(lambda/2) a;
        # cx a,b;
        # u1(-lambda/2) b;
        # cx a,b;
        # u1(lambda/2) b;
        # }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Id("lambda", 0, ""),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("a", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Prefix([
                            node.UnaryOperator('-'),
                            node.Id("lambda", 0, "")
                        ]),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("a", 0, ""),
                    node.Id("b", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Id("lambda", 0, ""),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("b", 0, "")
                ])
            ])
        ])
    }

    # controlled-U
    QuantumCircuit.definitions["cu3"] = {
        "print": False,
        "opaque": False,
        "n_args": 3,
        "n_bits": 2,
        "args": ["theta", "phi", "lambda"],
        "bits": ["c", "t"],
        # gate cu3(theta,phi,lambda) c, t
        # {
        #  u1((lambda-phi)/2) t;
        #  cx c,t;
        #  u3(-theta/2,0,-(phi+lambda)/2) t;
        #  cx c,t;
        #  u3(theta/2,phi,0) t;
        # }
        "body": node.GateBody([
            node.CustomUnitary([
                node.Id("u1", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.BinaryOp([
                            node.BinaryOperator('-'),
                            node.Id("lambda", 0, ""),
                            node.Id("phi", 0, "")
                        ]),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("t", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("c", 0, ""),
                    node.Id("t", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("u3", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Prefix([
                            node.UnaryOperator('-'),
                            node.Id("theta", 0, "")
                        ]),
                        node.Int(2)
                    ]),
                    node.Int(0),
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Prefix([
                            node.UnaryOperator('-'),
                            node.BinaryOp([
                                node.BinaryOperator('+'),
                                node.Id("phi", 0, ""),
                                node.Id("lambda", 0, "")
                            ]),
                        ]),
                        node.Int(2)
                    ])
                ]),
                node.PrimaryList([
                    node.Id("t", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("cx", 0, ""),
                node.PrimaryList([
                    node.Id("c", 0, ""),
                    node.Id("t", 0, "")
                ])
            ]),
            node.CustomUnitary([
                node.Id("u3", 0, ""),
                node.ExpressionList([
                    node.BinaryOp([
                        node.BinaryOperator('/'),
                        node.Id("theta", 0, ""),
                        node.Int(2)
                    ]),
                    node.Id("phi", 0, ""),
                    node.Int(0)
                ]),
                node.PrimaryList([
                    node.Id("t", 0, "")
                ])
            ])
        ])
    }
