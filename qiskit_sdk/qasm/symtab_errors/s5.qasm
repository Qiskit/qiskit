
gate cnot a,b { CX a, b; } // ok
gate cnot a,b { CX a, c; } // notok
