builtin.module @pbc_program 
{
  ^block1v1():
    pbc.circuit {circuit_name : "main", program_width : <2: si64>}
    {
      ^entry_block2v1():
        pivot_v0 = pbc.h_qubit_decl : pbc.h_qubit <0,0> !0;
        q0_v1 = pbc.h_qubit_decl : pbc.h_qubit <0,1> !1;
        v2 = pbc.pauli_rotation (q0_v1) {pbc_pauli_bases : [X], angle : 1.57079632679489661923132169164} : pbc.h_qubit <0,1>;
        v3 = pbc.pauli_rotation (v2) {pbc_pauli_bases : [Z], angle : -3.14159265358979323846264338328} : pbc.h_qubit <0,1>;
        pbc.return () : 
    }
}

outlined_attributes:
!0 = [builtin_given_names = builtin.given_names [pivot]]
!1 = [builtin_given_names = builtin.given_names [q0]]
