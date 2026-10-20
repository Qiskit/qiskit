builtin.module @pbc_program 
{
  ^block1v1():
    pbc.circuit {circuit_name : "main", program_width : <2: si64>}
    {
      ^entry_block2v1():
        pivot_v0 = pbc.h_qubit_decl : pbc.h_qubit <0,0> !0;
        q0_v1 = pbc.h_qubit_decl : pbc.h_qubit <0,1> !1;
        b_v2 = pbc.bool_decl : pbc.bool  !2;
        q0_v3, b_v4 = pbc.pauli_measurement (q0_v1) {pbc_pauli_bases : [X]} : pbc.h_qubit <0,1>, pbc.bool  !3;
        pbc.return () : 
    }
}

outlined_attributes:
!0 = [builtin_given_names = builtin.given_names [pivot]]
!1 = [builtin_given_names = builtin.given_names [q0]]
!2 = [builtin_given_names = builtin.given_names [b]]
!3 = [builtin_given_names = builtin.given_names [q0, b]]
