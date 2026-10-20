builtin.module @pbc_program 
{
  ^block1v1():
    pbc.circuit {circuit_name : "main", program_width : <14: si64>}
    {
      ^entry_block2v1():
        pivot_v0 = pbc.h_qubit_decl : pbc.h_qubit <0,0> !0;
        q0_v1 = pbc.h_qubit_decl : pbc.h_qubit <0,1> !1;
        q1_v2 = pbc.h_qubit_decl : pbc.h_qubit <0,2> !2;
        q2_v3 = pbc.h_qubit_decl : pbc.h_qubit <0,3> !3;
        q3_v4 = pbc.h_qubit_decl : pbc.h_qubit <0,4> !4;
        q4_v5 = pbc.h_qubit_decl : pbc.h_qubit <0,5> !5;
        q5_v6 = pbc.h_qubit_decl : pbc.h_qubit <0,6> !6;
        q6_v7 = pbc.h_qubit_decl : pbc.h_qubit <0,7> !7;
        q7_v8 = pbc.h_qubit_decl : pbc.h_qubit <0,8> !8;
        q8_v9 = pbc.h_qubit_decl : pbc.h_qubit <0,9> !9;
        q9_v10 = pbc.h_qubit_decl : pbc.h_qubit <0,10> !10;
        q10_v11 = pbc.h_qubit_decl : pbc.h_qubit <0,11> !11;
        pivot_2_v12 = pbc.h_qubit_decl : pbc.h_qubit <1,0> !12;
        q11_v13 = pbc.h_qubit_decl : pbc.h_qubit <1,1> !13;
        b_v14 = pbc.bool_decl : pbc.bool  !14;
        q0_v15, q11_v16, b_v17 = pbc.pauli_measurement (q0_v1, q11_v13) {pbc_pauli_bases : [X, Z]} : pbc.h_qubit <0,1>, pbc.h_qubit <1,1>, pbc.bool  !15;
        pbc.return () : 
    }
}

outlined_attributes:
!0 = [builtin_given_names = builtin.given_names [pivot]]
!1 = [builtin_given_names = builtin.given_names [q0]]
!2 = [builtin_given_names = builtin.given_names [q1]]
!3 = [builtin_given_names = builtin.given_names [q2]]
!4 = [builtin_given_names = builtin.given_names [q3]]
!5 = [builtin_given_names = builtin.given_names [q4]]
!6 = [builtin_given_names = builtin.given_names [q5]]
!7 = [builtin_given_names = builtin.given_names [q6]]
!8 = [builtin_given_names = builtin.given_names [q7]]
!9 = [builtin_given_names = builtin.given_names [q8]]
!10 = [builtin_given_names = builtin.given_names [q9]]
!11 = [builtin_given_names = builtin.given_names [q10]]
!12 = [builtin_given_names = builtin.given_names [pivot_2]]
!13 = [builtin_given_names = builtin.given_names [q11]]
!14 = [builtin_given_names = builtin.given_names [b]]
!15 = [builtin_given_names = builtin.given_names [q0, q11, b]]
