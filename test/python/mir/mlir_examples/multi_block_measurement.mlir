builtin.module @pbc_program 
{
  ^block1v1():
    pbc.circuit {circuit_name : "main", program_width : <24: si64>}
    {
      ^entry_block2v1():
        q0_v0 = pbc.h_qubit_decl : pbc.h_qubit <0,0> !0;
        q1_v1 = pbc.h_qubit_decl : pbc.h_qubit <0,1> !1;
        q2_v2 = pbc.h_qubit_decl : pbc.h_qubit <0,2> !2;
        q3_v3 = pbc.h_qubit_decl : pbc.h_qubit <0,3> !3;
        q4_v4 = pbc.h_qubit_decl : pbc.h_qubit <0,4> !4;
        q5_v5 = pbc.h_qubit_decl : pbc.h_qubit <0,5> !5;
        q6_v6 = pbc.h_qubit_decl : pbc.h_qubit <0,6> !6;
        q7_v7 = pbc.h_qubit_decl : pbc.h_qubit <0,7> !7;
        q8_v8 = pbc.h_qubit_decl : pbc.h_qubit <0,8> !8;
        q9_v9 = pbc.h_qubit_decl : pbc.h_qubit <0,9> !9;
        q10_v10 = pbc.h_qubit_decl : pbc.h_qubit <0,10> !10;
        q11_v11 = pbc.h_qubit_decl : pbc.h_qubit <0,11> !11;
        q12_v12 = pbc.h_qubit_decl : pbc.h_qubit <1,0> !12;
        q13_v13 = pbc.h_qubit_decl : pbc.h_qubit <1,1> !13;
        q14_v14 = pbc.h_qubit_decl : pbc.h_qubit <1,2> !14;
        q15_v15 = pbc.h_qubit_decl : pbc.h_qubit <1,3> !15;
        q16_v16 = pbc.h_qubit_decl : pbc.h_qubit <1,4> !16;
        q17_v17 = pbc.h_qubit_decl : pbc.h_qubit <1,5> !17;
        q18_v18 = pbc.h_qubit_decl : pbc.h_qubit <1,6> !18;
        q19_v19 = pbc.h_qubit_decl : pbc.h_qubit <1,7> !19;
        q20_v20 = pbc.h_qubit_decl : pbc.h_qubit <1,8> !20;
        q21_v21 = pbc.h_qubit_decl : pbc.h_qubit <1,9> !21;
        q22_v22 = pbc.h_qubit_decl : pbc.h_qubit <1,10> !22;
        q23_v23 = pbc.h_qubit_decl : pbc.h_qubit <1,11> !23;
        result_v24 = pbc.bool_decl : pbc.bool  !24;
        q0_v25, q1_v26, q2_v27, q3_v28, q4_v29, q5_v30, q6_v31, q7_v32, q8_v33, q9_v34, q10_v35, q11_v36, q12_v37, q13_v38, q14_v39, q15_v40, q16_v41, q17_v42, q18_v43, q19_v44, q20_v45, q21_v46, q22_v47, q23_v48, result_v49 = pbc.pauli_measurement (q0_v0, q1_v1, q2_v2, q3_v3, q4_v4, q5_v5, q6_v6, q7_v7, q8_v8, q9_v9, q10_v10, q11_v11, q12_v12, q13_v13, q14_v14, q15_v15, q16_v16, q17_v17, q18_v18, q19_v19, q20_v20, q21_v21, q22_v22, q23_v23) {pbc_pauli_bases : [I, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, I, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y]} : pbc.h_qubit <0,0>, pbc.h_qubit <0,1>, pbc.h_qubit <0,2>, pbc.h_qubit <0,3>, pbc.h_qubit <0,4>, pbc.h_qubit <0,5>, pbc.h_qubit <0,6>, pbc.h_qubit <0,7>, pbc.h_qubit <0,8>, pbc.h_qubit <0,9>, pbc.h_qubit <0,10>, pbc.h_qubit <0,11>, pbc.h_qubit <1,0>, pbc.h_qubit <1,1>, pbc.h_qubit <1,2>, pbc.h_qubit <1,3>, pbc.h_qubit <1,4>, pbc.h_qubit <1,5>, pbc.h_qubit <1,6>, pbc.h_qubit <1,7>, pbc.h_qubit <1,8>, pbc.h_qubit <1,9>, pbc.h_qubit <1,10>, pbc.h_qubit <1,11>, pbc.bool  !25;
        pbc.return () : 
    }
}

outlined_attributes:
!0 = [builtin_given_names = builtin.given_names [q0]]
!1 = [builtin_given_names = builtin.given_names [q1]]
!2 = [builtin_given_names = builtin.given_names [q2]]
!3 = [builtin_given_names = builtin.given_names [q3]]
!4 = [builtin_given_names = builtin.given_names [q4]]
!5 = [builtin_given_names = builtin.given_names [q5]]
!6 = [builtin_given_names = builtin.given_names [q6]]
!7 = [builtin_given_names = builtin.given_names [q7]]
!8 = [builtin_given_names = builtin.given_names [q8]]
!9 = [builtin_given_names = builtin.given_names [q9]]
!10 = [builtin_given_names = builtin.given_names [q10]]
!11 = [builtin_given_names = builtin.given_names [q11]]
!12 = [builtin_given_names = builtin.given_names [q12]]
!13 = [builtin_given_names = builtin.given_names [q13]]
!14 = [builtin_given_names = builtin.given_names [q14]]
!15 = [builtin_given_names = builtin.given_names [q15]]
!16 = [builtin_given_names = builtin.given_names [q16]]
!17 = [builtin_given_names = builtin.given_names [q17]]
!18 = [builtin_given_names = builtin.given_names [q18]]
!19 = [builtin_given_names = builtin.given_names [q19]]
!20 = [builtin_given_names = builtin.given_names [q20]]
!21 = [builtin_given_names = builtin.given_names [q21]]
!22 = [builtin_given_names = builtin.given_names [q22]]
!23 = [builtin_given_names = builtin.given_names [q23]]
!24 = [builtin_given_names = builtin.given_names [result]]
!25 = [builtin_given_names = builtin.given_names [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23, result]]
