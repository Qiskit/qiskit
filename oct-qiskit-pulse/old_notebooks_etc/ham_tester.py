#%%

H_d, H_c = get_hamiltonians(backend, subsystem_list, ['wq0'])
n_ctrls = len(H_c)
U_0 = identity(3)
U_targ = get_hadamard()

