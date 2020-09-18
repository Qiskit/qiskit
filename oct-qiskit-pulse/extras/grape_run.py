def main(H_d, H_c, U_0, U_targ):

    # Number of time slots
    n_ts = 256
    # Time allowed for the evolution
    evo_time = 2560

    # Fidelity error target
    fid_err_targ = 1e-10
    # Maximum iterations for the optisation algorithm
    max_iter = 200
    # Maximum (elapsed) time allowed in seconds
    max_wall_time = 120
    # Minimum gradient (sum of gradients squared)
    # as this tends to 0 -> local minima has been found
    min_grad = 1e-20# Fidelity error target

    # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
    p_type = 'RND'

    result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                out_file_ext=f_ext, init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True)