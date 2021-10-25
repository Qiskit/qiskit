import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# num_time_steps = [1, 2, 5, 10, 20]
# depths = [1, 2, 3]
num_time_steps = [1]
depths = [1]

# entanglements = ['full', 'linear', 'sca']
# entanglements = ['linear', 'sca']
e = 'linear'
# regularizations = ['none', 'perturb_diag', 'ridge']
reg = 'perturb_diag'

fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'e_bound',
              'e_grad']
fieldnames.extend(['fid_to_targ', 'true_error', 'true_energy', 'trained_energy'])


################## VarQRTE ###################


color_count = 0

snapshot_dir_real = os.path.join('..', 'output', 'real', e, reg)
with open(os.path.join(snapshot_dir_real, 'varqte_output.csv'), mode='r') as csv_file:
    reader = csv.DictReader(csv_file, fieldnames=fieldnames)
    counter = 0
    t = []
    num_params = []
    num_time_steps = []
    trained_error = []
    grad_error = []
    fid_to_target = []
    true_error = []
    true_energy = []
    trained_energy = []
    for line in reader:
        if counter != 0:
            t.append(np.around(float(line['t']), 3))
            num_params.append(np.around(float(line['num_params']), 3))
            num_time_steps.append(np.around(float(line['num_time_steps']), 3))
            trained_error.append(np.around(complex(line['e_bound']), 3))
            grad_error.append(np.around(complex(line['e_grad']), 3))
            fid_to_target.append(np.around(complex(line['fid_to_targ']), 3))
            true_error.append(np.around(float(line['true_error']), 3))
            true_energy.append(np.around(complex(line['true_energy']), 3))
            trained_energy.append(np.around(complex(line['trained_energy']), 3))
            # norm[i].append(np.around(float(line['1-norm']), 3))
            # loss[i].append(np.around(float(line['loss'].strip("()j").replace("-","")), 3))
        counter += 1


def get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


# colors = get_cmap((len(entanglements) * len(set(num_params))* len(regularizations)))
colors = get_cmap((len(set(num_time_steps))))
for i, n_p in enumerate(list(set(num_params))):
    color_count = 0
    # color = [(i+k)/(len(set(num_params)) + len(entanglements)), i/len(set(num_params)),
    #          k/len(entanglements)]
    time_steps_plot = []
    num_time_steps_plot = []
    trained_error_plot = []
    fid_to_target_plot = []
    true_error_plot = []
    grad_error_plot = []
    true_energy_plot = []
    trained_energy_plot = []
    if n_p == 8:
        snapshot_dir = '/Users/ouf/Documents/GitHub/qiskit-aqua/qiskit/working_files/varQTE' \
                       '/output/' \
                       'real/sca/perturb_diag'

        numpy_error = np.load(os.path.join(snapshot_dir, 'state_error.npy'), allow_pickle=True)
        numpy_energy = np.load(os.path.join(snapshot_dir, 'energy.npy'), allow_pickle=True)
        numpy_grad_error = np.load(os.path.join(snapshot_dir, 'grad_error.npy'), allow_pickle=True)
    for j in range(len(num_params)):
        if num_params[j] == n_p:
            time_steps_plot.append(t[j])
            num_time_steps_plot.append(num_time_steps[j])
            trained_error_plot.append(trained_error[j])
            fid_to_target_plot.append(fid_to_target[j])
            true_error_plot.append(true_error[j])
            grad_error_plot.append(grad_error[j])
            true_energy_plot.append(true_energy[j])
            trained_energy_plot.append(trained_energy[j])

            if t[j] == 1:
                if n_p == 8:
                    numpy_error_plot = numpy_error.item()[str(int(num_time_steps[j]))]
                    numpy_grad_error_plot = numpy_grad_error.item()[str(int(num_time_steps[j]))]
                    numpy_energy_plot = numpy_energy.item()[str(int(num_time_steps[j]))]
                size = len(set(num_time_steps)) * 20 - color_count * 20
                color = colors(color_count)
                color_count += 1
                """
                # For the same number of parameters print trained_error, true_error & fidelity
                """
                plt.figure(1)
                plt.title('Error ' + reg)
                # plt.plot(num_time_steps_plot, trained_error_plot, color=color, marker='o',
                #          linestyle='dashed', linewidth=2, markersize=6, label=str(n_p) + e)
                # plt.plot(num_time_steps_plot, true_error_plot, color=color, marker='x',
                # linestyle='dashed',
                #          linewidth=2, markersize=12)
                plt.scatter(time_steps_plot, trained_error_plot, color=color, marker='x', s=size,
                         label=r' $\#$ time steps ' + str(num_time_steps[j]))
                plt.scatter(time_steps_plot, true_error_plot, color=color, marker='*', s=size)
                if n_p == 8:
                    plt.scatter(time_steps_plot, numpy_error_plot, color=color, marker='D', s=size)
                plt.legend(loc='best')
                plt.xlabel('time steps')
                # plt.xticks(range(counter-1))
                # if reg == 'none':
                #     plt.ylim((0., 1.8))
                plt.figtext(0.01, 0.01, 'Cross: Bound, Star: Actual, Diamond: Numpy')
                plt.ylabel('error')


                plt.figure(2)

                plt.title('Fidelity '+ reg)
                # plt.plot(num_time_steps_plot, fid_to_target_plot, color=color, marker='o',
                #          linestyle='dashed', linewidth=2, markersize=6, label=str(n_p)+e)
                plt.scatter(time_steps_plot, fid_to_target_plot, color=color, marker='x', s=size,
                         label=r' $\#$ time steps ' + str(num_time_steps[j]))
                plt.legend(loc='best')
                plt.xlabel('time steps')
                # plt.xticks(range(counter-1))
                plt.ylim((0., 1.0))
                plt.ylabel('fidelity')

                plt.figure(3)

                plt.title('Energy '+ reg)
                # plt.plot(num_time_steps_plot, trained_energy_plot, color=color, marker='o',
                #          linestyle='dashed', linewidth=2, markersize=6, label=str(n_p) + e)
                # plt.plot(num_time_steps_plot, true_energy_plot, color=color, marker='x', linestyle='dashed',
                #          linewidth=2, markersize=12)
                plt.scatter(time_steps_plot, trained_energy_plot, color=color, marker='x',
                            s=size, label=r' $\#$ time steps ' + str(num_time_steps[j]))
                plt.scatter(time_steps_plot, true_energy_plot, color=color, marker='*',
                            s=size)
                if n_p == 8:
                    plt.scatter(time_steps_plot, numpy_energy_plot, color=color, marker='D', s=size)
                plt.legend(loc='best')
                plt.xlabel('time steps')
                # plt.xticks(range(counter-1))
                # plt.ylim((0.82, 1.01))
                plt.figtext(0.01, 0.01, 'Cross: Approx, Star: Analytic')
                plt.ylabel('error')

                plt.figure(4)

                plt.title('Gradient Error ' + reg)
                # plt.plot(num_time_steps_plot, fid_to_target_plot, color=color, marker='o',
                #          linestyle='dashed', linewidth=2, markersize=6, label=str(n_p)+e)
                plt.scatter(time_steps_plot, grad_error_plot, color=color, marker='x', s=size,
                            label=r' $\#$ time steps ' + str(num_time_steps[j]))
                if n_p == 8:
                    plt.scatter(time_steps_plot, numpy_grad_error_plot, color=color, marker='D', s=size)
                plt.legend(loc='best')
                plt.xlabel('time steps')
                # plt.xticks(range(counter-1))
                # plt.ylim((0., 1.0))
                plt.ylabel('gradient error')

                time_steps_plot = []
                num_time_steps_plot = []
                trained_error_plot = []
                fid_to_target_plot = []
                true_error_plot = []
                grad_error_plot = []
                true_energy_plot = []
                trained_energy_plot = []

    # plt.show()
    plt.savefig(os.path.join('..', 'output', 'real', e, reg, str(n_p) + 'grad_error.png'))
    plt.close()
    plt.figure(3)
    plt.savefig(os.path.join('..', 'output', 'real', e, reg, str(n_p) + 'energy.png'))
    plt.close()
    plt.figure(2)
    # plt.show()
    plt.savefig(os.path.join('..', 'output', 'real', e, reg, str(n_p) + 'fidelity.png'))
    plt.close()
    plt.figure(1)
    # plt.show()
    plt.savefig(os.path.join('..', 'output', 'real', e, reg, str(n_p) + 'error.png'))
    plt.close()

#
# ################## VarQITE ###################
#
# for m, reg in enumerate(regularizations):
#     color_count = 0
#     for k, e in enumerate(entanglements):
#         if reg == None:
#             reg = 'None'
#         snapshot_dir_imag = os.path.join('..', 'output', 'imag', e, reg)
#         with open(os.path.join(snapshot_dir_imag, 'varqte_output.csv'), mode='r') as csv_file:
#             reader = csv.DictReader(csv_file, fieldnames=fieldnames)
#             counter = 0
#             t = []
#             num_params = []
#             num_time_steps = []
#             trained_error = []
#             fid_to_target = []
#             true_error = []
#             true_energy = []
#             trained_energy = []
#             for line in reader:
#                 if counter != 0:
#                     t.append(np.around(float(line['t']), 3))
#                     num_params.append(np.around(float(line['num_params']), 3))
#                     num_time_steps.append(np.around(float(line['num_time_steps']), 3))
#                     trained_error.append(np.around(complex(line['e']), 3))
#                     fid_to_target.append(np.around(float(line['fid_to_targ']), 3))
#                     true_error.append(np.around(float(line['true_error']), 3))
#                     true_energy.append(np.around(complex(line['true_energy']), 3))
#                     trained_energy.append(np.around(complex(line['trained_energy']), 3))
#                     # norm[i].append(np.around(float(line['1-norm']), 3))
#                     # loss[i].append(np.around(float(line['loss'].strip("()j").replace("-","")), 3))
#                 counter += 1
#
#
#         def get_cmap(n, name='jet'):
#             '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
#             RGB color; the keyword argument name must be a standard mpl colormap name.'''
#             return plt.cm.get_cmap(name, n)
#
#
#         colors = get_cmap((len(entanglements) * len(set(num_params))))
#
#         for i, n_p in enumerate(set(num_params)):
#             # color = [(i+k)/(len(set(num_params)) + len(entanglements)), i/len(set(num_params)),
#             #          k/len(entanglements)]
#             color = colors(color_count)
#             color_count += 1
#             num_time_steps_plot = []
#             trained_error_plot = []
#             fid_to_target_plot = []
#             true_error_plot = []
#             true_energy_plot = []
#             trained_energy_plot = []
#             for j in range(len(num_params)):
#                 if num_params[j] == n_p and t[j] == 1:
#                     num_time_steps_plot.append(num_time_steps[j])
#                     trained_error_plot.append(trained_error[j])
#                     fid_to_target_plot.append(fid_to_target[j])
#                     true_error_plot.append(true_error[j])
#                     true_energy_plot.append(true_energy[j])
#                     trained_energy_plot.append(trained_energy[j])
#
#             """
#             # For the same number of parameters print trained_error, true_error & fidelity
#             """
#             plt.figure(1)
#             if i == 0:
#                 plt.title('Error ' + reg)
#             # plt.plot(num_time_steps_plot, trained_error_plot, color=color, marker='o',
#             #          linestyle='dashed', linewidth=2, markersize=6, label=str(n_p) + e)
#             # plt.plot(num_time_steps_plot, true_error_plot, color=color, marker='x',
#             # linestyle='dashed',
#             #          linewidth=2, markersize=12)
#             plt.scatter(num_time_steps_plot, trained_error_plot, color=color, marker='o', s=40,
#                         label=str(n_p) + ' ' + e)
#             plt.scatter(num_time_steps_plot, true_error_plot, color=color, marker='x', s=40)
#             plt.legend(loc='best')
#             plt.xlabel('number time steps')
#             # plt.xticks(range(counter-1))
#             # plt.ylim((0.82, 1.01))
#             plt.figtext(0.01, 0.01, 'Dots: Trained, Cross: Target')
#             plt.ylabel('error')
#
#             plt.figure(2)
#             if i == 0:
#                 plt.title('Fidelity ' + reg)
#             # plt.plot(num_time_steps_plot, fid_to_target_plot, color=color, marker='o',
#             #          linestyle='dashed', linewidth=2, markersize=6, label=str(n_p)+e)
#             plt.scatter(num_time_steps_plot, fid_to_target_plot, color=color, marker='o', s=40,
#                         label=str(n_p) + ' ' + e)
#             plt.legend(loc='best')
#             plt.xlabel('number time steps')
#             # plt.xticks(range(counter-1))
#             # plt.ylim((0.82, 1.01))
#             plt.ylabel('error')
#
#             plt.figure(3)
#             if i == 0:
#                 plt.title('Energy ' + reg)
#             # plt.plot(num_time_steps_plot, trained_energy_plot, color=color, marker='o',
#             #          linestyle='dashed', linewidth=2, markersize=6, label=str(n_p) + e)
#             # plt.plot(num_time_steps_plot, true_energy_plot, color=color, marker='x',
#             # linestyle='dashed',
#             #          linewidth=2, markersize=12)
#             plt.scatter(num_time_steps_plot, trained_energy_plot, color=color, marker='o',
#                         s=40, label=str(n_p) + ' ' + e)
#             plt.scatter(num_time_steps_plot, true_energy_plot, color=color, marker='x',
#                         s=40)
#             plt.legend(loc='best')
#             plt.xlabel('number time steps')
#             # plt.xticks(range(counter-1))
#             # plt.ylim((0.82, 1.01))
#             plt.figtext(0.01, 0.01, 'Dots: Trained, Cross: Target')
#             plt.ylabel('error')
#
#     plt.savefig(os.path.join('..', 'output', 'imag', reg +'energy.png'))
#     plt.close()
#     plt.figure(2)
#     plt.savefig(os.path.join('..', 'output', 'imag', reg +'fidelity.png'))
#     plt.close()
#     plt.figure(1)
#     plt.savefig(os.path.join('..', 'output', 'imag', reg + 'error.png'))
#     plt.close()




        # color1 = 'teal'
        # color2 = 'mediumvioletred'
        # color3 = 'pink'
        # fig, ax1 = plt.subplots()
        # # ax1.title('Restricted QBM')
        # plt.title('Error vs. Fidelity with ' + str(n_p) + ' Parameters')
        # ax1.plot(num_time_steps_plot, trained_error_plot, color=color1,
        #             label='trained Error')
        # ax1.plot(num_time_steps_plot, true_error_plot, color=color2,
        #             label='true Error')
        # # plt.legend(loc='best')
        # ax1.set_xlabel('number time steps')
        # # plt.xticks(range(counter-1))
        # # plt.ylim((0.82, 1.01))
        # ax1.set_ylabel('error', color=color1)
        # ax1.tick_params(axis='y', labelcolor=color1)
        # # plt.savefig(os.path.join(snapshot_dir, 'loss_plot.png'))
        #
        # ax2 = ax1.twinx()
        # # ax2.title('Restricted QBM - Distance')
        # ax2.plot(num_time_steps_plot, fid_to_target_plot, color=color3, label='fidelity')
        #
        # # ax2.set_xlabel('iteration')
        # plt.xticks(num_time_steps_plot)
        # # plt.ylim((0.82, 1.01))
        # ax2.set_ylabel('Fidelity to target', color=color2)
        # ax2.tick_params(axis='y', labelcolor=color2)
        # fig.tight_layout()
        # fig.legend(loc='center')
        # plt.show()
        # # plt.savefig(os.path.join(snapshot_dir_real, 'energy_fidelity_plot.png'))
        #
        #
        #
        # """
        # # For the same number of parameters print trained_error, true_error & true_energy,
        # trained_energy
        # """
        #
        # color1 = 'teal'
        # color2 = 'mediumvioletred'
        # color3 = 'pink'
        # color4 = 'red'
        # fig, ax1 = plt.subplots()
        # # ax1.title('Restricted QBM')
        # plt.title('Error vs. Energy with ' + str(n_p) + ' Parameters')
        # ax1.plot(num_time_steps_plot, trained_error_plot, color=color1,
        #             label='trained Error')
        # ax1.plot(num_time_steps_plot, true_error_plot, color=color2,
        #             label='true Error')
        # # plt.legend(loc='best')
        # ax1.set_xlabel('number time steps')
        # # plt.xticks(range(counter-1))
        # # plt.ylim((0.82, 1.01))
        # ax1.set_ylabel('error', color=color1)
        # ax1.tick_params(axis='y', labelcolor=color1)
        # # plt.savefig(os.path.join(snapshot_dir, 'loss_plot.png'))
        #
        # ax2 = ax1.twinx()
        # # ax2.title('Restricted QBM - Distance')
        # ax2.plot(num_time_steps_plot, trained_energy_plot, color=color3,
        #             label='trained Energy')
        # ax2.plot(num_time_steps_plot, true_energy_plot, color=color4,
        #             label='true Energy')
        #
        # # ax2.set_xlabel('iteration')
        # # plt.xticks(range(counter-1))
        # # plt.ylim((0.82, 1.01))
        # ax2.set_ylabel('energy', color=color2)
        # ax2.tick_params(axis='y', labelcolor=color2)
        # fig.legend(loc='center')
        # plt.xticks(num_time_steps_plot)
        # fig.tight_layout()
        # plt.show()
