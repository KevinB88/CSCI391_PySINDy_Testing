import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import filepaths as fp
from datetime import datetime
import os

# default data-path

# temp path for the time: 10/9 12:49 PM
# path = fp.default_mse_gr_fp
path = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/DataPath/default'

# Computation of the mean-squared error
def MSE(x_true, x_sim, print_label=False, label=None):
    mse_calc = mean_squared_error(x_true, x_sim)
    if print_label:
        print(f"Mean squared error ({label}) : {mse_calc}")
    return mse_calc


def plot_results(time, x_true, x_sim, mse_val, save_png=False, show_plot=True, experiment=None, data_path=path, alpha=None, SNR=None, units='kogan'):

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, x_true, label='True position', color='blue')
    plt.plot(time, x_sim, label='Simulated position', color='red')
    plt.fill_between(time, x_true, x_sim, color='green', alpha=0.5, label=f'Error = ({mse_val:.4f})')

    if alpha is not None:
        if alpha < 1e-4:
            plt.title(f'Position: True vs. Simulated, alpha-val={np.log10(alpha):.4f}')
        else:
            plt.title(f'Position: True vs. Simulated, alpha-val={alpha:.4f}')
    else:
        plt.title(f'Position: True vs. Simulated')

    plt.xlabel('Time: t')
    plt.ylabel('Angular Displacement : theta(t)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=3)

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if experiment is None:
        if SNR is None:
            png_path = os.path.join(data_path, f'MSE_graph_{current_time}_.png')
        else:
            png_path = os.path.join(data_path, f'MSE_graph_{current_time}_SNR={SNR}db_.png')
    else:
        if SNR is None:
            png_path = os.path.join(data_path, f'expr={experiment}_MSE_graph_{current_time}_.png')
        else:
            png_path = os.path.join(data_path, f'expr={experiment}_MSE_graph_{current_time}_SNR={SNR}db_.png')
    if save_png:
        plt.savefig(png_path, bbox_inches='tight')
    if show_plot:
        plt.show()


def plot_absolute_error(time, x_true, x_sim, save_png, experiment, SNR=None, data_path=path):
    plt.figure(figsize=(10, 4))
    absolute_error = np.abs(x_true - x_sim)
    plt.plot(time, absolute_error, label='Absolute error', color='blue')
    plt.title('Absolute Error Over Time')
    plt.xlabel('Time')
    plt.ylabel('Error')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if SNR is None:
        png_path = os.path.join(data_path, f'expr={experiment}_abs_error_{current_time}_.png')
    else:
        png_path = os.path.join(data_path, f'expr={experiment}_abs_error_{current_time}_SNR={SNR}db_.png')
    if save_png:
        plt.savefig(png_path, bbox_inches='tight')

    plt.show()


'''
current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")
if not os.path.exists(fp.noise_matrix_fp):
    os.makedirs(fp.noise_matrix_fp)
png_path = os.path.join(fp.noise_matrix_fp, f'expr={experiment}_noise_compare_graph.png')
if file_output:
    plt.savefig(png_path, bbox_inches='tight')
'''