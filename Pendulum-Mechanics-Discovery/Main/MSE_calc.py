import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import filepaths as fp
from datetime import datetime
import os


# Computation of the mean-squared error
def MSE(x_true, x_sim, label):
    mse_calc = mean_squared_error(x_true, x_sim)
    print(f"Mean squared error ({label}) : {mse_calc}")
    return mse_calc


def plot_results(time, x_true, x_sim, mse_val, save_png, experiment):

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, x_true, label='True position', color='blue')
    plt.plot(time, x_sim, label='Simulated position', color='red')
    plt.fill_between(time, x_true, x_sim, color='green', alpha=0.5, label=f'Error = ({mse_val:.4f})')
    plt.title('Position: True vs. Simulated')
    plt.xlabel('Time: t')
    plt.ylabel('Angular Displacement : theta(t)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=3)

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")
    if not os.path.exists(fp.gr_mse_fp):
        os.makedirs(fp.gr_mse_fp)
    png_path = os.path.join(fp.gr_mse_fp, f'expr={experiment}_MSE_graph_{current_time}_.png')
    if save_png:
        plt.savefig(png_path, bbox_inches='tight')


def plot_absolute_error(time, x_true, x_sim, save_png, experiment):
    plt.figure(figsize=(10, 4))
    absolute_error = np.abs(x_true - x_sim)
    plt.plot(time, absolute_error, label='Absolute error', color='blue')
    plt.title('Absolute Error Over Time')
    plt.xlabel('Time')
    plt.ylabel('Error')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")
    if not os.path.exists(fp.gr_mse_fp):
        os.makedirs(fp.gr_mse_fp)
    png_path = os.path.join(fp.gr_mse_fp, f'expr={experiment}_abs_error_{current_time}_.png')
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