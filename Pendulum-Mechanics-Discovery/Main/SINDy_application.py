import contextlib
import os
import io
import sys
import pysindy as ps
import pandas as pd
import filepaths as fp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import MSE_calc as mse_calc
import SNR_computation as snr
import time as t
from contextlib import redirect_stdout
from datetime import datetime


library_functions = [
    lambda x: 1,
    lambda x: x,
    lambda x: np.sin(x)
]

library_function_names = [
    lambda x: '1',
    lambda x: x,
    lambda x: "sin(" + x + ")"
]

custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)

def decibel_linear_conversion(x):
    return 10 ** (x / 10)


# extract trajectory data from the csv
# Leveraging the data that is theta_0 strictly less than 1

'''
    clarification on parameter usages:
    
    X_all : all of the original clean synthetic data 
    X_full : an additional assignment made in order to denote either the entire clean or noisy data
    X_part : a partition of X_full relative to a given number of time-steps 

'''


# extracting the time values from a csv
def SINDy_run_kogan():
    experiment_count = 2
    file_output = True
    noise = True
    # desired SNR values to attempts
    db_amount = 20
    train_time = 4000
    destination = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/DataPath/10-16-Experiment-Kogan-Data'
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")

    meta_data_filepath = os.path.join(destination, f'exp={experiment_count}_meta_data_{current_time}.txt')
    os.makedirs(destination, exist_ok=True)

    # print(f'Train time: {train_time} time steps.')
    sim_time = 10**4

    file = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/Main/simple_pendulum_energy=0.1.csv'

    df = pd.read_csv(file)
    # extracting all values across the first column from the csv file
    table_time = df.iloc[:, 0].to_numpy()
    # extracting all the original data-values, which are completely clean at this point
    # assuming that data-frame originally has two-columns
    table_data = df.iloc[:, [1, 2]]
    X_all = table_data[['x0', 'x1']].values
    full_time = len(df.iloc[:, 0])
    # print(f'Full time: {full_time} time steps.')
    # print(f'Full time: {full_time} time-steps.', file=file)
    # print(f'Data percentage: {(train_time /  full_time) * 100} %')

    if train_time < 0 or train_time > full_time:
        print(f'Insufficient train time, must 1 <= T <= {full_time}')
        return
    if sim_time < 0 or sim_time > full_time:
        print(f'Insufficient simulation time, must be 1 <= T <= {full_time}')
        return

    mean = 0
    std_dev = 0.10

    if noise:
        SNR = decibel_linear_conversion(db_amount)
    else:
        SNR = None
        db_amount = None

    if noise:
        X_full, alpha = snr.add_noise(SNR, X_all, mean, std_dev, 37, 'fro')
    else:
        alpha = None
        X_full = X_all

    # partitioning the time relative to the training amount
    X_part = X_full[:train_time]
    time_part = table_time[:train_time]

    # print("Alpha: ", alpha)
    # amplification factor

    # Theta = ps.PolynomialLibrary(degree=1) + ps.FourierLibrary(n_frequencies=1)
    # Theta = ps.PolynomialLibrary(degree=2)
    Theta = custom_library
    # Theta = ps.FourierLibrary(n_frequencies=2)

    model = ps.SINDy(feature_library=Theta)

    # print(model.differentiation_method)

    model.fit(X_part, t=time_part)

    with open(meta_data_filepath, 'w') as file:
        print(f'Train time: {train_time} time steps.\n', file=file)
        print(f'Full time: {full_time} time-steps.\n', file=file)
        print(f'Data percentage: {(train_time / full_time) * 100} %\n', file=file)
        print(f"Alpha: {alpha}\n", file=file)
        print(f"Optimizer used: {model.optimizer}\n", file=file)
        print(f"Differentiation model: {model.differentiation_method}\n", file=file)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            model.print()
        equations = output.getvalue()
        print(f'{equations}', file=file)
        print(f"Library of functions: {Theta.get_feature_names()}", file=file)

    time_ext = table_time[:sim_time]

    X_sim = model.simulate(X_part[0], t=time_ext)

    # ensure the sizes of both matrices in the computation for the MSE are equivalent
    X_og_portion = X_full[:len(X_sim[:, 0])]

    mse_result = mse_calc.MSE(X_og_portion[:, 0], X_sim[:, 0], 'angular displacement')
    mse_calc.plot_results(time_ext, X_og_portion[:, 0], X_sim[:, 0], mse_result, file_output,
                          experiment=experiment_count, alpha=alpha, SNR=db_amount, data_path=destination)
    mse_calc.plot_absolute_error(time_ext, X_og_portion[:, 0], X_sim[:, 0], file_output, experiment_count,
                                 SNR=db_amount, data_path=destination)


def SINDy_run_syn():
    data = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/DataPath/Pendulum_Data_10-14-2024-23-10-1728964751.csv'
    experiment = 6
    db_amount = 0
    file_output = False
    noise = False
    # the sample amount given to the model
    # the partition of time provided to the model for training
    train_part = 500

    # simulation time for the experiment, must always be <= # time steps from OG data
    sim_time = 1000

    '''
    Taking in the following time-partitions:
    
    20% 40% 60% 80 %
    100, 200, 300, 400
    '''

    df = pd.read_csv(data)
    time = np.linspace(0, train_part / 10, train_part)

    # time = df['time'].values
    X_all = df[['x0', 'x1']].values
    of_ts_count = len(X_all[:, 0])
    print("Number of time-steps in original data ", of_ts_count)
    if train_part < 0 or train_part > of_ts_count:
        print("Training sample is out of bounds!")
        return
    if sim_time < 0 or sim_time > of_ts_count:
        print("Simulation time is out of bounds!")
        return

    # possible desired SNR values to leverage
    # 20 ** 3, 16 ** 3, 12 ** 3, 8 ** 3, 4 ** 3

    mean = 0
    std_dev = 0.10
    SNR = decibel_linear_conversion(db_amount)

    alpha = 0
    if alpha == 0:
        alpha = None

    print(X_all)
    if noise:
        X_full, alpha = snr.add_noise(SNR, X_all, mean, std_dev, 37, 'fro')
    else:
        X_full = X_all

    # partitioning the time relative to the training amount
    X_part = X_full[:train_part]

    print("Alpha: ", alpha)
    # amplification factor

    Theta = ps.PolynomialLibrary(degree=1) + ps.FourierLibrary()
    # Theta = ps.FourierLibrary(n_frequencies=2)

    model = ps.SINDy(feature_library=Theta)

    print("Optimizer used:", model.optimizer)

    print(model.differentiation_method)

    model.fit(X_part, t=time)
    model.print()

    extended_time = np.linspace(0, sim_time/10, sim_time)

    X_sim = model.simulate(X_part[0], t=extended_time)

    # ensure the sizes of both matrices in the computation for the MSE are equivalent
    X_og_portion = X_full[:len(X_sim[:, 0])]

    mse_result = mse_calc.MSE(X_og_portion[:, 0], X_sim[:, 0], 'angular displacement')
    mse_calc.plot_results(extended_time, X_og_portion[:, 0], X_sim[:, 0], mse_result, file_output, experiment=experiment, alpha=alpha, SNR=db_amount)
    mse_calc.plot_absolute_error(extended_time, X_og_portion[:, 0], X_sim[:, 0], file_output, experiment, SNR=db_amount)


if __name__ == "__main__":
    SINDy_run_kogan()
    # SINDy_run_syn()

# df_noise_matrix = pd.DataFrame(noise_matrix, columns=['x0', 'x1'])
# if not os.path.exists(fp.noise_matrix_fp):
#     os.makedirs(fp.data_fp)
# export_filepath = os.path.join(fp.noise_matrix_fp, f'expr={experiment}_noise_matrix_{current_time}.csv')
# if file_output:
#     df_noise_matrix.to_csv(export_filepath, index=False)

#
# print(f'The noise matrix: {noise_matrix}')

# print(X)


# X_dot = model.differentiate(X, t=time)
# df_differentiated = pd.DataFrame(X_dot, columns=['x1_prime', 'x2_prime'])
#
# current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")
#
# if not os.path.exists(fp.data_fp):
#     os.makedirs(fp.data_fp)
# export_filepath = os.path.join(fp.data_fp, f'finite_difference_data_{current_time}.csv')
#
# # df_differentiated.to_csv(export_filepath, index=False)
#
# coefficients = model.coefficients()
# feature_names = model.get_feature_names()
#
# df_coefficients = pd.DataFrame(coefficients, columns=feature_names)
# state_variable_labels = ['x0 prime ', 'x2 prime']
# df_coefficients.insert(0, 'State Variable', state_variable_labels)
#
# current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")
#
# if not os.path.exists(fp.data_fp):
#     os.makedirs(fp.data_fp)
# export_filepath = os.path.join(fp.data_fp, f'coefficients_data_{current_time}.csv')
#
# # df_coefficients.to_csv(export_filepath, index=False)
#