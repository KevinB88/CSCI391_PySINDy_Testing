import os
import sys
import pysindy as ps
import pandas as pd
import filepaths as fp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import MSE_calc as mse_calc

# extract trajectory data from the csv
data = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/DataPath/Pendulum_Data_10-03-2024-01-10-1727932172.csv'

experiment = "11"
file_output = True

df = pd.read_csv(data)
time = np.linspace(0, 10, 100)
# time = df['time'].values
X_all = df[['x0', 'x1']].values

sub_extract = df.iloc[:100]

X = sub_extract[['x0', 'x1']].values

# adding Gaussian noise to the container as a test to assess the algorithms ability at uncovering equations from within the noisy data
mean = 0
std_dev = 0.15
noise_matrix = np.random.normal(mean, std_dev, size=X.shape)
# amplification factor
alpha = 0.2
noise_matrix *= alpha

current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")

df_noise_matrix = pd.DataFrame(noise_matrix, columns=['x0', 'x1'])
if not os.path.exists(fp.noise_matrix_fp):
    os.makedirs(fp.data_fp)
export_filepath = os.path.join(fp.noise_matrix_fp, f'expr={experiment}_noise_matrix_{current_time}.csv')
if file_output:
    df_noise_matrix.to_csv(export_filepath, index=False)

#
# print(f'The noise matrix: {noise_matrix}')
X = X + noise_matrix

# print(X)

Theta = ps.PolynomialLibrary(degree=1) + ps.FourierLibrary()
# Theta = ps.FourierLibrary(n_frequencies=2)

model = ps.SINDy(feature_library=Theta)

print("Optimizer used:", model.optimizer)

print(model.differentiation_method)

model.fit(X, t=time)
model.print()

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

extended_time = np.linspace(0, 50, 500)

X_sim = model.simulate(X[0], t=extended_time)
# print("Simulated trajectory: \n", X_sim)
#
# experiment will be conducted for the angular displacement
mse_result = mse_calc.MSE(X_all[:, 0], X_sim[:, 0], 'angular displacement')
mse_calc.plot_results(extended_time, X_all[:, 0], X_sim[:, 0], mse_result, True, experiment)
mse_calc.plot_absolute_error(extended_time, X_all[:, 0], X_sim[:, 0], True, experiment)

'''
After simulating the data for a longer time, 
the discrepancy between curves begin to grow slowly.

Computing the error between the points. 

The next step would be to test how well the algorithm 
will hold with noise 
'''
