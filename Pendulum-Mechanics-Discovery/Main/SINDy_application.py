import os

import pysindy as ps
import pandas as pd
import filepaths as fp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# extract trajectory data from the csv
data = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/DataPath/Pendulum_Data_09-28-2024-21-09-1727572905.csv'

df = pd.read_csv(data)
time = np.linspace(0, 10, 20)
# time = df['time'].values
X = df[['x0', 'x1']].values

# print(X)

Theta = ps.PolynomialLibrary(degree=2) + ps.FourierLibrary()

model = ps.SINDy(feature_library=Theta)

model.fit(X, t=time)
model.print()

X_dot = model.differentiate(X, t=time)
df_differentiated = pd.DataFrame(X_dot, columns=['x1_prime', 'x2_prime'])

current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")

if not os.path.exists(fp.data_fp):
    os.makedirs(fp.data_fp)
export_filepath = os.path.join(fp.data_fp, f'finite_difference_data_{current_time}.csv')

# df_differentiated.to_csv(export_filepath, index=False)

coefficients = model.coefficients()
feature_names = model.get_feature_names()

df_coefficients = pd.DataFrame(coefficients, columns=feature_names)
state_variable_labels = ['x0 prime ', 'x2 prime']
df_coefficients.insert(0, 'State Variable', state_variable_labels)

current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")

if not os.path.exists(fp.data_fp):
    os.makedirs(fp.data_fp)
export_filepath = os.path.join(fp.data_fp, f'coefficients_data_{current_time}.csv')

# df_coefficients.to_csv(export_filepath, index=False)

X_sim = model.simulate(X[0], t=time)
print("Simulated trajectory: \n", X_sim)

plt.plot(time, X, label='Original data')
plt.plot(time, X_sim, '--', label='Simulated data')
plt.xlabel('time')
plt.ylabel('angle (theta)')
plt.legend()
plt.show()


