import pysindy as ps
import numpy as np
import pandas as pd
import SNR_computation as snr
import MSE_calc as mse_calc
import matplotlib.pyplot as plt
import time


# for concatenating the columns of csv files
def cat_csv_cols():
    simple_pendulum_theta = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/Kogan-Data/energy=2/theta.csv'
    simple_pendulum_omega = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/Kogan-Data/energy=2/omega.csv'

    df_theta = pd.read_csv(simple_pendulum_theta, header=None)
    df_omega = pd.read_csv(simple_pendulum_omega, header=None)

    omega_column = df_omega.iloc[:, 1]

    df_theta.insert(2, 'omega', omega_column)
    headers = ['t', 'x0', 'x1']
    df_theta.to_csv('simple_pendulum_energy=2.5.csv', index=False, header=headers)


def experiment(data):

    # for n-number of time-steps
    time_partition = 10

    df = pd.read_csv(data)

    # extracting the original clean
    X_clean = df[['x0', 'x1']].values

    # Desired SNRs to test: 40, 20, 10, 0
    desired_SNR = 10**3
    mean = 0
    std_dev = 0.15
    # calculating noise with Gaussian distribution and using a desired SNR
    # use keyword 'fro' (Frobenius) when working with MxN matrices
    X_noisy, alpha = snr.add_noise(desired_SNR, X_clean, mean, std_dev, 37, 'fro')

    print("Alpha", alpha)

    full_time = df[['t']].values

    X_part = X_noisy[:time_partition]
    # partitioned time
    part_time = full_time[:time_partition]

    print(f'X noisy shape: {X_noisy.shape}')
    print(f'Part time noise: {part_time.shape}')

    # Establishing the library of functions
    Theta = ps.PolynomialLibrary(degree=1) + ps.FourierLibrary()

    model = ps.SINDy(feature_library=Theta)

    print("Optimizer used: ", model.optimizer)
    print("Differentiation method: ", model.differentiation_method)

    model.fit(X_part, t=part_time.flatten())
    model.print()

    # params: X_noisy[initial state], t=the time used for the experiment

    X_sim = model.simulate(X_part[0], t=full_time.flatten())

    # calculating the mean-squared error
    # mse = mse_calc.MSE(X_noisy[:, 0], X_sim[:, 0])
    # # plotting the data
    # mse_calc.plot_results(full_time.flatten(), X_noisy[:, 0], X_sim[:, 0], mse, show_plot=True)

    plt.plot(full_time.flatten(), X_sim[:, 0], label='Sim')
    plt.show()


if __name__ == "__main__":
    simple_pendulum = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/Kogan-Data/energy=1/simple_pendulum_energy=1.csv'

    cat_csv_cols()

    # experiment(simple_pendulum)


