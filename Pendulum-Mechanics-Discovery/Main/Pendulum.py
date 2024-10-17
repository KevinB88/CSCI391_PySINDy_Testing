import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import filepaths as fp
from datetime import datetime
import os

if __name__ == "__main__":
    k = 1000
    # simulating until t=10, for k time-steps
    t = np.linspace(0, 50, k)
    # length of the rod
    length = 3

    # gravitational acceleration constant
    g = 9.80665

    # angular frequency
    w = np.sqrt(g / length)

    # max angular displacement
    theta_max = 0.99

    # angle as a function of time
    theta_t = theta_max * np.cos(w * t)

    # angular velocity as a function of time
    theta_t_prime = -theta_max * w * np.sin(w * t)

    # plt.title('Angular versus time')
    # plt.plot(t, theta_t, color='red')
    # plt.xlabel('t')
    # plt.ylabel('Angle (Theta)')
    # plt.show()
    #
    # plt.title('Angular velocity versus time')
    # plt.plot(t, theta_t_prime)
    # plt.xlabel('t')
    # plt.ylabel('Angle (Theta)')
    # plt.show()

    # storing the data to a data-frame, for angular displacement and velocity
    df = pd.DataFrame({
        'x0': theta_t,
        'x1': theta_t_prime
    })
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")

    filepath = fp.data_fp

    if not os.path.exists(fp.data_fp):
        os.makedirs(fp.data_fp)

    export_filepath = os.path.join(filepath, f'Pendulum_Data_{current_time}.csv')

    df.to_csv(export_filepath, index=False)





