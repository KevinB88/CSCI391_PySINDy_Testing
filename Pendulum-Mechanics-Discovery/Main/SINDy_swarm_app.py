import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pysindy import SINDy
import pysindy as ps
from pysindy.optimizers import STLSQ
from sklearn.metrics import mean_squared_error
import SNR_computation as snr
import matplotlib.animation as animation


def animate_system(true_data, simulated_data, interval=200):
    """
    Animate the system evolution, comparing true and simulated data over time.
    Args:
        true_data: DataFrame of the true data.
        simulated_data: DataFrame of the simulated data.
        interval: Time interval between frames in milliseconds.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Extract initial data to initialize the scatter plot
    initial_X_true = true_data.iloc[0, 12:22].values
    initial_Y_true = true_data.iloc[0, 22:32].values
    initial_theta_true = true_data.iloc[0, 2:12].values

    initial_X_sim = simulated_data.iloc[0, 12:22].values
    initial_Y_sim = simulated_data.iloc[0, 22:32].values
    initial_theta_sim = simulated_data.iloc[0, 2:12].values

    # Initialize scatter plots with first frame's data
    scatter_true = ax[0].scatter(initial_X_true, initial_Y_true, c=initial_theta_true, cmap='viridis', alpha=0.75)
    scatter_sim = ax[1].scatter(initial_X_sim, initial_Y_sim, c=initial_theta_sim, cmap='viridis', alpha=0.75)

    ax[0].set_title("True Data")
    ax[0].set_xlabel('X Position')
    ax[0].set_ylabel('Y Position')
    ax[1].set_title("Simulated Data")
    ax[1].set_xlabel('X Position')
    ax[1].set_ylabel('Y Position')

    # Add colorbars (only once, outside the animation loop)
    colorbar_true = fig.colorbar(scatter_true, ax=ax[0])
    colorbar_true.set_label('Theta (Phase)')
    colorbar_sim = fig.colorbar(scatter_sim, ax=ax[1])
    colorbar_sim.set_label('Theta (Phase)')

    time_text = ax[ 0 ].text(0.02, 0.95, '', transform=ax[0].transAxes)

    initial_xlim, initial_ylim = ax[0].get_xlim(), ax[0].get_ylim()

    # Zoom factor for manual zooming
    zoom_factor = 1.1

    # Animation update function
    def update(frame):
        # Extract data for the current time step
        theta_true = true_data.iloc[frame, 2:12].values
        X_true = true_data.iloc[frame, 12:22].values
        Y_true = true_data.iloc[frame, 22:32].values

        theta_sim = simulated_data.iloc[frame, 2:12].values
        X_sim = simulated_data.iloc[frame, 12:22].values
        Y_sim = simulated_data.iloc[frame, 22:32].values

        # Update the scatter plots with new data
        scatter_true.set_offsets(np.c_[X_true, Y_true])
        scatter_true.set_array(theta_true)

        scatter_sim.set_offsets(np.c_[X_sim, Y_sim])
        scatter_sim.set_array(theta_sim)
        time_text.set_text(f'Time step: {frame}')

        return scatter_true, scatter_sim, time_text

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(true_data), interval=interval, blit=False)

    def on_key(event):
        if event.key == 'i':  # Zoom in
            ax[0].set_xlim([ x / zoom_factor for x in ax[0].get_xlim()])
            ax[0].set_ylim([ y / zoom_factor for y in ax[0].get_ylim()])
            ax[1].set_xlim([ x / zoom_factor for x in ax[1].get_xlim()])
            ax[1].set_ylim([ y / zoom_factor for y in ax[1].get_ylim()])
        elif event.key == 'o':  # Zoom out
            ax[0].set_xlim([ x * zoom_factor for x in ax[0].get_xlim()])
            ax[0].set_ylim([ y * zoom_factor for y in ax[0].get_ylim()])
            ax[1].set_xlim([ x * zoom_factor for x in ax[1].get_xlim()])
            ax[1].set_ylim([ y * zoom_factor for y in ax[1].get_ylim()])
        elif event.key == 'r':  # Reset zoom
            ax[0].set_xlim(initial_xlim)
            ax[0].set_ylim(initial_ylim)
            ax[1].set_xlim(initial_xlim)
            ax[1].set_ylim(initial_ylim)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()
def plot_comparison(true_data, simulated_data, time_step):
    """
    Plot scatter plots comparing true and simulated data at a given time step.
    Args:
        true_data: DataFrame of the true data.
        simulated_data: DataFrame of the simulated data.
        time_step: The time step (row) for which the data is plotted.
    """
    # Extract theta, X, Y values for the true data
    theta_true = true_data.iloc[time_step, 2:12].values  # True Theta
    X_true = true_data.iloc[time_step, 12:22].values  # True X
    Y_true = true_data.iloc[time_step, 22:32].values  # True Y

    # Extract theta, X, Y values for the simulated data
    theta_sim = simulated_data.iloc[time_step, 2:12].values  # Simulated Theta
    X_sim = simulated_data.iloc[time_step, 12:22].values  # Simulated X
    Y_sim = simulated_data.iloc[time_step, 22:32].values  # Simulated Y

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # True data scatter plot
    scatter_true = ax[0].scatter(X_true, Y_true, c=theta_true, cmap='viridis', alpha=0.75)
    ax[0].set_title(f"True Data at Time Step {time_step}")
    ax[0].set_xlabel('X Position')
    ax[0].set_ylabel('Y Position')
    colorbar_true = fig.colorbar(scatter_true, ax=ax[0])
    colorbar_true.set_label('Theta (Phase)')

    # Simulated data scatter plot
    scatter_sim = ax[1].scatter(X_sim, Y_sim, c=theta_sim, cmap='viridis', alpha=0.75)
    ax[1].set_title(f"Simulated Data at Time Step {time_step}")
    ax[1].set_xlabel('X Position')
    ax[1].set_ylabel('Y Position')
    colorbar_sim = fig.colorbar(scatter_sim, ax=ax[1])
    colorbar_sim.set_label('Theta (Phase)')

    plt.tight_layout()
    plt.show()

def compute_mse(true_data, simulated_data):
    """
    Compute the Mean Squared Error (MSE) between true and simulated data.
    Args:
        true_data: DataFrame of the true data.
        simulated_data: DataFrame of the simulated data.
    Returns:
        mse_theta, mse_X, mse_Y: MSE values for theta, X, and Y.
    """
    mse_theta = mean_squared_error(true_data.iloc[:, 2:12], simulated_data.iloc[:, 2:12])
    mse_X = mean_squared_error(true_data.iloc[:, 12:22], simulated_data.iloc[:, 12:22])
    mse_Y = mean_squared_error(true_data.iloc[:, 22:32], simulated_data.iloc[:, 22:32])

    print(f"Mean Squared Error (Theta): {mse_theta:.4f}")
    print(f"Mean Squared Error (X Positions): {mse_X:.4f}")
    print(f"Mean Squared Error (Y Positions): {mse_Y:.4f}")

    return mse_theta, mse_X, mse_Y


def SINDy_run_kogan(csv_file_path, threshold=0.05, noise=False, d_snr=None, output_csv_path='simulated_output.csv'):

    start_time = time.time()

    print("Loading data...")
    data = pd.read_csv(csv_file_path)
    time_column = data.iloc[:, 1].values  # Time is the second column
    theta = data.iloc[:, 2:12].values  # Theta columns (10)
    X = data.iloc[:, 12:22].values  # X positions (10)
    Y = data.iloc[:, 22:32].values  # Y positions (10)
    original_data = np.hstack([theta, X, Y])

    if noise:
        mean = 0
        st_dev = 0.1
        theta, alpha = snr.add_noise(d_snr, theta, mean, st_dev, 37, 'fro')
        print(f'Alpha value for theta matrix: {alpha}')
        X, alpha = snr.add_noise(d_snr, theta, mean, st_dev, 37, 'fro')
        print(f'Alpha value for X matrix: {alpha}')
        Y, alpha = snr.add_noise(d_snr, theta, mean, st_dev, 37, 'fro')
        print(f'Alpha value for Y matrix: {alpha}')

    data_load_time = time.time() - start_time
    print(f"Data loading complete. Time taken: {data_load_time:.2f} seconds")

    print("Training SINDy model...")
    model_start_time = time.time()

    # simulating with both linear and trig terms
    # Theta = ps.FourierLibrary(n_frequencies=1) + ps.PolynomialLibrary(degree=1)
    # running with linear terms only
    Theta = ps.PolynomialLibrary(degree=1)

    # simulating with trig terms only
    # Theta = ps.FourierLibrary(n_frequencies=1)

    model = ps.SINDy(optimizer=STLSQ(threshold=threshold))

    model.fit(original_data, t=time_column)

    print('Library of basis functions:')
    model.feature_library.get_feature_names()

    print('Governing equations:')
    model.print()

    model_training_time = time.time() - model_start_time
    print(f"Model training complete. Time taken: {model_training_time:.2f} seconds")

    print("Simulating data using learned model...")
    simulation_start_time = time.time()

    initial_conditions = original_data[0]  # Use the first row as the initial condition
    simulated_data = model.simulate(initial_conditions, time_column)

    simulation_time = time.time() - simulation_start_time
    print(f"Data simulation complete. Time taken: {simulation_time:.2f} seconds")

    print("Saving simulated data to CSV...")
    df_simulated = pd.DataFrame({
        'Index': range(len(time_column)),
        'Time': time_column
    })

    for i in range(10):
        df_simulated[f'Theta_{i}'] = simulated_data[:, i]
    for i in range(10, 21):
        df_simulated[f'X_{i}'] = simulated_data[:, i]
    for i in range(21, 30):
        df_simulated[f'Y_{i}'] = simulated_data[:, i]

    # Add the theta, X, and Y simulated values
    # for i in range(10):
    #     df_simulated[f'Theta_{i+1}'] = simulated_data[:, i]
    #     df_simulated[f'X_{i+1}'] = simulated_data[:, i+10]
    #     df_simulated[f'Y_{i+1}'] = simulated_data[:, i+20]

    # Save to CSV
    df_simulated.to_csv(output_csv_path, index=False)
    print(f"Simulated data saved to {output_csv_path}")

    # Step 5: Compare original and simulated data with scatter plots
    print("Generating scatter plots for comparison...")

    return model


if __name__ == "__main__":
    file_path = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/Main/swarm_data_J=0_433012_K=0_25_N=10.csv'
    # SINDy_run_kogan(file_path, threshold=0.085, d_snr=40, noise=False, output_csv_path='noisy_swarm_N=10_SNR=40.csv')

    simulated_output = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/Main/simulated_output.csv'
    true_output = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/Main/swarm_data_J=0_433012_K=0_25_N=10.csv'

    sim_df = pd.read_csv(simulated_output)
    tru_df = pd.read_csv(true_output)

    compute_mse(tru_df, sim_df)
    # plot_comparison(tru_df, sim_df, 50)
    # animate_system(tru_df, sim_df)
