import numpy as np

'''
parameters:

SNR : desired signal to noise ratio
clean : input data

mean, std_dev, and seed are all params for noise matrix
'''


def add_noise(SNR, clean_matrix, mean, std_dev, seed=None, norm_spec=None):

    # seed may be fixed to retain a uniform noise data matrix for experiments
    np.random.seed(seed)
    mean = mean
    std_dev = std_dev
    noise_matrix = np.random.normal(mean, std_dev, size=clean_matrix.shape)

    # calculating the power
    signal_power = np.linalg.norm(clean_matrix, norm_spec) ** 2
    noise_power = np.linalg.norm(noise_matrix, norm_spec) ** 2

    # computing alpha from the desired SNR
    alpha = np.sqrt(signal_power / (SNR * noise_power))
    output_matrix = clean_matrix + alpha * noise_matrix
    return output_matrix, alpha

