import numpy as np
import pandas as pd
import os
from datetime import datetime
import pysindy as ps


def generate_sindy_library(input_filepath, output_filepath, libraries=['polynomial', 'fourier'], degree=2, include_iteraction=True, frequencies=None, rounding_digits=3):
    data_read = pd.read_csv(input_filepath)
    X = data_read.values
    selected_libraries = []

    for library_type in libraries:
        if library_type == 'polynomial':
            poly_library = ps.PolynomialLibrary(degree=degree, include_interaction=include_iteraction)
            selected_libraries.append(poly_library)
        elif library_type == 'fourier':
            if frequencies is None:
                frequencies = [1, 2]
            fourier_library = ps.FourierLibrary(n_frequencies=frequencies)
            selected_libraries.append(fourier_library)
        elif library_type == 'identity':
            identity_library = ps.IdentityLibrary()
            selected_libraries.append(identity_library)
        else:
            raise ValueError(f'Unsupported library type: {library_type}')

    if len(selected_libraries) > 1:
        combined_library = ps.GeneralizedLibrary(selected_libraries)
    else:
        combined_library = selected_libraries[0]

    combined_library.fit(X)

    feature_names = combined_library.get_feature_names(input_features=data_read.columns)

    transformed_X = combined_library.transform(X)

    transformed_X = np.round(transformed_X, decimals=rounding_digits)

    date = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")

    transformed_df = pd.DataFrame(transformed_X, columns=feature_names)

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    output_desintation = os.path.join(output_filepath, f'library_fitted_data_{date}.csv')
    transformed_df.to_csv(output_desintation, index=False)


def generate_fixed_set(csv_filepath, output_file_destination):
    data = pd.read_csv(csv_filepath)

    x_0 = data['x0'].values
    x_1 = data['x1'].values

    library_matrix = np.column_stack((
        np.ones_like(x_0),
        x_0,
        x_1,
        x_0 ** 2,
        x_0 * x_1,
        x_1 ** 2,
        np.sin(x_0),
        np.cos(x_0),
        np.sin(x_1),
        np.cos(x_1)
    ))

    library_df = pd.DataFrame(library_matrix, columns=[
        '1.0', 'x_0', 'x_1', 'x_0^2', 'x_0x_1', 'x_1^2', 'sin(x_0)', ' cos(x_0)', 'sin(x_1)', 'cos(x_1)'
    ])

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%m-%s")

    if not os.path.exists(output_file_destination):
        os.makedirs(output_file_destination)

    export_filepath = os.path.join(output_file_destination, f'library_fitted_data_{current_time}.csv')

    library_df.to_csv(export_filepath, index=False)


destination = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/DataPath/Latex-Data'
filepath = "/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/DataPath/Latex-Data/Pendulum_Data_09-28-2024-21-09-1727572905.csv"


if __name__ == "__main__":
    generate_sindy_library(filepath, destination, frequencies=1)

