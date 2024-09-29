
'''
9/28/24
9:04 PM

The following tool has been programmed to quickly convert the contents of a csv into a data matrix adhering
to the latex syntax
'''

import pandas as pd
import os
import filepaths as fp


def csv_to_latex_matrix(file_path, matrix_type='bmatrix'):
    df = pd.read_csv(file_path, header=None)
    rows = df.shape[0]

    latex_matrix = f"\\begin{{{matrix_type}}}\n"

    for i in range(rows):
        row_data = " & ".join(map(str, df.iloc[i, :,]))
        latex_matrix += f"{row_data} \\\\ \n"

    latex_matrix += f"\\end{{{matrix_type}}}\n"

    return latex_matrix


def save_latex_to_file(latex_code, directory, filename):
    file_path = os.path.join(directory, filename)

    with open(file_path, 'w') as f:
        f.write(latex_code)

    print(f"Latex matrix saved to {file_path}")


csv_file_path = '/Users/kbedoya88/Desktop/QC24-Fall-Semester/Computer-Science-Research/Equation-Discovery/Project/Setup/391_SINDy_Testing/Pendulum-Mechanics-Discovery/DataPath/Latex-Data/library_fitted_data_09-29-2024-15-09-1727637695.csv'
latex_code = csv_to_latex_matrix(csv_file_path)

destination = fp.latex_data_fp
filename = 'library-fitted-data-matrix.txt'
save_latex_to_file(latex_code, destination, filename)
