# LASSO regression with coordinate-descent minimization

import numpy as np


def soft_thresholding(z, lambda_param):
    return np.sign(z) * max(np.abs(z) - lambda_param, 0)


def compute_residual(X, y, beta, j):
    y_predict = X.dot(beta)
    residual = y - y_predict + X[:, j] * beta[j]
    return residual


def lasso_coordinate_decent(X, y, lambda_param, num_iterations=1000, tolerance=1e-6):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)

    for iteration in range(num_iterations):
        beta_old = beta.copy()
        for j in range(n_features):
            residual = compute_residual(X, y, beta, j)
            X_j = X[:, j]
            rho_j = X_j.dot(residual)
            z_j = np.sum(X_j ** 2)

            if z_j != 0:
                beta[j] = soft_thresholding(rho_j / z_j, lambda_param / z_j)

        if np.max(np.abs(beta - beta_old)) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
    return beta


if __name__ == "__main__":
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])

    y = np.array([1, 2, 3, 4])
    lambda_p = 0.01

    beta = lasso_coordinate_decent(X, y, lambda_p)
    print(f'Coefficients: {beta}')

