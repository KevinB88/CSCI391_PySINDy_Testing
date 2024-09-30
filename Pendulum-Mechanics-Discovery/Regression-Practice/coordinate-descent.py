import numpy as np
from sklearn.linear_model import Lasso

dot_X = np.array([[0.5], [1.0], [1.5]])

Theta_X = np.array([[1, 1, 1],
                    [1, 2, 4],
                    [1, 3, 9]])

# regularization 'hyper' parameter
lambda_param = 0.1

# Performing sparse regression using Lasso
''' 
    The alpha parameter corresponds to the sparsity of the solution. 
'''
model = Lasso(alpha=lambda_param, fit_intercept=False)
model.fit(Theta_X, dot_X)
Xi = model.coef_
print("Sparse Coefficients (Xi):")
print(Xi)

'''
LASSO: a type of linear regression with L1-norm regularization 
These tools can be accessed from the sklearn module

Points/Concepts to clarify: what exactly is LASSO,
provide the details of a lienar regression and why it was used in this case.


'''