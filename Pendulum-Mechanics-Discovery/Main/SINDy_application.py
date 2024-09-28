import pysindy as ps
import pandas as pd
import filepaths as fp
import numpy as np
import matplotlib.pyplot as plt

# extract trajectory data from the csv
df = pd.read_csv(fp.data_set)

time = df['time'].values
X = df[['x0', 'x1']].values

print(X)

Theta = ps.PolynomialLibrary(degree=3) + ps.FourierLibrary()

model = ps.SINDy(feature_library=Theta)

model.fit(X, t=time)
model.print()

X_sim = model.simulate(X[0], t=time)
print("Simulated trajectory: \n", X_sim)

plt.plot(time, X, label='Original data')
plt.plot(time, X_sim, '--', label='Simulated data')
plt.legend()
plt.show()


