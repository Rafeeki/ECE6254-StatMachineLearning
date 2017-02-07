import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(100)
plt.hist(X)
plt.title("Histogram of X")

## Show the plot
plt.show()