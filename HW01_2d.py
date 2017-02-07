import numpy as np
import matplotlib.pyplot as plt

B = [3, 4, 5, 6]

for b in B:
	m = 10^b;
	X = np.random.randn(100)
	
	plt.hist(X)
	plt.title("Histogram of X")

	## Show the plot
	plt.show()
