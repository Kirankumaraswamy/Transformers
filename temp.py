import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


data = np.arange(0.1, 20, 1)
plt.plot(data)
plt.xticks(np.arange(0, 20, 2))
plt.show()