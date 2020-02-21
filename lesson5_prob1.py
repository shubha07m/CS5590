import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

sns.set()


housedata = pd.read_csv('newhouse.csv')

X = housedata['GarageArea']
Y = housedata['SalePrice']

fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(X, Y)
ax.set_xlabel('GarageArea')
ax.set_ylabel('SalePrice')
plt.show()