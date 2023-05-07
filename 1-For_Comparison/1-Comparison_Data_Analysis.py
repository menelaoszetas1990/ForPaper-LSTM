import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Comparison.csv')

df.info()
print(df.describe())
print(df.head())

plt.plot(df.hist)
plt.show()
print('end')
