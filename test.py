import math

from sklearn.metrics import mean_squared_error, mean_absolute_error

x = [1, 3, 5, 7]
y = [4, 4, 4, 4]

z = mean_absolute_error(x, y)
print(z)
z = mean_squared_error(x, y)
print(z)
z = math.sqrt(mean_squared_error(x, y))
print(z)
