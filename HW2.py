from matplotlib import pyplot as plt
import numpy as np
num_data = 500
num_x = 500
sample = np.random.uniform(-1, 1, 2)


def optimized_func(x, x_1, x_2):
    a = x_1+x_2
    b = -x_1*x_2
    return a*x+b


g_values = []
bias = []
total_bias = []
total_variance = []
mean_g = []
for i in range(num_data):
    sample = np.random.uniform(-1, 1, 2)
    for j in range(num_x):
        x = np.random.uniform(-1, 1, 1)
        func = optimized_func(x, sample[0], sample[1])
        g_values.append(func)
        bias.append((func-x**2)**2)
    total_variance.append(np.var(g_values))
    total_bias.append(sum(bias))
    mean_g.append(np.mean(g_values))
print(total_variance)
print(total_bias)
print(mean_g)