import numpy as np
import matplotlib.pyplot as plt

# Seed
# Q1
np.random.seed(1001)
# Initialize w*, first dimension is bias
weight_vector = np.zeros(11)
weight_vector[1:] = np.random.rand(10)


# Generation of training set, the first dimension is bias (threshold)
def gen_training_inputs(n):
    training_set_1 = np.random.choice([-1, 1], (n, 11))
    training_set_1[:, 0] = 1
    return training_set_1


def gen_set_with_label(sample, weight):
    labels = np.sign(np.dot(sample, weight))
    train_set = np.column_stack((labels, sample))
    return train_set


def Perceptron_algorithm(x, y):
    weights = np.zeros(11)
    weights[1:] = np.random.rand(10)
    iterations_train = 0
    while True:
        errors = 0
        for xi, label in zip(x, y):
            if (np.dot(xi, weights) * label) <= 0:
                weights += xi * label
                errors += 1
        iterations_train += 1
        if errors == 0:
            break
    return iterations_train, weights


iterations_num_list = []
upper_bound_list = []
for _ in range(1000):
    weight_vector = np.zeros(11)
    weight_vector[1:] = np.random.rand(10)
    training_input = gen_training_inputs(100)
    # Initialization
    training_set = gen_set_with_label(training_input, weight_vector)
    iterations, weight_vector = Perceptron_algorithm(training_set[:, 1:], training_set[:, 0])
    iterations_num_list.append(iterations)
    # Training and recording iteration required.

    rho = min(training_set[:, 0] * np.dot(training_set[:, 1:], weight_vector))
    r_value = max(np.linalg.norm(training_set[:, 1:], axis=1))
    upper_bound_list.append(r_value ** 2 * np.linalg.norm(weight_vector) / rho ** 2)
    # calculating the upper bound.

# Calculate the average number of iterations
average_iterations = np.mean(iterations_num_list)
print(average_iterations)

plt.hist(iterations_num_list, bins=30, color='blue', edgecolor='black')

plt.xlabel('Number of iters')
plt.ylabel('Frequency')
plt.title('Histogram for Number of iterations')
plt.show()
diff = zip(iterations_num_list, upper_bound_list)
diff_list = []
for i, j in diff:
    diff_list.append(np.log10(np.abs(i - j)))
plt.hist(diff_list, color='red', bins=30)

plt.xlabel('log10(Difference)')
plt.ylabel('Frequency')
plt.title('Histogram for log10 of difference of Number of iterations')
plt.show()


# Q3

def probability_no_heads(n, mu, m):
    # For 1 coin
    p_no_heads_single = (1 - mu) ** n
    # Probability that at least one coin out of M shows no heads
    p_at_least_one_no_heads = 1 - (1 - p_no_heads_single) ** m
    return p_at_least_one_no_heads


N = 10
mu_values = [0.05, 0.8]
M_values = [1, 10 ** 4, 10 ** 6]

# Compute probabilities
results = {}
for mu in mu_values:
    for M in M_values:
        p = probability_no_heads(N, mu, M)
        results[(mu, M)] = p
print(results)

# Q4
# As given:
N = 6
M = 2
mu = 0.5

epsilons = np.linspace(0, 1, 200)

from scipy.stats import binom


def binom_tail(N, mu, epsilon):
    # Probability of getting more than (mu + epsilon)N heads or less than (mu - epsilon)N heads
    k_min = np.ceil((mu + epsilon) * N)
    k_max = np.floor((mu - epsilon) * N)
    tail_prob = binom.cdf(k_max, N, mu) + (1 - binom.cdf(k_min - 1, N, mu))
    return tail_prob


# Calculate the exact probability and Hoeffding-Union upper bound
exact_probs = [1 - (1 - binom_tail(N, mu, eps)) ** M for eps in epsilons]
hoeffding_bounds = [2 * np.exp(-2 * N * eps ** 2) for eps in epsilons]

# Plot the exact probability and the Hoeffding bound
plt.plot(epsilons, exact_probs, label='Exact Probability')
plt.plot(epsilons, hoeffding_bounds, label='Hoeffding Bound')
plt.title('Exact Probability and Hoeffding Bound With Union upper bound')
plt.xlabel('Epsilon')
plt.ylabel('Probability')
plt.legend()
plt.show()
