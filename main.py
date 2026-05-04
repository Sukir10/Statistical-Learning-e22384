import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log, pi, e

mu_T, sigma_T = 0, 1
mu_P, sigma_P = 0, 2

print("Probabilities:")
print("P(X ≤ 0) =", norm.cdf(0))
print("P(X > 1) =", 1 - norm.cdf(1))

h_X = 0.5 * log(2 * pi * e)
h_P = 0.5 * log(2 * pi * e * 4)
print("\nEntropy:")
print("h(X) =", h_X)
print("h(Y) =", h_X + h_P)

t = np.linspace(-4, 4, 400)
p = np.linspace(-8, 8, 400)
T, P = np.meshgrid(t, p)

joint = norm.pdf(T, mu_T, sigma_T) * norm.pdf(P, mu_P, sigma_P)

plt.contourf(T, P, joint, levels=25)
plt.title("Joint Density")
plt.show()
