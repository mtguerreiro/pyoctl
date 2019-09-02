import numpy as np
import matplotlib.pyplot as plt
import pyoctl.opt as optctl

# --- Input ---
A = np.array([[0.9974, 0.0539], [-0.1078, 1.1591]])
B = np.array([[0.0013], [0.0539]])
Q = np.array([[0.25, 0], [0, 0.05]])
R = np.array([[0.05]])
H = np.array([[0, 0], [0, 0]])
N = 200
xi = np.array([2, 1])

# --- Algorithm ---
# Gains
F = optctl.dynprg_gains(A, B, Q, R, H, N)

# Simulation
x, u = optctl.dynprg_sim(A, B, Q, R, H, F, xi, N)

# --- Plots ---
plt.ion()

plt.figure()
plt.plot(F)
plt.title('Gains')

plt.figure()
plt.plot(x)
plt.plot(u)
plt.title('States and control')
