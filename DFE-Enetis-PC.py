import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rc('font', size=12)

beta = 1.5
gamma = 1

omega_1 = 0.4
omega_3 = 0.2

alpha_1 = 1.25
alpha_2 = 0.6

N_star = (- (omega_1 + omega_3 - alpha_1 + alpha_2) + ((omega_1 + omega_3 - alpha_1 + alpha_2) ** 2 + 4 * (alpha_1 - omega_1) * (alpha_2)) ** (1/2)) / (2 * (alpha_1 - omega_1))
B_star = 1 - N_star
alpha_star = alpha_1 * N_star + alpha_2
omega_star = omega_1 * B_star + omega_3

def R_0(q_star, p, c):
    return ((1 - q_star) * beta * N_star * (alpha_star + gamma + (1 - p) * omega_star)) / ((gamma + alpha_star) * (gamma + omega_star) - alpha_star * omega_star) + (beta * (N_star * q_star + (1 - c) * B_star) * (alpha_star + (1 - p) * (omega_star + gamma))) / ((gamma + alpha_star) * (gamma + omega_star) - alpha_star * omega_star)



fig, ax = fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 5))

q_1s = np.arange(0, 1, 0.01)

R_0s = []
R_0s1 = []
for q_1 in q_1s * B_star:
    p = c = 1
    R = R_0(q_1, p, c)
    R_0s.append(R)
    p = c = 0
    R = R_0(q_1, p, c)
    R_0s1.append(R)

R_0s = np.array(R_0s)
x_interp = np.interp(1, R_0s[::-1], q_1s[::-1])
ys_interp = np.arange(0.5, 1.05, 0.1)
xs_interp = np.array([x_interp for y in ys_interp])
xs = q_1s
ys = R_0s
ys1 = R_0s1
trig = np.ones(len(xs))
ax[0, 0].plot(xs, trig, color='#000000', linestyle='--')
ax[0, 0].plot(xs, ys, color='#0062E2')
ax[0, 0].set_title(r'Impact of $q_{1}$ for an Influenza-like illness', fontsize=15)
ax[0, 0].set_xlabel(r'$q_{1}$', fontsize=15)
ax[0, 0].set_ylabel(r'Basic Reproduction Number ($\mathscr{R}_{0}$)', fontsize=15)
ax[0, 0].set_yticks(np.arange(0.5, 1.6, 0.1))

q_3s = np.arange(0, 1, 0.01)

R_0s = []
R_0s1 = []
for q_3 in q_3s:
    p = c = 1
    R = R_0(q_3, p, c)
    R_0s.append(R)
    p = c = 0
    R = R_0(q_3, p, c)
    R_0s1.append(R)

R_0s = np.array(R_0s)
x_interp = np.interp(1, R_0s[::-1], q_1s[::-1])
ys_interp = np.arange(0.5, 1.05, 0.1)
xs_interp = np.array([x_interp for y in ys_interp])
xs = q_3s
ys = R_0s
ys1 = R_0s1
trig = np.ones(len(xs))
ax[0, 1].plot(xs, trig, color='#000000', linestyle='--')
ax[0, 1].plot(xs_interp, ys_interp, color='#FF0000', linestyle=':')
ax[0, 1].plot(xs, ys, color='#0062E2')
ax[0, 1].set_title(r'Impact of $q_{3}$ for an Influenza-like illness', fontsize=15)
ax[0, 1].set_xlabel(r'$q_{3}$', fontsize=15)
ax[0, 1].set_ylabel(r'Basic Reproduction Number ($\mathscr{R}_{0}$)', fontsize=15)
ax[0, 1].set_yticks(np.arange(0.5, 1.6, 0.1))

beta = 8.2
q_1s = np.arange(0, 1, 0.01)

R_0s = []
R_0s1 = []
for q_1 in q_1s * B_star:
    p = c = 1
    R = R_0(q_1, p, c)
    R_0s.append(R)
    p = c = 0
    R = R_0(q_1, p, c)
    R_0s1.append(R)

R_0s = np.array(R_0s)
xs = q_1s
ys = R_0s
ys1 = R_0s1
trig = np.ones(len(xs))
ax[1, 0].plot(xs, trig, color='#000000', linestyle='--')
ax[1, 0].plot(xs, ys, color='#0062E2')
ax[1, 0].set_title(r'Impact of $q_{1}$ for an COVID-19-like illness', fontsize=15)
ax[1, 0].set_xlabel(r'$q_{1}$', fontsize=15)
ax[1, 0].set_ylabel(r'Basic Reproduction Number ($\mathscr{R}_{0}$)', fontsize=15)

q_3s = np.arange(0, 1, 0.01)

R_0s = []
R_0s1 = []
for q_3 in q_3s:
    p = c = 1
    R = R_0(q_3, p, c)
    R_0s.append(R)
    p = c = 0
    R = R_0(q_3, p, c)
    R_0s1.append(R)

R_0s = np.array(R_0s)
xs = q_3s
ys = R_0s
ys1 = R_0s1
trig = np.ones(len(xs))
ax[1, 1].plot(xs, trig, color='#000000', linestyle='--')
ax[1, 1].plot(xs, ys, color='#0062E2')
ax[1, 1].set_title(r'Impact of $q_{3}$ for an COVID-19-like illness', fontsize=15)
ax[1, 1].set_xlabel(r'$q_{3}$', fontsize=15)
ax[1, 1].set_ylabel(r'Basic Reproduction Number ($\mathscr{R}_{0}$)', fontsize=15)


plt.show()
