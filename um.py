import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib import font_manager


plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rc('font', size=12)


beta = 8.2
gamma = 1
nu = 1/40

omega_1 = 0.4
omega_2 = 8
omega_3 = 0.2

alpha_1 = 1.25
alpha_2 = 0.6


def model(t, X, p, c, beta):
    S_N, I_N, R_N, S_B, I_B, R_B = X

    lambda_N = beta * (I_N + (1 - p) * I_B)
    lambda_B = (1 - c) * lambda_N
    omega = omega_1 * (S_B + I_B + R_B) + omega_2 * (I_N + I_B) + omega_3
    alpha = alpha_1 * (S_N + I_N + R_N) + alpha_2

    dS_N_dt = -lambda_N * S_N + nu * R_N + alpha * S_B - omega * S_N
    dI_N_dt = lambda_N * S_N - gamma * I_N + alpha * I_B - omega * I_N
    dR_N_dt = gamma * I_N - nu * R_N + alpha * R_B - omega * R_N
    dS_B_dt = -lambda_B * S_B + nu * R_B - alpha * S_B + omega * S_N
    dI_B_dt = lambda_B * S_B - gamma * I_B - alpha * I_B + omega * I_N
    dR_B_dt = gamma * I_B - nu * R_B - alpha * R_B + omega * R_N
    return [dS_N_dt, dI_N_dt, dR_N_dt, dS_B_dt, dI_B_dt, dR_B_dt]


S_N0 = 0.999
I_N0 = 0.001
R_N0 = 0
S_B0 = 0
I_B0 = 0
R_B0 = 0
X0 = [S_N0, I_N0, R_N0, S_B0, I_B0, R_B0]


t_span = (0, 60)
t_span1 = (0, 300)


beta = 8.2
p = 1
c = 1
solution = solve_ivp(model, t_span, X0, max_step=0.02, args=(p, c, beta))

beta = 8.2
p = 0
c = 0
solution1 = solve_ivp(model, t_span, X0, max_step=0.02, args=(p, c, beta))

beta = 1.5
p = 1
c = 1
solution2 = solve_ivp(model, t_span1, X0, max_step=0.1, args=(p, c, beta))

beta = 1.5
p = 0
c = 0
solution3 = solve_ivp(model, t_span1, X0, max_step=0.1, args=(p, c, beta))


S_N, I_N, R_N, S_B, I_B, R_B = solution.y
S_N1, I_N1, R_N1, S_B1, I_B1, R_B1 = solution1.y
S_N2, I_N2, R_N2, S_B2, I_B2, R_B2 = solution2.y
S_N3, I_N3, R_N3, S_B3, I_B3, R_B3 = solution3.y



fig, ax = fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))


ax[0].plot(S_N + S_B, I_N + I_B, label='$p = c = 1$', color='#2B32FF')
ax[0].plot(S_N1 + S_B1, I_N1 + I_B1, label='$p = c = 0$', color='#FF602B', 
           linestyle=':')
i=0
for S, I, color in [
    (S_N + S_B, I_N + I_B, '#2B32FF'),
    (S_N1 + S_B1, I_N1 + I_B1, '#FF602B')
]:
    n_points = len(S)
    idx = n_points // 60 + i
    i -= 5
    ax[0].annotate(
        '', 
        xy=(S[idx], I[idx]), 
        xytext=(S[idx - 1], I[idx - 1]), 
        arrowprops=dict(arrowstyle='simple', color=color, lw=1, mutation_scale=20)
    )

ax[0].set_yticks(np.arange(0, 0.7, 0.1))
ax[0].set_xlabel('Proportion of susceptibles ($S$)', size=14)
ax[0].set_ylabel('Proportion of infectious ($I$)', size=14)
ax[0].set_title('COVID-19-like illness', size=14)
ax[0].legend()
ax[0].grid(False)

ax[1].plot(S_N2 + S_B2, I_N2 + I_B2, label='$p = c = 1$', color='#2B32FF')
ax[1].plot(S_N3 + S_B3, I_N3 + I_B3, label='$p = c = 0$', color='#FF602B', 
           linestyle=':')
i=0
for S, I, color in [
    (S_N2 + S_B2, I_N2 + I_B2, '#2B32FF'),
    (S_N3 + S_B3, I_N3 + I_B3, '#FF602B')
]:
    n_points = len(S)
    idx = n_points // 24 + i
    i -= 50
    ax[1].annotate(
        '', 
        xy=(S[idx], I[idx]), 
        xytext=(S[idx - 1], I[idx - 1]), 
        arrowprops=dict(arrowstyle='simple', color=color, lw=1, mutation_scale=20)
    )

ax[1].set_yticks(np.arange(0, 0.07, 0.01))
ax[1].set_xlabel('Proportion of susceptible ($S$)', size=14)
ax[1].set_ylabel('Proportion of infectious ($I$)', size=14)
ax[1].set_title('Influenza-like illness', size=14)
ax[1].legend()
ax[1].grid(False)

plt.show()