import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rc('font', size=12)

gamma = 1
nu = 1/40

omega_1 = 0.4
omega_2 = 8
omega_3 = 0.2

alpha_1 = 1.25
alpha_2 = 0.6


def model(t, X, q_1, q_2, q_3, p, c, beta, r_1, r_2):
    S_N, I_N, R_N, S_B, I_B, R_B = X

    lambda_N = beta * (I_N + (1 - p) * I_B)
    lambda_B = (1 - c) * lambda_N
    omega = omega_1 * (S_B + I_B + R_B) + omega_2 * (I_N + I_B) + omega_3
    alpha = alpha_1 * (S_N + I_N + R_N) + alpha_2
    q = q_1 * (S_B + I_B + R_B) + q_2 * (I_N + I_B) + q_3
    r = r_1 * (S_N + I_N + R_N) + r_2

    dS_N = -lambda_N * S_N + nu * R_N + alpha * S_B - omega * S_N
    dI_N = (1 - q) * lambda_N * S_N - gamma * I_N + alpha * I_B - omega * I_N
    dR_N = gamma * I_N - nu * R_N + alpha * R_B - omega * R_N + r * gamma * I_B
    dS_B = -lambda_B * S_B + nu * R_B - alpha * S_B + omega * S_N
    dI_B = lambda_B * S_B - gamma * I_B - alpha * I_B + omega * I_N + q * lambda_N * S_N
    dR_B = (1 - r) * gamma * I_B - nu * R_B - alpha * R_B + omega * R_N
    
    return [dS_N, dI_N, dR_N, dS_B, dI_B, dR_B]


S_N0 = 0.999
I_N0 = 0.001
R_N0 = 0
S_B0 = 0
I_B0 = 0
R_B0 = 0
X0 = [S_N0, I_N0, R_N0, S_B0, I_B0, R_B0]


t_span = (0, 30)
t_span1 = (0, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes0 = axes[0, 0]
axes1 = axes[0, 1]
axes2 = axes[1, 0]
axes3 = axes[1, 1]

solution = solve_ivp(model, t_span, X0, max_step=0.01, args = [0, 0, 0, 1, 1, 8.2, 0, 0])

solution1 = solve_ivp(model, t_span, X0, max_step=0.01, args = [1, 0, 0, 1, 1, 8.2, 0, 0])

solution2 = solve_ivp(model, t_span, X0, max_step=0.01, args = [0, 1, 0, 1, 1, 8.2, 0, 0])

solution3 = solve_ivp(model, t_span, X0, max_step=0.01, args = [0, 0, 1, 1, 1, 8.2, 0, 0])


solution4 = solve_ivp(model, t_span1, X0, max_step=0.1, args = [0, 0, 0, 1, 1, 1.5, 0, 0])

solution5 = solve_ivp(model, t_span1, X0, max_step=0.1, args = [1, 0, 0, 1, 1, 1.5, 0, 0])

solution6 = solve_ivp(model, t_span1, X0, max_step=0.1, args = [0, 1000, 0, 1, 1, 1.5, 0, 0])

solution7 = solve_ivp(model, t_span1, X0, max_step=0.1, args = [0, 0, 1, 1, 1, 1.5, 0, 0])

S_N, I_N, R_N, S_B, I_B, R_B = solution.y

S_N1, I_N1, R_N1, S_B1, I_B1, R_B1 = solution1.y

S_N2, I_N2, R_N2, S_B2, I_B2, R_B2 = solution2.y

S_N3, I_N3, R_N3, S_B3, I_B3, R_B3 = solution3.y

S_N4, I_N4, R_N4, S_B4, I_B4, R_B4 = solution4.y

S_N5, I_N5, R_N5, S_B5, I_B5, R_B5 = solution5.y

S_N6, I_N6, R_N6, S_B6, I_B6, R_B6 = solution6.y

S_N7, I_N7, R_N7, S_B7, I_B7, R_B7 = solution7.y

t = solution.t
axes0.plot(t, I_N1 + I_B1, label='$Q = (1, 0, 0)$', color='#15D300')
axes0.plot(t, I_N2 + I_B2, label='$Q = (0, 1, 0)$', color='#FF6600')
axes0.plot(t, I_N3 + I_B3, label='$Q = (0, 0, 1)$', color='#001DDA')
axes0.plot(t, I_N + I_B, label='$Q = (0, 0, 0)$', color='#000000', linestyle=':')
axes0.set_yticks(np.arange(0, 0.6, 0.1))
axes0.set_xticks(np.arange(0, 31, 5))
axes0.set_xlabel('$t$', size=15)
axes0.set_ylabel('Proportion of infected ($I$)', size=15)
axes0.set_title('COVID-19-like illness', size=15)
axes0.legend()

t1 = solution4.t
axes1.plot(t1, I_N5 + I_B5, label='$Q = (1, 0, 0)$', color='#15D300')
axes1.plot(t1, I_N6 + I_B6, label='$Q = (0, 1, 0)$', color='#FF6600')
axes1.plot(t1, I_N7 + I_B7, label='$Q = (0, 0, 1)$', color='#001DDA')
axes1.plot(t1, I_N4 + I_B4, label='$Q = (0, 0, 0)$', color='#000000', linestyle=':')
axes1.set_yticks(np.arange(0, 0.025, 0.005))
axes1.set_xticks(np.arange(0, 201, 50))
axes1.set_xlabel('$t$', size=15)
axes1.set_ylabel('Proportion of infected ($I$)', size=15)
axes1.set_title('Influenza-like illness', size=15)
axes1.legend()

axes2.plot(S_N1 + S_B1, I_N1 + I_B1, label='$Q = (1, 0, 0)$', color='#15D300')
axes2.plot(S_N2 + S_B2, I_N2 + I_B2, label='$Q = (0, 1, 0)$', color='#FF6600')
axes2.plot(S_N3 + S_B3, I_N3 + I_B3, label='$Q = (0, 0, 1)$', color='#001DDA')
axes2.plot(S_N + S_B, I_N + I_B, label='$Q = (0, 0, 0)$', color='#000000', linestyle=':')
axes2.set_yticks(np.arange(0, 0.6, 0.1))
axes2.set_xticks(np.arange(0, 1.1, 0.2))
axes2.set_xlabel('Proportion of susceptibles ($S$)', size=15)
axes2.set_ylabel('Proportion of infected ($I$)', size=15)
axes2.set_title('COVID-19-like illness', size=15)
axes2.legend()

axes3.plot(S_N5 + S_B5, I_N5 + I_B5, label='$Q = (1, 0, 0)$', color='#15D300')
axes3.plot(S_N6 + S_B6, I_N6 + I_B6, label='$Q = (0, 1, 0)$', color='#FF6600')
axes3.plot(S_N7 + S_B7, I_N7 + I_B7, label='$Q = (0, 0, 1)$', color='#001DDA')
axes3.plot(S_N4 + S_B4, I_N4 + I_B4, label='$Q = (0, 0, 0)$', color='#000000', linestyle=':')
axes3.set_yticks(np.arange(0, 0.025, 0.005))
axes3.set_xticks(np.arange(0.5, 1.01, 0.1))
axes3.set_xlabel('Proportion of susceptibles ($S$)', size=15)
axes3.set_ylabel('Proportion of infected ($I$)', size=15)
axes3.set_title('Influenza-like illness', size=15)
axes3.legend()



plt.show()
