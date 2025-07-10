import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Configuração dos Gráficos (do seu código original) ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 24
})

# --- 1. Parâmetros do Sistema e Controladores (do seu código original) ---
R_a = 0.2
L_a = 0.004
k = 0.265
k_a = 0.8e-3
M = 5.75
g = 9.81
r = 0.0235
Y = 2.3
V_a_max = 120.0
T_load_static = M * g * r
I_a_max = 10.0  # Limite para o Item C

J_motor = 0.023
J_total = J_motor + M * r**2

Kpp = 85.1
Kpv = 4.94
Kiv = 0.151
Kpc = 1.0
Kic = 50.0

I_ff = T_load_static / k

# --- 2. Função de Dinâmica do Sistema (do seu código original) ---
def system_dynamics(t, state, omega_y, T_hold):
    y, omega_m, i_a, integ_v, integ_c = state
    if t < T_hold:
        y_ref = 0.0
    else:
        y_ref = Y * np.sin(omega_y * (t - T_hold))
    
    error_p = y_ref - y
    omega_ref = Kpp * error_p
    
    error_v = omega_ref - omega_m
    i_a_ref_unlimited = (Kpv * error_v) + (Kiv * integ_v) + I_ff
    i_a_ref = np.clip(i_a_ref_unlimited, -I_a_max, I_a_max)
    
    # Lógica Anti-windup para o integrador de velocidade
    d_integ_v_dt = error_v if i_a_ref == i_a_ref_unlimited else 0.0
    
    error_c = i_a_ref - i_a
    v_a_unlimited = (Kpc * error_c) + (Kic * integ_c)
    v_a = np.clip(v_a_unlimited, -V_a_max, V_a_max)

    # Lógica Anti-windup para o integrador de corrente
    d_integ_c_dt = error_c if v_a == v_a_unlimited else 0.0
    
    e_a = k * omega_m
    di_a_dt = (v_a - e_a - i_a * R_a) / L_a
    T_e = k * i_a
    T_net = T_e - T_load_static
    d_omega_dt = (T_net - k_a * omega_m) / J_total
    dy_dt = omega_m * r
    
    return [dy_dt, d_omega_dt, di_a_dt, d_integ_v_dt, d_integ_c_dt]

# --- 3. Execução da Simulação para o Caso de Teste ---
# Frequência para o caso de erro médio (~40%), como na sua Figura 2.
omega_y_teste = 0.77
T_hold_duration = 1.0

print(f"--- Simulando o sistema para ω_y = {omega_y_teste:.2f} rad/s ---")

period = 2 * np.pi / omega_y_teste
t_final = T_hold_duration + 5 * period
t_eval = np.linspace(0, t_final, int(1000 * t_final))
s0 = [0.0, 0.0, 0.0, 0.0, 0.0]

sol = solve_ivp(
    system_dynamics, 
    [0, t_final], 
    s0, 
    t_eval=t_eval, 
    args=(omega_y_teste, T_hold_duration), 
    method='RK45'
)

# --- 4. Pós-processamento para Obter a Referência de Velocidade ---
# A referência de velocidade (saída do controlador de posição) precisa ser recalculada
# usando os resultados da simulação para poder ser plotada.
y_sim = sol.y[0]
t_sim = sol.t

y_ref = np.where(t_sim >= T_hold_duration, Y * np.sin(omega_y_teste * (t_sim - T_hold_duration)), 0.0)
erro_pos = y_ref - y_sim
wm_ref = Kpp * erro_pos # Esta é a referência de velocidade gerada pela malha de posição

# A velocidade medida é o segundo estado da simulação
wm_medida = sol.y[1]

# --- 5. Geração do Gráfico de Justificativa ---
print("--- Gerando o gráfico de justificativa do sensor de velocidade ---")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(t_sim, wm_ref, 'k--', label='Velocidade de Referência ($\\omega_{m,ref}$)')
ax.plot(t_sim, wm_medida, 'b-', label='Velocidade Medida ($\\omega_m$)')

ax.set_title(f'Rastreamento da Malha de Velocidade ($\\omega_y$ = {omega_y_teste:.2f} rad/s)')
ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Velocidade Angular (rad/s)')
ax.legend()
ax.grid(True)
ax.set_xlim(0, T_hold_duration + 4.5 * period)

plt.tight_layout()
output_filename = 'item_I_grafico_justificativa_sensor_velocidade.pdf'
try:
    fig.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"--> Gráfico salvo com sucesso como: {output_filename}")
except Exception as e:
    print(f"Erro ao salvar o gráfico: {e}")

plt.show()