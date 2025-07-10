import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,          # Tamanho da fonte padrão para itens não especificados
    'axes.titlesize': 18,     # Tamanho do título dos eixos (ax.set_title)
    'axes.labelsize': 18,     # Tamanho dos rótulos dos eixos (xlabel, ylabel)
    'xtick.labelsize': 16,    # Tamanho dos números no eixo X
    'ytick.labelsize': 16,    # Tamanho dos números no eixo Y
    'legend.fontsize': 14,    # Tamanho da fonte da legenda
    'figure.titlesize': 24    # Tamanho do título principal da figura (fig.suptitle)
})

# --- 1. Parâmetros do Sistema e Controladores ---
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
I_a_max = 30.0 # Limite para o Item D

# Inércia total
J_motor = 0.023
J_total = J_motor + M * r**2

# Ganhos do controlador
Kpp = 85.1
Kpv = 4.94
Kiv = 0.151
Kpc = 1.0
Kic = 50.0

# Corrente de feedforward
I_ff = T_load_static / k

# --- 2. Função de Dinâmica do Sistema ---
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
    d_integ_v_dt = error_v if i_a_ref == i_a_ref_unlimited else 0.0
    error_c = i_a_ref - i_a
    v_a_unlimited = (Kpc * error_c) + (Kic * integ_c)
    v_a = np.clip(v_a_unlimited, -V_a_max, V_a_max)
    d_integ_c_dt = error_c if v_a == v_a_unlimited else 0.0
    e_a = k * omega_m
    di_a_dt = (v_a - e_a - i_a * R_a) / L_a
    T_e = k * i_a
    T_net = T_e - T_load_static
    d_omega_dt = (T_net - k_a * omega_m) / J_total
    dy_dt = omega_m * r
    return [dy_dt, d_omega_dt, di_a_dt, d_integ_v_dt, d_integ_c_dt]

# --- 3. Função de Simulação ---
def simulate_system_ode(omega_y, T_hold=1.0):
    period = 2 * np.pi / omega_y
    t_final = T_hold + 5 * period 
    t_eval = np.linspace(0, t_final, int(500 * (t_final)))
    s0 = [0.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(system_dynamics, [0, t_final], s0, t_eval=t_eval, args=(omega_y, T_hold), method='RK45')
    y_sim = sol.y[0]
    y_ref = np.where(sol.t >= T_hold, Y * np.sin(omega_y * (sol.t - T_hold)), 0.0)
    start_time_for_error = T_hold + 2.5 * period
    start_index = np.argmax(sol.t >= start_time_for_error)
    if start_index < len(y_ref) and start_index > 0:
        tracking_error = np.max(np.abs(y_ref[start_index:] - y_sim[start_index:]))
        error_percent = (tracking_error / Y) * 100 if Y > 0 else 0
    else:
        error_percent = 100.0
    return error_percent, sol

# --- 4. Função de Plotagem ---
def plot_results(sol, omega_y, error_percent, T_hold=1.0, save_filename=None, is_limit_case=False):
    period = 2 * np.pi / omega_y
    y_sim, i_a_sim = sol.y[0], sol.y[2]
    y_ref = np.where(sol.t >= T_hold, Y * np.sin(omega_y * (sol.t - T_hold)), 0.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    title_text = f'Resposta para $\omega_y$ = {omega_y:.2f} rad/s (Erro = {error_percent:.2f}%)'
    if is_limit_case:
        title_text = f'Resposta na Frequência Limite Teórica ($\omega_y$ = {omega_y:.2f} rad/s, Erro = {error_percent:.2f}%)'

    ax1.plot(sol.t, y_ref, 'k--', label='Referência (y_ref)')
    ax1.plot(sol.t, y_sim, 'r-', label='Resposta do Sistema (y)')
    ax1.set_title(title_text)
    ax1.set_ylabel('Posição (m)'); ax1.grid(True); ax1.legend()

    ax2.plot(sol.t, i_a_sim, 'b-', label='Corrente de Armadura ($I_a$)')
    ax2.axhline(I_a_max, color='orange', linestyle='--', label=f'Limite Máx ({I_a_max} A)')
    ax2.axhline(-I_a_max, color='orange', linestyle='--')
    ax2.set_title('Corrente de Armadura'); ax2.set_ylabel('Corrente (A)'); ax2.set_xlabel('Tempo (s)'); ax2.grid(True); ax2.legend()
    
    ax1.set_xlim(0, T_hold + 4.5 * period)
    plt.tight_layout()
    
    if save_filename:
        try:
            fig.savefig(f"{save_filename}.pdf", format='pdf', bbox_inches='tight')
            print(f"--> Gráfico salvo com sucesso como: {save_filename}.pdf")
        except Exception as e:
            print(f"Erro ao salvar o gráfico: {e}")
    plt.show()

# --- 5. Execução Principal ---
T_hold_duration = 1.0
print("--- Item D: Iniciando busca por frequências... ---")
omegas = np.linspace(0.1, 2.5, 350)
results = {"low": {"freq": 0, "error": 100}, "mid": {"freq": 0, "error": 100}, "high": {"freq": 0, "error": 0, "found": False}}
for w in omegas:
    error, _ = simulate_system_ode(w, T_hold=T_hold_duration)
    if np.isnan(error): continue
    if error < 5 and abs(error - 5) < abs(results["low"]["error"] - 5): results["low"] = {"freq": w, "error": error}
    if abs(error - 40) < abs(results["mid"]["error"] - 40): results["mid"] = {"freq": w, "error": error}
    if error > 80 and not results["high"]["found"]: results["high"] = {"freq": w, "error": error, "found": True}
print("--- Busca Concluída ---")
print(f"Frequência com erro < 5%: w = {results['low']['freq']:.2f} rad/s (Erro: {results['low']['error']:.2f}%)")
print(f"Frequência com erro ~40%: w = {results['mid']['freq']:.2f} rad/s (Erro: {results['mid']['error']:.2f}%)")
print(f"Frequência com erro > 80%: w = {results['high']['freq']:.2f} rad/s (Erro: {results['high']['error']:.2f}%)")

# Plotagem para os 3 níveis de erro
filename_map = {"low": "baixo", "mid": "medio", "high": "alto"}
for label, res_data in results.items():
    freq = res_data['freq']
    if freq > 0:
        print(f"\n--- Gerando gráficos para '{label}' (w = {freq:.2f} rad/s) ---")
        error, sol_data = simulate_system_ode(freq, T_hold=T_hold_duration)
        filename_suffix = filename_map.get(label, "desconhecido")
        filename_to_save = f"item_D_grafico_{filename_suffix}"
        plot_results(sol_data, freq, error, T_hold=T_hold_duration, save_filename=filename_to_save)

# Plotagem para a frequência limite teórica
omega_limite_teorico = 1.61
print(f"\n--- Gerando gráfico para a Frequência Limite Teórica (w = {omega_limite_teorico:.2f} rad/s) ---")
error_limite, sol_data_limite = simulate_system_ode(omega_limite_teorico, T_hold=T_hold_duration)
filename_limite = "item_D_grafico_limite_teorico"
plot_results(sol_data_limite, omega_limite_teorico, error_limite, T_hold=T_hold_duration, save_filename=filename_limite, is_limit_case=True)