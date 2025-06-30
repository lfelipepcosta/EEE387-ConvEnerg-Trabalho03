import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parâmetros do Sistema ---
# Parâmetros da Máquina
Ra = 0.2         # Resistência de armadura [Ohm]
La_sobre_Ra = 0.020 # Constante de tempo da armadura [s]
La = La_sobre_Ra * Ra # Indutância da armadura [H]
k1 = 0.265       # Constante de tensão [V/(rad/s)]
k2 = 0.265       # Constante de torque [Nm/A]
ka = 0.8e-3      # Coeficiente de atrito [Nm/(rad/s)]
Va_max = 120.0   # Tensão máxima de armadura [V]
Ia_max = 10.0    # Corrente máxima de armadura [A]

# Parâmetros da Carga e Projeto
M = 5.75         # Massa [kg]
g = 9.81         # Aceleração da gravidade [m/s^2]
r = 0.0235       # Raio da polia [m]
Y = 2.3          # Amplitude do movimento [m] (0.4 * 5.75)
J_motor = 0.023  # Inércia do motor [kg m^2]
J_carga = M * r**2 # Inércia da carga refletida no eixo
J_total = J_motor + J_carga # Inércia total

# Torque estático da carga
T_carga_static = M * g * r

# --- 2. Ganhos do Controlador em Cascata ---
Kpc = 1.0
Kic = 50.0
Kpv = 4.94
Kiv = 0.15
Kpp = 85.1

# --- 3. Simulação para Medição de Energia ---
# Configurações da simulação
dt = 0.0001
T_sim = 60
time = np.arange(0, T_sim, dt)

# Variáveis de estado do sistema
ia = 0.0
wm = 0.0
pos_y = 0.0
va = 0.0

# Variáveis de estado do controlador
int_erro_i = 0.0
int_erro_w = 0.0

# Máquina de Estados
current_state = "DESCENDING"
descent_start_time = 0.0
descent_end_time = 0.0
stop_time_bottom = 0.0
stop_time_top = 0.0
hold_duration = 0.0

# Variáveis para cálculo de energia
energia_subida_J = 0.0
energia_descida_J = 0.0

print(f"Iniciando simulação para UM CICLO COMPLETO...")

# Loop da Simulação
for i in range(len(time)):
    t = time[i]

    # Lógica da Máquina de Estados
    if current_state == "DESCENDING":
        y_ref = -Y
        ia_ref_pos = 0.0
        if pos_y <= -Y:
            current_state = "BREAKING"
            descent_end_time = t
            descent_duration = descent_end_time - descent_start_time
            hold_duration = 10 * descent_duration
            print(f"Tempo {t:.2f}s: Estado -> BREAKING.")

    elif current_state == "BREAKING":
        y_ref = -Y
        ia_ref_pos = Ia_max
        if wm >= -0.01:
            current_state = "HOLDING_BOTTOM"
            stop_time_bottom = t
            print(f"Tempo {t:.2f}s: Estado -> HOLDING_BOTTOM.")

    elif current_state == "HOLDING_BOTTOM":
        y_ref = -Y
        if t - stop_time_bottom > hold_duration:
            current_state = "ASCENDING"
            print(f"Tempo {t:.2f}s: Estado -> ASCENDING.")

    elif current_state == "ASCENDING":
        y_ref = +Y
        if pos_y >= Y:
            current_state = "HOLDING_TOP"
            stop_time_top = t
            print(f"Tempo {t:.2f}s: Estado -> HOLDING_TOP.")

    elif current_state == "HOLDING_TOP":
        y_ref = +Y
        if t - stop_time_top > hold_duration:
            print(f"Tempo {t:.2f}s: Fim do primeiro ciclo completo. Interrompendo a simulação.")
            break

    # Lógica de Controle em Cascata
    if current_state not in ["DESCENDING", "BREAKING"]:
        erro_pos = y_ref - pos_y
        wm_ref = Kpp * erro_pos
        erro_w = wm_ref - wm
        int_erro_w += erro_w * dt
        ia_ref_ff = T_carga_static / k2
        ia_ref_pos = Kpv * erro_w + Kiv * int_erro_w + ia_ref_ff

    ia_ref_sat = np.clip(ia_ref_pos, -Ia_max, Ia_max)
    erro_i = ia_ref_sat - ia

    if abs(va) >= Va_max and (np.sign(erro_i) * np.sign(va) > 0):
        pass
    else:
        int_erro_i += erro_i * dt

    va_control = Kpc * erro_i + Kic * int_erro_i
    ea = k1 * wm
    va = va_control + ea
    va = np.clip(va, -Va_max, Va_max)

    # Modelo da Planta (Motor + Carga)
    dia_dt = (va - Ra * ia - ea) / La
    ia += dia_dt * dt
    Te = k2 * ia
    dwm_dt = (Te - T_carga_static - ka * wm) / J_total
    wm += dwm_dt * dt
    pos_y += (wm * r) * dt

    # MUDANÇA: Cálculo da Energia Instantânea
    potencia_instantanea = va * ia
    if current_state == "ASCENDING":
        energia_subida_J += potencia_instantanea * dt
    if current_state == "BREAKING":
        energia_descida_J += potencia_instantanea * dt


# --- 4. Exibir Resultados de Energia ---
print("\n--- Análise de Energia (Item F) para UM CICLO ---")
print(f"Energia Gasta na Subida (Motor): {energia_subida_J:.2f} Joules")
print(f"Energia na Descida (Gerador/Frenagem): {energia_descida_J:.2f} Joules")
print(f"Balanço Energético do Ciclo (Gasto Líquido): {energia_subida_J + energia_descida_J:.2f} Joules")