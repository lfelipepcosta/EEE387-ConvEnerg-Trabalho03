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

# --- 3. Simulação ---
dt = 0.0001
T_sim = 180 # Tempo total da simulação
time = np.arange(0, T_sim, dt)

# Arrays para armazenar resultados do gráfico de posição
y_out = np.zeros_like(time)

# Variáveis de estado
ia, wm, pos_y, va = 0.0, 0.0, 0.0, 0.0
int_erro_i, int_erro_w = 0.0, 0.0

# Máquina de Estados e variáveis de lógica
current_state = "DESCENDING"
descent_start_time, descent_end_time = 0.0, 0.0
stop_time_bottom, stop_time_top = 0.0, 0.0
hold_duration = 0.0
pos_parada_inferior = 0.0

# MUDANÇA: Lógica para rastrear energia de eventos separados
energias_descida = []
energias_subida = []
energia_evento_atual = 0.0

# Loop da Simulação
for i in range(len(time)):
    t = time[i]

    # --- Lógica da Máquina de Estados ---
    if current_state == "DESCENDING":
        y_ref = -Y
        ia_ref_pos = 0.0
        if pos_y <= -Y:
            current_state = "BRAKING"
            descent_end_time = t
            descent_duration = descent_end_time - descent_start_time
            hold_duration = 10 * descent_duration
            energia_evento_atual = 0.0

    elif current_state == "BRAKING":
        y_ref = -Y
        ia_ref_pos = Ia_max
        if wm >= -0.01:
            energias_descida.append(energia_evento_atual)
            pos_parada_inferior = pos_y
            current_state = "HOLDING_BOTTOM"
            stop_time_bottom = t

    elif current_state == "HOLDING_BOTTOM":
        y_ref = pos_parada_inferior
        if t - stop_time_bottom > hold_duration:
            current_state = "ASCENDING"
            energia_evento_atual = 0.0

    elif current_state == "ASCENDING":
        y_ref = +Y
        if pos_y >= Y:
            energias_subida.append(energia_evento_atual)
            current_state = "HOLDING_TOP"
            stop_time_top = t

    elif current_state == "HOLDING_TOP":
        y_ref = +Y
        if t - stop_time_top > hold_duration:
            current_state = "DESCENDING"
            descent_start_time = t

    # --- Lógica de Controle em Cascata ---
    if current_state not in ["DESCENDING", "BRAKING"]:
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
    
    # --- Modelo da Planta (Motor + Carga) ---
    dia_dt = (va - Ra * ia - ea) / La
    ia += dia_dt * dt
    Te = k2 * ia
    dwm_dt = (Te - T_carga_static - ka * wm) / J_total
    wm += dwm_dt * dt
    pos_y += (wm * r) * dt
    
    # Armazena resultado de posição para o gráfico
    y_out[i] = pos_y
    
    # Cálculo de Energia para o evento atual
    potencia_instantanea = va * ia
    if current_state == "ASCENDING" or current_state == "BRAKING":
        energia_evento_atual += potencia_instantanea * dt

# --- 4. Exibir Resultados ---

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
fig.suptitle('Simulação da Manobra - Posição vs. Tempo', fontsize=16)

ax.plot(time, y_out, label='Posição da Carga (y)')
ax.axhline(y=Y, color='k', linestyle=':', label=f'Limite Superior (+Y = {Y}m)')
ax.axhline(y=-Y, color='k', linestyle=':', label=f'Início da Frenagem (-Y = {-Y}m)')
ax.set_ylabel('Posição (m)'), ax.legend(), ax.grid(True)
ax.set_xlabel('Tempo (s)')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

print("--- Análise de Energia (Item F) ---")
num_pares = min(len(energias_descida), len(energias_subida))

for i in range(num_pares):
    print(f"\n--- Descida {i+1} ---")
    print(f"  Energia Recuperada (Gerador): {energias_descida[i]:.2f} Joules")
    print(f"--- Subida {i+1} ---")
    print(f"  Energia Gasta (Motor): {energias_subida[i]:.2f} Joules")

# Imprime qualquer descida extra que não teve uma subida correspondente
if len(energias_descida) > num_pares:
    print(f"\n--- Descida {len(energias_descida)} ---")
    print(f"  Energia Recuperada (Gerador): {energias_descida[-1]:.2f} Joules")