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
# Malha de Corrente (PI)
Kpc = 1.0
Kic = 50.0
# Malha de Velocidade (PI)
Kpv = 4.94
Kiv = 0.15
# Malha de Posição (P)
Kpp = 85.1

# --- 3. Simulação ---
# Configurações da simulação
dt = 0.0001  # Passo de tempo pequeno para estabilidade
T_sim = 60   # Tempo total de simulação [s]
time = np.arange(0, T_sim, dt)

# Arrays para armazenar resultados
y_out = np.zeros_like(time)
y_ref_out = np.zeros_like(time)
ia_out = np.zeros_like(time)
wm_out = np.zeros_like(time)
va_out = np.zeros_like(time)

# Variáveis de estado do sistema
ia = 0.0
wm = 0.0 # Velocidade angular mecânica [rad/s]
pos_y = 0.0 # Posição linear [m]
va = 0.0

# Variáveis de estado do controlador (termos integrais)
int_erro_i = 0.0
int_erro_w = 0.0

# Máquina de Estados para a Lógica de Controle
states = ["DESCENDING", "BRAKING", "HOLDING_BOTTOM", "ASCENDING", "HOLDING_TOP"]
current_state = "DESCENDING"

# Variáveis auxiliares para a lógica
descent_start_time = 0.0
descent_end_time = 0.0
stop_time_bottom = 0.0
stop_time_top = 0.0
hold_duration = 0.0

print(f"Iniciando simulação... Estado inicial: {current_state}")

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
            print(f"Tempo {t:.2f}s: Estado -> BRAKING. Duração da descida: {descent_duration:.2f}s. Tempo de parada: {hold_duration:.2f}s")
    elif current_state == "BRAKING":
        y_ref = -Y
        ia_ref_pos = Ia_max
        if wm >= -0.01:
            current_state = "HOLDING_BOTTOM"
            stop_time_bottom = t
            print(f"Tempo {t:.2f}s: Estado -> HOLDING_BOTTOM")
    elif current_state == "HOLDING_BOTTOM":
        y_ref = -Y
        if t - stop_time_bottom > hold_duration:
            current_state = "ASCENDING"
            print(f"Tempo {t:.2f}s: Estado -> ASCENDING")
    elif current_state == "ASCENDING":
        y_ref = +Y
        if pos_y >= Y:
            current_state = "HOLDING_TOP"
            stop_time_top = t
            print(f"Tempo {t:.2f}s: Estado -> HOLDING_TOP")
    elif current_state == "HOLDING_TOP":
        y_ref = +Y
        if t - stop_time_top > hold_duration:
            current_state = "DESCENDING"
            descent_start_time = t
            print(f"Tempo {t:.2f}s: Estado -> DESCENDING (reiniciando ciclo)")

    # --- Lógica de Controle em Cascata ---
    # Somente ativa as malhas externas se não estiver em modo de corrente direta
    if current_state not in ["DESCENDING", "BRAKING"]:
        erro_pos = y_ref - pos_y
        wm_ref = Kpp * erro_pos
        
        erro_w = wm_ref - wm
        int_erro_w += erro_w * dt

        ia_ref_ff = T_carga_static / k2
        ia_ref_pos = Kpv * erro_w + Kiv * int_erro_w + ia_ref_ff
    
    # Malha de Corrente (PI)
    ia_ref_sat = np.clip(ia_ref_pos, -Ia_max, Ia_max)
    erro_i = ia_ref_sat - ia
    
    # Lógica Anti-windup: usa o 'va' da iteração ANTERIOR
    # Impede que o termo integral cresça se a saída já estiver saturada
    if abs(va) >= Va_max and (np.sign(erro_i) * np.sign(va) > 0):
        pass # Não integra
    else:
        int_erro_i += erro_i * dt
        
    # Saída do controlador PI de corrente
    va_control = Kpc * erro_i + Kic * int_erro_i
    
    # Força contra-eletromotriz
    ea = k1 * wm
    
    # Tensão de armadura (recalculada para a iteração ATUAL)
    va = va_control + ea
    
    # Saturação final da fonte de tensão
    va = np.clip(va, -Va_max, Va_max)
    
    # --- Modelo da Planta (Motor + Carga) ---
    dia_dt = (va - Ra * ia - ea) / La
    ia += dia_dt * dt
    
    Te = k2 * ia
    dwm_dt = (Te - T_carga_static - ka * wm) / J_total
    wm += dwm_dt * dt
    
    pos_y += (wm * r) * dt
    
    # Armazena resultados
    y_out[i] = pos_y
    y_ref_out[i] = y_ref
    ia_out[i] = ia
    wm_out[i] = wm
    va_out[i] = va

# --- 4. Plotar Resultados ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Simulação da Manobra de Frenagem e Onda Quadrada=', fontsize=16)

# Posição
axs[0].plot(time, y_out, label='Posição da Carga (y)')
axs[0].plot(time, y_ref_out, '--', label='Referência (y_ref)')
axs[0].set_ylabel('Posição (m)')
axs[0].legend()
axs[0].grid(True)

# Corrente de Armadura
axs[1].plot(time, ia_out, label='Corrente de Armadura (Ia)')
axs[1].axhline(y=Ia_max, color='r', linestyle='--', label=f'Limite Máx ({Ia_max}A)')
axs[1].axhline(y=-Ia_max, color='r', linestyle='--')
axs[1].axhline(y=T_carga_static/k2, color='g', linestyle=':', label='Corrente Estática (~5A)')
axs[1].set_ylabel('Corrente (A)')
axs[1].legend()
axs[1].grid(True)

# Velocidade Angular
axs[2].plot(time, wm_out, label='Velocidade Angular (ωm)')
axs[2].set_xlabel('Tempo (s)')
axs[2].set_ylabel('Velocidade (rad/s)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('item_E_grafico.pdf')

print("Gráfico salvo com sucesso como 'item_E_grafico.pdf'")