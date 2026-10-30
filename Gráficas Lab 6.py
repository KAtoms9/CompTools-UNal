import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATOS EXPERIMENTALES (Tabla 1)
# ==========================================
# Masa giratoria M (kg) y Radio R (m)
M = 0.4485
R = 0.219

# Datos obtenidos
v2 = np.array([6.024, 6.268, 6.480])      # Variable independiente (x): v^2 [m^2/s^2]
Fre = np.array([10.7331, 12.6833, 12.6883]) # Variable dependiente (y): Fre [N]

# Incertidumbres (de la sección 3.3 del informe)
delta_v2 = np.array([0.1, 0.1, 0.1])      # Incertidumbre en x
delta_Fre = np.array([0.0005, 0.0005, 0.0005]) # Incertidumbre en y

# ==========================================
# 2. AJUSTE LINEAL Y PENDIENTE TEÓRICA
# ==========================================
# Ajuste lineal por mínimos cuadrados: y = m*x + b
m_exp, b_exp = np.polyfit(v2, Fre, 1)

# Pendiente teórica: m_teo = M / R
m_teo = M / R

# Línea de tendencia para graficar
x_trend = np.linspace(min(v2) - 0.2, max(v2) + 0.2, 100)
y_trend = m_exp * x_trend + b_exp

# Coeficiente de determinación R^2
y_pred = m_exp * v2 + b_exp
r_squared = 1 - (np.sum((Fre - y_pred)**2) / np.sum((Fre - np.mean(Fre))**2))

# ==========================================
# 3. GENERACIÓN DE LA GRÁFICA
# ==========================================
plt.figure(figsize=(8, 5), dpi=300)

# Graficar datos experimentales con barras de error
plt.errorbar(v2, Fre, xerr=delta_v2, yerr=delta_Fre, fmt='o', color='navy', 
             ecolor='crimson', elinewidth=1.5, capsize=4, capthick=1.5, 
             label='Datos experimentales', zorder=3)

# Graficar ajuste lineal
plt.plot(x_trend, y_trend, '--', color='black', linewidth=1.5,
         label=f'Ajuste lineal: $y = {m_exp:.4f}x + ({b_exp:.4f})$\n$R^2 = {r_squared:.4f}$')

# Configuración de etiquetas y título
plt.title('Gráfica de Fuerza del Resorte ($F_{re}$) vs. Velocidad al Cuadrado ($v^2$)', fontsize=12, fontweight='bold')
plt.xlabel('Velocidad al cuadrado $v^2$ (m$^2$/s$^2$)', fontsize=11)
plt.ylabel('Fuerza del resorte $F_{re}$ (N)', fontsize=11)

# Estilo de cuadrícula y leyenda
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

# Ajuste de límites para mejor visualización
plt.xlim(min(v2) - 0.3, max(v2) + 0.3)
plt.ylim(min(Fre) - 1.0, max(Fre) + 1.0)

plt.tight_layout()

# Mostrar resultados en consola
print(f"--- RESULTADOS DE LA PARTE 1 ---")
print(f"Pendiente Experimental (m_exp): {m_exp:.4f} N/(m^2/s^2)")
print(f"Pendiente Teórica (M/R):       {m_teo:.4f} kg/m")
print(f"Error Porcentual:               {abs(m_exp - m_teo)/m_teo * 100:.2f}%")
print(f"Coeficiente R^2:               {r_squared:.4f}")

# Mostrar gráfico
plt.show()
