import numpy as np
import matplotlib.pyplot as plt

def simular_guerrilla(N=10000, nu=0.01, pasos=1000000):
    """
    Simula el proceso de coalescencia y fragmentación de unidades de ataque.
    
    Parámetros:
    - N: Número total de elementos base (fuerza total conservada).
    - nu: Probabilidad de fragmentación.
    - pasos: Número de iteraciones de la simulación.
    
    Retorna:
    - Un diccionario con {tamaño_unidad: cantidad_de_unidades} en el estado estacionario.
    """
    # counts almacena {fuerza_S: número_de_unidades_con_esa_fuerza}
    # Iniciamos con N unidades de fuerza 1
    counts = {1: N}
    
    for _ in range(pasos):
        # Decidir si ocurre fragmentación o coalescencia
        if np.random.rand() < nu:
            # --- FRAGMENTACIÓN ---
            # Elegir una unidad proporcional a su fuerza (S)
            r = np.random.randint(1, N + 1)
            acc = 0
            for s, c in counts.items():
                acc += s * c
                if acc >= r:
                    s1 = s
                    break
            
            # Fragmentar la unidad elegida en s1 unidades de fuerza 1
            if s1 > 1:
                counts[s1] -= 1
                if counts[s1] == 0:
                    del counts[s1]
                counts[1] = counts.get(1, 0) + s1
                
        else:
            # --- COALESCENCIA ---
            # Elegir la primera unidad
            r1 = np.random.randint(1, N + 1)
            acc = 0
            for s, c in counts.items():
                acc += s * c
                if acc >= r1:
                    s1 = s
                    break
            
            # Elegir la segunda unidad
            r2 = np.random.randint(1, N + 1)
            acc = 0
            for s, c in counts.items():
                acc += s * c
                if acc >= r2:
                    s2 = s
                    break
            
            # Lógica de fusión
            if s1 == s2:
                # Si las fuerzas son iguales, verificamos si elegimos EXACTAMENTE la misma unidad.
                # La probabilidad de elegir la misma unidad dentro del mismo grupo de tamaño s1 es 1 / cantidad.
                if np.random.rand() < (1.0 / counts[s1]):
                    continue # Se eligió la misma unidad, no ocurre nada
                else:
                    # Se eligieron dos unidades distintas con la misma fuerza
                    counts[s1] -= 2
                    if counts[s1] == 0:
                        del counts[s1]
                    counts[s1 + s2] = counts.get(s1 + s2, 0) + 1
            else:
                # Se eligieron unidades de diferentes fuerzas
                counts[s1] -= 1
                if counts[s1] == 0:
                    del counts[s1]
                counts[s2] -= 1
                if counts[s2] == 0:
                    del counts[s2]
                counts[s1 + s2] = counts.get(s1 + s2, 0) + 1
                
    return counts

def analizar_y_graficar():
    """
    Ejecuta la simulación para diferentes valores de nu, grafica la 
    distribución log-log y calcula el exponente alfa.
    """
    N_total = 20000 
    pasos_sim = 3_000_000 # Un tiempo largo para asegurar el estado estacionario
    valores_nu = [0.01, 0.05, 0.1]
    
    plt.figure(figsize=(10, 6))
    
    for nu in valores_nu:
        print(f"Simulando para ν = {nu}...")
        distribucion = simular_guerrilla(N=N_total, nu=nu, pasos=pasos_sim)
        
        # Extraer fuerzas (S) y cantidad (n)
        S = np.array(sorted(distribucion.keys()))
        n = np.array([distribucion[s] for s in S])
        
        # Graficar en escala Log-Log
        plt.loglog(S, n, 'o', label=f'ν = {nu}', markersize=5, alpha=0.7)
        
        # Calcular alfa (pendiente de la regresión lineal en escala logarítmica)
        # Se omite la cola extrema ruidosa para un mejor ajuste (e.g., n > 1)
        filtro = n > 2 
        if np.sum(filtro) > 2:
            log_S = np.log10(S[filtro])
            log_n = np.log10(n[filtro])
            pendiente, _ = np.polyfit(log_S, log_n, 1)
            alfa = -pendiente
            print(f"  -> Exponente α estimado para ν={nu}: {alfa:.3f}")

    plt.xlabel('Fuerza de ataque (S)', fontsize=12)
    plt.ylabel('Número de unidades n(S)', fontsize=12)
    plt.title('Distribución de Unidades de Ataque (Ley de Potencias)', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

# Ejecutar el análisis
if __name__ == '__main__':
    analizar_y_graficar()
