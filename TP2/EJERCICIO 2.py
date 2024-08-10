#EJERCICIO 2

# Importo librerías
import numpy as np
import matplotlib.pyplot as plt

# funcion objetivo hiperboloide eliptico
def funcion_objetivo(x):
    return np.sin(x)+np.sin(x**2)

def solve_PSO(funcion_objetivo, num_particulas, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup):

    # inicializacion
    particulas = np.random.uniform(limite_inf, limite_sup, num_particulas)  # posiciones iniciales de las particulas

    velocidades = np.zeros((num_particulas))  # inicializacion de la matriz de velocidades en cero

    # inicializacion de pbest y gbest
    pbest = particulas.copy()  # mejores posiciones personales iniciales

    fitness_pbest = np.empty(num_particulas)  # mejores fitness personales iniciales
    for i in range(num_particulas):
        fitness_pbest[i] = funcion_objetivo(particulas[i])

    gbest = pbest[np.argmin(fitness_pbest)]  # mejor posicion global inicial
    fitness_gbest = np.min(fitness_pbest)  # fitness global inicial

    g_bests = []

    # busqueda
    for iteracion in range(cantidad_iteraciones):
        for i in range(num_particulas):  # iteracion sobre cada partícula
            r1, r2 = np.random.rand(), np.random.rand()  # generacion dos numeros aleatorios

            # actualizacion de la velocidad de la particula en cada dimension
            velocidades[i] = (w * velocidades[i] + c1 * r1 * (pbest[i] - particulas[i]) + c2 * r2 * (gbest - particulas[i]))

            particulas[i] = particulas[i] + velocidades[i]  # cctualizacion de la posicion de la particula en cada dimension

            # mantenimiento de las partículas dentro de los limites
            particulas[i] = np.clip(particulas[i], limite_inf, limite_sup)

            fitness = funcion_objetivo(particulas[i])  # Evaluacion de la funcion objetivo para la nueva posicion

            # actualizacion el mejor personal
            if fitness > fitness_pbest[i]:
                fitness_pbest[i] = fitness  # actualizacion del mejor fitness personal
                pbest[i] = particulas[i].copy()  # actualizacion de la mejor posicion personal

                # actualizacion del mejor global
                if fitness > fitness_gbest:
                    fitness_gbest = fitness  # actualizacion del mejor fitness global
                    gbest = particulas[i].copy()  # actualizacion de la mejor posicion global
                    
        g_bests.append(gbest)

        # imprimir el mejor global en cada iteracion
        print(f"Iteración {iteracion + 1}: Mejor posición global {gbest}, Valor {fitness_gbest}")

    # resultado
    solucion_optima = gbest  # mejor posicion global final
    valor_optimo = fitness_gbest  # mejor fitness global final

    print("\nSolucion optima (x):", solucion_optima)
    print("Valor optimo:", valor_optimo)

    return solucion_optima, valor_optimo, g_bests

##################
# CON 2 PARTÍCULAS
##################

# parametros
num_particulas = 2  # numero de particulas
cantidad_iteraciones = 30  # maximo numero de iteraciones
c1 = 1.49  # componente cognitivo
c2 = 1.49  # componente social
w = 0.5  # factor de inercia
limite_inf = 0  # limite inferior de busqueda
limite_sup = 10  # limite superior de busqueda

solucion_optima, valor_optimo, g_bests = solve_PSO(funcion_objetivo, num_particulas, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup)

# GRÁFICO ÓPTIMO
fig, ax = plt.subplots(figsize=(10,6))

x = np.linspace(limite_inf, limite_sup, 100)
y = funcion_objetivo(x)

ax.plot(x, y)
ax.scatter(solucion_optima, valor_optimo, s=50, c='green')
ax.axvline(x=solucion_optima, c='green', linewidth=1)
ax.axhline(y=valor_optimo, c='green', linewidth=1)
ax.set_title('Maximización de función con algoritmo PSO')
ax.set_xlabel('Variable independiente')
ax.set_ylabel('Variable dependiente')
plt.tight_layout()
plt.show()

# GRÁFICO GBESTS
fig, ax = plt.subplots(figsize=(10,6))

x = np.arange(cantidad_iteraciones)
y = g_bests

ax.plot(x, y)
ax.set_title('Gbest en función de las iteraciones')
ax.set_xlabel('Cantidad de iteraciones')
ax.set_ylabel('Gbest')
plt.tight_layout()
plt.show()

##################
# CON 4 PARTÍCULAS
##################

# parametros
num_particulas = 4  # numero de particulas
cantidad_iteraciones = 30  # maximo numero de iteraciones
c1 = 1.49  # componente cognitivo
c2 = 1.49  # componente social
w = 0.5  # factor de inercia
limite_inf = 0  # limite inferior de busqueda
limite_sup = 10  # limite superior de busqueda

solucion_optima, valor_optimo, g_bests = solve_PSO(funcion_objetivo, num_particulas, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup)

# GRÁFICO ÓPTIMO
fig, ax = plt.subplots(figsize=(10,6))

x = np.linspace(limite_inf, limite_sup, 100)
y = funcion_objetivo(x)

ax.plot(x, y)
ax.scatter(solucion_optima, valor_optimo, s=50, c='green')
ax.axvline(x=solucion_optima, c='green', linewidth=1)
ax.axhline(y=valor_optimo, c='green', linewidth=1)
ax.set_title('Maximización de función con algoritmo PSO')
ax.set_xlabel('Variable independiente')
ax.set_ylabel('Variable dependiente')
plt.tight_layout()
plt.show()

# GRÁFICO GBESTS
fig, ax = plt.subplots(figsize=(10,6))

x = np.arange(cantidad_iteraciones)
y = g_bests

ax.plot(x, y)
ax.set_title('Gbest en función de las iteraciones')
ax.set_xlabel('Cantidad de iteraciones')
ax.set_ylabel('Gbest')
plt.tight_layout()
plt.show()

##################
# CON 6 PARTÍCULAS
##################

# parametros
num_particulas = 6  # numero de particulas
cantidad_iteraciones = 30  # maximo numero de iteraciones
c1 = 1.49  # componente cognitivo
c2 = 1.49  # componente social
w = 0.5  # factor de inercia
limite_inf = 0  # limite inferior de busqueda
limite_sup = 10  # limite superior de busqueda

solucion_optima, valor_optimo, g_bests = solve_PSO(funcion_objetivo, num_particulas, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup)

# GRÁFICO ÓPTIMO
fig, ax = plt.subplots(figsize=(10,6))

x = np.linspace(limite_inf, limite_sup, 100)
y = funcion_objetivo(x)

ax.plot(x, y)
ax.scatter(solucion_optima, valor_optimo, s=50, c='green')
ax.axvline(x=solucion_optima, c='green', linewidth=1)
ax.axhline(y=valor_optimo, c='green', linewidth=1)
ax.set_title('Maximización de función con algoritmo PSO')
ax.set_xlabel('Variable independiente')
ax.set_ylabel('Variable dependiente')
plt.tight_layout()
plt.show()

# GRÁFICO GBESTS
fig, ax = plt.subplots(figsize=(10,6))

x = np.arange(cantidad_iteraciones)
y = g_bests

ax.plot(x, y)
ax.set_title('Gbest en función de las iteraciones')
ax.set_xlabel('Cantidad de iteraciones')
ax.set_ylabel('Gbest')
plt.tight_layout()
plt.show()

##################
# CON 10 PARTÍCULAS
##################

# parametros
num_particulas = 10  # numero de particulas
cantidad_iteraciones = 30  # maximo numero de iteraciones
c1 = 1.49  # componente cognitivo
c2 = 1.49  # componente social
w = 0.5  # factor de inercia
limite_inf = 0  # limite inferior de busqueda
limite_sup = 10  # limite superior de busqueda

solucion_optima, valor_optimo, g_bests = solve_PSO(funcion_objetivo, num_particulas, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup)

# GRÁFICO ÓPTIMO
fig, ax = plt.subplots(figsize=(10,6))

x = np.linspace(limite_inf, limite_sup, 100)
y = funcion_objetivo(x)

ax.plot(x, y)
ax.scatter(solucion_optima, valor_optimo, s=50, c='green')
ax.axvline(x=solucion_optima, c='green', linewidth=1)
ax.axhline(y=valor_optimo, c='green', linewidth=1)
ax.set_title('Maximización de función con algoritmo PSO')
ax.set_xlabel('Variable independiente')
ax.set_ylabel('Variable dependiente')
plt.tight_layout()
plt.show()

# GRÁFICO GBESTS
fig, ax = plt.subplots(figsize=(10,6))

x = np.arange(cantidad_iteraciones)
y = g_bests

ax.plot(x, y)
ax.set_title('Gbest en función de las iteraciones')
ax.set_xlabel('Cantidad de iteraciones')
ax.set_ylabel('Gbest')
plt.tight_layout()
plt.show()



