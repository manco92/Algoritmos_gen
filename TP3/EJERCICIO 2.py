# EJERCICIO 2

# Importo librerías
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import pandas as pd

# función objetivo a maximizar
def f(x):
    return 500 * x[0] + 400 * x[1]

# primera restriccion
def g1(x):
    return 300 * x[0] + 400 * x[1] - 127000 <= 0

# segunda restriccion
def g2(x):
    return 20 * x[0] + 10 * x[1] - 4270 <= 0

# tercera restriccion
def g3(x):
    return x[0] >= 0 and x[1] >= 0

def solve_pso(f, g1, g2, g3, n_particles, n_dimensions, max_iterations, c1, c2, w, verbose=True):

    # inicialización de particulas
    x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las particulas
    v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las particulas
    pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
    pbest_fit = -np.inf * np.ones(n_particles)  # mector para las mejores aptitudes personales (inicialmente -infinito)
    gbest = np.zeros(n_dimensions)  # mejor solución global
    gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)
    g_bests = []

    # inicializacion de particulas factibles
    for i in range(n_particles):
        while True:  # bucle para asegurar que la particula sea factible
            x[i] = np.random.uniform(0, 10, n_dimensions)  # inicializacion posicion aleatoria en el rango [0, 10]
            if g1(x[i]) and g2(x[i]) and g3(x[i]):  # se comprueba si la posicion cumple las restricciones
                break  # Salir del bucle si es factible
        v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
        pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
        fit = f(x[i])  # calculo la aptitud de la posicion inicial
        if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
            pbest_fit[i] = fit  # se actualiza el mejor valor personal

    # Optimizacion
    for _ in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
        for i in range(n_particles):
            fit = f(x[i])  # Se calcula la aptitud de la posicion actual
            # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
            if fit > pbest_fit[i] and g1(x[i]) and g2(x[i]) and g3(x[i]):
                pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
                pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
                if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                    gbest_fit = fit  # Se actualizar la mejor aptitud global
                    gbest = x[i].copy()  # Se actualizar la mejor posicion global

            # actualizacion de la velocidad de la particula
            v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
            x[i] += v[i]  # Se actualiza la posicion de la particula

            # se asegura de que la nueva posicion esté dentro de las restricciones
            if not (g1(x[i]) and g2(x[i]) and g3(x[i])):
                # Si la nueva posicion no es válida, revertir a la mejor posicion personal
                x[i] = pbest[i].copy()

        g_bests.append(gbest)

    # Se imprime la mejor solucion encontrada y también su valor optimo
    mejor_solucion = gbest
    valor_optimo = gbest_fit
    if verbose:
        print(f"Mejor solucion: [{gbest[0]:.0f}, {gbest[1]:.0f}]")
        print(f"Valor optimo: {gbest_fit}")

    return mejor_solucion, valor_optimo, g_bests

# parametros
n_particles = 10  # numero de particulas en el enjambre
n_dimensions = 2  # dimensiones del espacio de busqueda (x1 y x2)
max_iterations = 80  # numero máximo de iteraciones para la optimizacion
c1 = c2 = 2  # coeficientes de aceleracion
w = 0.5  # Factor de inercia

solucion_optima, valor_optimo, g_bests = solve_pso(f, g1, g2, g3, n_particles, n_dimensions, max_iterations, c1, c2, w)

fig, ax = plt.subplots(figsize=(12,6))

x = np.arange(len(g_bests))
y = g_bests

ax.plot(x, np.array(g_bests)[:, 0], color='red', label='Impresora 1')
ax.plot(x, np.array(g_bests)[:, 1], color='blue', label='Impresora 2')
ax.set_title('Gbest en función de las iteraciones')
ax.set_xlabel('Cantidad de iteraciones')
ax.set_ylabel('Gbest')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# segunda restriccion
def g2(x):
    return 20 * x[0] + 11 * x[1] - 4270 <= 0

# parametros
n_particles = 10  # numero de particulas en el enjambre
n_dimensions = 2  # dimensiones del espacio de busqueda (x1 y x2)
max_iterations = 80  # numero máximo de iteraciones para la optimizacion
c1 = c2 = 2  # coeficientes de aceleracion
w = 0.5  # Factor de inercia

solucion_optima, valor_optimo, g_bests = solve_pso(f, g1, g2, g3, n_particles, n_dimensions, max_iterations, c1, c2, w)

# parametros
n_particles = 50  # numero de particulas en el enjambre
n_dimensions = 2  # dimensiones del espacio de busqueda (x1 y x2)
max_iterations = 80  # numero máximo de iteraciones para la optimizacion
c1 = c2 = 2  # coeficientes de aceleracion
w = 0.5  # Factor de inercia

soluciones_optimas = []
for _ in tqdm(range(1000)):
    solucion_optima, valor_optimo, g_bests = solve_pso(f, g1, g2, g3, n_particles, n_dimensions, max_iterations, c1, c2, w, verbose=False)
    soluciones_optimas.append(solucion_optima)

soluciones_optimas = pd.DataFrame(np.array(soluciones_optimas), columns=['Impresora 1', 'Impresora 2'])

fig, ax = plt.subplots(1,2, figsize=(10,5))
for i,n in enumerate(['Impresora 1', 'Impresora 2']):
    sns.kdeplot(x=soluciones_optimas[n], data=soluciones_optimas, ax=ax[i], fill=True)
    ax[i].set_title(f'Distribución de producción de {n}')
    ax[i].set_xlim((0, 300))
    ax[i].set_ylabel('')
    ax[i].set_xlabel('')

plt.tight_layout()
plt.show()