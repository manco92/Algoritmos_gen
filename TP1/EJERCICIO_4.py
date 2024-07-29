## EJERCICIO 3

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Parámetros
TAMANIO_POBLACION = 128*4
LONGITUD_CROMOSOMA = 16
TASA_MUTACION = 0.09
TASA_CRUCE = 0.7
GENERACIONES = 50
PRECISION = 3

np.random.seed(22)
random.seed(22)

###################################################################
# Conversión de cromosoma binario a decimal con precisión y rango
###################################################################
def binario_a_decimal(cromosoma, precision, rango):
    entero = int(cromosoma, 2)
    max_entero = 2**len(cromosoma) - 1
    decimal = rango[0] + (entero / max_entero) * (rango[1] - rango[0])
    return round(decimal, precision)

###################################################################
# Aptitud
###################################################################
def aptitud(cromosoma):
    n = len(cromosoma) // 2
    x = binario_a_decimal(cromosoma[:n], PRECISION, rango=(-10, 10))
    y = binario_a_decimal(cromosoma[n:], PRECISION, rango=(0, 20))
    c = 7.7+0.15*x+0.22*y-0.05*x**2-0.016*y**2-0.007*x*y
    return c

###################################################################
# Inicializar la población
###################################################################
def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    poblacion = []
    for z in range(tamanio_poblacion):
        cromosoma = ""
        for t in range(longitud_cromosoma):
            cromosoma = cromosoma+str(random.randint(0, 1))
        poblacion.append(cromosoma)
    return poblacion

###################################################################
# Seleccion por ruleta
###################################################################
def seleccion_ruleta(poblacion, aptitud_total):
    seleccion = random.uniform(0, aptitud_total)
    aptitud_actual = 0
    for individuo in poblacion:
        aptitud_actual = aptitud_actual+aptitud(individuo)
        if aptitud_actual > seleccion:
            return individuo
        
###################################################################
# Cruce monopunto con probabilidad de cruza pc = 0.92
###################################################################
def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):
    if random.random() < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return descendiente1, descendiente2

###################################################################
# mutacion
###################################################################
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = ""
    for bit in cromosoma:
        if random.random() < tasa_mutacion:
            cromosoma_mutado = cromosoma_mutado+str(int(not int(bit)))
        else:
            cromosoma_mutado = cromosoma_mutado+bit
    return cromosoma_mutado

###################################################################
# aplicacion de operadores geneticos
###################################################################
def algoritmo_genetico(tamaño_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones):
    poblacion = inicializar_poblacion(tamaño_poblacion, longitud_cromosoma)

    gens_vals = []

    for generacion in tqdm(range(generaciones)):
        print("Generación:", generacion + 1)

        # Calcular aptitud total para luego
        aptitud_total = 0
        for cromosoma in poblacion:
            aptitud_total = aptitud_total+aptitud(cromosoma)

        print("Aptitud total:", aptitud_total)

        # ..................................................................
        # seleccion
        # de progenitores con el metodo ruleta
        # se crea una lista vacia de progenitores primero
        progenitores = []
        for _ in range(tamaño_poblacion):
            progenitores.append(seleccion_ruleta(poblacion, aptitud_total))
            #progenitores.append(seleccion_torneo(poblacion, TAMANIO_TORNEO))

        # ..................................................................
        # Cruce
        descendientes = []
        for i in range(0, tamaño_poblacion, 2):
            descendiente1, descendiente2 = cruce_mono_punto(progenitores[i], progenitores[i + 1], tasa_cruce)
            descendientes.extend([descendiente1, descendiente2])

        # ..................................................................
        # mutacion
        descendientes_mutados = []
        for descendiente in descendientes:
            descendientes_mutados.append(mutacion(descendiente, tasa_mutacion))

        # Aqui se aplica elitismo
        # se reemplazar los peores cromosomas con los mejores progenitores
        poblacion.sort(key=aptitud)
        descendientes_mutados.sort(key=aptitud, reverse=True)
        for i in range(len(descendientes_mutados)):
            if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                poblacion[i] = descendientes_mutados[i]

        # mostrar el mejor individuo de la generacion
        mejor_individuo = max(poblacion, key=aptitud)
        x_mejor = binario_a_decimal(mejor_individuo[:longitud_cromosoma//2], PRECISION, rango=(-10, 10))
        y_mejor = binario_a_decimal(mejor_individuo[longitud_cromosoma//2:], PRECISION, rango=(0, 20))
        print("Mejor individuo:", mejor_individuo, "-> x:", x_mejor, ", y:", y_mejor, "Aptitud:", aptitud(mejor_individuo))
        print("_________________________________________________________________________________")

        gens_vals.append(aptitud(mejor_individuo))

    return mejor_individuo, x_mejor, y_mejor, gens_vals

###################################################################
# algoritmo genetico ejecucion principal
###################################################################
print("_________________________________________________________________________________")
print("_________________________________________________________________________________")
print()
mejor_solucion, x_mejor, y_mejor, gens_vals = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
print("Mejor solución:", "-> x:", x_mejor, ", y:", y_mejor, "Aptitud:", aptitud(mejor_solucion))

###################################################################
# imprimo gráfico 1
###################################################################
def func(x, y):
    return 7.7+0.15*x+0.22*y-0.05*x**2-0.016*y**2-0.007*x*y

x1 = np.arange(-10, 10)
x2 = np.arange(0, 20)
y = func(x1, x2)

# Gráfico de variable x
fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].plot(x1, y)
ax[0].scatter(x_mejor, aptitud(mejor_solucion), c='red', s=20)
ax[0].axhline(y=aptitud(mejor_solucion), color='red', linewidth=0.5)
ax[0].axvline(x=x_mejor, color='red', linewidth=0.5)
ax[0].set_title("Distribución de concentración de contaminante")
ax[0].set_xlabel('Variable X')
ax[0].set_ylabel('Concentración')

# Gráfico de variable y
ax[1].plot(x2, y)
ax[1].scatter(y_mejor, aptitud(mejor_solucion), c='red', s=20)
ax[1].axhline(y=aptitud(mejor_solucion), color='red', linewidth=0.5)
ax[1].axvline(x=y_mejor, color='red', linewidth=0.5)
ax[1].set_title("Distribución de concentración de contaminante")
ax[1].set_xlabel('Variable Y')
ax[1].set_ylabel('Concentración')

plt.show()

###################################################################
# imprimo gráfico 2
###################################################################
x = np.arange(len(gens_vals))
y = gens_vals

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(x, y)
ax.set_title("Evolución de mejores aptitudes en cada generación")
ax.set_xlabel('Generación')
ax.set_ylabel('Mejor aptitud')
plt.show()