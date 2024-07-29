## EJERCICIO 3

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Parámetros
TAMANIO_POBLACION = 16
LONGITUD_CROMOSOMA = 10
TASA_MUTACION = 0.07
TASA_CRUCE = 0.85
GENERACIONES = 20
TAMANIO_TORNEO = 3
PRECISION = 2

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
    x = binario_a_decimal(cromosoma, PRECISION, rango=(0, 10))
    g = 2*x/(4+0.8*x+x**2+0.2*x**3)
    return g

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
# Seleccion por torneo
###################################################################
def seleccion_torneo(poblacion, tamaño_torneo):
    participantes = random.sample(poblacion, tamaño_torneo)
    mejor_individuo = max(participantes, key=aptitud)
    return mejor_individuo
        
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
    
    for generacion in range(generaciones):
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
            progenitores.append(seleccion_torneo(poblacion, TAMANIO_TORNEO))

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
        mejor_x = binario_a_decimal(mejor_individuo, PRECISION, rango=(0, 10))
        print("Mejor individuo:", mejor_individuo, "Valor X:", mejor_x, "Aptitud:", aptitud(mejor_individuo))
        print("_________________________________________________________________________________")

        gens_vals.append(aptitud(mejor_individuo))
    
    return mejor_individuo, mejor_x, gens_vals

###################################################################
# algoritmo genetico ejecucion principal
###################################################################
print("_________________________________________________________________________________")
print("_________________________________________________________________________________")
print()
mejor_solucion, mejor_x, gens_vals = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
print("Mejor solución:", mejor_x, "Aptitud:", aptitud(mejor_solucion))

###################################################################
# imprimo gráfico 1
###################################################################
def func(x):
    return 2*x/(4+0.8*x+x**2+0.2*x**3)

x = np.arange(-1, 20)
y = func(x)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(x, y)
ax.scatter(mejor_x, aptitud(mejor_solucion), c='red', s=20)
ax.axhline(y=aptitud(mejor_solucion), color='red', linewidth=0.5)
ax.axvline(x=mejor_x, color='red', linewidth=0.5)
ax.set_title("Crecimiento de una levadura")
ax.set_xlabel('Concentración de alimento')
ax.set_ylabel('Tasa de crecimiento')
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