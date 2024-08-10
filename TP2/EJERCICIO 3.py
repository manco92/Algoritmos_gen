#EJERCICIO 3

# Importo librerías
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyswarm import pso

# funcion objetivo hiperboloide eliptico
def funcion_objetivo(a, b, x, y):
    return (x-a)**2+(y+b)**2

def solve_PSO(funcion_objetivo, num_particulas, dim, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup, input_a, input_b):

    # inicializacion
    particulas = np.random.uniform(limite_inf, limite_sup, (num_particulas, dim))  # posiciones iniciales de las particulas

    velocidades = np.zeros((num_particulas, dim))  # inicializacion de la matriz de velocidades en cero

    # inicializacion de pbest y gbest
    pbest = particulas.copy()  # mejores posiciones personales iniciales

    fitness_pbest = np.empty(num_particulas)  # mejores fitness personales iniciales
    for i in range(num_particulas):
        fitness_pbest[i] = funcion_objetivo(input_a, input_b, particulas[i][0], particulas[i][1])

    gbest = pbest[np.argmin(fitness_pbest)]  # mejor posicion global inicial
    fitness_gbest = np.min(fitness_pbest)  # fitness global inicial

    g_bests = []

    # busqueda
    for iteracion in range(cantidad_iteraciones):
        for i in range(num_particulas):  # iteracion sobre cada partícula
            r1, r2 = np.random.rand(), np.random.rand()  # generacion dos numeros aleatorios

            # actualizacion de la velocidad de la particula en cada dimension
            for d in range(dim):
                velocidades[i][d] = (w * velocidades[i][d] + c1 * r1 * (pbest[i][d] - particulas[i][d]) + c2 * r2 * (gbest[d] - particulas[i][d]))

            for d in range(dim):
                particulas[i][d] = particulas[i][d] + velocidades[i][d]  # cctualizacion de la posicion de la particula en cada dimension

                # mantenimiento de las partículas dentro de los limites
                particulas[i][d] = np.clip(particulas[i][d], limite_inf, limite_sup)

            fitness = funcion_objetivo(input_a, input_b, particulas[i][0], particulas[i][1])  # Evaluacion de la funcion objetivo para la nueva posicion

            # actualizacion el mejor personal
            if fitness < fitness_pbest[i]:
                fitness_pbest[i] = fitness  # actualizacion del mejor fitness personal
                pbest[i] = particulas[i].copy()  # actualizacion de la mejor posicion personal

                # actualizacion del mejor global
                if fitness < fitness_gbest:
                    fitness_gbest = fitness  # actualizacion del mejor fitness global
                    gbest = particulas[i].copy()  # actualizacion de la mejor posicion global

        g_bests.append(gbest)

        # imprimir el mejor global en cada iteracion
        print(f"Iteración {iteracion + 1}: Mejor posición global {gbest}, Valor {fitness_gbest}")

    # resultado
    solucion_optima = gbest  # mejor posicion global final
    valor_optimo = fitness_gbest  # mejor fitness global final

    print("\nSolucion optima (x, y):", solucion_optima)
    print("Valor optimo:", valor_optimo)

    return solucion_optima, valor_optimo, g_bests

def plot_3d(funcion_objetivo, limite_inf, limite_sup, min_x, min_y):
    # Crear una malla de puntos para x y y
    x = np.linspace(limite_inf, limite_sup, 400)
    y = np.linspace(limite_inf, limite_sup, 400)
    X, Y = np.meshgrid(x, y)
    Z = funcion_objetivo(input_a, input_b, X, Y)

    # Mínimo teórico de la función
    min_z = funcion_objetivo(input_a, input_b, min_x, min_y)

    # Configurar la gráfica 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la superficie
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Graficar las líneas que indican el mínimo
    ax.plot([min_x, min_x], [min_y, min_y], [-100, min_z], color='r', linestyle='--')  # Línea vertical desde el mínimo hasta el plano XY
    ax.plot([min_x, min_x], [-100, min_y], [min_z, min_z], color='r', linestyle='--')  # Línea en el plano XY desde el eje Y al mínimo
    ax.plot([-100, min_x], [min_y, min_y], [min_z, min_z], color='r', linestyle='--')  # Línea en el plano XY desde el eje X al mínimo

    # Graficar el punto que indica el mínimo
    ax.scatter(min_x, min_y, min_z, color='red', s=50, label='Mínimo')

    # Etiquetas de los ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Título de la gráfica
    ax.set_title('Gráfico 3D con el Mínimo de la Función')

    # Mostrar la leyenda
    ax.legend()

    # Mostrar la gráfica
    plt.show()

# parametros
num_particulas = 20  # numero de particulas
dim = 2  # dimensiones
cantidad_iteraciones = 10  # maximo numero de iteraciones
c1 = 2.0  # componente cognitivo
c2 = 2.0  # componente social
w = 0.7  # factor de inercia
limite_inf = -100  # limite inferior de busqueda
limite_sup = 100  # limite superior de busqueda

input_a = input("Introduzca el valor del parámetro a")
input_a = max(min(float(input_a), 60), -60)
print("a: ", input_a)

input_b = input("Introduzca el valor del parámetro b")
input_b = max(min(float(input_b), 60), -60)
print("b: ", input_b)

solucion_optima, valor_optimo, g_bests = solve_PSO(funcion_objetivo, num_particulas, dim, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup, input_a, input_b)


# GRÁFICO ÓPTIMO
fig, ax = plt.subplots(1,2,figsize=(12,6))

x1 = np.linspace(limite_inf, limite_sup, 100)
x2 = np.linspace(limite_inf, limite_sup, 100)
y = funcion_objetivo(input_a, input_b, x1, x2)

# Variable X
ax[0].plot(x1, y)
ax[0].scatter(solucion_optima[0], valor_optimo, s=50, c='green')
ax[0].axvline(x=solucion_optima[0], c='green', linewidth=1)
ax[0].axhline(y=valor_optimo, c='green', linewidth=1)
ax[0].set_title('Minimización de función con algoritmo PSO (Variable X)')
ax[0].set_xlabel('Variable independiente X')
ax[0].set_ylabel('Variable dependiente')

# Variable Y
ax[1].plot(x2, y)
ax[1].scatter(solucion_optima[1], valor_optimo, s=50, c='green')
ax[1].axvline(x=solucion_optima[1], c='green', linewidth=1)
ax[1].axhline(y=valor_optimo, c='green', linewidth=1)
ax[1].set_title('Minimización de función con algoritmo PSO (Variable Y)')
ax[1].set_xlabel('Variable independiente Y')
ax[1].set_ylabel('Variable dependiente')

plt.tight_layout()
plt.show()

# GRÁFICO 3D
plot_3d(funcion_objetivo, limite_inf, limite_sup, solucion_optima[0], solucion_optima[1])

# GRÁFICO GBESTS
fig, ax = plt.subplots(1,2,figsize=(12,6))

x = np.arange(cantidad_iteraciones)
y1 = np.array(g_bests)[:, 0]
y2 = np.array(g_bests)[:, 1]

# Variable X
ax[0].plot(x, y1)
ax[0].set_title('Gbest (Variable X) en función de las iteraciones')
ax[0].set_xlabel('Cantidad de iteraciones')
ax[0].set_ylabel('Gbest')

# Variable Y
ax[1].plot(x, y2)
ax[1].set_title('Gbest (Variable Y) en función de las iteraciones')
ax[1].set_xlabel('Cantidad de iteraciones')
ax[1].set_ylabel('Gbest')

plt.tight_layout()
plt.show()

###########################
# CON COEF DE INERCIA W = 0
###########################

# parametros
num_particulas = 20  # numero de particulas
dim = 2  # dimensiones
cantidad_iteraciones = 10  # maximo numero de iteraciones
c1 = 2.0  # componente cognitivo
c2 = 2.0  # componente social
w = 0  # factor de inercia
limite_inf = -100  # limite inferior de busqueda
limite_sup = 100  # limite superior de busqueda

solucion_optima, valor_optimo, g_bests = solve_PSO(funcion_objetivo, num_particulas, dim, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup, input_a, input_b)

# GRÁFICO ÓPTIMO
fig, ax = plt.subplots(1,2,figsize=(12,6))

x1 = np.linspace(limite_inf, limite_sup, 100)
x2 = np.linspace(limite_inf, limite_sup, 100)
y = funcion_objetivo(x1, x2, input_a, input_b)

# Variable X
ax[0].plot(x1, y)
ax[0].scatter(solucion_optima[0], valor_optimo, s=50, c='green')
ax[0].axvline(x=solucion_optima[0], c='green', linewidth=1)
ax[0].axhline(y=valor_optimo, c='green', linewidth=1)
ax[0].set_title('Minimización de función con algoritmo PSO (Variable X)')
ax[0].set_xlabel('Variable independiente X')
ax[0].set_ylabel('Variable dependiente')

# Variable Y
ax[1].plot(x2, y)
ax[1].scatter(solucion_optima[1], valor_optimo, s=50, c='green')
ax[1].axvline(x=solucion_optima[1], c='green', linewidth=1)
ax[1].axhline(y=valor_optimo, c='green', linewidth=1)
ax[1].set_title('Minimización de función con algoritmo PSO (Variable Y)')
ax[1].set_xlabel('Variable independiente Y')
ax[1].set_ylabel('Variable dependiente')

plt.tight_layout()
plt.show()

# GRÁFICO 3D
plot_3d(funcion_objetivo, limite_inf, limite_sup, solucion_optima[0], solucion_optima[1])

# GRÁFICO GBESTS
fig, ax = plt.subplots(1,2,figsize=(12,6))

x = np.arange(cantidad_iteraciones)
y1 = np.array(g_bests)[:, 0]
y2 = np.array(g_bests)[:, 1]

# Variable X
ax[0].plot(x, y1)
ax[0].set_title('Gbest (Variable X) en función de las iteraciones')
ax[0].set_xlabel('Cantidad de iteraciones')
ax[0].set_ylabel('Gbest')

# Variable Y
ax[1].plot(x, y2)
ax[1].set_title('Gbest (Variable Y) en función de las iteraciones')
ax[1].set_xlabel('Cantidad de iteraciones')
ax[1].set_ylabel('Gbest')

plt.tight_layout()
plt.show()

#############
# CON PYSWARM
#############

# funcion objetivo hiperboloide eliptico
def funcion_objetivo(x):
    return (x[0] - input_a) ** 2 + (x[1] + input_b) ** 2

# parametros
num_particulas = 20  # numero de particulas
dim = 2  # dimensiones
cantidad_iteraciones = 10  # maximo numero de iteraciones
c1 = 2.0  # componente cognitivo
c2 = 2.0  # componente social
w = 0.7  # factor de inercia

lb = [-100, -100]  # limite inf
ub = [100, 100]  # limite sup

# Llamada a la función pso
solucion_optima, valor_optimo = pso(
    funcion_objetivo,
    lb,
    ub,
    swarmsize=num_particulas,
    maxiter=cantidad_iteraciones,
    omega=w,
    phip=c1,
    phig=c2,
    debug=False)

# Resultados
print("\nSolución óptima (x, y):", solucion_optima)
print("Valor óptimo:", valor_optimo)

# GRÁFICO ÓPTIMO
fig, ax = plt.subplots(1,2,figsize=(12,6))

x1 = np.linspace(limite_inf, limite_sup, 100)
x2 = np.linspace(limite_inf, limite_sup, 100)
y = [funcion_objetivo(x) for x in np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))]

# Variable X
ax[0].plot(x1, y)
ax[0].scatter(solucion_optima[0], valor_optimo, s=50, c='green')
ax[0].axvline(x=solucion_optima[0], c='green', linewidth=1)
ax[0].axhline(y=valor_optimo, c='green', linewidth=1)
ax[0].set_title('Minimización de función con algoritmo PSO (Variable X)')
ax[0].set_xlabel('Variable independiente X')
ax[0].set_ylabel('Variable dependiente')

# Variable Y
ax[1].plot(x2, y)
ax[1].scatter(solucion_optima[1], valor_optimo, s=50, c='green')
ax[1].axvline(x=solucion_optima[1], c='green', linewidth=1)
ax[1].axhline(y=valor_optimo, c='green', linewidth=1)
ax[1].set_title('Minimización de función con algoritmo PSO (Variable Y)')
ax[1].set_xlabel('Variable independiente Y')
ax[1].set_ylabel('Variable dependiente')

plt.tight_layout()
plt.show()


