
from pyswarm import pso

# función objetivo
def funcion_objetivo(x):
    return x[0]**2 + x[1]**2

lb = [-100, -100]  # limite inf
ub = [100, 100]  # limite sup

num_particulas = 10  # numero de particulas
cantidad_iteraciones = 20  # numero maximo de iteraciones

# Llamada a la función pso
solucion_optima, valor_optimo = pso(funcion_objetivo, lb, ub, swarmsize=num_particulas, maxiter=cantidad_iteraciones, debug=False)

# Resultados
print("\nSolución óptima (x, y):", solucion_optima)
print("Valor óptimo:", valor_optimo)
