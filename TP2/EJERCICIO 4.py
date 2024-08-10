#EJERCICIO 4

# Importo librerías
import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
import optuna
import contextlib # Esto es para evitar los prints excesivos
import os

# Función objetivo
def funcion_objetivo(x):
    x1, x2 = x
    eq1 = (3 * x1 + 2 * x2 - 9) ** 2  # error cuadrático de la primera ecuación
    eq2 = (x1 - 5 * x2 - 4) ** 2  # error cuadrático de la segunda ecuación
    return eq1 + eq2

# Definir la función que será optimizada por Optuna
def optimizacion(trial):
    # Sugerir valores para los hiperparámetros
    num_particulas = trial.suggest_int('num_particulas', 10, 100)
    cant_iteraciones = trial.suggest_int('cant_iteraciones', 50, 300)
    w = trial.suggest_uniform('w', 0.1, 0.9)
    c1 = trial.suggest_uniform('c1', 0.5, 3.0)
    c2 = trial.suggest_uniform('c2', 0.5, 3.0)
    lb = trial.suggest_uniform('lb', -100, 0)
    ub = trial.suggest_uniform('ub', 0, 100)

    # Límites para x1 y x2
    lb_ = [lb, lb]
    ub_ = [ub, ub]

    # Ejecutar PSO
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            best_position, best_value = pso(
                funcion_objetivo,
                lb=lb_,
                ub=ub_,
                swarmsize=num_particulas,
                maxiter=cant_iteraciones,
                omega=w,
                phip=c1,
                phig=c2
            )

    # El objetivo de la optimización es minimizar el valor de la función objetivo
    return best_value

# Crear un estudio de Optuna y optimizar
study = optuna.create_study(direction='minimize')
study.optimize(optimizacion, n_trials=200)

# Imprimir los resultados
print("Mejor valor objetivo:", study.best_value)
print("Mejores hiperparámetros:", study.best_params)

# Mejores hiperparámetros
print(study.best_params)

# Con los mejores hiperparámetros hallados, resuelvo con PSO
best_position, best_value = pso(
                funcion_objetivo,
                lb=[study.best_params['lb']]*2,
                ub=[study.best_params['ub']]*2,
                swarmsize=study.best_params['num_particulas'],
                maxiter=study.best_params['cant_iteraciones'],
                omega=study.best_params['w'],
                phip=study.best_params['c1'],
                phig=study.best_params['c2'],
                debug=False)

# Verifico las igualdades de mis dos ecuaciones
x_1 = best_position[0]
x_2 = best_position[1]

print("Verificación de igualdades:")
print(np.round(3*x_1+2*x_2, 3) == 9)
print(np.round(x_1-5*x_2, 3) == 4)