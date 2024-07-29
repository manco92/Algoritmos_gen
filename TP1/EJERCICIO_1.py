## EJERCICIO 1

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

np.random.seed(22)
random.seed(22)

LONGITUD_CROMOSOMA = 5
p = 0.09

pre_A = []
post_A = []
B = np.random.rand(20)
mat_chromosomes = np.random.randint(2, size=(20, LONGITUD_CROMOSOMA))
for i, chromosome in enumerate(mat_chromosomes):
    pre_A.append(''.join(map(str, chromosome)))
    # Si la probabilidad es menor a p, elijo aleatoriamente un alelo del cromosoma y lo cambio.
    if B[i] < p:
        pos = np.random.randint(LONGITUD_CROMOSOMA)
        chromosome[pos] = 0 if chromosome[pos]==1 else 1
        print(f"Se cambió el alelo {pos} del cromosoma {i}")
    post_A.append(''.join(map(str, chromosome)))

# Muestro los cromosomas A antes y después de la mutación.
probs = [f"{b:0.4f}" for b in B]
print("-----")
print("Probabilidades:")
print(probs)
print("-----")
print("Antes de la mutación:")
print(pre_A)
print("-----")
print("Después de la mutación:")
print(post_A)