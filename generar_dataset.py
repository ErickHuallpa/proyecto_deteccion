import csv
import random
import numpy as np

ruta_csv = 'datos/dataset.csv'
num_por_clase = 500

def ruido(valor, sigma=0.02):
    return max(0, min(1, np.random.normal(valor, sigma)))

with open(ruta_csv, mode='w', newline='') as archivo:
    writer = csv.writer(archivo)
    writer.writerow(['left_ear', 'right_ear', 'avg_ear', 'movimiento', 'label'])

    for _ in range(num_por_clase):
        le = ruido(random.uniform(0.10, 0.20), 0.03)
        re = ruido(random.uniform(0.10, 0.20), 0.03)
        avg = (le + re) / 2
        movimiento = max(0, random.gauss(1.5, 0.7))
        writer.writerow([le, re, avg, movimiento, 0])

    for _ in range(num_por_clase):
        le = ruido(random.uniform(0.08, 0.15), 0.02)
        re = ruido(random.uniform(0.08, 0.15), 0.02)
        avg = (le + re) / 2
        movimiento = max(0, random.gauss(0.1, 0.05))
        writer.writerow([le, re, avg, movimiento, 1])

print(f"Dataset simulado menos perfecto guardado en {ruta_csv}")
