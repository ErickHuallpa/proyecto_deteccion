import csv
import random

ruta_csv = 'datos/dataset.csv'
num_por_clase = 500

with open(ruta_csv, mode='w', newline='') as archivo:
    writer = csv.writer(archivo)
    writer.writerow(['left_ear', 'right_ear', 'avg_ear', 'movimiento', 'label'])

    for _ in range(num_por_clase):
        le = random.uniform(0.26, 0.32)
        re = random.uniform(0.26, 0.32)
        avg = (le + re) / 2
        movimiento = random.uniform(0.5, 2.0)
        writer.writerow([le, re, avg, movimiento, 0])

    for _ in range(num_por_clase):
        le = random.uniform(0.10, 0.17)
        re = random.uniform(0.10, 0.17)
        avg = (le + re) / 2
        movimiento = random.uniform(0.1, 1.0)
        writer.writerow([le, re, avg, movimiento, 1])

    for _ in range(num_por_clase):
        le = random.uniform(0.18, 0.23)
        re = random.uniform(0.18, 0.23)
        avg = (le + re) / 2
        movimiento = random.uniform(10.0, 25.0)
        writer.writerow([le, re, avg, movimiento, 2])

print(f"Dataset simulado guardado en {ruta_csv}")
