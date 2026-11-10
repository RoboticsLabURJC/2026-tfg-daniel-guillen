"""Entrena la red que adivina el giro del cubo (src/tfg/red_giro.py) con el
dataset de scripts/recoger_dataset.py.

Basado en el tutorial oficial de PyTorch "Learn the Basics":
https://pytorch.org/tutorials/beginner/basics/intro.html

Uso: uv run python scripts/entrenar_red_giro.py
"""
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from tfg.red_giro import RedGiro

# Fotos con el robot del día 15 (llega "limpio") y con redes a medio aprender.
DATASETS = ["datasets/giro_dia15.pt", "datasets/giro_vision.pt"]
RED = "datasets/red_giro.pt"
FIGURA = "datasets/ejemplos_dataset.png"
device = "cuda" if torch.cuda.is_available() else "cpu"


def a_grados(giro):
  """De (sin, cos) de 4 veces el giro a grados, de -45° a 45°."""
  return torch.rad2deg(torch.atan2(giro[..., 0], giro[..., 1]) / 4)

# Cargar los dos ficheros y juntarlos.
datos = [torch.load(fichero) for fichero in DATASETS]
# Transformada: se guardan en half (media precisión) para ocupar menos.
imagenes = torch.cat([d["imagenes"] for d in datos]).float()  # Imágenes
giros = torch.cat([d["giros"] for d in datos])                # Etiqueta
distancias = torch.cat([d["distancias"] for d in datos])      # Distancia
robot = torch.cat([d["robots"] for d in datos])               # de qué robot es cada foto

# Separar entrenamiento y test por robots (no por fotos sueltas): las fotos
# seguidas de un mismo robot son casi iguales.
es_test = robot % 5 == 0  # 1 de cada 5 robots para el test

# TensorDataset junta cada imagen con su etiqueta: dataset[i] = (imagen, giro).
training_data = TensorDataset(imagenes[~es_test], giros[~es_test])
test_data = TensorDataset(imagenes[es_test], giros[es_test])
test_distancias = distancias[es_test]
train_distancias = distancias[~es_test]
print(f"entrenamiento: {len(training_data)} imágenes, test: {len(test_data)}")

# Ver el dataset (figura para el blog)

figure = plt.figure(figsize=(8, 8))
rangos_figura = [(0, 0.05), (0.05, 0.10), (0.10, 10)]  # una fila por rango de distancia
for fila, (cerca, lejos) in enumerate(rangos_figura):
  en_rango = ((train_distancias >= cerca) & (train_distancias < lejos)).nonzero().squeeze(1)
  for col in range(3):
    sample_idx = en_rango[torch.randint(len(en_rango), size=(1,))].item()
    img, giro = training_data[sample_idx]
    figure.add_subplot(3, 3, fila * 3 + col + 1)
    plt.title(f"giro {a_grados(giro):.0f}°, a {train_distancias[sample_idx] * 100:.0f} cm")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray", interpolation="nearest")
plt.tight_layout()
plt.savefig(FIGURA, dpi=150)
print("figura guardada en", FIGURA)

# DataLoader: reparte el dataset en tandas de 512 y las baraja
batch_size = 512

RANGOS = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 10)]
PROPORCION = [0.40, 0.30, 0.20, 0.10]
pesos = torch.zeros(len(training_data))
for (cerca, lejos), proporcion in zip(RANGOS, PROPORCION):
  m = (train_distancias >= cerca) & (train_distancias < lejos)
  pesos[m] = proporcion / m.sum()  # cada foto del rango: su parte entre las que hay
  print(f"  {cerca * 100:.0f}–{lejos * 100:.0f} cm: {int(m.sum())} fotos de entrenamiento, "
        f"{proporcion:.0%} de cada tanda")
# El sampler elige qué fotos van en cada tanda según su peso (sustituye a shuffle).
sampler = WeightedRandomSampler(pesos, num_samples=len(training_data), replacement=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=sampler)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# La red, la pérdida y el optimizador
model = RedGiro().to(device)
# Regresión (2 números), no clasificación: error cuadrático medio en vez de
# CrossEntropyLoss como en el tutorial.
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_loop(dataloader, model, loss_fn, optimizer):
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    pred = model(X)          # lo que dice la red
    loss = loss_fn(pred, y)  # cuánto se equivoca

    loss.backward()          # autograd: cómo cambiar cada peso para equivocarse menos
    optimizer.step()         # cambiar los pesos
    optimizer.zero_grad()    # borrar esos cálculos para la siguiente tanda

    if batch % 100 == 0:
      print(f"  pérdida: {loss.item():.4f}  [{batch * batch_size:>6d}/{len(dataloader.dataset)}]")


def test_loop(dataloader, model):
  """Error en grados de cada imagen de test (de 0° a 45°)."""
  model.eval()
  errores = []
  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X.to(device)).cpu()
      diferencia = a_grados(pred) - a_grados(y)
      # El cubo es igual cada 90°: -44° y 44° están a 2°, no a 88°.
      diferencia = torch.remainder(diferencia + 45, 90) - 45
      errores.append(diferencia.abs())
  return torch.cat(errores)

# Entrenar

epochs = 30
for t in range(epochs):
  print(f"Época {t + 1}")
  train_loop(train_dataloader, model, loss_fn, optimizer)
  error = test_loop(test_dataloader, model)
  print(f"  test: error medio {error.mean():.1f}°, mediana {error.median():.1f}°")

# Error según lo lejos que está la pinza del cubo (al azar sería 22,5° de media).
print("\nError en el test según la distancia pinza–cubo:")
for cerca, lejos in RANGOS:
  m = (test_distancias >= cerca) & (test_distancias < lejos)
  print(f"  {cerca * 100:.0f}–{lejos * 100:.0f} cm: mediana {error[m].median():.1f}°, "
        f"media {error[m].mean():.1f}°, {int(m.sum())} imágenes")

# Guardar

torch.save(model.state_dict(), RED)
print("red guardada en", RED)
