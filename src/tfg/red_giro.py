"""Red pequeña que adivina el giro del cubo respecto a la pinza mirando la
imagen de profundidad de la cámara de la muñeca.

Basada en el tutorial oficial de PyTorch "Learn the Basics" (Build the Neural
Network): https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
Cambios respecto al tutorial: capas de convolución para la imagen (como la CNN
del actor) y 2 salidas (sin y cos de 4 veces el giro) en vez de 10 clases.
"""
from torch import nn


class RedGiro(nn.Module):
  def __init__(self):
    super().__init__()
    # Parte que mira la imagen: (1, 32, 32) -> (16, 16, 16) -> (32, 8, 8).
    self.convoluciones = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
      nn.ELU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
      nn.ELU(),
    )
    self.flatten = nn.Flatten()  # (32, 8, 8) -> 2048 números en fila
    # Parte que decide el giro a partir de lo que ha visto.
    self.lineales = nn.Sequential(
      nn.Linear(32 * 8 * 8, 128),
      nn.ELU(),
      nn.Linear(128, 2),  # sin y cos de 4 veces el giro
    )

  def forward(self, x):
    x = self.convoluciones(x)
    x = self.flatten(x)
    return self.lineales(x)
