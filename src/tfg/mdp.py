"""Funciones propias de la tarea (recompensas, observaciones...) que no trae mjlab."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def finger_contact_grasp(
  env: ManagerBasedRlEnv,
  left_sensor_name: str,
  right_sensor_name: str,
) -> torch.Tensor:
  """1 si los dos dedos de la pinza tocan el objeto a la vez, 0 si no.

  Usa dos sensores de contacto (dedo izquierdo-objeto y dedo derecho-objeto).
  Exigir los dos dedos evita premiar empujar el objeto con un solo dedo.
  """
  left: ContactSensor = env.scene[left_sensor_name]
  right: ContactSensor = env.scene[right_sensor_name]
  assert left.data.found is not None and right.data.found is not None
  left_touch = (left.data.found > 0).any(dim=-1)
  right_touch = (right.data.found > 0).any(dim=-1)
  return (left_touch & right_touch).float()

def grasped_in_air(
  env: ManagerBasedRlEnv,
  left_sensor_name: str,
  right_sensor_name: str,
  object_ground_sensor_name: str,
) -> torch.Tensor:
  """1 si el objeto está agarrado con los dos dedos y no toca la mesa, 0 si no."""
  grasped = finger_contact_grasp(env, left_sensor_name, right_sensor_name)
  ground: ContactSensor = env.scene[object_ground_sensor_name]
  on_ground = (ground.data.found > 0).any(dim=-1)
  return grasped * (~on_ground).float()
