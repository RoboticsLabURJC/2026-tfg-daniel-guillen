"""Funciones propias de la tarea (recompensas, observaciones...) que no trae mjlab."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
)

from mjlab.utils.lab_api.math import quat_apply, quat_inv


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


def target_height_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[tuple[int, tuple[float, float]]],
) -> dict[str, torch.Tensor]:
  """Sube la altura del objetivo según la iteración del entrenamiento.

  Cada etapa es (iteración en la que empieza, (altura mínima, altura máxima)).
  """
  iteration = env.common_step_counter // 24  # 24 pasos por iteración (num_steps_per_env)

  height = stages[0][1]
  for start_iteration, stage_height in stages:
    if iteration >= start_iteration:
      height = stage_height

  command = env.command_manager.get_term(command_name)
  command.cfg.target_position_range.z = height
  return {"target_height_max": torch.tensor(height[1])}


def released_at_target(
  env: ManagerBasedRlEnv,
  left_sensor_name: str,
  right_sensor_name: str,
  object_ground_sensor_name: str,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  """0-1: objeto cerca del objetivo, en el suelo y sin agarrar con los dos dedos."""
  released = 1 - finger_contact_grasp(env, left_sensor_name, right_sensor_name)
  ground: ContactSensor = env.scene[object_ground_sensor_name]
  on_ground = (ground.data.found > 0).any(dim=-1)
  obj = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  # Distancia al cuadrado cubo-objetivo (x² + y² + z²) y premio gaussiano:
  # 1 en el objetivo, 0.37 a una distancia std, casi 0 a 2·std.
  distance2 = torch.sum(
    torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
  )
  near = torch.exp(-distance2 / std**2)
  return near * released * on_ground.float()


def target_position_base(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Target position in EE frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (LiftingCommand, MultiCubeLiftingCommand)):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand or "
      f"MultiCubeLiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.root_link_pos_w
  ee_quat_w = robot.data.root_link_quat_w
  target_pos_w = command.target_pos
  target_pos_rel_w = target_pos_w - ee_pos_w
  target_pos_ee = quat_apply(quat_inv(ee_quat_w), target_pos_rel_w)
  return target_pos_ee
