"""Funciones propias de la tarea (recompensas, observaciones...) que no trae mjlab."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.metrics_manager import MetricsTermCfg
  from mjlab.managers.reward_manager import RewardTermCfg
  
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
  command_name: str,
  object_name: str,
  min_height: float,
) -> torch.Tensor:
  """0-1 si el objeto está agarrado con los dos dedos: crece con la altura sobre
  el destino y llega a 1 a min_height por encima. 0 si no está agarrado."""
  grasped = finger_contact_grasp(env, left_sensor_name, right_sensor_name)
  obj = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  # Cuánto ha subido el cubo sobre el destino, como fracción de min_height
  # (0 a su altura, 1 a min_height por encima). clamp lo deja entre 0 y 1.
  height = obj.data.root_link_pos_w[:, 2] - command.target_pos[:, 2]
  high_enough = torch.clamp(height / min_height, 0.0, 1.0)
  return grasped * high_enough


def grasped_off_ground(
  env: ManagerBasedRlEnv,
  left_sensor_name: str,
  right_sensor_name: str,
  object_ground_sensor_name: str,
) -> torch.Tensor:
  """1 si el objeto está agarrado con los dos dedos y no toca la mesa, 0 si no.

  Versión fácil de grasped_in_air (la primera que tuvo el YAM, commit 3256aba):
  paga en cuanto el objeto se despega de la mesa, sin mirar la altura.
  """
  grasped = finger_contact_grasp(env, left_sensor_name, right_sensor_name)
  ground: ContactSensor = env.scene[object_ground_sensor_name]
  on_ground = (ground.data.found > 0).any(dim=-1)
  return grasped * (~on_ground).float()


def grasp_centered(
  env: ManagerBasedRlEnv,
  left_sensor_name: str,
  right_sensor_name: str,
  object_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """0-1: objeto agarrado con los dos dedos y centrado entre las yemas.

  Mide la distancia en horizontal (x, y) entre el centro de las yemas
  (asset_cfg.geom_names) y el centro del objeto: 1 centrado, 0.37 a `std`.
  Agarrarlo cerca del borde hace que se escurra al levantarlo.
  """
  grasped = finger_contact_grasp(env, left_sensor_name, right_sensor_name)
  robot: Entity = env.scene[asset_cfg.name]
  pads = robot.data.geom_pos_w[:, asset_cfg.geom_ids].mean(dim=1)
  obj = env.scene[object_name].data.root_link_pos_w
  distance2 = torch.sum(torch.square(pads[:, :2] - obj[:, :2]), dim=-1)
  return grasped * torch.exp(-distance2 / std**2)


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


def gripper_vel_after_place(
  env: ManagerBasedRlEnv,
  left_sensor_name: str,
  right_sensor_name: str,
  object_ground_sensor_name: str,
  command_name: str,
  object_name: str,
  std: float,
  max_value: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Velocidad² de la pinza (como mucho max_value), con el cubo ya soltado."""
  placed = released_at_target(
    env, left_sensor_name, right_sensor_name, object_ground_sensor_name,
    command_name, object_name, std,
  )
  robot: Entity = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  vel2 = torch.sum(torch.square(vel), dim=-1)
  return placed * torch.clamp(vel2, max=max_value)


def bring_object_xy_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  """0-1: objeto cerca del objetivo solo en horizontal (x, y), sin mirar la altura.

  Como bring_object_reward de mjlab, pero así no castiga llevar el cubo alto.
  """
  obj = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  # Distancia al cuadrado cubo-objetivo solo en el plano (x² + y²).
  distance2 = torch.sum(
    torch.square(command.target_pos[:, :2] - obj.data.root_link_pos_w[:, :2]), dim=-1
  )
  return torch.exp(-distance2 / std**2)


class released_after_lift:
  """released_at_target, pero solo si el cubo ha estado levantado antes.

  Cada robot tiene una casilla "lo he levantado": se marca cuando tiene el cubo
  agarrado y despegado de la mesa (grasped_off_ground) y se vacía al empezar un
  episodio o cuando el comando pone un cubo y un destino nuevos. Así empujar el
  cubo hasta el destino no da puntos: hay que cogerlo, llevarlo y soltarlo.
  Hace falta una clase (y no una función) para guardar la casilla de un paso a
  otro; mjlab llama a reset() al empezar cada episodio.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg  # No se usa.
    self.lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Último destino visto: si cambia, el comando ha recolocado el cubo.
    self.last_target = torch.zeros(env.num_envs, 3, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_sensor_name: str,
    right_sensor_name: str,
    object_ground_sensor_name: str,
    command_name: str,
    object_name: str,
    std: float,
  ) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    new_target = (command.target_pos != self.last_target).any(dim=-1)
    self.lifted[new_target] = False
    self.last_target[:] = command.target_pos

    lifted_now = grasped_off_ground(
      env, left_sensor_name, right_sensor_name, object_ground_sensor_name
    )
    self.lifted |= lifted_now > 0

    return self.lifted * released_at_target(
      env, left_sensor_name, right_sensor_name, object_ground_sensor_name,
      command_name, object_name, std,
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self.lifted[env_ids] = False


class task_success_rate:
  """Métrica: fracción de encargos terminados con el cubo bien colocado.

  Un encargo termina cuando el comando cambia el destino de sitio. Justo
  antes, se mira si el cubo estaba soltado, apoyado y a menos de
  `threshold` del destino. El último encargo del episodio no cuenta porque
  se corta a medias. Solo mide: no cambia el entrenamiento.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    del cfg  # No se usa.
    n = env.num_envs
    self.completed = torch.zeros(n, device=env.device)  # Encargos terminados.
    self.successes = torch.zeros(n, device=env.device)  # De ellos, bien hechos.
    self.was_placed = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.last_target = torch.zeros(n, 3, device=env.device)
    # Tras un reset el destino cambia, pero no termina ningún encargo.
    self.fresh = torch.ones(n, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_sensor_name: str,
    right_sensor_name: str,
    object_ground_sensor_name: str,
    command_name: str,
    object_name: str,
    threshold: float,
  ) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    new_target = (command.target_pos != self.last_target).any(dim=-1)
    finished = new_target & ~self.fresh
    self.completed += finished.float()
    self.successes += (finished & self.was_placed).float()
    self.fresh &= ~new_target
    self.last_target[:] = command.target_pos

    # ¿Está ahora el cubo bien colocado? Se usa si el encargo acaba en el
    # siguiente paso (al cambiar de encargo el cubo se mueve a su nuevo sitio).
    grasped = finger_contact_grasp(env, left_sensor_name, right_sensor_name) > 0
    ground: ContactSensor = env.scene[object_ground_sensor_name]
    on_ground = (ground.data.found > 0).any(dim=-1)
    error = torch.norm(
      command.target_pos - env.scene[object_name].data.root_link_pos_w, dim=-1
    )
    self.was_placed = (error < threshold) & on_ground & ~grasped

    return self.successes / torch.clamp(self.completed, min=1.0)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.completed[env_ids] = 0.0
    self.successes[env_ids] = 0.0
    self.was_placed[env_ids] = False
    self.fresh[env_ids] = True


def target_position_base(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Target position in robot base (root link) frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (LiftingCommand, MultiCubeLiftingCommand)):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand or "
      f"MultiCubeLiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  root_link_pos_w = robot.data.root_link_pos_w
  root_link_quat_w = robot.data.root_link_quat_w
  target_pos_w = command.target_pos
  target_pos_rel_w = target_pos_w - root_link_pos_w
  target_pos_root_link = quat_apply(quat_inv(root_link_quat_w), target_pos_rel_w)
  return target_pos_root_link
