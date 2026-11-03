"""Acciones propias de la tarea que no trae mjlab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class GripperSpeedLimitActionCfg(JointPositionActionCfg):
  """Como JointPositionActionCfg, pero la pinza se mueve como la real: va hacia
  la posición pedida a una velocidad máxima."""

  gripper_joint: str
  """Articulación que mueve la red para abrir y cerrar la pinza."""
  max_speed: float
  """Velocidad máxima del objetivo de la pinza (rad/s)."""
  gripper_range: tuple[float, float]
  """Posición de la pinza abierta y cerrada (rad)."""

  def build(self, env: ManagerBasedRlEnv) -> GripperSpeedLimitAction:
    return GripperSpeedLimitAction(self, env)


class GripperSpeedLimitAction(JointPositionAction):
  cfg: GripperSpeedLimitActionCfg

  def __init__(self, cfg: GripperSpeedLimitActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._gripper = self._target_names.index(cfg.gripper_joint)  # Columna de la pinza.
    self._max_step = cfg.max_speed * env.step_dt  # Lo que puede avanzar en un paso.
    self._gripper_target = torch.zeros(self.num_envs, device=self.device)

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)  # Objetivos normales: 0.3 + 0.5 * acción, etc.
    # Lo que pide la red para la pinza, sin salirse de su recorrido.
    desired = self._processed_actions[:, self._gripper].clamp(*self.cfg.gripper_range)
    # El objetivo de la pinza se acerca a lo pedido, como mucho max_step por paso.
    step = (desired - self._gripper_target).clamp(-self._max_step, self._max_step)
    self._gripper_target += step
    self._processed_actions[:, self._gripper] = self._gripper_target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)  # Todos los robots.
    # Al empezar un episodio, el objetivo de la pinza empieza donde está la pinza.
    joint_id = self._target_ids[self._gripper]
    self._gripper_target[env_ids] = self._entity.data.joint_pos[env_ids, joint_id]
