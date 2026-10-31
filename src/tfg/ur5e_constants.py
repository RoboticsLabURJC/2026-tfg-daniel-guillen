"""ur5e constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

UR5E_XML: Path = (
  Path(__file__).parents[2] / "models" / "ur5e_2f85" / "ur5e_2f85.xml"
)
assert UR5E_XML.exists()

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(UR5E_XML))
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for key in list(spec.keys):
    spec.delete(key)
  return spec

##
# Actuator config.
##

EFFECTIVE_INERTIAS = {
  "shoulder_pan_joint": 1.0321,
  "shoulder_lift_joint": 1.8142,
  "elbow_joint": 1.0851,
  "wrist_1_joint": 0.1565,
  "wrist_2_joint": 0.1292,
  "wrist_3_joint": 0.1011,
}

# Armature de las articulaciones del UR5e en el XML de Menagerie.
ARMATURE_UR5E = 0.1

# Par máximo: forcerange del XML de Menagerie (clases size3 y size1).
UR5E_SIZE3 = ElectricActuator(
  reflected_inertia=ARMATURE_UR5E,
  velocity_limit=3.14,
  effort_limit=150.0,
)
UR5E_SIZE1 = ElectricActuator(
  reflected_inertia=ARMATURE_UR5E,
  velocity_limit=3.14,
  effort_limit=28.0,
)

NATURAL_FREQ = 2 * 2.0 * 3.1415926535  # 2Hz
DAMPING_RATIO = 2.0

# Per-joint PD gains using effective inertia, and actuator configs.
_ARM_JOINTS: dict[str, ElectricActuator] = {
  "shoulder_pan_joint": UR5E_SIZE3,
  "shoulder_lift_joint": UR5E_SIZE3,
  "elbow_joint": UR5E_SIZE3,
  "wrist_1_joint": UR5E_SIZE1,
  "wrist_2_joint": UR5E_SIZE1,
  "wrist_3_joint": UR5E_SIZE1,
}

ARM_ACTUATORS = tuple(
  BuiltinPositionActuatorCfg(
    target_names_expr=(name,),
    stiffness=EFFECTIVE_INERTIAS[name] * NATURAL_FREQ**2,
    damping=2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS[name] * NATURAL_FREQ,
    effort_limit=motor.effort_limit,
    armature=motor.reflected_inertia,
  )
  for name, motor in _ARM_JOINTS.items()
)

##
# Gripper actuator.
##

# Solo se mueve el dedo izquierdo; el derecho lo sigue por la <equality> del
# XML de la pinza (como right_finger en el YAM). Mismos kp, kv y fuerza que el
# motor original de Robotiq en Menagerie (2f85.xml). 0 = abierta, 0.8 = cerrada.
GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("gripper_left_driver_joint",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=5.0,
)

##
# Keyframe config.
##

# Pinza mirando hacia abajo, con grasp_site en (0.45, 0, 0.20)
# respecto a la base (como el YAM, que la tiene a 0.20 de altura).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "shoulder_pan_joint": 2.8392,
    "shoulder_lift_joint": -1.6602,
    "elbow_joint": 2.0153,
    "wrist_1_joint": -1.9259,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": -0.3024,
    # Pinza a medias (como el YAM): hueco de 5.5 cm entre yemas para un cubo de
    # 4 cm. Valores de las piezas del mecanismo con el dedo motor en 0.3,
    # medidos en simulación, para que empiece en una postura que encaja.
    "gripper_.*_driver_joint": 0.3,
    "gripper_.*_spring_link_joint": 0.3,
    "gripper_.*_follower_joint": -0.29,
    "gripper_.*_coupler_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", "gripper_.*_pad[12]"),
  contype={
    "(wrist_3|gripper|camera)_.*": 1,
    ".*": 0,
  },
  conaffinity={
    "(wrist_3|gripper|camera)_.*": 1,
    ".*": 0,
  },
  condim={
    "gripper_.*_pad[12]": 6,
    ".*": 3,
  },
  priority={
    "gripper_.*_pad[12]": 1,
    ".*": 0,
  },
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(*ARM_ACTUATORS, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)


def get_ur5e_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


UR5E_ACTION_SCALE: dict[str, float] = {}
for a in ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    UR5E_ACTION_SCALE[n] = 0.25 * e / s
# La pinza aparte: con la fórmula saldría 0.25 * 5 / 100 = 0.0125 rad y no
# podría moverse. Objetivo = 0.3 (inicial) + 0.5 * acción: 0 = a medias,
# -0.6 = abierta del todo, +1 = cerrada (llega al tope de 0.8). Antes empezaba
# abierta con escala 0.8: las acciones negativas no hacían nada y la red
# aprendió a dejarla abierta y empujar el cubo.
UR5E_ACTION_SCALE["gripper_left_driver_joint"] = 0.5


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_ur5e_robot_cfg())

  viewer.launch(robot.spec.compile())
