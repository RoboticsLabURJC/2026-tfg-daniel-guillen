import colorsys
from typing import Any, Callable, Literal

import mujoco

from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from tfg.lift_cube_env_cfg import make_lift_cube_env_cfg
from tfg.pick_place_env_cfg import make_pick_place_env_cfg
from mjlab.tasks.manipulation.mdp import MultiCubeLiftingCommandCfg
from tfg import mdp as tfg_mdp
from tfg.actions import GripperSpeedLimitActionCfg
from tfg.ur5e_constants import UR5E_ACTION_SCALE, get_ur5e_robot_cfg

def get_cube_spec(
  cube_size: float = 0.02,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec

def yam_lift_cube_env_cfg(
  play: bool = False,
  make_base_cfg: Callable[[], ManagerBasedRlEnvCfg] = make_lift_cube_env_cfg,
) -> ManagerBasedRlEnvCfg:
  # make_base_cfg: tarea base sobre la que se pone el YAM (lift o pick and place).
  cfg = make_base_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"
    elif sensor.name == "left_finger_cube_contact":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "lf_down"
    elif sensor.name == "right_finger_cube_contact":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "rf_down"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    # Higher command resampling frequency for more dynamic play.
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)
    # Sin currículo: usar directamente la altura final del objetivo.
    cfg.commands["lift_height"].target_position_range.z = (0.20, 0.40)

  return cfg


def yam_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  play: bool = False,
  make_base_cfg: Callable[[], ManagerBasedRlEnvCfg] = make_lift_cube_env_cfg,
) -> ManagerBasedRlEnvCfg:
  cfg = yam_lift_cube_env_cfg(play=play, make_base_cfg=make_base_cfg)

  camera_names = ["robot/camera_d405"]
  cam_kwargs = {
    "robot/camera_d405": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
      **shared_cam_kwargs,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs
    )

  camera_obs = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )
  cfg.observations["camera"] = camera_obs

  if cam_type == "rgb":
    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.0, 1.0),
      },
    )

  # Pop privileged info from actor observations.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")

  # Add goal_position to actor observations.
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
    # NOTE: No noise for goal position.
  )

  return cfg

def _cube_color(i: int, n: int) -> tuple[float, float, float, float]:
  """Generate a distinct color for cube i of n using HSV hue rotation."""
  h = i / max(n, 1)
  r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
  return (r, g, b, 1.0)


def yam_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Pick and place con el YAM y la cámara de profundidad de la muñeca.

  De momento es igual que el lift con cámara de profundidad; la tarea propia
  está en pick_place_env_cfg.py.
  """
  cfg = yam_lift_cube_vision_env_cfg(
    cam_type="depth", play=play, make_base_cfg=make_pick_place_env_cfg
  )
  
  actor_obs = cfg.observations["actor"]
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=tfg_mdp.target_position_base,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=()),
    },
    # NOTE: No noise for goal position.
  )
  
  if play:
    cfg.commands["lift_height"].target_position_range.z = (0.02, 0.02)
    cfg.commands["lift_height"].resampling_time_range = (4.0, 6.0)
  
  return cfg

def ur5e_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Pick and place con el UR5e (pinza Robotiq 2F-85 y cámara D405).

  Parte de la tarea del YAM y cambia solo lo que es propio del robot.
  """
  cfg = yam_pick_place_env_cfg(play=play)

  cfg.scene.entities["robot"] = get_ur5e_robot_cfg()
  
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # La pinza se mueve como la Robotiq 2F-85 real: a 150 mm/s como mucho
  # (85 mm de recorrido = 0.8 rad del dedo → 1.41 rad/s, se cierra en ~0.6 s).
  # Sin límite se cerraba en 0.06 s, diez veces más rápido que la real.
  cfg.actions["joint_pos"] = GripperSpeedLimitActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=joint_pos_action.actuator_names,
    scale=UR5E_ACTION_SCALE,
    use_default_offset=True,
    gripper_joint="gripper_left_driver_joint",
    max_speed=1.41,
    gripper_range=(0.0, 0.8),
  )

  fingertip_geoms = r"gripper_.*_pad[12]"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "wrist_3_link"
    elif sensor.name == "left_finger_cube_contact":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "gripper_left_pad"
    elif sensor.name == "right_finger_cube_contact":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "gripper_right_pad"

  # Penalizaciones de articulaciones: solo las que mueve la red. Las piezas
  # pasivas de la pinza (coupler, spring_link, follower) trabajan pegadas a su
  # tope y se mueven solas al cerrar; castigarlas no tiene sentido.
  arm_joints = ("shoulder_.*", "elbow_joint", "wrist_.*")
  cfg.rewards["joint_pos_limits"].params["asset_cfg"].joint_names = arm_joints
  # La pinza tampoco va en el castigo de velocidad: sus dedos giran y se
  # cierran a ~5.7 rad/s (el umbral es 0.5), así que cerrarla costaba mucho
  # más de lo que da coger el cubo y la red aprendía a no moverla. En el YAM
  # el dedo se desliza a ~0.03 m/s y nunca llegaba al umbral.
  cfg.rewards["joint_vel_hinge"].params["asset_cfg"].joint_names = arm_joints
  # Castigo por velocidad más fuerte que en el YAM: con −1 el UR5e movía las
  # articulaciones a 1–2.6 rad/s (umbral 0.5) y a veces se le caía el cubo al
  # arrancar hacia el destino. Se sube cuando ya sabe coger y llevar el cubo.
  # Con −10 seguía pareciendo brusco al llevarlo lejos (zonas grandes): −15.
  # En play no hay currículums (cfg.curriculum vacío).
  if not play:
    cfg.curriculum["joint_vel_hinge_weight"].params["stages"] = [
      {"step": 0, "weight": -0.01},
      {"step": 500 * 24, "weight": -0.1},
      {"step": 1000 * 24, "weight": -1.0},
      {"step": 2000 * 24, "weight": -3.0},
      {"step": 4000 * 24, "weight": -10.0},
      {"step": 5000 * 24, "weight": -15.0},
    ]

  # Acercarse al cubo con una campana estrecha (5 cm en vez de 20): con la ancha,
  # quedarse a 5 cm ya daba el 94 % del premio y no centraba la pinza sobre el
  # cubo, así que nunca llegaba a cogerlo.
  cfg.rewards["lift"].params["reaching_std"] = 0.05

  # Con el cubo ya en el destino, la pinza abría y cerraba los dedos ~5 veces
  # por segundo porque no le costaba nada. Ahora moverla le cuesta.
  # Tope de 16 (unos 4 rad/s, lo que mueve los dedos al bailar): al soltar,
  # los dedos siguen abriéndose a ~12 rad/s unos pasos después de dejar de
  # tocar el cubo y, sin tope, eso costaba hasta −25 en un solo paso.
  cfg.rewards["gripper_still"] = RewardTermCfg(
    func=tfg_mdp.gripper_vel_after_place,
    weight=-0.1,
    params={
      "left_sensor_name": "left_finger_cube_contact",
      "right_sensor_name": "right_finger_cube_contact",
      "object_ground_sensor_name": "cube_ground_contact",
      "command_name": "lift_height",
      "object_name": "cube",
      "std": 0.05,
      "max_value": 16.0,
      "asset_cfg": SceneEntityCfg("robot", joint_names=("gripper_left_driver_joint",)),
    },
  )

  # Zonas del cubo y del destino: la misma para los dos, dentro de lo que ve
  # la cámara de la muñeca en reposo (x 0.31–0.66, |y| 0.14 cerca y 0.19
  # lejos), para que siempre vea el cubo al empezar. Antes eran dos zonas
  # separadas (cubo a la derecha, y −0.2 a −0.05; destino a la izquierda,
  # y 0.05 a 0.2; x 0.35–0.55).
  command = cfg.commands["lift_height"]
  command.object_pose_range.x = (0.33, 0.62)
  command.object_pose_range.y = (-0.15, 0.15)
  command.target_position_range.x = (0.33, 0.62)
  command.target_position_range.y = (-0.15, 0.15)

  cfg.viewer.body_name = "base"

  return cfg


def yam_multi_cube_seg_env_cfg(
  num_cubes: int = 3,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Multi-cube task: depth + segmentation mask for goal conditioning."""
  cfg = make_lift_cube_env_cfg()

  # La recompensa de agarre en el aire está pensada para un solo cubo: se quita aquí.
  cfg.rewards.pop("grasped_in_air")
  cfg.curriculum.pop("target_height")
  assert cfg.scene.sensors is not None
  cfg.scene.sensors = tuple(
    s for s in cfg.scene.sensors if not s.name.endswith("_finger_cube_contact") 
      and s.name != "cube_ground_contact"
  )

  cube_names = [f"cube_{i}" for i in range(num_cubes)]
  entities: dict[str, EntityCfg] = {"robot": get_yam_robot_cfg()}
  for i, name in enumerate(cube_names):
    color = _cube_color(i, num_cubes)
    entities[name] = EntityCfg(
      spec_fn=lambda c=color: get_cube_spec(rgba=c),
    )
  cfg.scene.entities = entities

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.commands = {
    "lift_height": MultiCubeLiftingCommandCfg(
      entity_names=tuple(cube_names),
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
    ),
  }

  cfg.rewards["lift"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_staged_position_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "reaching_std": 0.2,
      "bringing_std": 0.3,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )
  cfg.rewards["lift_precise"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_bring_object_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "std": 0.05,
    },
  )

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 55 + num_cubes * 120)

  cam_cfg = CameraSensorCfg(
    name="camera_d405",
    camera_name="robot/camera_d405",
    height=32,
    width=32,
    data_types=("depth", "segmentation"),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  cam_terms = {
    "depth": ObservationTermCfg(
      func=manipulation_mdp.camera_depth,
      params={
        "sensor_name": "camera_d405",
        "cutoff_distance": 0.5,
      },
    ),
    "target_mask": ObservationTermCfg(
      func=manipulation_mdp.camera_target_cube_mask,
      params={
        "sensor_name": "camera_d405",
        "command_name": "lift_height",
      },
    ),
  }
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms,
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  for group_name in ("actor", "critic"):
    obs = cfg.observations[group_name]
    obs.terms.pop("ee_to_cube", None)
    obs.terms.pop("cube_to_goal", None)
    obs.terms["goal_position"] = ObservationTermCfg(
      func=manipulation_mdp.target_position,
      params={
        "command_name": "lift_height",
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (
      4.0,
      4.0,
    )

  return cfg
