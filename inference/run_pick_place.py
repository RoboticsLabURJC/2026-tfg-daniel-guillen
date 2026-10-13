import time
import onnxruntime as ort
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('inference/scene/yam_lift_cube/scene.xml')
d = mujoco.MjData(m)

policy = ort.InferenceSession('inference/policies/yam_pick_place.onnx')
for entrada in policy.get_inputs():
  print(entrada.name, entrada.shape)

for salida in policy.get_outputs():
  print(salida.name, salida.shape)
  
policy_meta = policy.get_modelmeta()
meta_map = policy_meta.custom_metadata_map

action_scale = np.array(meta_map['action_scale'].split(','), dtype=float)
default_joint_pos = np.array(meta_map['default_joint_pos'].split(','), dtype=float)

mujoco.mj_resetDataKeyframe(m, d, 0)

# Colocar el cubo delante del robot, dentro de la zona del entrenamiento
# (x 0.2–0.4, y -0.2–-0.05). En qpos, después de las 8 articulaciones del brazo
# van las 3 posiciones del cubo y luego su orientación (4 números).
d.qpos[8:11] = [0.3, -0.12, 0.02]
mujoco.mj_forward(m, d)  # recalcular posiciones con el cubo ya colocado

last_action = np.zeros(7)

# Destino donde dejar el cubo, en el mundo (metros). Tiene que estar dentro
# de la zona del entrenamiento: x 0.2–0.4, y 0.05–0.2, z 0.02 (en el suelo).
target = np.array([0.3, 0.12, 0.02])

# Cámara de profundidad de la muñeca, igual que en el entrenamiento
# (src/tfg/env_cfgs.py): 32x32 píxeles y solo ve los grupos 0 y 3.
renderer = mujoco.Renderer(m, height=32, width=32)
renderer.enable_depth_rendering()
scene_option = mujoco.MjvOption()
scene_option.geomgroup[:] = [1, 0, 0, 1, 0, 0]

with mujoco.viewer.launch_passive(m, d) as viewer:

  while viewer.is_running():
    step_start = time.time()
    
    joint_pos = d.qpos[:8] - default_joint_pos
    joint_vel = d.qvel[:8]

    # Objetivo visto desde la pinza: la flecha pinza → objetivo, girada al
    # sistema de referencia de la pinza (traspuesta de su rotación).
    ee_pos = d.site('robot/grasp_site').xpos
    ee_rot = d.site('robot/grasp_site').xmat.reshape(3, 3)
    goal_position = ee_rot.T @ (target - ee_pos)

    obs = np.concatenate([joint_pos, joint_vel, last_action, goal_position])
    
    obs = obs.reshape(1, 26).astype(np.float32)
    
    # Foto de profundidad desde la cámara de la muñeca (en metros).
    renderer.update_scene(d, camera='robot/camera_d405', scene_option=scene_option)
    depth = renderer.render()

    # Igual que mjlab (camera_depth): recortar entre 0.01 y 0.5 m, dividir
    # entre 0.5 para que quede entre 0 y 1, y darle la forma [1, 1, 32, 32].
    depth = np.clip(depth, 0.01, 0.5) / 0.5
    camera = depth.reshape(1, 1, 32, 32).astype(np.float32)
    
    actions = policy.run(None, {'obs': obs, 'camera': camera})[0][0]
    last_action = actions
    
    d.ctrl[:] = actions * action_scale + default_joint_pos[:7]

    for _ in range(4):
      mujoco.mj_step(m, d)
    
    viewer.sync()
    
    # Para que el tiempo no vaya demasiado rápido y la ejecución de mujoco 
    # coincida con el tiempo real.
    time_until_next_step = 4 * m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
