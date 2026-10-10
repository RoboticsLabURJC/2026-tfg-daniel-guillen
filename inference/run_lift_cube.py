import time
import onnxruntime as ort
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('inference/scene/yam_lift_cube/scene.xml')
d = mujoco.MjData(m)

policy = ort.InferenceSession('inference/policies/yam_lift_cube.onnx')
for entrada in policy.get_inputs():
  print(entrada.name, entrada.shape)

for salida in policy.get_outputs():
  print(salida.name, salida.shape)
  
policy_meta = policy.get_modelmeta()
meta_map = policy_meta.custom_metadata_map

action_scale = np.array(meta_map['action_scale'].split(','), dtype=float)
default_joint_pos = np.array(meta_map['default_joint_pos'].split(','), dtype=float)

mujoco.mj_resetDataKeyframe(m, d, 0)

last_action = np.zeros(7)

with mujoco.viewer.launch_passive(m, d) as viewer:

  while viewer.is_running():
    step_start = time.time()
    
    joint_pos = d.qpos[:8] - default_joint_pos
    joint_vel = d.qvel[:8]
    
    obs = np.concatenate([joint_pos, joint_vel, last_action])
    print(obs.shape)

    # Step the physics.
    mujoco.mj_step(m, d)
    
    viewer.sync()
    
    # Para que el tiempo no vaya demasiado rápido y la ejecución de mujoco 
    # coincida con el tiempo real.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
