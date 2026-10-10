import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('inference/scene/yam_lift_cube/scene.xml')
d = mujoco.MjData(m)

mujoco.mj_resetDataKeyframe(m, d, 0)

with mujoco.viewer.launch_passive(m, d) as viewer:

  while viewer.is_running():
    step_start = time.time()

    # Step the physics.
    mujoco.mj_step(m, d)
    
    viewer.sync()
    
    # Para que el tiempo no vaya demasiado rápido y la ejecución de mujoco 
    # coincida con el tiempo real.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
