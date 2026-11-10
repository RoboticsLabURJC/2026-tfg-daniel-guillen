"""Recoge un dataset de imágenes de la cámara de la muñeca con el giro del cubo.

Varias redes mueven 1024 robots y, cada pocos pasos, se guarda lo que ve la
cámara de cada robot junto con el giro exacto del cubo respecto a la pinza (lo
sabe la simulación). Ese dataset sirve para entrenar una red pequeña que
adivine el giro mirando solo la imagen.

Uso: uv run python scripts/recoger_dataset.py
"""
import os
from dataclasses import asdict

import torch

import tfg  # noqa: F401  (al importarlo se registran las tareas en mjlab)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from tfg import mdp as tfg_mdp

TAREA = "Tfg-Pick-Place-Cube-Ur5e-Depth"
# Redes que mueven el brazo: las de un entrenamiento con visión a medio
# aprender, que llegan al cubo de formas distintas (descentrado, tapado por un
# dedo...). Las fotos del robot del día 15, que llega "limpio", están en
# datasets/giro_dia15.pt.
REDES = [
  "logs/rsl_rl/ur5e_pick_place/2026-11-08_13-55-15/model_1000.pt",
  "logs/rsl_rl/ur5e_pick_place/2026-11-08_13-55-15/model_2000.pt",
  "logs/rsl_rl/ur5e_pick_place/2026-11-08_13-55-15/model_3000.pt",
  "logs/rsl_rl/ur5e_pick_place/2026-11-08_13-55-15/model_4400.pt",
]
RUIDO = 0.3           # empujones al azar en las acciones de la mitad de los robots
N = 1024              # robots a la vez
SEGUNDOS = 30         # tiempo simulado con cada red
CADA = 5              # guardar 1 de cada 5 pasos (los seguidos casi no cambian)
# De cerca (< 5 cm) hay muchísimas fotos y casi iguales (cubo ya cogido o justo
# debajo): solo se guarda 1 de cada 4. Las de 5 cm o más se guardan todas.
CERCA = 0.05
GUARDAR_CERCA = 0.25
SALIDA = "datasets/giro_vision.pt"

# 1. Configuración de la tarea. Sin play=True: en play el episodio no se acaba
# nunca y el cubo no vuelve a salir con otro giro al azar.
cfg = load_env_cfg(TAREA)
cfg.scene.num_envs = N
# Episodios cortos: cada 3 s el cubo vuelve a salir en otro sitio y con otro
# giro, así hay muchas imágenes acercándose al cubo (lo que interesa).
cfg.episode_length_s = 3.0

# 2. Crear la simulación y adaptarla a RSL-RL.
cfg_rl = load_rl_cfg(TAREA)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env_rl = RslRlVecEnvWrapper(env, clip_actions=cfg_rl.clip_actions)
runner = load_runner_cls(TAREA)(env_rl, asdict(cfg_rl), device="cuda:0")

# Para calcular el giro con la misma función que usa el crítico (gripper_cube_yaw).
pinza = SceneEntityCfg("robot", site_names=("grasp_site",))
pinza.resolve(env.scene)  # traduce el nombre "grasp_site" a su número
robot = env.scene["robot"]
cubo = env.scene["cube"]

imagenes, giros, distancias, robots = [], [], [], []
numero_robot = torch.arange(N, device="cuda:0")
con_ruido = (numero_robot % 2 == 0).unsqueeze(1)  # la mitad de los robots: (N, 1)
obs = env_rl.get_observations()

for red in REDES:
  # 3. Cargar la red que mueve los robots (solo el actor).
  runner.load(red, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  politica = runner.get_inference_policy(device="cuda:0")
  print("moviendo con", red)

  # 4. Mover los robots y guardar.
  with torch.no_grad():  # solo miramos, no se entrena nada
    for paso in range(int(SEGUNDOS / env.step_dt)):
      acciones = politica(obs)
      acciones = acciones + con_ruido * RUIDO * torch.randn_like(acciones)
      obs, _, _, _ = env_rl.step(acciones)
      if paso % CADA != 0:
        continue
      # Distancia pinza–cubo: para elegir qué fotos guardar y para medir
      # después en qué distancias acierta la red.
      pos_pinza = robot.data.site_pos_w[:, pinza.site_ids[0]]
      distancia = (pos_pinza - cubo.data.root_link_pos_w).norm(dim=-1)
      guardar = (distancia >= CERCA) | (torch.rand(N, device="cuda:0") < GUARDAR_CERCA)
      imagenes.append(obs["camera"][guardar].half())  # (n, 1, 32, 32); half = mitad de memoria
      giros.append(tfg_mdp.gripper_cube_yaw(env, "cube", pinza)[guardar])  # (n, 2): sin y cos de 4·diferencia
      distancias.append(distancia[guardar])
      robots.append(numero_robot[guardar])  # de qué robot es cada foto (para separar el test)

# 5. Juntar todas las tandas y guardar.
datos = {
  "imagenes": torch.cat(imagenes).cpu(),
  "giros": torch.cat(giros).cpu(),
  "distancias": torch.cat(distancias).cpu(),
  "robots": torch.cat(robots).cpu(),
}
for nombre, valor in datos.items():
  print(nombre, tuple(valor.shape))
os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
torch.save(datos, SALIDA)
print("guardado en", SALIDA)
