from gym.envs.registration import register
#from gym.scoreboard.registration import add_task, add_group

import os
import os.path as osp
import subprocess

register(
    id='QuadruppedWalk-v1',
    entry_point='quadruppedEnv:QuadruppedWalker',
    max_episode_steps=10000,
    reward_threshold=2500.0,
    tags={ "pg_complexity": 8*1000000 },
    )

from roboschool.gym_pendulums import RoboschoolInvertedPendulum
from roboschool.gym_pendulums import RoboschoolInvertedPendulumSwingup
from roboschool.gym_pendulums import RoboschoolInvertedDoublePendulum
from roboschool.gym_reacher import RoboschoolReacher
from roboschool.gym_mujoco_walkers import RoboschoolHopper
from roboschool.gym_mujoco_walkers import RoboschoolWalker2d
from roboschool.gym_mujoco_walkers import RoboschoolHalfCheetah
from roboschool.gym_mujoco_walkers import RoboschoolAnt
from roboschool.gym_mujoco_walkers import RoboschoolHumanoid
from roboschool.gym_humanoid_flagrun import RoboschoolHumanoidFlagrun
from roboschool.gym_humanoid_flagrun import RoboschoolHumanoidFlagrunHarder
from roboschool.gym_atlas import RoboschoolAtlasForwardWalk
from roboschool.gym_pong import RoboschoolPong
from quadruppedEnv.robot import QuadruppedWalker

