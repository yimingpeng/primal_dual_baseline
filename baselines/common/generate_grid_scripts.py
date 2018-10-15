#!/usr/bin/env python

"""Description:
"""
__author__ = "Yiming Peng"
__copyright__ = "Copyright 2018, baselines"
__credits__ = ["Yiming Peng"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Yiming Peng"
__email__ = "yiming.peng@ecs.vuw.ac.nz"
__status__ = "Prototype"

# Scripts for generating GCP startup scripts
import os

person = 'yimingpeng'
#person = 'achen'

if person == 'achen':
    f = open("../../grid_scripts/aaron_template.sh")
    f2 = open("../../grid_scripts/run_aaron_grid_ex_template.sh")
else:
    f = open("../../grid_scripts/template.sh")
    f2 = open("../../grid_scripts/run_grid_ex_template.sh")
# algorithms = ["ppo", "ppo_rac", "ppo_nac_fisher",
#               "ppo_nac_advantage_fisher", "ppo_nac_advantage",
#               "ppo_dual_rac", "ppo_dual_nac_fisher", "ppo_dual_nac_advantage"]
algorithms = ["ars", "dual_nac_advantage", "dual_nac_fisher",
              "dual_rac", "nac_advantage",
              "nac_advantage_fisher", "nac_fisher", "ppo_linear", "rac"]
# bullet_problems = ["HalfCheetah", "Hopper", "InvertedDoublePendulum",
#                    "InvertedPendulum", "InvertedPendulumSwingup", "Reacher",
#                    "Walker2D"]
bullet_problems = ["InvertedDoublePendulum",
                   "InvertedPendulum", "InvertedPendulumSwingup"]
gym_problems = ["MountainCarContinuous"]
seeds = range(5)
# Generate for Bullet problems
for algorithm in algorithms:
    for problem in bullet_problems:
        directory = "../../grid_scripts/" + str(algorithm)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f1 = open(directory + "/" + algorithm + "_" +
                  problem + ".sh", 'w')
        for line in f:
            # if 'source activate cmaes_baselines' in line and algorithm == "DDPG":
            #         line = line.replace("cmaes_baselines", "ddpg_baselines")
            if 'pyName="run_pybullet.py"' in line:
                if algorithm == "DDPG":
                    line = line.replace("run_pybullet.py", "main.py")
            if "$experimentFolder/$experimentName/ppo/" in line:
                line = "cd $experimentFolder/$experimentName/" + algorithm.lower() + "/\n"
            if "BipedalWalker-v2" in line:
                if algorithm == "DDPG":
                    line = "python $pyName --env-id " + problem + "BulletEnv-v0" + " --seed $SGE_TASK_ID\n"
                else:
                    line = "python $pyName --env " + problem + "BulletEnv-v0" + " --seed $SGE_TASK_ID\n"
            f1.write(line)
        f1.close()
        f.seek(0)
    # f3 = open(directory + "/run_grid_ex_" + algorithm + ".sh", 'w')
    # for line in f2:
    #     if "ACKTR" in line:
    #         line = line.replace("ACKTR", algorithm)
    #     f3.write(line)
    # f3.close()
    f2.seek(0)

# Generate for gym control problems
for algorithm in algorithms:
    for problem in gym_problems:
        directory = "../../grid_scripts/" + str(algorithm)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f1 = open(directory + "/" + algorithm + "_" +
                  problem + ".sh", 'w')
        for line in f:
            if 'pyName="run_pybullet.py"' in line:
                if algorithm == "DDPG":
                    line = line.replace("run_pybullet.py", "main.py")
                else:
                    line = 'pyName="run_gym_ctrl.py"'
            if "$experimentFolder/$experimentName/ppo/" in line:
                line = "cd $experimentFolder/$experimentName/" + algorithm.lower() + "/\n"
            if "BipedalWalker-v2" in line:
                if algorithm == "DDPG":
                    line = "python $pyName --env-id " + problem + "-v2" + " --seed $SGE_TASK_ID\n"
                else:
                    line = "python $pyName --env " + problem + "-v2" + " --seed $SGE_TASK_ID\n"
            f1.write(line)
        f1.close()
        f.seek(0)

    # f3 = open(directory + "/run_grid_ex_" + algorithm + ".sh", 'w')
    # for line in f2:
    #     if "ACKTR" in line:
    #         line = line.replace("ACKTR", algorithm)
    #     f3.write(line)
    # f3.close()
    f2.seek(0)
f.close()

import glob
from functools import reduce

all_list = []
for algorithm in algorithms:
    all_files = glob.glob("../../grid_scripts/" + algorithm + "/*.sh")
    all_list.extend(list(map(lambda x: "".join(["qsub -t 1-5:1 ./",x.split("/")[-2], "/", x.split("/")[-1]]), all_files)))
# new_list = reduce(lambda x,y:x.extend(y), all_list)
command = reduce(lambda x,y: "".join([x, "\n", y]), all_list)
with open("../../grid_scripts/run.sh", "w") as f3:
    command = reduce(lambda x,y: "".join([x, "\n", y]), all_list)
    f3.write(command)



