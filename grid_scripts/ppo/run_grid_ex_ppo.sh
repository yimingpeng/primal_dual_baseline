#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/ppo/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/primal_dual_baseline &
# cd ./primal_dual_baseline/grid_scripts/ppo/

# setting the grid env
need sgegrid
qsub -t 1-5:1 ppo_HalfCheetah.sh
qsub -t 1-5:1 ppo_Hopper.sh
qsub -t 1-5:1 ppo_InvertedDoublePendulum.sh
qsub -t 1-5:1 ppo_InvertedPendulum.sh
qsub -t 1-5:1 ppo_InvertedPendulumSwingup.sh
qsub -t 1-5:1 ppo_Reacher.sh
qsub -t 1-5:1 ppo_Walker2D.sh
qsub -t 1-5:1 ppo_BipedalWalker.sh
qsub -t 1-5:1 ppo_BipedalWalkerHardcore.sh
qsub -t 1-5:1 ppo_LunarLanderContinuous.sh





