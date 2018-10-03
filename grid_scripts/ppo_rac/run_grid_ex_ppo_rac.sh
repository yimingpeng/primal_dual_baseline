#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/ppo_rac/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/primal_dual_baseline &
# cd ./primal_dual_baseline/grid_scripts/ppo_rac/

# setting the grid env
need sgegrid
qsub -t 1-5:1 ppo_rac_HalfCheetah.sh
qsub -t 1-5:1 ppo_rac_Hopper.sh
qsub -t 1-5:1 ppo_rac_InvertedDoublePendulum.sh
qsub -t 1-5:1 ppo_rac_InvertedPendulum.sh
qsub -t 1-5:1 ppo_rac_InvertedPendulumSwingup.sh
qsub -t 1-5:1 ppo_rac_Reacher.sh
qsub -t 1-5:1 ppo_rac_Walker2D.sh
qsub -t 1-5:1 ppo_rac_BipedalWalker.sh
qsub -t 1-5:1 ppo_rac_BipedalWalkerHardcore.sh
qsub -t 1-5:1 ppo_rac_LunarLanderContinuous.sh





