#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/ppo_nac_advantage/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/primal_dual_baseline &
# cd ./primal_dual_baseline/grid_scripts/ppo_nac_advantage/

# setting the grid env
need sgegrid
qsub -t 1-10:1 ppo_nac_advantage_HalfCheetah.sh
qsub -t 1-10:1 ppo_nac_advantage_Hopper.sh
qsub -t 1-10:1 ppo_nac_advantage_InvertedDoublePendulum.sh
qsub -t 1-10:1 ppo_nac_advantage_InvertedPendulum.sh
qsub -t 1-10:1 ppo_nac_advantage_InvertedPendulumSwingup.sh
qsub -t 1-10:1 ppo_nac_advantage_Reacher.sh
qsub -t 1-10:1 ppo_nac_advantage_Walker2D.sh
qsub -t 1-10:1 ppo_nac_advantage_BipedalWalker.sh
qsub -t 1-10:1 ppo_nac_advantage_BipedalWalkerHardcore.sh
qsub -t 1-10:1 ppo_nac_advantage_LunarLanderContinuous.sh





