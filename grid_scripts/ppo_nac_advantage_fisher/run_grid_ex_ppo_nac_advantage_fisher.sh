#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/ppo_nac_advantage_fisher/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/primal_dual_baseline &
# cd ./primal_dual_baseline/grid_scripts/ppo_nac_advantage_fisher/

# setting the grid env
need sgegrid
qsub -t 1-10:1 ppo_nac_advantage_fisher_HalfCheetah.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_Hopper.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_InvertedDoublePendulum.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_InvertedPendulum.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_InvertedPendulumSwingup.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_Reacher.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_Walker2D.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_BipedalWalker.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_BipedalWalkerHardcore.sh
qsub -t 1-10:1 ppo_nac_advantage_fisher_LunarLanderContinuous.sh





