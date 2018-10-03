#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/ppo_dual_nac_fisher/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/primal_dual_baseline &
# cd ./primal_dual_baseline/grid_scripts/ppo_dual_nac_fisher/

# setting the grid env
need sgegrid
qsub -t 1-10:1 ppo_dual_nac_fisher_HalfCheetah.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_Hopper.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_InvertedDoublePendulum.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_InvertedPendulum.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_InvertedPendulumSwingup.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_Reacher.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_Walker2D.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_BipedalWalker.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_BipedalWalkerHardcore.sh
qsub -t 1-10:1 ppo_dual_nac_fisher_LunarLanderContinuous.sh





