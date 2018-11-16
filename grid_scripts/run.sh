qsub -t 1-30:1 ./ppo_linear/ppo_linear_InvertedPendulumSwingup.sh
qsub -t 1-30:1 ./ppo_linear/ppo_linear_BipedalWalker.sh
qsub -t 1-30:1 ./ppo_linear/ppo_linear_LunarLanderContinuous.sh
qsub -t 1-30:1 ./ppo_linear/ppo_linear_InvertedDoublePendulum.sh
qsub -t 1-30:1 ./ppo_linear/ppo_linear_MountainCarContinuous.sh
qsub -t 1-30:1 ./ppo_linear/ppo_linear_InvertedPendulum.sh