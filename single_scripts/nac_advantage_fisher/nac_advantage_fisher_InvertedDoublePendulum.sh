#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/nac_advantage_fisher/

for i in 1 2 3 4 5
do
  echo "Looping ... i is set to $i"
python $pyName --env InvertedDoublePendulumBulletEnv-v0 --seed $SGE_TASK_ID
done