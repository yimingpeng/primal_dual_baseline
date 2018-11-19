#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd $experimentName/nac_fisher/

for i in 1 2 3 4 5
do
python $pyName --env InvertedPendulumBulletEnv-v0 --seed $SGE_TASK_ID
done