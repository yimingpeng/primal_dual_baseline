#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/nac_advantage_fisher/

for i in {0..5}
do
	( python $pyName --env InvertedPendulumSwingupBulletEnv-v0 --seed $i  &> InvertedPendulumSwingup_"$i".out)
     echo "Complete the process $i"
done