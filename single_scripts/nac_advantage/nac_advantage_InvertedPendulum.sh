#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/nac_advantage/

for i in {0..5}
do
	( python $pyName --env InvertedPendulumBulletEnv-v0 --seed $i  &> $i.out)
     echo "Complete the process $i"
done