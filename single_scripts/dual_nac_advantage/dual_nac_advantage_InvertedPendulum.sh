#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/dual_nac_advantage/

for i in {0..5}
do
	( python $pyName --env InvertedPendulumBulletEnv-v0 --seed $i  &)
done