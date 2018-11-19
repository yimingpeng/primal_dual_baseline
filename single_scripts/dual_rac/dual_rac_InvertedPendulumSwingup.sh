#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/dual_rac/

for i in {0..5}
do
	( python $pyName --env InvertedPendulumSwingupBulletEnv-v0 --seed $i  &)
done