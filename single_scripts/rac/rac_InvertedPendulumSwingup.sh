#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/rac/

for i in {0..5}
do
	( python $pyName --env InvertedPendulumSwingupBulletEnv-v0 --seed $SGE_TASK_ID  &)
done