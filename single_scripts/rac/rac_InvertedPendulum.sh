#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/rac/

for i in {1..5} ;
do
	( python $pyName --env InvertedPendulumBulletEnv-v0 --seed $SGE_TASK_ID  &)
; done