#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/ppo_linear/

for i in {1..5} ;
do
	( python $pyName --env InvertedDoublePendulumBulletEnv-v0 --seed $SGE_TASK_ID  &)
; done