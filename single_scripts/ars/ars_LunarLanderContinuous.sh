#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/ars/

for i in {1..5}
do
	( python $pyName --env LunarLanderContinuous-v2 --seed $SGE_TASK_ID  &)
done