#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/nac_advantage/

for i in {0..5}
do
	( python $pyName --env LunarLanderContinuous-v2 --seed $i &> LunarLanderContinuous_"$i".out)
     echo "Complete the process $i"
done