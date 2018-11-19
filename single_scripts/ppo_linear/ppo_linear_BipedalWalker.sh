#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/ppo_linear/

for i in {0..5}
do
	( python $pyName --env BipedalWalker-v2 --seed $i &> BipedalWalker_"$i".out)
     echo "Complete the process $i"
done