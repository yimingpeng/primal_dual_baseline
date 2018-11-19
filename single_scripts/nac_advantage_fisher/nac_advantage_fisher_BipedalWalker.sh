#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/nac_advantage_fisher/

for i in {0..5}
do
	( python $pyName --env BipedalWalker-v2 --seed $i &> $i.out)
     echo "Complete the process $i"
done