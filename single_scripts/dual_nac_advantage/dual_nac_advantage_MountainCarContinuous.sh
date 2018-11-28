#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/dual_nac_advantage/

for i in {0..5}
do
	( python $pyName --env MountainCarContinuous-v0 --seed $i &> MountainCarContinuous_"$i".out)
     echo "Complete the process $i"
done