#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/dual_rac/

for i in {0..5}
do
	( python $pyName --env MountainCarContinuous-v0 --seed $i  &)
done