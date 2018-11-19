#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/nac_fisher/

for i in {0..5}
do
	( python $pyName --env MountainCarContinuous-v0 --seed $i  &)
done