#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/ars/

for i in {0..5}
do
	( python $pyName --env BipedalWalker-v2 --seed $i  &)
done