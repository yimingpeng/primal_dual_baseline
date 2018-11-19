#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd $experimentName/rac/

for i in 1 2 3 4 5
do
python $pyName --env MountainCarContinuous-v0 --seed $SGE_TASK_ID
done