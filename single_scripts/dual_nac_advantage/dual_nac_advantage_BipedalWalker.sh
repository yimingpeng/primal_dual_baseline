#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/dual_nac_advantage/

for i in 1 2 3 4 5
do
  echo "Looping ... i is set to $i"
python $pyName --env BipedalWalker-v2 --seed $SGE_TASK_ID
done