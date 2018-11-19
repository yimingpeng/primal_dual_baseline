#!/bin/bash

experimentName="baselines"

pyName="run_gym_ctrl.py"

cd ../../$experimentName/nac_advantage_fisher/

for i in {1..5} ;
do
	( python $pyName --env MountainCarContinuous-v0 --seed $SGE_TASK_ID  &)
; done