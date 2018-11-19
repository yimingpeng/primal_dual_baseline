#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd $experimentName/ppo/

for i in {1..5} ;
do
     (python $pyName --env BipedalWalker-v2 --seed $SGE_TASK_ID &)
; done