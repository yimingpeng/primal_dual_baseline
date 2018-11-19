#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd $experimentName/ppo/

for i in 1 2 3 4 5
do
  echo "Looping ... i is set to $i"
  python $pyName --env BipedalWalker-v2 --seed $i &
done