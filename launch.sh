#!/bin/bash

for i in {1..5}; do 
      ./run.sh 0 python3 torch_runner.py --file=whirl_baselines/3m_torch with name=test3_3m_torch label=test3_3m_torch &
done
