#!/bin/bash

for i in {1..5}; do 
      #./run.sh 0 python3 torch_runner.py --file=whirl_baselines/3m_torch with name=test3_3m_torch label=test3_3m_torch &
      #./run.sh ${i} python3 torch_runner.py --file=whirl_baselines/2s3z_torch with name=test3_2s3z_torch label=test3_2s3z_torch &
      #./run.sh ${i} python3 torch_runner.py --file=whirl_baselines/2s3z_torch_cnn with name=test3_2s3z_torch_cnn label=test3_2s3z_torch_cnn &
      ./run.sh $(($i%2)) python3 torch_runner.py --file=whirl_baselines/5m_vs_6m_torch with name=5m_vs_6m_torch label=5m_vs_6m_torch &
done
