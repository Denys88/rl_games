#!/bin/bash

#for i in {1..1}; do 
#      ./run_servers.sh 3 python3 torch_runner.py --file=whirl_baselines/3m_torch with name=test3_3m_torch label=test3_3m_torch &
#done

for i in {1..5}; do 
      #./run_servers.sh ${i} python3 torch_runner.py --file=whirl_baselines/3m_torch_cnn with name=test3_3m_torch_cnn label=test3_3m_torch_cnn &
      #./run_servers.sh ${i} python3 torch_runner.py --file=whirl_baselines/2s3z_torch_cnn with name=test3_2s3z_torch_cnn label=test3_2s3z_torch_cnn &
      #./run_servers.sh ${i} python3 torch_runner.py --file=whirl_baselines/2s3z_torch with name=test3_2s3z_torch label=test3_2s3z_torch &
      #./run_servers.sh ${i} python3 torch_runner.py --file=whirl_baselines/5m_vs_6m_torch with name=test3_5m_vs_6m_torch label=test3_5m_vs_6m_torch &
      #./run_servers.sh ${i} python3 torch_runner.py --file=whirl_baselines/5m_vs_6m_torch_cnn with name=test3_5m_vs_6m_torch_cnn label=test3_5m_vs_6m_torch_cnn &
      ./run_servers.sh ${i} python3 torch_runner.py --file=whirl_baselines/corridor_torch with name=corridor_torch label=corridor_torch &
done
