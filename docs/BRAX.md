# Brax (https://github.com/google/brax)  

## How to run:  
* **Ant** ```python runner.py --train --file rl_games/configs/brax/ppo_ant.yaml```
* **Humanoid** ```python runner.py --train --file rl_games/configs/brax/ppo_humanoid.yaml```
## Visualization of the trained policy:  
* **brax_visualization.ipynb**

## Results:  
* **Ant** fps step: 1692066.6 fps total: 885603.1  
![Ant](pictures/brax/brax_ant.jpg)  
* **Humanoid** fps step: 1244450.3 fps total: 661064.5  
![Humanoid](pictures/brax/brax_humanoid.jpg)  
* **ur5e** fps step: 1116872.3 fps total: 627117.0  
![Humanoid](pictures/brax/brax_ur5e.jpg)  


![Alt Text](pictures/brax/humanoid.gif)
![Alt Text](pictures/brax/ur5e.gif)