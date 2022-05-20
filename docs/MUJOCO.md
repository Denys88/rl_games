# Mujoco (https://github.com/deepmind/mujoco)  

## How to run:  
* **Humanoid** 

```bash
poetry install -E mujoco
poetry run python runner.py --train --file rl_games/configs/mujoco/humanoid.yaml
```

## Results:  
* **Humanoid-v3**
![Humanoid](pictures/mujoco/humanoid.jpg)  
* **HalfCheetah-v3**
![HalfCheetah](pictures/mujoco/half_cheetah.jpg)  
* **Hopper-v3**  
![Hopper](pictures/mujoco/hopper.jpg)
* **Walker-v3**  
![Walker](pictures/mujoco/walker.jpg) 

