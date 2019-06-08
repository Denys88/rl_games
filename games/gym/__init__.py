import env_configurations
import games.gym.env_config

for name, config in games.gym.env_config.configurations.items():
    env_configurations.register(name, config)

print ('init gym configs')