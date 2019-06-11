import env_configurations
import games.other.env_config

for name, config in games.other.env_config.configurations.items():
    env_configurations.register(name, config)

print ('init other configs')