import env_configurations
import games.openai.env_config

for name, config in games.openai.env_config.configurations.items():
    env_configurations.register(name, config)

print ('init gym configs')