import gym

gym.envs.register(
     id='DMSoccerEnv-v0',
     entry_point='rl_games.envs.self_play.dm_soccer:DMSoccerEnv',
     max_episode_steps=100500,
)