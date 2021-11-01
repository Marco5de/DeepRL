import gym

from stable_baselines3 import A2C
from stable_baselines3 import PPO

env = gym.make('CartPole-v1')

# model = A2C('MlpPolicy', env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)


obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()