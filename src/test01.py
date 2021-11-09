import gym
import sys

from stable_baselines3 import A2C
from stable_baselines3 import PPO

ENV_NAMES = ["CartPole-v1", "MountainCar-v0", "MsPacman-v0", "Hopper-v3"]


env = gym.make(ENV_NAMES[0])

print(f"Env.action_space: {env.action_space} Env.observation_space: {env.observation_space}\n")

for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f"~~~\nObs: {observation}\nReward: {reward}\nDone: {done}\nInfo: {info}\nAction: {action}\n~~~\n")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()




sys.exit(0)


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