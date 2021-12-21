import time, sys
import gym
import pybulletgym

from src.PPO import PPO

ENV_NAMES = ["AntPyBulletEnv-v0", "HalfCheetahPyBulletEnv-v0", "Walker2DPyBulletEnv-v0", "HopperPyBulletEnv-v0",
             "InvertedPendulumPyBulletEnv-v0", "InvertedDoublePendulumPyBulletEnv-v0",
             "InvertedPendulumSwingupPyBulletEnv-v0", "HumanoidPyBulletEnv-v0", "HumanoidFlagrunPyBulletEnv-v0",
             "HumanoidFlagrunHarderPyBulletEnv-v0", "AtlasPyBulletEnv-v0", "PusherPyBulletEnv-v0",
             ]


def train():
    agent = PPO(ENV_NAMES[0], surrogate_objective="clipped", render_env=False, base_lr=0.001)


    for i in range(10):
        agent()


def test_env(env_name: str) -> None:
    env = gym.make(env_name)
    env.render()
    print(f"Env.action_space: {env.action_space} Env.observation_space: {env.observation_space}\n")
    for i_episode in range(1):
        observation = env.reset()
        for t in range(100):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(f"~~~\nObs: {observation}\nReward: {reward}\nDone: {done}\nInfo: {info}\nAction: {action}\n~~~\n")
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            time.sleep(0.1)
    env.close()


if __name__ == "__main__":
    train()
