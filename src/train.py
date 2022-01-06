import sys, time, os
from datetime import datetime
import gym
import pybulletgym

from src.PPO import PPO

ENV_NAMES = ["AntPyBulletEnv-v0", "HalfCheetahPyBulletEnv-v0", "Walker2DPyBulletEnv-v0", "HopperPyBulletEnv-v0",
             "InvertedPendulumPyBulletEnv-v0", "InvertedDoublePendulumPyBulletEnv-v0",
             "InvertedPendulumSwingupPyBulletEnv-v0", "HumanoidPyBulletEnv-v0", "HumanoidFlagrunPyBulletEnv-v0",
             "HumanoidFlagrunHarderPyBulletEnv-v0", "AtlasPyBulletEnv-v0", "PusherPyBulletEnv-v0",
             ]

SAVE_MODEL_FREQ = 100


def train():
    ctime = datetime.now()
    dt_string = "model_" + ctime.strftime("%d_%m_%y-%H_%M_%s")
    dt_string = os.path.join("res", "model", dt_string)
    print(dt_string)
    agent = PPO(ENV_NAMES[0], model_save_dir=dt_string, surrogate_objective="clipped", render_env=True, base_lr=0.001)


    for i in range(10000):
        agent()

        if i % SAVE_MODEL_FREQ == 0:
            agent.save_model()


def env_test(env_name: str) -> None:
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
