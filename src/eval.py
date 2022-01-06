import os, time
from datetime import datetime
import gym
import torch

from src.PPO import PPO
from src.train import ENV_NAMES

RENDER_ENV = False
MODEL_DIR_PATH = "res/model/model_06_01_22-13_43_1641473027"
NUM_EPISODES = 100


def eval():
    ctime = datetime.now()
    dt_string = "model_" + ctime.strftime("%d_%m_%y-%H_%M_%s")
    dt_string = os.path.join("res", "model", dt_string)

    agent = PPO(ENV_NAMES[0], model_save_dir=dt_string, surrogate_objective="clipped", render_env=True, base_lr=0.001)
    agent.load_model(MODEL_DIR_PATH)

    env = gym.make(ENV_NAMES[0])

    cum_rewards = 0.0

    if RENDER_ENV:
        env.render()

    for episode in range(NUM_EPISODES):
        obs = env.reset()

        done = False
        while not done:
            action, _ = agent.evaluate_policy(torch.tensor(obs))
            obs, reward, done, info = env.step(action.detach().numpy())
            cum_rewards += reward
            if RENDER_ENV:
                time.sleep(0.01)

    print(f"Average reward= {cum_rewards / NUM_EPISODES}")


if __name__ == "__main__":
    eval()
