import os
from datetime import datetime

from src.PPO import PPO
from src.train import ENV_NAMES

RENDER_ENV = False
MODEL_DIR_PATH = "res/model/model_06_01_22-13_43_1641473027"
NUM_EPISODES = 100


def eval():
    ctime = datetime.now()
    dt_string = "model_" + ctime.strftime("%d_%m_%y-%H_%M_%s")
    dt_string = os.path.join("res", "model", dt_string)

    agent = PPO(ENV_NAMES[0], model_save_dir=dt_string, surrogate_objective="clipped", render_env=RENDER_ENV,
                base_lr=0.001)
    agent.load_model(MODEL_DIR_PATH)

    avg_episodic_reward, ep_length = agent.validation(num_episodes=NUM_EPISODES)

    print(f"Average reward= {avg_episodic_reward}\n"
          f"Episode lengths: {ep_length}\n")


if __name__ == "__main__":
    eval()
