import os
from datetime import datetime

from src.lib.PPO import PPO
from src.train import ENV_NAMES

RENDER_ENV = False
MODEL_DIR_PATH = ""
NUM_EPISODES = 1
ENV_IDX = 0

def eval():
    ctime = datetime.now()
    dt_string = "model_" + ctime.strftime("%d_%m_%y-%H_%M_%s")
    dt_string = os.path.join("res", "model", dt_string)

    agent = PPO(ENV_NAMES[ENV_IDX], model_save_dir=dt_string, surrogate_objective="clipped", render_env=RENDER_ENV,)
    agent.load_model(MODEL_DIR_PATH)

    avg_episodic_reward, ep_length = agent.validation(num_episodes=NUM_EPISODES)

    print(f"Average reward= {avg_episodic_reward}\n"
          f"Episode lengths: {ep_length}\n")


if __name__ == "__main__":
    eval()
