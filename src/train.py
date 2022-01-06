import sys, time, os
from datetime import datetime
import gym
import pybulletgym
from torch.utils.tensorboard import SummaryWriter

from src.PPO import PPO

ENV_NAMES = ["AntPyBulletEnv-v0",  # 0
             "HalfCheetahPyBulletEnv-v0",  # 1
             "Walker2DPyBulletEnv-v0",  # 2
             "HopperPyBulletEnv-v0",  # 3
             "InvertedPendulumPyBulletEnv-v0",  # 4
             "InvertedDoublePendulumPyBulletEnv-v0",  # 5
             "InvertedPendulumSwingupPyBulletEnv-v0",  # 6
             "HumanoidPyBulletEnv-v0",  # 7
             "HumanoidFlagrunPyBulletEnv-v0",  # 8
             "HumanoidFlagrunHarderPyBulletEnv-v0",  # 9
             "AtlasPyBulletEnv-v0",  # 10
             "PusherPyBulletEnv-v0",  # 11
             "ReacherPyBulletEnv-v0",  # 12
             ]

SAVE_MODEL_FREQ = 100
EVAL_FREQ = 10
TRAIN_STEPS = 10000
ENV_IDX = 12


def train():
    ctime = datetime.now()
    dt_string = "model_" + ctime.strftime("%d_%m_%y-%H_%M_%s")
    dt_string = os.path.join("res", "model", dt_string)
    print(dt_string)
    agent = PPO(ENV_NAMES[ENV_IDX], model_save_dir=dt_string, surrogate_objective="clipped", render_env=False,
                base_lr=2.5e-4)

    writer = SummaryWriter(log_dir=os.path.join("res/log_dir", dt_string))

    for i in range(TRAIN_STEPS):
        policy_net_loss, value_net_loss = agent()
        writer.add_scalar("loss/policy", policy_net_loss, global_step=i)
        writer.add_scalar("loss/value", value_net_loss, global_step=i)

        if i % SAVE_MODEL_FREQ == 0 and i != 0:
            agent.save_model()

        if i % EVAL_FREQ == 0:
            avg_reward, episode_lengths = agent.validation(num_episodes=3)
            writer.add_scalar("episode_reward", avg_reward, global_step=i // EVAL_FREQ)
            print(f"Current average reward per episode= {avg_reward}\n"
                  f"Episode lengths: {episode_lengths}")


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
