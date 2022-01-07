import time, os
from datetime import datetime
import gym
# import pybulletgym # todo uncomment
from torch.utils.tensorboard import SummaryWriter

from lib.PPO import PPO

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

             # Mujoco Envs
             "InvertedPendulumMuJoCoEnv-v0",  # 13
             "ReacherMuJoCoEnv-v0",  # 14
             "AntMuJoCoEnv-v0",  # 15
             "Pendulum-v0",  # 16
             ]

SAVE_MODEL_FREQ = 100
LOG_FREQ = 1
TRAIN_STEPS = 800
ENV_IDX = 16
RENDER_ENV = False


def train():
    ctime = datetime.now()
    dt_string = "model_" + ctime.strftime("%d_%m_%y-%H_%M_%s")
    dt_string = os.path.join("res", "model", dt_string)
    print(dt_string)
    agent = PPO(ENV_NAMES[ENV_IDX], model_save_dir=dt_string, surrogate_objective="clipped", render_env=RENDER_ENV)

    writer = SummaryWriter(log_dir=os.path.join("res/log_dir", dt_string))

    for i in range(TRAIN_STEPS):
        t1 = time.time_ns()
        policy_net_loss, value_net_loss, avg_return, avg_len = agent()
        t2 = time.time_ns()
        duration_ns = t2 - t1

        writer.add_scalar("loss/policy", policy_net_loss, global_step=agent.c_time_step)
        writer.add_scalar("loss/value", value_net_loss, global_step=agent.c_time_step)
        writer.add_scalar("reward/avg_ep_reward", avg_return, global_step=agent.c_time_step)
        writer.add_scalar("reward/avg_ep_len", avg_len, global_step=agent.c_time_step)
        writer.add_scalar("proc_time_ms", duration_ns * 1e-6, global_step=i)

        if i % SAVE_MODEL_FREQ == 0 and i != 0:
            agent.save_model()

        if i % LOG_FREQ == 0:
            print(f"~~~Train Summary {i + 1} / {TRAIN_STEPS} - {i / TRAIN_STEPS * 100:.1f}% complete~~~\n"
                  f"Current time step= {agent.c_time_step}\n"
                  f"Current avg. episodic reward= {avg_return}\n"
                  f"Avg episode length= {avg_len}\n"
                  f"Policy-loss= {policy_net_loss:.4f} Value-loss= {value_net_loss:.4f}\n"
                  f"~~~ END Summary~~~ ")


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
