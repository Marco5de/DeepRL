import torch
import torch.nn as nn
import gym

class PPO(nn.Module):

    def __init__(self, gym_name):
        super().__init__()
        self.env = gym.make(gym_name)

    def forward(self):
        pass


    def evaluate_policy(self, observation: torch.Tensor):
        # mean = NNactor(observation)
        # distribution = MultiGaussian(mean, covMat)
        # sample
        # logprobs
        return None

    def rollout_buffer(self, N: int, T: int):
        # todo: alle werte irgendwo speichern und dann wieder rausgeben
        for actor in range(N):
            observation = self.env.reset()
            for t in range(T):
                # get action - implement action getter from current policy
                action = self.env.action_space.sample()

                observation, action, done, info = self.env.step(action)
                print(observation)
                pass







def __main__():
    ENV_NAMES = ["CartPole-v1", "MountainCar-v0", "MsPacman-v0", "Hopper-v3"]

    print("Hello World")
    ppo = PPO(ENV_NAMES[0])
    ppo.rollout_buffer(5, 2)


if __name__ == "__main__":
    __main__()