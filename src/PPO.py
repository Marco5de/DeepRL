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
        """
        Evaluates the actor policy network to generate an action
        Is used for the on-policy algorithm

        :param observation last observation that was observed from the env, used to evaluate pi(a|s)
        :return: action sampled from the current policy
        """
        # mean = NNactor(observation)
        # distribution = MultiGaussian(mean, covMat)
        # sample
        # logprobs
        # todo: for now return random action
        return self.env.action_space.sample()

    def rollout_buffer(self, N: int, T: int):
        # todo: alle werte irgendwo speichern und dann wieder rausgeben

        rollout_buffer = RolloutBuffer()

        for actor in range(N):
            observation = self.env.reset()
            for t in range(T):
                # get action - implement action getter from current policy
                action = self.eval_policy()

                observation, action, done, info = self.env.step(action)
                print(observation)
                pass



class RolloutBuffer:
    """
    Implementation of a rollout buffer which is filled when sampling data from the environment in the PPO algorithm
    """

    def __init__(self):
        # todo add missing values that must be tracked
        self.actions = []
        self.observations = []
        self.rewards = []



def __main__():
    ENV_NAMES = ["CartPole-v1", "MountainCar-v0", "MsPacman-v0", "Hopper-v3"]

    print("Hello World")
    ppo = PPO(ENV_NAMES[0])
    ppo.rollout_buffer(5, 2)


if __name__ == "__main__":
    __main__()