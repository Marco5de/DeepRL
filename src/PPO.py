import torch
import torch.nn as nn
import gym

from src.ANN import PolicyNetwork, ValueFunctionNetwork


class PPO(nn.Module):

    def __init__(self, gym_name: str, render_env: bool):
        super().__init__()
        self.env = gym.make(gym_name)

        self.render_env = render_env

        self.state_space_dim = 10
        self.action_space_dim = 3
        self.cov_mat = torch.tensor([[1.0, 0, 0],
                                     [0, 1.0, 0],
                                     [0, 0, 1.0]])

        # todo: dont hardcode dims and so on
        self.policy_network = PolicyNetwork(input_dim=self.state_space_dim,
                                            output_dim=self.action_space_dim,
                                            covariance_mat=self.cov_mat)

        self.value_func_network = ValueFunctionNetwork(input_dim=self.state_space_dim)

    def forward(self):
        pass

    def evaluate_policy(self, observation: torch.Tensor):
        """
        Evaluates the actor policy network to generate an action
        Is used for the on-policy algorithm

        :param observation last observation that was observed from the env, used to evaluate pi(a|s)
        :return: action sampled from the current policy and its log probability
        """
        action_sample, log_prob = self.policy_network(observation)
        return action_sample, log_prob

    def rollout_buffer(self, N: int, T: int):
        rollout_buffer = RolloutBuffer()

        for actor in range(N):
            observation = self.env.reset()
            rollout_buffer.observations.append(observation)

            # episode of length T
            length = 0
            for t in range(T):
                if self.render_env:
                    self.env.render()

                # get action - implement action getter from current policy
                action, log_prob = self.eval_policy()
                rollout_buffer.actions.append(action)
                rollout_buffer.log_probabilities.append(log_prob)

                observation, reward, done, info = self.env.step(action)
                rollout_buffer.rewards.append(reward)

                length = t
                if done:  # if environment sequence is over break from this loop!
                    break

            # mark length of the episode and the final reward
            rollout_buffer.episode_length.append(length)

        return rollout_buffer


class RolloutBuffer:
    """
    Implementation of a rollout buffer which is filled when sampling data from the environment in the PPO algorithm
    """

    def __init__(self):
        # todo add missing values that must be tracked
        self.actions = []
        self.log_probabilities = []
        self.observations = []
        self.rewards = []
        self.episode_length = []


def __main__():
    ENV_NAMES = ["CartPole-v1", "MountainCar-v0", "MsPacman-v0", "Hopper-v3"]

    print("Hello World")
    ppo = PPO(ENV_NAMES[0])
    ppo.rollout_buffer(5, 2)


if __name__ == "__main__":
    __main__()
