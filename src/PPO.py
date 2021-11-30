import torch
import torch.nn as nn
import torch.optim
import gym
from typing import Tuple

from src.ANN import PolicyNetwork, ValueFunctionNetwork
from src.util import kullback_leibler_div


class Hyperparameter:
    def __init__(self):
        # todo find value from paper / impl
        self.epsilon_clip = 0.1
        self.gamma = 0.95
        self.beta = 1.0
        self.d_target_ratio = 1.5
        self.d_target = 0.1
        self.var = 1.0
        self.N = 10
        self.T = 10
        self.K = 5


class RolloutBuffer:
    """
    Implementation of a rollout buffer which is filled when sampling data from the environment in the PPO algorithm
    """

    def __init__(self):
        self.actions = []
        self.log_probabilities = []
        self.observations = []
        self.rewards = []
        self.episode_length = []

    def to_tensor(self):
        """
        Conversion from python lists to torch.Tensor

        :return: actions, log_probabilities, observations, rewards, episode length as torch.Tensor
        """
        tensor_actions = torch.tensor(self.actions)
        tensor_log_probs = torch.tensor(self.log_probabilities)
        tensor_observations = torch.tensor(self.observations)
        tensor_rewards = torch.tensor(self.rewards)
        tensor_episode_length = torch.tensor(self.episode_length)

        return tensor_actions, tensor_log_probs, tensor_observations, tensor_rewards, tensor_episode_length


class PPO(nn.Module):
    _valid_surrogate_objectives = ["clipped", "adaptive_KL", "policy_gradient"]

    def __init__(self, gym_name: str, surrogate_objective: str, render_env: bool, base_lr: float):
        super().__init__()
        assert surrogate_objective in self._valid_surrogate_objectives, "Specified surrogate objective is not valid!"
        self.env = gym.make(gym_name)

        self.render_env = render_env

        self.surrogate_objective = surrogate_objective
        self.hyperparameter = Hyperparameter()

        self.state_space_dim = self.env.observation_space.n
        self.action_space_dim = self.env.action_space.n
        # todo: macht es sinn, dass die covariance für alle dimensionen gleich ist?
        self.cov_mat = torch.diag(torch.tensor([self.hyperparameter.var] * self.state_space_dim))

        self.policy_network = PolicyNetwork(input_dim=self.state_space_dim,
                                            output_dim=self.action_space_dim,
                                            covariance_mat=self.cov_mat)
        self.policy_network_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=base_lr)

        self.value_func_network = ValueFunctionNetwork(input_dim=self.state_space_dim)
        self.value_func_network_optimizer = torch.optim.Adam(self.value_func_network.parameters(), lr=base_lr)

    def forward(self) -> None:
        """
        Learning is done in here, this function must be called repeatedly for #training_steps
        :return:
        """
        # sample data - Run policy π_θ_old in environment for T time steps using N actors
        rollout_buffer = self.sample_data(N=10, T=10)
        actions, log_probs, observations, rewards, episode_length = rollout_buffer.to_tensor()

        # compute advantage estimates A_1, \dots, A_T
        value, log_probs = self.evaluate_value_function(observations, actions)
        # todo: trick to normalize advantage functions
        advtange_values = self.compute_advantage_values(value, rewards)

        # Optimize surrogate L w.r.t. θ with K epochs and minibatch size M ≤ N T
        self.optimize_neural_networks(observations, actions, K=10)

    def optimize_neural_networks(self, observations, actions, K) -> None:
        """
        Optimize surrogate L w.r.t. θ with K epochs and minibatch size M ≤ N T
        :return:
        """
        for i in range(K):
            value, log_probs = self.evaluate_value_function(observations,
                                                            actions)  # this expression changes as the NN is updated!

            # compute surrogate objective function
            surrogate = 0.0
            if self.surrogate_objective == "clipped":
                surrogate = self.clipped_surrogate_function()
            elif self.surrogate_objective == "adaptive_KL":
                surrogate = self.adaptive_KL_surrogate_function()
            elif self.surrogate_objective == "policy_gradient":
                surrogate = self.policy_gradient_surrogate_funcion()

            policy_net_loss = - surrogate # maximization!
            self.policy_network_optimizer.zero_grad()
            policy_net_loss.backward()
            self.policy_network_optimizer.step()

            # todo: was nimmt man also Value function loss?
            value_function_net_loss = None
            self.value_func_network_optimizer.zero_grad()
            value_function_net_loss.backward()
            self.value_func_network_optimizer.step()

    def compute_advantage_values(self, value: torch.Tensor, rewards: torch.Tensor, episode_lengths: torch.Tensor) \
            -> torch.Tensor:
        """
        Paper specifies a couple of estimators are they equivalent?
        Implementations contain other, different estimators, which one to use?

        :param value:
        :param rewards:
        :param episode_lengths:
        :return: advantage values torch.Tensor
        """
        # todo: lese https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/
        number_of_episodes = episode_lengths.size()

        # todo check if this index magic is working!
        discounted_rewards = []
        offset = 0
        for i in range(number_of_episodes):
            sum = 0
            length = episode_lengths[-(i + 1)]  # reverse
            for j in range(length):
                # length is the length of the current episode
                sum = rewards[-(j + offset + 1)] + self.hyperparameter.gamma * sum
                discounted_rewards.insert(0, sum)
            offset += length
        discounted_rewards = torch.tensor(discounted_rewards)

        return discounted_rewards - value

    def evaluate_value_function(self, observations: torch.Tensor, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimation of the value function V(s) by evaluating the NN
        :param observations:
        :param actions:
        :return:
        """
        value = self.value_func_network(observations)
        _, _, distribution = self.policy_network(observations)
        log_probs = distribution.log_prob(actions)

        return value, log_probs

    def evaluate_policy(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the actor policy network to generate an action
        Is used for the on-policy algorithm

        :param observation last observation that was observed from the env, used to evaluate pi(a|s)
        :return: action sampled from the current policy and its log probability
        """
        action_sample, log_prob = self.policy_network(observation)
        return action_sample, log_prob

    def sample_data(self, N: int, T: int) -> RolloutBuffer:
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

    def clipped_surrogate_function(self, old: torch.Tensor, new: torch.Tensor, advantage_values: torch.Tensor) \
            -> torch.Tensor:
        """
        Implements the clipped surrogate function
        PPO paper section (3)
        :return: value of clipped surrogate function
        """
        r_t = torch.exp(new) / torch.exp(old)

        surrogate = torch.min(torch.tensor([
            r_t * advantage_values,
            torch.clamp(r_t, 1 - self.hyperparameter.epsilon_clip, 1 + self.hyperparameter.epsilon_clip)
        ]))

        return surrogate

    def adaptive_KL_surrogate_function(self, old: torch.Tensor, new: torch.Tensor,
                                       advantage_values: torch.Tensor, clip: bool = True) -> torch.Tensor:
        """
        :param clip specifies whether the clipping method (see paper and clipped surrogate function) is applied
        :return:
        """
        surrogate = 0.0
        if clip:
            surrogate = self.clipped_surrogate_function(old, new, advantage_values)
        else:
            r_t = torch.exp(new) / torch.exp(old)
            surrogate = r_t * advantage_values

        kl_penalty = kullback_leibler_div(old, self.cov_mat, new, self.cov_mat)
        combined = surrogate - self.hyperparameter.beta * kl_penalty
        # todo im Paper ist da noch eine Expectation, aber ich denke mal in der Implementierung ist das über den Batch drin
        # updated beta is used for the next policy update!
        if kl_penalty < (self.hyperparameter.d_target / self.hyperparameter.d_target_ratio):
            self.hyperparameter.beta = 0.5 * self.hyperparameter.beta
        else:
            self.hyperparameter.beta = 2 * self.hyperparameter.beta

        return combined

    def policy_gradient_surrogate_funcion(self) -> torch.Tensor:
        """
        todo: implement
        :return:
        """
        return 0


def __main__():
    ENV_NAMES = ["CartPole-v1", "MountainCar-v0", "MsPacman-v0", "Hopper-v3"]

    print("Hello World")
    ppo = PPO(ENV_NAMES[0])
    ppo.rollout_buffer(5, 2)


if __name__ == "__main__":
    __main__()
