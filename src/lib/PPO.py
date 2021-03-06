import os

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import gym
from typing import Tuple, List
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from lib.ANN import PolicyNetwork, ValueFunctionNetwork
from lib.util import kullback_leibler_div
from lib.Hyperparameter import Hyperparameter


class RolloutBuffer:
    """
    Implementation of a rollout buffer which is filled when sampling data from the environment in the PPO algorithm
    """

    def __init__(self):
        self.actions = []
        self.log_probabilities = []
        self.observations = []
        self.rewards = []
        self.unnormed_rewards = []
        self.episode_length = []

    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Conversion from python lists to torch.Tensor

        :return: actions, log_probabilities, observations, rewards, episode length as torch.Tensor
        """
        tensor_actions = torch.stack(self.actions)
        tensor_log_probs = torch.tensor(self.log_probabilities)
        tensor_observations = torch.tensor(np.stack(self.observations))
        tensor_rewards = torch.tensor(np.array(self.rewards))
        tensor_episode_length = torch.tensor(self.episode_length)
        tensor_unnormed_rewards = torch.tensor(np.array(self.unnormed_rewards))

        return tensor_actions, tensor_log_probs, tensor_observations, tensor_rewards, tensor_episode_length, tensor_unnormed_rewards


class PPO(nn.Module):
    _valid_surrogate_objectives = ["clipped", "adaptive_KL", "policy_gradient"]

    def __init__(self, gym_name: str, model_save_dir: str, surrogate_objective: str, render_env: bool):
        """
        Ctor
        :param gym_name: name of the environment in ENV_NAMES
        :param model_save_dir:
        :param surrogate_objective: surrogate function in _valid_surrogate_objectives
        :param render_env: bool whether to render environment during training
        """
        super().__init__()
        assert surrogate_objective in self._valid_surrogate_objectives, "Specified surrogate objective is not valid!"
        self.env = gym.make(gym_name)
        # ensure compatibility with both pybullet-gym and openai-gym
        try:
            self.state_space_dim = len(self.env.robot.observation_space.high)
            self.action_space_dim = len(self.env.robot.action_space.high)

            self.action_space_lim = (self.env.robot.action_space.low, self.env.robot.action_space.high)
            self.state_space_lim = (self.env.robot.observation_space.low, self.env.robot.observation_space.high)
        except:
            self.state_space_dim = len(self.env.observation_space.low)
            self.action_space_dim = len(self.env.action_space.low)

            self.action_space_lim = (self.env.action_space.low, self.env.action_space.high)
            self.state_space_lim = (self.env.observation_space.low, self.env.observation_space.high)

        print(f"Action space limits (low, high)= {self.action_space_lim}\nState space limits (low, high)= "
              f"{self.state_space_lim}")
        print(f"Action space dim= {self.action_space_dim} State space dim= {self.state_space_dim}")

        self.env = DummyVecEnv([lambda: gym.make(gym_name)])
        # norm both seems to perform best
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True,
                                clip_obs=self.state_space_lim[1])

        self.model_save_dir = model_save_dir
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        else:
            raise Exception("Directory already exists")

        self.render_env = render_env
        if self.render_env:
            self.env.render()

        self.surrogate_objective = surrogate_objective
        self.hyperparameter = Hyperparameter()

        # covariance chosen the same for all dimension (assuming that the action space is normed!
        self.cov_mat = torch.diag(torch.tensor([self.hyperparameter.var] * self.action_space_dim))

        self.policy_network = PolicyNetwork(input_dim=self.state_space_dim,
                                            output_dim=self.action_space_dim,
                                            covariance_mat=self.cov_mat,
                                            activation=torch.relu)
        self.policy_network_optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                                         lr=self.hyperparameter.base_lr)

        self.value_func_network = ValueFunctionNetwork(input_dim=self.state_space_dim,
                                                       activation=torch.relu)
        self.value_func_network_optimizer = torch.optim.Adam(self.value_func_network.parameters(),
                                                             lr=self.hyperparameter.base_lr)

        self.c_time_step = 0

    def forward(self) -> Tuple[float, float, float, float]:
        """
        Learning is done in here, this function must be called repeatedly for #training_steps
        :return:
        """
        # sample data - Run policy ??_??_old in environment for T time steps using N actors
        rollout_buffer = self.sample_data(N=self.hyperparameter.N, T=self.hyperparameter.T)
        self.c_time_step += len(rollout_buffer.observations)
        actions, log_probs, observations, rewards, episode_length, unnormed_rewards = rollout_buffer.to_tensor()

        advantage_vals, rewards_togo = self.compute_advantage_values(observations, actions, rewards, episode_length)

        # Optimize surrogate L w.r.t. ?? with K epochs and minibatch size M ??? N T
        polidcy_net_loss, value_net_loss = self.optimize_neural_networks(observations, actions, log_probs,
                                                                         advantage_vals, rewards_togo,
                                                                         K=self.hyperparameter.K)

        avg_return, avg_len = self.compute_avg_episodic_returns(unnormed_rewards, episode_length)

        return polidcy_net_loss, value_net_loss, avg_return, avg_len

    def optimize_neural_networks(self, observations, actions, log_probs, advantage_values, rewards_togo, K) \
            -> Tuple[float, float]:
        """
        Optimize surrogate L w.r.t. ?? with K epochs and minibatch size M ??? N T
        :return:
        """
        avg_policy_net_loss = 0.0
        avg_value_net_loss = 0.0

        for step in range(K):
            # this expression changes as the NN is updated!
            value, current_log_probs = self.evaluate_value_function(observations, actions)

            # compute surrogate objective function
            policy_net_loss = 0.0
            if self.surrogate_objective == "clipped":
                policy_net_loss = self.clipped_surrogate_function(old=log_probs, new=current_log_probs,
                                                                  advantage_values=advantage_values)
            elif self.surrogate_objective == "adaptive_KL":
                policy_net_loss = self.adaptive_KL_surrogate_function(old=log_probs, new=current_log_probs,
                                                                      advantage_values=advantage_values, clip=True)
            elif self.surrogate_objective == "policy_gradient":
                policy_net_loss = self.policy_gradient_surrogate_funcion()

            self.policy_network_optimizer.zero_grad()
            policy_net_loss.backward(retain_graph=True)
            self.policy_network_optimizer.step()

            value_function_net_loss = nn.HuberLoss()(value.squeeze(), rewards_togo)
            self.value_func_network_optimizer.zero_grad()
            value_function_net_loss.backward()
            self.value_func_network_optimizer.step()

            avg_policy_net_loss += policy_net_loss.detach()
            avg_value_net_loss += value_function_net_loss.detach()

        return avg_policy_net_loss / K, avg_value_net_loss / K

    def compute_rewards_togo(self, rewards: torch.Tensor, episode_lengths: torch.Tensor) \
            -> torch.Tensor:
        """
        Implementation of the advantage function estimator
        see https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/ for reference

        :param rewards:
        :param episode_lengths:
        :return: advantage values torch.Tensor
        """
        discounted_rewards = []
        offset = 0
        for i in range(self.hyperparameter.N):
            sum = 0
            length = episode_lengths[-(i + 1)].item()  # reverse
            for j in range(length):
                # length is the length of the current episode
                sum = rewards[-(j + offset + 1)] + self.hyperparameter.gamma * sum
                discounted_rewards.insert(0, sum)
            offset += length
        discounted_rewards = torch.tensor(discounted_rewards)

        return discounted_rewards

    def evaluate_value_function(self, observations: torch.Tensor, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimation of the value function V(s) by evaluating the NN
        :param observations:
        :param actions:
        :return:
        """
        value = self.value_func_network(observations)
        mean = self.policy_network(observations, return_mean=True)
        distribution = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        log_probs = distribution.log_prob(actions)

        return value, log_probs.squeeze()

    def evaluate_policy(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the actor policy network to generate an action
        Is used for the on-policy algorithm

        :param observation last observation that was observed from the env, used to evaluate pi(a|s)
        :return: action sampled from the current policy and its log probability
        """
        action_sample, log_prob = self.policy_network(observation)
        return action_sample, log_prob

    def compute_advantage_values(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                                 episode_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the advantage values and the reward to go
        :param observations:        observations from the rollout buffer as torch.Tensor
        :param actions:             actions from the rollout buffer as torch.Tensor
        :param rewards:             rewards from the rollout buffer as torch.Tensor
        :param episode_length:      episode length of each episode
        :return:                    advantage_values and reward-to-go
        """
        # compute advantage estimates A_1, \dots, A_T
        value, _ = self.evaluate_value_function(observations, actions)

        rewards_togo = self.compute_rewards_togo(rewards, episode_length)
        # graph of values is not required --> detach
        advantage_vals = rewards_togo - value.detach().squeeze()
        # normalize https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
        advantage_vals = (advantage_vals - advantage_vals.mean()) / \
                         (advantage_vals.std() + self.hyperparameter.numeric_stable)
        return advantage_vals, rewards_togo

    def sample_data(self, N: int, T: int) -> RolloutBuffer:
        """
        Computation of the rollout buffer, sampling data from the environment
        :param N:   number of actors in parallel / different episodes that are considered
        :param T:   nubmer of timesteps per episode
        :return:    RolloutBuffer, includes observation, action, reward, episode_length
        """
        rollout_buffer = RolloutBuffer()

        for actor in range(N):
            observation = self.env.reset()
            # episode of length T
            length = 0
            for t in range(T):
                # convert observation to torch tensor
                rollout_buffer.observations.append(observation)
                observation = torch.tensor(observation)
                # get action - implement action getter from current policy
                action, log_prob = self.evaluate_policy(observation)
                rollout_buffer.actions.append(action)
                rollout_buffer.log_probabilities.append(log_prob)

                observation, reward, done, info = self.env.step(action.detach().numpy())
                rollout_buffer.rewards.append(reward)
                rollout_buffer.unnormed_rewards.append(self.env.unnormalize_reward(reward))

                length = t + 1  # iteration over t start at 0
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
        ratios = torch.exp(new) / torch.exp(old)
        policy_loss1 = ratios * advantage_values
        policy_loss2 = torch.clamp(ratios, 1 - self.hyperparameter.epsilon_clip,
                                   1 + self.hyperparameter.epsilon_clip) * advantage_values
        return - torch.min(policy_loss1, policy_loss2).mean()

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
        # updated beta is used for the next policy update!
        if kl_penalty < (self.hyperparameter.d_target / self.hyperparameter.d_target_ratio):
            self.hyperparameter.beta = 0.5 * self.hyperparameter.beta
        else:
            self.hyperparameter.beta = 2 * self.hyperparameter.beta

        return combined

    def policy_gradient_surrogate_funcion(self, old: torch.Tensor, new: torch.Tensor,
                                          advantage_values: torch.Tensor) -> torch.Tensor:
        """
        todo: implement
        :return:
        """
        raise NotImplementedError

    def compute_avg_episodic_returns(self, rewards: torch.Tensor, episode_lengths: torch.Tensor) -> Tuple[float, float]:
        """
        Computes the average episodic returns
        :param rewards:             unnormalized rewards from the env
        :param episode_lengths:     length of each of the episodes
        :return:                    average episodic reward, average episode length
        """
        cum_return = 0.0
        cum_len = 0.0
        offset = 0

        for episode_len in episode_lengths:
            sum = 0.0
            for i in range(episode_len):
                sum += rewards[i + offset]
                offset = i + 1
            cum_len += episode_len
            cum_return += sum

        avg_return = cum_return / len(episode_lengths)
        avg_len = cum_len / len(episode_lengths)

        return avg_return, avg_len

    def validation(self, num_episodes) -> Tuple[float, List[int]]:
        """
        Perform validation step on env. by evaluating the cuurrent policy and returning the average reward
        :param num_episodes: number of episodes that are simulated
        :return: average reward of the episodes and length of the episodes
        """
        # don't update env stats and norm for reward is not required during validation
        self.env.training = False
        self.env.norm_reward = False

        cum_rewards = 0.0
        episode_lengths = []
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            t = 0

            while not done:
                action, _ = self.evaluate_policy(torch.tensor(obs))
                obs, reward, done, _ = self.env.step(action.detach().numpy())
                cum_rewards += reward
                t += 1

            episode_lengths.append(t)
            print(episode_lengths)

        # training will be resumed and for that normalization is expected
        self.env.training = True
        self.env.norm_reward = True
        return cum_rewards / num_episodes, episode_lengths

    def save_model(self) -> None:
        """
        Saves the policy and value function network on the specified path
        self.model_save_dir/policy_net.pth
        self.model_save_dir/value_net.pth
        :return: None
        """
        torch.save(self.policy_network.state_dict(), os.path.join(self.model_save_dir, "policy_net.pth"))
        torch.save(self.value_func_network.state_dict(), os.path.join(self.model_save_dir, "value_net.pth"))
        self.env.save(os.path.join(self.model_save_dir, "vec_normalize.pkl"))
        print(f"Saved model parameters to {self.model_save_dir}")

    def load_model(self, model_path: str) -> None:
        """
        Load model located at model_path/policy_net.pth and model_path/value_net.pth
        :param model_path: path to the model directory
        :return: None
        """
        policy_net_path = os.path.join(model_path, "policy_net.pth")
        value_net_path = os.path.join(model_path, "value_net.pth")
        env_stats = os.path.join(model_path, "vec_normalize.pkl")
        self.env = VecNormalize.load(env_stats, self.env)
        self.policy_network.load_state_dict(torch.load(policy_net_path))
        self.value_func_network.load_state_dict(torch.load(value_net_path))
