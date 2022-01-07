import torch
import torch.nn as nn
import torch.distributions


class PolicyNetwork(nn.Module):
    """
    Implements the network that is used to model a general statistical policy
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layer_widths: list = [64, 64],
                 activation: callable = torch.tanh,
                 covariance_mat: torch.Tensor = None):
        """
        Ctor
        :param input_dim: dimension of the state space
        :param output_dim: dimension of the action space
        :param hidden_layer_widths: number of units in the respective hidden layer, length must be 2
        :param activation: callable activation function used throughout the network
        :param covariance_mat: dimension state_space_dim x state_space_dim
        """
        super().__init__()

        assert len(hidden_layer_widths) == 2
        assert covariance_mat is not None

        self.activation = activation
        self.covariance_mat = covariance_mat

        self.input = nn.Linear(input_dim, hidden_layer_widths[0])
        self.hidden = nn.Linear(hidden_layer_widths[0], hidden_layer_widths[1])
        self.output = nn.Linear(hidden_layer_widths[1], output_dim)

    def forward(self, t: torch.Tensor, return_mean: bool = False):
        """
        Implements the forward pass of the policy network

        :param t: input tensor
        :param return_distribution if true the multivariate gaussian is also returned (see VF eval)
        :return: output sampled from the distribution outputted by the NN
        """
        t = self.activation(self.input(t))
        t = self.activation(self.hidden(t))
        t = self.output(t)

        if return_mean:
            return t

            # t are now the mean parameters, use that to create a multivariate gaussian to be sampled
        distribution = torch.distributions.MultivariateNormal(t, self.covariance_mat)
        sample = distribution.sample()
        # get log prob for the sample, detach as graph is not required!
        log_prob = distribution.log_prob(sample).detach()

        return sample, log_prob


class ValueFunctionNetwork(nn.Module):
    """
    Implements a network to approximate / model the value function
    """

    def __init__(self, input_dim: int, hidden_layer_widths: list = [64, 64], activation: callable = torch.tanh):
        """
        Ctor
        :param input_dim: dimension of the state space
        :param hidden_layer_widths: number of units in the hidden layers, length must be 2
        :param activation: callable activation function used in the network
        """
        super().__init__()

        assert len(hidden_layer_widths) == 2

        self.activation = activation

        self.input = nn.Linear(input_dim, hidden_layer_widths[0])
        self.hidden = nn.Linear(hidden_layer_widths[0], hidden_layer_widths[1])
        # the output of the value function is always a scalar
        self.output = nn.Linear(hidden_layer_widths[1], 1)

        self.distribution = None

    def forward(self, t: torch.Tensor):
        """
        Implements the forward pass of the value function network

        :param t: input tensor
        :return: value function value
        """
        t = self.activation(self.input(t))
        t = self.activation(self.hidden(t))
        t = self.output(t)

        return t


class ActorCriticNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_layer_widths: list = [64, 64],
                 covariance_mat: torch.Tensor = None):
        super().__init__()

        assert len(hidden_layer_widths) == 2
        assert covariance_mat is not None

        self.covariance_mat = covariance_mat

        self.common_base = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_widths[0]),
            nn.Tanh(),
            nn.Linear(hidden_layer_widths[0], hidden_layer_widths[1]),
            nn.Tanh()
        )

        self.policy_head = nn.Linear(hidden_layer_widths[1], output_dim)
        self.value_head = nn.Linear(hidden_layer_widths[1], 1)

        self.distribution = None

    def eval_policy(self, obs: torch.Tensor):
        mean = self.policy_head(self.common_base(obs))

        # t are now the mean parameters, use that to create a multivariate gaussian to be sampled
        self.distribution = torch.distributions.MultivariateNormal(mean, self.covariance_mat)
        sample = self.distribution.sample()

        # get log prob for the sample, detach as graph is not required!
        log_prob = self.distribution.log_prob(sample).detach()

        return sample, log_prob

    def eval_value(self, obs: torch.Tensor):
        return self.value_head(self.common_base(obs))


def test_ANN():
    # Make sure that conv-mat is float
    cov_mat = torch.tensor([[1.0, 0, 0],
                            [0, 1.0, 0],
                            [0, 0, 1.0]])
    policy_net = PolicyNetwork(input_dim=10,
                               output_dim=3,
                               covariance_mat=cov_mat)

    actor_critic = ActorCriticNetwork(input_dim=10, output_dim=3, covariance_mat=cov_mat)

    t = torch.normal(torch.arange(0.0, 10.0))
    sample, log_prob = policy_net(t)
    print(f"Output of policy net: sample={sample}, log_prob={log_prob}")

    value_func_net = ValueFunctionNetwork(10)
    value = value_func_net(t)
    print(f"Value function network= {value}")

    sample, log_prob = actor_critic.eval_policy(t)
    print(log_prob)
    print(sample.shape, log_prob.shape)

    val = actor_critic.eval_value(t)
    print(val.shape)


if __name__ == "__main__":
    test_ANN()
