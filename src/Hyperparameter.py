import yaml

ENV_TO_KEY = {
    "AntPyBulletEnv-v0": "AntBulletEnv-v0",
    "HalfCheetahPyBulletEnv-v0": "HalfCheetahBulletEnv-v0",
    "Walker2DPyBulletEnv-v0": "Walker2DBulletEnv-v0",
    "InvertedPendulumPyBulletEnv-v0": "Pendulum-v0"
}


class Hyperparameter:
    def __init__(self):
        # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
        self.epsilon_clip = 0.2  # cliprange
        self.gamma = 0.99
        self.beta = 1.0
        self.d_target_ratio = 1.5
        self.d_target = 0.1
        self.var = 0.1  # try different variance!
        self.N = 32  # minibatches
        self.T = 1024  # nsteps
        self.K = 10  # noptepochs
        self.numeric_stable = 1e-9
        self.base_lr = 2.5e-4

    def read_yaml(self, env_name):
        with open("res/hyperparameter.yaml") as f:
            self.hp_dict = yaml.safe_load(f)

        self.T = self.hp_dict[ENV_TO_KEY[env_name]]["n_steps"]
        self.N = self.hp_dict[ENV_TO_KEY[env_name]]["nminibatches"]
        self.gamma = self.hp_dict[ENV_TO_KEY[env_name]]["gamma"]
        self.K = self.hp_dict[ENV_TO_KEY[env_name]]["noptepochs"]
        self.epsilon_clip = self.hp_dict[ENV_TO_KEY[env_name]]["cliprange"]
        self.base_lr = self.hp_dict[ENV_TO_KEY[env_name]]["learning_rate"]
