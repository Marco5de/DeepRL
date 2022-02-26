# Deep Reinforcement Learning Project - Winterterm 21/22

## How to run the code?
First, clone the repo
```
git clone https://github.com/Marco5de/DeepRL.git
cd DeepRL
```
I recommend installing all of the required modules in a fresh virtualenv.
```
virtualenv venv
source venv/bin/activate
```
then install most of the required modules directly using pip
```
pip install -r requirements.txt
```
install all of the external dependencies beginning with openAI gym
```
cd src
mkdir extern && cd extern
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
cd ..
```
and pybullet-gym
```
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
cd ../../../
```
with that the installation of all required modules is complete.  
Now to run a training run the `train.py` script which in turn is using the small lib implementing the PPO algorithm.
```
python3 src/train.py
```

## Changing between environments and hyperparameters
Most of the settings for the training are contained in the `train.py` script as global variables.
The following contains a brief description for each of the available settings
* `SAVE_MODEL_FREQ`: model is saved every `SAVE_MODEL_FREQ`
* `LOG_FREQ`: log to stdout every `LOG_FREQ` iterations, note that no logging lib is used but simple print output
* `TRAIN_STEPS`: total number of training iterations, note that this does not directly correspond to the number of training steps
often found in literature as each train_step corresponds to a total of `N * T` timesteps but also may depend on the environment sequence length 
* `ENV_IDX`: Specifies which environment is used, see `ENV_NAMES` for an enumeration of the respective environments 
* `RENDER_ENV`: Specifies if the environment is rendered during training, note that the pybulletgym rendering works different from 
the openai-gym rendering as the `env.render()` function must only be called once!
  
The hyperparameters can theoretically be read from a yaml in the same format as shown in [link](https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml).
Note that not all options are currently implemented and the `read_yaml()` is currently unused!
The simplest way is to manually adjust the default initialization in the ctor of the class.
The following lists the hyperparameters for the environments that were considered in the report.

| Hyperparameter   | Pendulum-v0             | AntPyBulletEnv-v0    
|---------|----------------------|--------------|
| `epsilon_clip` |  0.2      |     |
| `gamma` | 0.99 | |
| `beta` | 1.0 | |
| `d_target_ratio` | 1.5 | |
| `d_target` | 0.25 | |
| `var` | 0.5 | |
| `N` | 2048 | |
| `T` | 200 | |
| `K` | 10 | |
| `numeric_stable` | 1e-10 | |
| `base_lr` | 3e-4 | |

## Logging with Tensorboard
The implementation is using tensorboard to log the loss function values of the policy and value function network.
Additionally, the processing time, average episode length and average episodic reward are all logged.
The default output location is `src/res/log_dir`.
To visualize the data in tensorboard simply start a server with the log directory
```
tensorboard --port 6006 --logdir srd/res/log_dir
```

## Saved models
The default path for model is `src/res/model` each model consists of the two MLP checkpoints `policy_net.pth` and `value_net.pth`
respectively which are simply saved and loaded using the PyTorch library.
Additionally, the normalization of the environment must be preserved which is why a `vec_normalize.pkl` is also present.
To save and load a model simply use the functionality provided in the `PPO.py` class: `load_model()`, `save_model()`.

## Compile report from LaTeX
The report including all resources can be found in the `report` directory.
The LaTeX code is compiled using `latexmk`, thus it is assumed that it is installed alongside all required LaTeX packages.
All of them should be contained in a full `texlive` installation.
The easiest way to build the report is using the provided `Makefile`.
To clean up the generated build files use `make clean`.


TODO - for eval remove sampling!

## Ant

Best run so far - Run 3

### Run1
self.epsilon_clip = 0.2  # cliprange
self.gamma = 0.99
self.var = 0.25  # try different variance!
self.N = 256 * 8  # nsteps #todo: ist das mult mit n_env sinnvoll?
self.T = 32   # minibatches / max steps per episode
self.K = 10  # noptepochs
self.base_lr = 2.5e-4

results:
around 1 mio: 35-37
End: no significant improvement

### Run2
self.epsilon_clip = 0.2  # cliprange
self.gamma = 0.99
self.var = 0.5  # try different variance!
self.N = 256 * 8  # nsteps #todo: ist das mult mit n_env sinnvoll?
self.T = 32   # minibatches / max steps per episode
self.K = 10  # noptepochs
self.base_lr = 2.5e-4

results:
around 1 mio: 30-33 
End:

### Run3
self.epsilon_clip = 0.2  # cliprange
self.gamma = 0.99
self.var = 0.1  # try different variance!
self.N = 256 * 8  # nsteps #todo: ist das mult mit n_env sinnvoll?
self.T = 32   # minibatches / max steps per episode
self.K = 10  # noptepochs
self.base_lr = 2.5e-4

results:
around 1 mio: 37.5-40 (1.17 - 1.25)
End:


