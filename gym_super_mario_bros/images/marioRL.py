from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sys import argv
import optuna
global_last_info = None

# Taken from https://github.com/nicknochnack/MarioRL/blob/main/Mario%20Tutorial.ipynb
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)

        return True

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

        self.last_info = None  # Initialize last_info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # If last_info is None (i.e., the first step), set it to the current info
        if self.last_info is None:
            self.last_info = info

        global global_last_info
        global_last_info = self.last_info

        # Modify the reward
        reward += custom_reward(info, self.last_info)
        

        self.last_info = info


        truncated = False  # Set a default value or compute it based on your needs
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.last_info = None
        return self.env.reset(**kwargs)


def custom_reward(info, last_info):
    reward = 0

    # Distance Reward
    reward += (info['x_pos'] - last_info['x_pos'])  

    # Life Penalty
    if info['life'] < last_info['life']:
        reward -= 200  # Increased penalty for losing a life

    # Completion Bonus
    if info['flag_get']:
        reward += 500  # Increased bonus for completing the level

    # Time Penalty
    # Changed the penalty to a smaller value to not overly penalize the agent for time spent.
    reward -= 0.1

    # Penalty for standing still
    if info['x_pos'] == last_info['x_pos']:
        reward -= 1

    return reward


# Setup environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info


def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    n_steps = trial.suggest_int('n_steps', 128, 2048, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00001, 0.1)
    
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = SkipFrame(env, skip=4)
    env = CustomRewardWrapper(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])  # type: ignore
    env = VecFrameStack(env, 4, channels_order="last")

    model = PPO("CnnPolicy", env, verbose=0, 
                learning_rate=learning_rate, 
                n_steps=n_steps, 
                gamma=gamma, 
                ent_coef=ent_coef, 
                tensorboard_log="./logs")

    logger = TrainAndLoggingCallback(check_freq=1000, save_path="./models")
    model.learn(total_timesteps=1000, callback=logger)  # reduce timesteps for quicker trials

    # Access the global variable
    global global_last_info
    final_x_pos = global_last_info['x_pos'] if global_last_info else 0

    return final_x_pos



if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)  # You can adjust n_trials as needed

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

# env = SkipFrame(env, skip=4)
# env = CustomRewardWrapper(env)
# # Preprocess environment
# env = GrayScaleObservation(env, keep_dim=True)
# env = DummyVecEnv([lambda: env]) # type: ignore
# env = VecFrameStack(env, 4, channels_order="last")

# # Run training or model based on CLI value
# # If value given then run that model
# if len(argv) != 2:
#     # Create model and logger
#     logger = TrainAndLoggingCallback(check_freq=100000, save_path="./models")
#     model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.00001, n_steps=512, tensorboard_log="./logs")


#     model.learn(total_timesteps=10000000, callback=logger)
# else:
#     # Load model from CLI
#     model = PPO.load(argv[1])

#     observation = env.reset()

#     # Run agent in environment
#     while True:
#         action, _ = model.predict(observation)
#         state, reward, truncated, done, info = env.step(action)
#         if done:
#             observation = env.reset()
#         env.render()