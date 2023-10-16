from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sys import argv
from stable_baselines3.common.vec_env import SubprocVecEnv

global_last_info = None

# https://python.plainenglish.io/build-an-ai-model-to-play-super-mario-7607b1ec1e17
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
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(model_path)
            print(f"Saved model at timestep {self.n_calls}")

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

        truncated = False  
        return obs, reward, done, truncated, info


def custom_reward(info, last_info):
    reward = 0

    # Distance Reward
    reward += (info['x_pos'] - last_info['x_pos'])  

    # Life Penalty
    if info['life'] < last_info['life']:
        reward -= 1000  # Increased penalty for losing a life

    # Completion Bonus
    if info['flag_get']:
        reward += 5000  # Increased bonus for completing the level
    
    if last_info['stage'] < info['stage']:
        reward += 10000

    if last_info['score'] < info['score']:
        reward += 100
    
    if last_info['coins'] < info['coins']:
        reward += 100

    return reward


# Setup environment

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


def make_env():
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = SkipFrame(env, skip=4)
    env = CustomRewardWrapper(env)
    env = GrayScaleObservation(env, keep_dim=True)

    return env


if __name__ == '__main__':
    num_envs = 1
    env = SubprocVecEnv([make_env for i in range(num_envs)])
    env = VecFrameStack(env, n_stack=4, channels_order="last")


    params = {
        "verbose": 1,
        "tensorboard_log": "./logs"
    }

    if len(argv) != 2:
        # Create model and logger
        logger = TrainAndLoggingCallback(check_freq=100000, save_path="./models")
        model = PPO("MlpPolicy", env, **params)
    else:
        # Load model from CLI and logger
        model = PPO.load(argv[1], env=env)
        logger = TrainAndLoggingCallback(check_freq=100000, save_path=os.path.dirname(argv[1]))

    try:
        model.learn(total_timesteps=2500000, callback=logger)
        final_model_path = os.path.join(logger.save_path, "final_model")
        model.save(final_model_path)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving the current model...")
        interrupted_model_path = os.path.join(logger.save_path, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"Model saved to {interrupted_model_path}")