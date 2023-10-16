import gym
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from sys import argv
from marioRL import make_env  # import the make_env function from your training script

def main(model_path):
    # Create environment
    env = make_env()
    
    # Vectorize and stack frames in the environment
    env = DummyVecEnv([lambda: env])  # vectorize environment
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    
    # Load trained model
    model = PPO.load(model_path, env=env)
    
    # Reset environment
    obs = env.reset()
    
    # Run agent in the environment
    while True:
        action, _ = model.predict(obs)
        print(action)
        obs, reward, done, info = env.step(action)
        env.render()  # render the environment to the screen
        if done:
            obs = env.reset()

if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: python run.py <path_to_model>")
    else:
        model_path = argv[1]
        main(model_path)
