from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

env = gym.make('SuperMarioBros-v0',
               apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Define the actions based on SIMPLE_MOVEMENT
RIGHT = 1
LEFT = 2
RIGHT_JUMP = 3

done = True
env.reset()

action_sequence = [RIGHT] * 10 + [RIGHT_JUMP] * 10 + [RIGHT] * 10 + [LEFT] * 10
action_index = 0

for step in range(5000):
    action = action_sequence[action_index]
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    if done:
        env.reset()

    action_index = (action_index + 1) % len(action_sequence)

env.close()
