from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym
import cv2 as cv
import numpy as np
import string
from gym.spaces import Box
import matplotlib.pyplot as plt


# ------- STABLE BASELINE --------
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# --------------------------------


PRINT_GRID      = True
PRINT_LOCATIONS = False

SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.8
SKY_COLOUR = None

# Number of frames for which the jump button will be held
JUMP_FRAMES = 15


MASK_COLOUR = np.array([252, 136, 104])

image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "block": ["block1.png", "block2.png", "block3.png"],
        "stair": ["block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        "mushroom": ["mushroom_red.png"],
    }
}

def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# PRINTING THE GRID (for debug purposes)

colour_map = {
    (104, 136, 252): " ", # sky blue colour
    (0,     0,   0): " ", # black
    (252, 252, 252): "'", # white / cloud colour
    (248,  56,   0): "M", # red / mario colour
    (228,  92,  16): "%", # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()),reverse=True)
DEFAULT_LETTER = "?"

def _get_colour(colour): # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]
    
    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER

def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x+i, y)] = name_str[i%len(name_str)]
                pixels[(x+i, y+height-1)] = name_str[(i+height-1)%len(name_str)]
            for i in range(1, height-1):
                pixels[(x, y+i)] = name_str[i%len(name_str)]
                pixels[(x+width-1, y+i)] = name_str[(i+width-1)%len(name_str)]

    # print the screen to terminal
    print("-"*SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
        print("".join(line))

def detect_ground(frame):
    sky_row = 10
    sky_color = frame[sky_row, frame.shape[1] // 2]  

    ground_row = int(frame.shape[0] * 0.9)  
    ground_color = frame[ground_row, frame.shape[1] // 2 + 15]

    is_missing_ground = np.array_equal(sky_color, ground_color)

    return is_missing_ground


################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions


        if stop_early and locations:
            break
    
    #      [((x,y), (width,height))]
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

from nes_py import NESEnv
_reset = NESEnv.reset

def reset(*args, **kwargs):
    obs_info = _reset(*args, **kwargs)
    if isinstance(obs_info, tuple):
        obs, info = obs_info
        return obs
    else:
        return obs_info


NESEnv.reset = reset

################################################################################
# WRAPPERS


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


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

        self.last_info = None  # Initialize last_info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # If last_info is None (i.e., the first step), set it to the current info
        if self.last_info is None:
            self.last_info = info

        # Modify the reward
        reward += custom_reward(info, self.last_info)
        

        self.last_info = info


        truncated = False  # Set a default value or compute it based on your needs
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.last_info = None
        return self.env.reset(**kwargs)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv.resize(observation, self.shape, interpolation=cv.INTER_AREA)
        return observation

################################################################################
def make_env():
    env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = JoypadSpace(env, RIGHT_ONLY)
    obs = env.reset()

    env = ResizeObservation(env, 84)
    env = SkipFrame(env, skip=5)
    env = CustomRewardWrapper(env)

    return DummyVecEnv([lambda: env])

def custom_reward(info, last_info):
    reward = 0

    # Distance Reward
    reward += 20 * (info['x_pos'] - last_info['x_pos'])  


    # Coin Reward
    reward += (info['coins'] - last_info['coins']) * 5  # Assuming 5 points for each coin

    # Life Penalty
    if info['life'] < last_info['life']:
        reward -= 1000  # Assuming a big penalty for losing a life

    # Completion Bonus
    if info['flag_get']:
        reward += 1000

    # Time Penalty
    reward -= 0.5


    return reward


env = make_env()

hyperparameters = {
    # General
    "learning_rate": 0.00008,         # Learning rate
    "batch_size": 256,               # Batch size
    "gamma": 0.99,                  # Discount factor
    "ent_coef": 0.015,               # Entropy coefficient

    # Other
    "verbose": 1,                   # Verbosity level during training
}


# Create the PPO model with the specified hyperparameters
model = PPO("MlpPolicy", env, **hyperparameters)

model.load("ppo_mario")
model.learn(total_timesteps=10000)
model.save("ppo_mario")



def plot_learning_curve(episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, marker='o')
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.show()

# Store episode rewards for plotting
episode_rewards = []

num_episodes = 10
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs = np.ascontiguousarray(obs)  # Ensure the observation has positive strides
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1} finished. Total reward: {total_reward}")

# Plot the learning curve
plot_learning_curve(episode_rewards)

env.close()
