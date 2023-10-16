from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string
import numpy as np


PRINT_GRID      = True
PRINT_LOCATIONS = False
IMAGE_DIR = './images/'
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.8
SKY_COLOUR = None

# Number of frames for which the jump button will be held
JUMP_FRAMES = 15

# Define basic actions for clarity
RIGHT = 1
RIGHT_JUMP = 2
################################################################################
# TEMPLATES FOR LOCATING OBJECTS

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
    # Prepend the directory path to the filename
    filepath = f"{IMAGE_DIR}{filename}"
    image = cv.imread(filepath)
    # -------------------------------------------
    assert image is not None, f"File {filepath} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None  # this is important for avoiding a problem where some things match everything
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
# LOCATE STAIRS 
def is_stair_in_front(mario_location, block_locations, mario_width):
    mario_x, mario_y = mario_location
    
    for block_location, block_dimensions, block_name in block_locations:
        if block_name != "stair":
            continue
        block_x, block_y = block_location
        
        if block_x > mario_x and (block_x - mario_x) < mario_width + 15:
            
            # Check if the stair block is higher than Mario
            if block_y < mario_y:
                return True, block_y - mario_y
    return False, 0

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

def make_action(screen, info, step, env, prev_action, done):

    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]
    enemy_locations = object_locations["enemy"]
    block_locations = object_locations["block"]

    if mario_locations:
        mario_location, mario_dimensions, mario_name = mario_locations[0]
        mario_x, mario_y = mario_location
        mario_width, mario_height = mario_dimensions

        # Check if there's an enemy in front of Mario
        for enemy_location, enemy_dimensions, enemy_name in enemy_locations:
            enemy_x, enemy_y = enemy_location
            enemy_width, enemy_height = enemy_dimensions

            if mario_y + mario_height > enemy_y and mario_y < enemy_y + enemy_height: 
                if enemy_x - (mario_x + mario_width) < 15: 
                    done = jump(10, done)
                    return RIGHT, done
 
        for block_location, block_dimensions, block_name in block_locations:
            block_x, block_y = block_location

            if block_name == "pipe":
                # Check if the pipe is in front of Mario
                if block_x > mario_x and block_x - (mario_x + mario_width) < 40:
                    done = jump(20,done)
                    return RIGHT, done
                    
        for block_location, block_dimensions, block_name in block_locations:
            block_x, block_y = block_location

            if block_name == "stair":
                if block_x > mario_x and block_x - (mario_x + mario_width) < 7:
                    env.step(RIGHT)
                    done = jump(20,done)
                    
        if detect_ground(screen):
            done = jump(55,done)
            return RIGHT, done

    return RIGHT, done


def jump(x, done):
    for i in range(x):
        if done:  # Check if done is true before each step
            return True  # Return True if terminated early
        _, _, done, _, _ = env.step(RIGHT_JUMP)  # Update done variable
    return False 


################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
SKY_COLOUR = env.reset()[0][0]
obs = env.reset()
done = False
action = RIGHT

for step in range(100000):
    if done:
        obs = env.reset()  # reset environment if done
        action = RIGHT  # reset action if done

    obs, reward, done, truncated, info = env.step(action)  # take a step
    if not done:
        action, done = make_action(obs, info, step, env, action, done)  # get next action

env.close()