# https://terbium.io/2018/11/wave-function-collapse/

import random
import io
import base64
from collections import namedtuple
import numpy as np
from PIL import Image
from IPython import display
from enum import Enum, auto

# display partial and final states of generated pattern
def blend_many(ims):
    '''
    blends a sequence of images
    '''
    current, *ims = ims
    for i, im in enumerate(ims):
        current = Image.blend(current, im, 1/(i+2))
    return current


def blend_tiles(choices, tiles):
    '''
    given a list of states (True if ruled out, False if not)
    for each tile, and a list of tiles, return a blend of all
    the tiles that haven't been ruled out
    '''
    to_blend = [tiles[i].bitmap for i in range(len(choices)) if choices[i]]
    return blend_many(to_blend)


def show_state(potential, tiles):
    '''
    given a list of states for each tile for each position of the image,
    return an image representing the state of the global image
    '''
    rows = []
    for row in potential:
        rows.append([np.asarray(blend_tiles(t, tiles)) for t in row])

    rows = np.array(rows)
    n_rows, n_cols, tile_height, tile_width, _ = rows.shape
    images = np.swapaxes(rows, 1, 2)
    return Image.fromarray(images.reshape(n_rows*tile_height, n_cols*tile_width, 4))


# use to find indices of celss that are True in bool arrays
def find_true(array):
    '''
    like np.nonzero
    '''
    transform = int if len(np.asarray(array).shape) == 1 else tuple
    return list(map(transform, np.transpose(np.nonzero(array))))


# load bitmaps
straight_image = Image.open(io.BytesIO(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAMklEQVQYlWNQVrT9r6xo+988UfN/0yqN/4evOP0/fMXpf9Mqjf/miZr/YfIMowrpqxAAjKLGXfWE8ZAAAAAASUVORK5CYII=')))
bend_image = Image.open(io.BytesIO(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAANklEQVQYlWNQVrT9TwxmIFmheaImXoyisGmVBk6MofDwFSesmHKFRFvdtEoDv2fQFWINHnwKAQHMxl1/fce/AAAAAElFTkSuQmCC')))
blank_image = Image.open(io.BytesIO(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFElEQVQYlWNQVrT9TwxmGFVIX4UAoDOWARI9hF0AAAAASUVORK5CYII=')))
cross_image = Image.open(io.BytesIO(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAU0lEQVQYlWNQVrT9r6xo+988UfN/0yqN/4evOP0/fMXpf9Mqjf/miZr/YfIMRCs0T9T8D8PYFMIwQ9Mqjf/IGFkhMmaASRDCxCsk2mqiPUP1cAQAKI/idfPNuccAAAAASUVORK5CYII=')))
t_image = Image.open(io.BytesIO(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAWUlEQVQYlWNQVrT9r6xo+988UfN/0yqN/4evOP0/fMXpf9Mqjf/miZr/YfIMRCs0T9T8D8PYFMIwQ9Mqjf/IGFkhMmaASRDCxCtEtwIXRvEMPgwPHkKYaIUAow/UaQFDAc4AAAAASUVORK5CYII=')))

# define representation for these tiles
# we need a way of telling algorithm which tiles can go next to which
# our only constraint is pipe/no pipe
# so we'll keep track of which sides of the tile have pipes
# can replace this True/False choice with any enum to describe more complex constraints
Tile = namedtuple('Tile', ('name', 'bitmap', 'sides', 'weight'))

# turn 5 tiles we have into 12 by generating rotations manually
tiles = [
    Tile('straight_ud', straight_image,
         [False, True, False, True], 1/2),
    Tile('straight_lr', straight_image.transpose(Image.ROTATE_90),
         [True, False, True, False], 1/2),
    Tile('bend_br', bend_image,
         [True, False, False, True], 1/4),
    Tile('bend_tr', bend_image.transpose(Image.ROTATE_90),
         [True, True, False, False], 1/4),
    Tile('bend_tl', bend_image.transpose(Image.ROTATE_180),
         [False, True, True, False], 1/4),
    Tile('bend_bl', bend_image.transpose(Image.ROTATE_270),
         [False, False, True, True], 1/4),
    Tile('t_u', t_image,
         [True, True, True, False], 1/4),
    Tile('t_l', t_image.transpose(Image.ROTATE_90),
         [False, True, True, True], 1/4),
    Tile('t_d', t_image.transpose(Image.ROTATE_180),
         [True, False, True, True], 1/4),
    Tile('t_r', t_image.transpose(Image.ROTATE_270),
         [True, True, False, True], 1/4),
    Tile('blank', blank_image,
         [False, False, False, False], 1),
    Tile('cross', cross_image,
         [True, True, True, True], 1)
]

# create dedicated array with tile weights
# easy to quickly find the weight of given tile
weights = np.asarray([t.weight for t in tiles])

# generate 30 by 30 element image
# initially create image as a grid where each element is a superposition of all possible tiles
# state represented by an array of 30x30x12 (12 states)
# call this array potential
# for example, the value potential[12][2][5] is True iff the element at coords (12, 5) can be tile 5
# goal is to reduct potential until we've picked one tile for each element
potential = np.full((30, 30, len(tiles)), True)
# so far looks like blend of all possible states
display.display(show_state(potential, tiles))

# WFC algorithm
# how do we reduce possibilities?
# initially, we can just choose a location at random and assign a random tile to that location
# may reduce possibilities for other locations around that initial one
# once we have propagated these changes, we are ready to pick the next tile

# 1. if all locations are decided, we are done
# 2. if any location has no choices left, we've failed
# 3. otherwise, find the undecided location with the fewest remaining choices
# 4. (collapse step) pick a tile at random among those that are still possible for this location
#    mark this tile as the only remaining possibility for this location
# 5. (propagation step) this choice is likely to rule out some possible tiles at neighbouring locations
#    we need to update our potential to reflect this
# 6. loop

def location_with_fewest_choices(potential):
    num_choices = np.sum(potential, axis=2, dtype='float32')
    num_choices[num_choices == 1] = np.inf
    candidate_locations = find_true(num_choices == num_choices.min())
    location = random.choice(candidate_locations)
    if num_choices[location] == np.inf:
        return None
    return location


# constraint propagation
class Direction(Enum):
    RIGHT = 0; UP = 1; LEFT = 2; DOWN = 3

    def reverse(self):
        return {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }[self]


def neighbours(location, height, width):
    res = []
    x, y = location
    if x != 0:
        res.append((Direction.UP, x-1, y))
    if y != 0:
        res.append((Direction.LEFT, x, y-1))
    if x < height - 1:
        res.append((Direction.DOWN, x+1, y))
    if y < width - 1:
        res.append((Direction.RIGHT, x, y+1))
    return res


# builds a set of th esides that are still possible at the source location
def add_constraint(potential, location, incoming_direction, possible_tiles):
    neighbour_constraint = {t.sides[incoming_direction.value] for t in possible_tiles}
    outgoing_direction = incoming_direction.reverse()
    changed = False
    for i_p, p in enumerate(potential[location]):
        if not p:
            continue
        if tiles[i_p].sides[outgoing_direction.value] not in neighbour_constraint:
            potential[location][i_p] = False
            changed = True
    if not np.any(potential[location]):
        raise Exception(f"No patterns left at {location}")
    return changed


# mark the cell we just collapsed as needing an update
# for each cell that needs an update U
#   for each of that cell's neighbours N
#       remove any tile that is no longer compatible with U's possible tiles
#       if we did remove any tiles, we need to propagate any new constraints 
#       so mark N as needing an update on the next iteration
# if any cells still need to be updated, go back to step 2, otherwise we are done
def propagate(potetial, start_location):
    height, width = potential.shape[:2]
    needs_update = np.full((height, width), False)
    needs_update[start_location] = True
    while np.any(needs_update):
        needs_update_next = np.full((height, width), False)
        locations = find_true(needs_update)
        for location in locations:
            possible_tiles = [tiles[n] for n in find_true(potential[location])]
            for neighbour in neighbours(location, height, width):
                neighbour_direction, neighbour_x, neighbour_y = neighbour
                neighbour_location = (neighbour_x, neighbour_y)
                was_updated = add_constraint(potential, neighbour_location, neighbour_direction, possible_tiles)
                needs_update_next[location] |= was_updated
        needs_update = needs_update_next


def run_iteration(old_potential):
    potential = old_potential.copy()
    to_collapse = location_with_fewest_choices(potential)
    if to_collapse is None:
        raise StopIteration()
    elif not np.any(potential[to_collapse]):
        raise Exception(f"No choices left at {to_collapse}")
    else:
        nonzero = find_true(potential[to_collapse])
        tile_probs = weights[nonzero]/sum(weights[nonzero])
        selected_tile = np.random.choice(nonzero, p=tile_probs)
        potential[to_collapse] = False
        potential[to_collapse][selected_tile] = True
        propagate(potential, to_collapse)
    return potential


p = potential
images = [show_state(p, tiles)]
while True:
    try:
        p = run_iteration(p)
        images.append(show_state(p, tiles))
    except StopIteration as e:
        break
    except Exception as e:
        print(e)
        break


out = io.BytesIO()
images[0].save(out, format='gif', save_all=True, append_images=images[1:], duration=50, loop=0)
images[-1]