#!/usr/bin/env python3

import numpy as np
import pickle
from car_racing import *
from concurrent.futures import *
from mlp import *
import pyglet

N_CTRL_PTS = 8


env = CarRacing()
env.reset()
env2 = CarRacing()
env2.reset()

user_action = np.array( [0.0, 0.0, 0.0] ) # steer [-1, 1], gas [0, 1], brake [0, 1]
restart = False


# Fitness function
def fitness(env, dna, dna2):
    global N_CTRL_PTS, restart
    _, _, done, state  = env.fast_reset()
    _, _, done2, state2  = env2.fast_reset()

    max_reward2 = 0
    env2.car.hull.color = (0.2,0.2,0.8)
    
    [reward, on_road, laps] = state[3:6]
    step = 1
    while True:
        while not restart:
            angle_deltas2 = state2[2]
            inp2 = np.append(angle_deltas2[:N_CTRL_PTS], state2[6:12])
            reaction2 = dna2.feed(inp2) # steer [-1, 1], gas [0, 1], brake [0, 1]
            _, _, done2, state2 = env2.step(reaction2)

            angle_deltas = state[2]
            inp = np.append(angle_deltas[:N_CTRL_PTS], state[6:12])
            reaction = dna.feed(inp) # steer [-1, 1], gas [0, 1], brake [0, 1]
            _, _, done, state = env.step(user_action)

            step += 1
            env.set_car2(env2.car)
            env.render()
        env.fast_reset()
        env2.fast_reset()
        env2.car.hull.color = (0.2,0.2,0.8)
        restart = False



dnas = None
last_fittest_dna = None
try:
    with open('saved_dnas', 'rb') as f:
        saved = pickle.load(f)
        dnas = saved['dnas']
        last_fittest_dna = saved['last_fittest_dna']
except BaseException as e:
    print('Error reading file:', e)
    exit()

from pyglet.window import key
a = user_action
def key_press(k, mod):
    global restart
    if k==0xff0d:    restart = True
    if k==key.LEFT:  a[0] = -1.0
    if k==key.RIGHT: a[0] = +1.0
    if k==key.UP:    a[1] = +1.0
    if k==key.DOWN:  a[2] = +0.5
def key_release(k, mod):
    if k==key.LEFT  and a[0]==-1.0: a[0] = 0
    if k==key.RIGHT and a[0]==+1.0: a[0] = 0
    if k==key.UP:                   a[1] = 0
    if k==key.DOWN:                 a[2] = 0
env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
fitness(env, last_fittest_dna, dnas[-1])


