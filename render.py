#!/usr/bin/env python3

import numpy as np
import pickle
from car_racing import *
from concurrent.futures import *
from mlp import *

N_CTRL_PTS = 8
LAPS = 2


env = CarRacing()
env.reset()
env2 = CarRacing()
env2.reset()


# Fitness function
def fitness(env, dna, dna2):
    _, _, _, state  = env.fast_reset()
    _, _, _, state2  = env2.fast_reset()

    max_reward2 = 0
    env2.car.hull.color = (0.2,0.2,0.8)
    
    step = 1
    max_reward = 0
    max_reward_step = 0
    while state.on_road and state.laps < 2:
        if step - max_reward_step > 4.5*FPS: # didnt increase the reward fast enough
            break

        reaction2 = dna2.feed(state2.as_array(N_CTRL_PTS))
        _, _, done2, state2 = env2.step(reaction2)
        if not done2:
            max_reward2 = np.maximum(max_reward2, state2.reward)

        reaction = dna.feed(state.as_array(N_CTRL_PTS))
        _, _, _, state = env.step(reaction)

        if state.reward > max_reward:
            max_reward = state.reward
            max_reward_step = step
        step += 1
        env.set_car2(env2.car)
        env.render()

    return max_reward, max_reward2



# Initialize dnas and fitnesses
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


# Render once
fit = fitness(env, dnas[-1], last_fittest_dna)
print('Previous Fittest:', '{:.2f}'.format(fit[1]))
print('Current  Fittest:', '{:.2f}'.format(fit[0]))


