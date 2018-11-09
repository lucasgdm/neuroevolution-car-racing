import numpy as np
import pickle
from car_racing import *
from concurrent.futures import *
from mlp import *

N_CTRL_PTS = 8


env = CarRacing()
env.reset()
env2 = CarRacing()
env2.reset()


# Fitness function
def fitness(env, dna, dna2):
    global N_CTRL_PTS
    _, _, done, state  = env.fast_reset()
    _, _, done2, state2  = env2.fast_reset()

    max_reward2 = 0
    env2.car.hull.color = (0.2,0.2,0.8)
    
    [reward, on_road, laps] = state[3:6]
    step = 1
    max_reward = 0
    max_reward_step = 0
    while on_road and laps < 5:
        if step - max_reward_step > 4.5*FPS: # didnt increase the reward fast enough
            break

        angle_deltas2 = state2[2]
        inp2 = np.append(angle_deltas2[:N_CTRL_PTS], state2[6:12])
        reaction2 = dna2.feed(inp2) # steer [-1, 1], gas [0, 1], brake [0, 1]
        _, _, done2, state2 = env2.step(reaction2)
        if not done2:
            max_reward2 = np.maximum(max_reward2, state2[3])

        angle_deltas = state[2]
        inp = np.append(angle_deltas[:N_CTRL_PTS], state[6:12])
        reaction = dna.feed(inp) # steer [-1, 1], gas [0, 1], brake [0, 1]
        _, _, done, state = env.step(reaction)

        [reward, on_road, laps] = state[3:6]
        if reward > max_reward:
            max_reward = reward
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


