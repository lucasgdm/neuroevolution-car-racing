import os, sys
import numpy as np
import pickle
from car_racing import *
from concurrent.futures import *
from mlp import *
from multiprocessing import Process, Pipe, Pool, Queue
import multiprocessing.sharedctypes

NTHREADS = 3
POP = 100
MUTATION_RATE = 1 # How many matrix coefficients are expected to be mutated
N_CTRL_PTS = 8
MLP_LAYERS = (N_CTRL_PTS + 6, 8, 4)
SIGMA = 0.07      # mutation magnitude - might be dynamically tweaked
                  # 95% will have mutation between -2*sigma and 2*sigma

np.random.seed(11)
task_queue, result_queue = Queue(), Queue()


# Fitness function
def fitness(env, dna, render=False):
    _, _, done, state  = env.fast_reset()
    
    [reward, on_road, laps] = state[3:6]
    step = 1
    max_reward = 0
    max_reward_step = 0
    while on_road and laps < 5:
        if step - max_reward_step > 4.5*FPS: # didnt increase the reward fast enough
            break
        angle_deltas = state[2]
        inp = np.append(angle_deltas[:N_CTRL_PTS], state[6:12])
        reaction = dna.feed(inp) # steer [-1, 1], gas [0, 1], brake [0, 1]
        _, _, done, state = env.step(reaction)

        [reward, on_road, laps] = state[3:6]
        if reward > max_reward:
            max_reward = reward
            max_reward_step = step
        step += 1
        if render:
            env.render()
    return max_reward

# Utility to get a percentage of the population in absolute numbers
def pct(p):
    return POP*p//100

# Parallel fitness measurement
def measure_fitness(dnas):
    fitnesses = np.array([-100] * len(dnas))
    for item in enumerate(dnas):
        task_queue.put(item)
    for _ in range(len(dnas)):
        i, fit = result_queue.get()
        fitnesses[i] = fit
    return fitnesses

# Entry function for child processes
def process_entry(task_queue, result_queue):
    penv = CarRacing()
    penv.seed(5)
    penv.reset()
    np.random.seed(5)

    while True:
        i, dna = task_queue.get()
        result_queue.put((i, fitness(penv, dna)))



#
# INITIALIZATION - load from file or start from scratch
#
dnas = None
epoch = 0
fitnesses = np.array([0] * POP)
indices = list(range(POP))
history = []

def initialize_dnas():
    global dnas, epoch, history
    try:
        with open('saved_dnas', 'rb') as f:
            saved = pickle.load(f)
            # Check if the layers haven't changed
            if saved['mlp_layers'] != MLP_LAYERS:
                raise Exception('Incompatible MLP layers')
            dnas, epoch, history = saved['dnas'], saved['epoch'], saved['history']
            pop_diff = saved['pop'] - POP
            # check if the population hasn't changed
            if   pop_diff > 0:
                dnas = dnas[:POP]
            elif pop_diff < 0:
                dnas.extend([dnas[-1]] * pop_diff)
            print('Loaded saved dnas')
    except IOError as e:
        dnas = np.array([MyMLP(MLP_LAYERS) for _ in range(POP)])
        print('Generating new dnas')
    except BaseException as e:
        dnas = np.array([MyMLP(MLP_LAYERS) for _ in range(POP)])
        print('Generating new dnas due to error:', str(e))


#
# SELECTION METHODS
#
def mp_n_tournaments(n):
    global dnas, fitnesses, indices
    np.random.shuffle(indices)
    children = [copy.deepcopy(dnas[i]) for i in indices[:n]]
    for child in children:
        child.mutate(MUTATION_RATE, SIGMA)
    children_fit = measure_fitness(children)
    for child, child_fit in zip(children, children_fit):
        other = np.random.randint(POP)
        if child_fit > fitnesses[other]:
            dnas[other], fitnesses[other] = child, child_fit

def replace_with_fittest(fittest):
    global dnas, fitnesses
    for i in range(POP):
        if i != fittest:
            dnas[i] = copy.deepcopy(dnas[fittest])
    fitnesses = measure_fitness(dnas)



#
# MAIN
#
if __name__ == '__main__':
    print('Pop       :', POP)
    print('Layers    :', MLP_LAYERS)
    print('Sigma     :', SIGMA)
    print('Mut. rate :', MUTATION_RATE)

    processes = [Process(target=process_entry, args=(task_queue, result_queue)) for _ in range(NTHREADS)]
    for p in processes:
        p.start()

    initialize_dnas()
    fitnesses = measure_fitness(dnas)

    if epoch == 0:
        random_dna = np.random.randint(POP)
        last_fittest_dna = copy.deepcopy(dnas[random_dna])
    else:
        last_fittest_dna = copy.deepcopy(dnas[-1])


    print('               Fitnesses')
    print('{:4}  {:>8} {:>8} {:>8} {:>8}'.format('Iter', '100 %ile', '75 %ile', '50 %ile', '25 %ile'))

    f = open('saved_dnas', 'wb', buffering=0)
    # Main loop
    while True:
        # Selection
        mp_n_tournaments(POP)

        # Order by ascending fitness
        idx = np.argsort(fitnesses)
        fitnesses = fitnesses[idx]
        dnas = dnas[idx]

        # Bookkeep fitness distribution for later analysis
        history.append(fitnesses)

        epoch += 1
        print('{:4}  {:8.2f} {:8.2f} {:8.2f} {:8.2f}'.format(
            epoch, fitnesses[-1], fitnesses[pct(75)], fitnesses[pct(50)], fitnesses[pct(25)]))

        if epoch % 5 == 0:
            # Serialize
            f.seek(0)
            pickle.dump(
                    {'dnas': dnas,
                    'epoch': epoch,
                    'mlp_layers': MLP_LAYERS,
                    'pop': POP,
                    'last_fittest_dna': last_fittest_dna,
                    'history': history},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)
            f.truncate()

            # Render fittest
            os.system(sys.executable + ' render.py > /dev/null &')

            last_fittest_dna = copy.deepcopy(dnas[-1])



