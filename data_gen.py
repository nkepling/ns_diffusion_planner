import torch
import numpy as np
import gymnasium as gym
from value_iteration import value_iteration
import ns_gym
from tqdm import tqdm
import warnings

def is_reachable(env, state, goal):
    """Check if a particular state is reachable from the start state.

    TODO: DFS or BFS to check if a state is reachable from the start state.
    """

    # warnings.warn("This function is not implemented yet.")
    return True


def generate_episode_trace(env, num_episodes, max_steps):
    """For a particular environment, generate a dataset of state, action, reward, next_state, and done tuples.
    """
    pass

def generate_new_map(size=8,max_steps=1000,max_holes=4,min_holes=0,seed=None):
    """Generate a new map for the FrozenLake environment.
    """    
    if seed is not None:
        np.random.seed(seed)

    map = np.zeros((size,size),dtype=np.uint8)

    map[0,0] = 1 # always start at 0,0
    map[size-1,size-1] = 2 # always end at size-1,size-1

    num_holes = np.random.randint(min_holes,max_holes+1)
    for _ in range(num_holes):

        x = np.random.randint(0,size)
        y = np.random.randint(0,size)

        while map[x,y] != 0:
            x = np.random.randint(0,size)
            y = np.random.randint(0,size)

        map[x,y] = 3

    if not is_reachable(map,(0,0),(size-1,size-1)):
        return generate_new_map(size,max_steps,max_holes,min_holes,seed)
    
    map_to_str = {0: 'F', 1: 'S', 2: 'G', 3: 'H'}

    map_str = ["".join([map_to_str[map[i,j]] for j in range(size)]) for i in range(size)]

    return map_str


def make_gym_env(p, map):
    env = gym.make('FrozenLake-v1', desc=map)
    param_name = "P"
    scheduler = ns_gym.schedulers.ContinuousScheduler()
    update_fn = ns_gym.update_functions.DistributionStepWiseUpdate(scheduler=scheduler,update_values=[p])
    parameter_map = {param_name: update_fn}
    ns_env = ns_gym.wrappers.NSFrozenLakeWrapper(env,parameter_map,change_notification=True,delta_change_notification=True)
    return ns_env

def generate_data(p, num_episdodes, max_steps, map_size, max_holes, min_holes):
    """Generate data for the FrozenLake environment.
    """
    total_number_of_maps = np.sum([np.math.factorial(map_size**2)/(np.math.factorial(map_size**2-i)*np.math.factorial(i)) for i in range(min_holes,max_holes+1)])

    print(f"Total number of maps: {total_number_of_maps}")
    map = generate_new_map(size=map_size, max_steps=max_steps, max_holes=max_holes, min_holes=min_holes)
    print(map)

    # TODO add loop to generate data for all maps

    # ns_env = make_gym_env(p, map)

if __name__ == "__main__":
    
    generate_data(0.1, 100, 1000, 6, 6, 1)






