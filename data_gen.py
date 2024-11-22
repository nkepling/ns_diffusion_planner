import torch
import numpy as np
import gymnasium as gym
from value_iteration import value_iteration
import ns_gym
from visualize import visualize_value_map, visualize_frozen_lake, visualize_frozen_lake_rectangles
import pandas as pd 
from collections import defaultdict
import csv
import os

def in_graph(graph, start):
  return np.all((start >= [0, 0]) & (start < graph.shape))
  
def dfs(graph, start, goal):
  graph[tuple(start)] = True
  actions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
  for action in actions:
    newstart = start + action
    if in_graph(graph, newstart) and not graph[tuple(newstart)]: #... and not already visited or a hole
      graph = dfs(graph, start + action, goal)
  return graph
  

def is_reachable(map, start, goal):
    """Check if a particular state is reachable from the start state."""
    graph = (map > 0) # generate a boolean matrix for searching 
    graph[goal] = False
    graph = dfs(graph, np.array(start), np.array(goal))
    return graph[goal]

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
        seed = seed+1
        return generate_new_map(size,max_steps,max_holes,min_holes,seed)
    
    map_to_str = {0: 'F', 1: 'S', 2: 'G', 3: 'H'}

    map_str = ["".join([map_to_str[map[i,j]] for j in range(size)]) for i in range(size)]

    return map_str,seed


def make_gym_env(p, map):
    env = gym.make('FrozenLake-v1', desc=map)
    param_name = "P"
    scheduler = ns_gym.schedulers.ContinuousScheduler()
    update_fn = ns_gym.update_functions.DistributionNoUpdate(scheduler=scheduler)
    parameter_map = {param_name: update_fn}
    ns_env = ns_gym.wrappers.NSFrozenLakeWrapper(env,parameter_map,change_notification=True,delta_change_notification=True,initial_prob_dist=[p,(1-p)/2,(1-p)/2])
    return ns_env


def get_sample(ns_env,map_size,gamma=0.9,theta=1e-6):
    """Compute value map and concatinate it wiht one-hot encoded actions. 
    """
    policy, V = value_iteration(ns_env,gamma=gamma,theta=theta)

    # policy = policy.reshape((map_size,map_size))
    # V = V.reshape((map_size,map_size))

    # # one-hot encode actions

    # one_hot = np.zeros((4,map_size,map_size))

    # for i in range(map_size):
    #     for j in range(map_size):
    #         action = policy[i,j]
    #         one_hot[action,i,j] = 1

    # concat Value map and one-hot encoded actions

    V = V.reshape((1,map_size,map_size))

    #X = np.concatenate([V,one_hot],axis=0) # concat along the channel dimension (first dimension)

    # Ensuure dim (channels, height, width) on when we include batch size (batch_size, channels, height, width)

    return V

class MetaData:
    def __init__(self):
        self.meta_data_dict = defaultdict(list)


    def store_metadata(self,id,seed):
        """Store metadata for the dataset.
        """
        self.meta_data_dict["id"].append(id)
        self.meta_data_dict["seed"].append(seed)

    def save_metadata(self,file_path):
        """Save metadata to disk.
        """
        # Check if the file exists and is empty or not
        file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

        # Append data to CSV, including the header only if the file is empty
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.meta_data_dict.keys())
            
            # Write header only if the file is new or empty
            if not file_exists:
                writer.writeheader()
                
            # Append the data
            writer.writerows([dict(zip(self.meta_data_dict, t)) for t in zip(*self.meta_data_dict.values())])

    


def generate_data(p, num_episodes, max_steps, map_size, max_holes, min_holes,save_path,meta_data_logger,meta_data_path,visualize=False,start_seed=None):
    """Generate data for the FrozenLake environment.
    """
    total_number_of_maps = np.sum([np.math.factorial(map_size**2)/(np.math.factorial(map_size**2-i)*np.math.factorial(i)) for i in range(min_holes,max_holes+1)])

    print(f"Total number of possible maps: {total_number_of_maps}")

    id = 0

    if start_seed is not None:
        seed = start_seed
    else:
        seed = 0

    while id < num_episodes:
        map,seed = generate_new_map(size=map_size, max_steps=max_steps, max_holes=max_holes, min_holes=min_holes,seed = seed)
        
        meta_data_logger.store_metadata(id,seed)

        ns_env = make_gym_env(p, map)
        out = ns_env.reset()

        if visualize:
            visualize_frozen_lake_rectangles(map)
            visualize_value_map(ns_env,map_size)

        X = get_sample(ns_env,map_size)

        # compress X and save it to disk
        np.savez_compressed(save_path+f"_{id}.npy",X)

        if id % 1000 == 0:
            print(f"Generated {id} samples")
            meta_data_logger.save_metadata(meta_data_path)

        id += 1
        seed += 1

    
    
    meta_data_logger.save_metadata(meta_data_path)
    print(f"Generated {num_episodes} samples")


if __name__ == "__main__":
    save_path = "data/p1/p1"
    meta_data_path = "data/p1_metadata.csv"

    # meta_data_dict = defaultdict(list)
    meta_data_logger = MetaData()

    generate_data(1, 20, 1000, 10, 4, 4, save_path=save_path,meta_data_logger=meta_data_logger,meta_data_path=meta_data_path,visualize=False)

    print("Data generation complete")
