import torch
import numpy as np
import gymnasium as gym
import concurrent.futures
from value_iteration import q_value_iteration
import ns_gym
from visualize import visualize_value_map, visualize_frozen_lake_rectangles
from collections import defaultdict
import csv
import os
import math


def in_graph(graph, start):
    return np.all((start >= [0, 0]) & (start < graph.shape))


def dfs(graph, start, goal):
    graph[tuple(start)] = True
    actions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    for action in actions:
        newstart = start + action
        # ... and not already visited or a hole
        if in_graph(graph, newstart) and not graph[tuple(newstart)]:
            graph = dfs(graph, start + action, goal)
    return graph


def is_reachable(map, start, goal):
    """Check if a particular state is reachable from the start state."""
    graph = (map > 0)  # generate a boolean matrix for searching
    graph[goal] = False
    graph = dfs(graph, np.array(start), np.array(goal))
    return graph[goal]


def generate_new_map(size=8, max_steps=1000, max_holes=4, min_holes=0, seed=None):
    """Generate a new map for the FrozenLake environment.
    """
    if seed is not None:
        np.random.seed(seed)

    map = np.zeros((size, size), dtype=np.uint8)

    map[0, 0] = 1  # always start at 0,0
    map[size-1, size-1] = 2  # always end at size-1,size-1

    num_holes = np.random.randint(min_holes, max_holes+1)
    for _ in range(num_holes):

        x = np.random.randint(0, size)
        y = np.random.randint(0, size)

        while map[x, y] != 0:
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)

        map[x, y] = 3

    if not is_reachable(map, (0, 0), (size-1, size-1)):
        seed = seed+1
        return generate_new_map(size, max_steps, max_holes, min_holes, seed)

    map_to_str = {0: 'F', 1: 'S', 2: 'G', 3: 'H'}

    map_str = ["".join([map_to_str[map[i, j]] for j in range(size)])
               for i in range(size)]

    return map_str, seed


def make_gym_env(p, map):
    env = gym.make('FrozenLake-v1', desc=map)
    param_name = "P"
    scheduler = ns_gym.schedulers.ContinuousScheduler()
    update_fn = ns_gym.update_functions.DistributionNoUpdate(
        scheduler=scheduler)
    parameter_map = {param_name: update_fn}
    ns_env = ns_gym.wrappers.NSFrozenLakeWrapper(
        env, parameter_map, change_notification=True, delta_change_notification=True, initial_prob_dist=[p, (1-p)/2, (1-p)/2])
    return ns_env


def get_sample(ns_env, map_size, gamma=0.9, theta=1e-6):
    """Compute q-value map"""
    policy, QV = q_value_iteration(ns_env, gamma=gamma, theta=theta)
    return QV


class MetaData:
    def __init__(self):
        self.meta_data_dict = defaultdict(list)

    def store_metadata(self, id, seed):
        """Store metadata for the dataset.
        """
        self.meta_data_dict["id"].append(id)
        self.meta_data_dict["seed"].append(seed)

    def save_metadata(self, file_path):
        """Save metadata to disk.
        """
        # Check if the file exists and is empty or not
        file_exists = os.path.isfile(
            file_path) and os.path.getsize(file_path) > 0

        # Append data to CSV, including the header only if the file is empty
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(
                file, fieldnames=self.meta_data_dict.keys())

            # Write header only if the file is new or empty
            if not file_exists:
                writer.writeheader()

            # Append the data
            writer.writerows([dict(zip(self.meta_data_dict, t))
                             for t in zip(*self.meta_data_dict.values())])


def process_map(id, seed, p, map_size, max_steps, max_holes, min_holes, save_path):
    # Generate a new map with the given parameters
    map, seed = generate_new_map(
        map_size, max_steps, max_holes, min_holes, seed)

    # Initialize the environment with the generated map
    ns_env = make_gym_env(p, map)
    ns_env.reset()

    # Collect and process samples from the environment
    X = get_sample(ns_env, map_size)
    X = X.reshape(map_size, map_size, ns_env.action_space.n)
    X = np.transpose(X, axes=[2, 0, 1])
    X = torch.tensor(X)

    # Save the processed data to a file
    save_file = f"{save_path}_{id}.pt"
    torch.save(X, save_file)

    # Return metadata for logging
    return id, seed, save_file


def generate_data_parallel(p, num_maps, max_steps, map_size,
                           max_holes, min_holes, save_path, meta_data_logger,
                           meta_data_path, num_workers, start_seed=None, max_in_flight=1000):
    # Initialize the starting seed
    seed = start_seed if start_seed is not None else 0

    # Calculate the total number of possible maps
    total_number_of_maps = np.sum([
        math.comb(map_size**2, i) for i in range(min_holes, max_holes + 1)
    ])
    print(f"Total number of possible maps: {total_number_of_maps}")

    # Set the number of maps to generate if not specified
    if num_maps is None:
        num_maps = int(total_number_of_maps)

    completed = 0  # Counter for completed tasks

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_id = {}
        id_iter = iter(range(num_maps))
        in_flight = 0

        # Submit initial batch of tasks
        for _ in range(min(max_in_flight, num_maps)):
            id = next(id_iter)
            future = executor.submit(
                process_map, id, seed + id, p, map_size, max_steps,
                max_holes, min_holes, save_path
            )
            future_to_id[future] = id
            in_flight += 1

        while future_to_id:
            # As futures complete, submit new tasks
            for future in concurrent.futures.as_completed(future_to_id):
                id = future_to_id.pop(future)
                in_flight -= 1
                try:
                    id_result, seed_result, save_file = future.result()
                    # Log metadata safely in the main process
                    meta_data_logger.store_metadata(id_result, seed_result)
                    completed += 1  # Increment completed counter

                    # Save metadata periodically
                    if completed % 100 == 0:
                        print(f"Generated {completed}/{num_maps} samples")
                        meta_data_logger.save_metadata(meta_data_path)
                except Exception as e:
                    print(f"Error processing map {id}: {e}")

                # Submit new task if any are left
                try:
                    next_id = next(id_iter)
                    future = executor.submit(
                        process_map, next_id, seed + next_id, p, map_size,
                        max_steps, max_holes, min_holes, save_path
                    )
                    future_to_id[future] = next_id
                    in_flight += 1
                except StopIteration:
                    pass  # No more tasks to submit

                if in_flight == 0:
                    break  # All tasks have been completed

        # Save any remaining metadata after all tasks are completed
        meta_data_logger.save_metadata(meta_data_path)


def generate_data(p, num_maps, max_steps, map_size,
                  max_holes, min_holes,
                  save_path, meta_data_logger, meta_data_path,
                  visualize=False, start_seed=None):
    """Generate data for the FrozenLake environment.
    """
    total_number_of_maps = np.sum([math.factorial(map_size**2)/(math.factorial(
        map_size**2-i)*math.factorial(i)) for i in range(min_holes, max_holes+1)])

    print(f"Total number of possible maps: {total_number_of_maps}")

    if num_maps is None:
        num_maps = round(total_number_of_maps)

    if start_seed is not None:
        seed = start_seed
    else:
        seed = 0

    for id in range(num_maps):
        map, seed = generate_new_map(
            map_size, max_steps, max_holes, min_holes, seed)

        meta_data_logger.store_metadata(id, seed)

        ns_env = make_gym_env(p, map)
        ns_env.reset()

        X = get_sample(ns_env, map_size)

        if visualize:
            visualize_frozen_lake_rectangles(map)
            visualize_value_map(X, map_size)

        X = X.reshape(map_size, map_size, ns_env.action_space.n)
        X = np.transpose(X, axes=[2, 0, 1])

        # compress X and save it to disk
        X = torch.tensor(X)
        torch.save(X, save_path + f"_{id}.pt")

        if id % 1000 == 0:
            print(f"Generated {id} samples")
            meta_data_logger.save_metadata(meta_data_path)

        seed += 1

    meta_data_logger.save_metadata(meta_data_path)
    print(f"Generated {num_maps} samples")


if __name__ == "__main__":
    import argparse
    from utils import parse_config
    parser = argparse.ArgumentParser(description='Generate Q-value map data.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file.')
    args = parser.parse_args()
    config = parse_config(args.config)

    p = config['p']
    num_maps = config['num_maps']
    max_steps = config['max_steps']

    map_size = config['map_size']
    max_holes = config['max_holes']
    min_holes = config['min_holes']

    save_path = config['save_path']
    meta_data_path = config['meta_data_path']
    num_workers = config['num_workers']

    parallel = config['parallel']

    meta_data_logger = MetaData()

    if parallel:
        generate_data_parallel(p, num_maps, max_steps,
                               map_size, max_holes, min_holes,
                               save_path=save_path,
                               meta_data_logger=meta_data_logger,
                               meta_data_path=meta_data_path,
                               num_workers=num_workers)
    else:
        generate_data(p, num_maps, max_steps, map_size, max_holes,
                      min_holes, save_path, meta_data_logger, meta_data_path)

    print("Data generation complete")
