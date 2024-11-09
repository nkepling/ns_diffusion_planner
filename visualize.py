from value_iteration import value_iteration
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


def visualize_value_map(env,map_size):
    """Visualize the value map for a particular environment.
    """
    policy, V = value_iteration(env)
    
    fig, ax = plt.subplots(figsize=(10, 10))


    display_matrix = np.copy(V)
    display_matrix.shape = (map_size, map_size)

    display_matrix[(map_size-1,map_size-1)] = 1


    

    # Plot the value map with viridis colormap
    im = ax.imshow(display_matrix, cmap='plasma', origin='upper')
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(-0.5, map_size, 1), minor=True)
    # Adjust plot settings
    # ax.invert_yaxis()
    # ax.grid(True)
    plt.show()


    # dispaly groudn truth map


# def visualize_frozen_lake(map_layout):
#     # Define colors for each type of cell
#     color_map = {
#             'S': 0,    # Start
#             'F': 1,    # Frozen
#             'H': 2,    # Hole
#             'G': 3     # Goal
#         }
        
#     # Map the layout to numeric values based on color_map
#     fig, ax = plt.subplots(figsize=(10, 10))
#     lake_array = np.array([[color_map[cell] for cell in row] for row in map_layout])

#     im = ax.imshow(lake_array, cmap='tab20c', origin='upper')
#     fig.colorbar(im, ax=ax)
#     ax.grid(True)

#     plt.show()

def visualize_frozen_lake(map_layout):
    # Define color mappings for each cell type
    color_map = {
        'S': 0,  # Start
        'F': 1,  # Frozen
        'H': 2,  # Hole
        'G': 3   # Goal
    }

    # Convert the map layout to a 2D numpy array of integers based on color_map
    lake_array = np.array([[color_map[cell] for cell in row] for row in map_layout])

    # Create a custom colormap with specific colors for each cell type
    custom_cmap = ListedColormap(['green', 'lightblue', 'red', 'yellow'])

    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(lake_array, cmap=custom_cmap, origin='upper')

    # Set ticks with an offset to place grid lines between cells
    ax.set_xticks(np.arange(-0.5, lake_array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, lake_array.shape[0], 1), minor=True)

    # Draw grid lines at minor ticks to show boundaries between cells
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    # Turn off tick labels for a cleaner look
    # ax.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)

    # Set limits to match the grid size exactly
    ax.set_xlim(-0.5, lake_array.shape[1] - 0.5)
    ax.set_ylim(-0.5, lake_array.shape[0] - 0.5)

    plt.show()


def visualize_frozen_lake_rectangles(map_layout):
    # Define colors for each type of cell
    color_map = {
        'S': 'green',      # Start
        'F': 'lightblue',  # Frozen
        'H': 'red',        # Hole
        'G': 'yellow'      # Goal
    }

    # Set up the plot
    rows = len(map_layout)
    cols = len(map_layout[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Draw each cell as a colored rectangle
    for i in range(rows):
        for j in range(cols):
            cell_type = map_layout[i][j]
            color = color_map[cell_type]
            rect = Rectangle((j, rows - i - 1), 1, 1, color=color)
            ax.add_patch(rect)

    # Set grid lines between cells
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(which="major", color="black", linestyle='-', linewidth=2)

    # Hide the ticks for a cleaner look
    ax.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)

    # Set limits to match the grid size exactly
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    plt.show()
    
if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make('FrozenLake-v1')
    visualize_value_map(env)
    