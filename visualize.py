from value_iteration import q_value_iteration
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


def visualize_value_map(QV, map_size):
    """
    Visualize the Q-values for each state in a grid environment.
    Each square contains four triangles representing the Q-values
    for the possible actions in that state.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop over the grid to plot Q-values
    for row in range(map_size):
        for col in range(map_size):
            # Extract Q-values for this state
            q_values = QV[row * map_size + col]

            # Normalize the Q-values for better visualization (optional)

            # Define the coordinates of the square
            x, y = col, row
            square_size = 1
            center = (x + square_size / 2, y + square_size / 2)

            # Action 0: Left Triangle
            ax.add_patch(patches.Polygon(
                [center,
                 (x, y + square_size),
                 (x, y)],
                color=plt.cm.plasma(q_values[0]),
            ))

            # Action 1: Bottom Triangle
            ax.add_patch(patches.Polygon(
                [(center[0], center[1]),
                 (x, y + square_size),
                 (x + square_size, y + square_size)],
                color=plt.cm.plasma(q_values[1]),
            ))

            # Action 2: Right Triangle
            ax.add_patch(patches.Polygon(
                [center,
                 (x + square_size, y + square_size),
                 (x + square_size, y)],
                color=plt.cm.plasma(q_values[2]),
            ))

            # Action 3: Top Triangle
            ax.add_patch(patches.Polygon(
                [center,
                 (x, y),
                 (x + square_size, y)],
                color=plt.cm.plasma(q_values[3]),
            ))

    # Set up the grid and labels
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, map_size, 1))
    ax.set_yticks(np.arange(0, map_size, 1))
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Add colorbar to show Q-value scale
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(
        vmin=np.min(QV), vmax=np.max(QV)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

    plt.show()


def visualize_frozen_lake(map_layout):
    # Define color mappings for each cell type
    color_map = {
        'S': 0,  # Start
        'F': 1,  # Frozen
        'H': 2,  # Hole
        'G': 3   # Goal
    }

    # Convert the map layout to a 2D numpy array of integers based on color_map
    lake_array = np.array([[color_map[cell] for cell in row]
                          for row in map_layout])

    # Create a custom colormap with specific colors for each cell type
    custom_cmap = ListedColormap(['green', 'lightblue', 'red', 'yellow'])

    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(lake_array, cmap=custom_cmap, origin='upper')

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
    ax.tick_params(which="both", left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    # Set limits to match the grid size exactly
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    plt.show()
