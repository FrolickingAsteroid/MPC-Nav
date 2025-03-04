import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.graph_objects as go
import time

class Auxiliary(object):
    def show_world(self, init, gpath, width=20, height=20, title="MPC Navigation"):
        """
        Visualizes the initial position, goal path using Matplotlib in a 2D plot.
        """

        # Create a figure and an axes.
        fig, ax = plt.subplots()

        # Plot the initial and goal path
        ax.plot([init[0]] + gpath[:, 0].tolist(),
                [init[1]] + gpath[:, 1].tolist(),
                'b-', label='Path')  # 'b-' is a blue line

        # Set the limits of x and y to be centered around the path with a fixed width
        half_width = width / 2
        ax.set_xlim([init[0] - half_width, init[0] + half_width])
        ax.set_ylim([init[1] - half_width, init[1] + half_width])

        # Customize the plot
        ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()

        self.__arrows = []

        # Add grid
        ax.grid(True)

        # Optional: Remove axis spines for a cleaner plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Show the plot
        plt.ion()
        plt.show()

        return fig, ax

    def update_world(self, fig, ax, mpc, title=None, path=[]):
        """
        Updates the figure dynamically with new data, focusing on updating arrows that represent the path and MPC predictions.
        """

        # Clear previous arrows if any
        for arrow in self.__arrows:
            arrow.remove()
        self.__arrows = []

        # Update the title of the plot
        ax.set_title(title)

        # Update arrows for the path
        for step in path:
            arrow = ax.arrow(step['x'],
                             step['y'],
                             np.cos(step['theta']) * 0.2,  # Adjust the length and direction based on your data scale
                             np.sin(step['theta']) * 0.2,
                             color='blue',
                             head_width=0.1)  # Adjust head width based on your plot scale
            self.__arrows.append(arrow)

        # Update MPC predicted trajectory as arrows
        xs = mpc.sol.value(mpc.pos_x)
        ys = mpc.sol.value(mpc.pos_y)
        thetas = mpc.sol.value(mpc.pos_theta)

        for i in range(len(xs)):
            arrow = ax.arrow(xs[i],
                             ys[i],
                             np.cos(thetas[i]) * 0.4,  # Adjust this as needed
                             np.sin(thetas[i]) * 0.4,
                             color='red',
                             head_width=0.2)  # Adjust head width based on your plot scale
            self.__arrows.append(arrow)

        # Force redraw of the figure to show updates
        fig.canvas.draw_idle()
        plt.pause(0.001)  # Give time for the plot to update

        return fig, ax



    def simulate_step(self, dt, state, control, configs, noise={'nx':0.01,'ny':0.01,'ntheta':0.01,'nvt':0.05,'nvr':0.05}):
        sim_state = {}
        deviation = 0.05

        sim_state['vt'] = max(configs['min_vt'],min(configs['max_vt'], state['vt'] + control['A']*dt + np.random.normal(scale=noise['nvt'], loc=0.0)*dt))
        sim_state['vr'] = max(configs['min_vr'],min(configs['max_vr'], state['vr'] + control['Phi']*dt + np.random.normal(scale=noise['nvr'], loc=0.0)*dt))
        sim_state['x']  = state['x']  + sim_state['vt']*math.cos(state['theta'])*dt + np.random.normal(scale=noise['nx'], loc=0.0)*dt
        sim_state['y']  = state['y']  + sim_state['vt']*math.sin(state['theta'])*dt + np.random.normal(scale=noise['ny'], loc=0.0)*dt
        sim_state['theta'] = state['theta'] + sim_state['vr']*dt + np.random.normal(scale=noise['ntheta'], loc=0.0)*dt
        return sim_state
