#!/usr/bin/env python3
"""
custom_rrt.py

A custom RRT* approach for a nonholonomic vehicle in 2D, parsing obstacles
from a MuJoCo environment.xml. Returns an (x,y) path or raises an error if no path.

Author: ChatGPT
Date: ...
"""

import os
import math
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mujoco

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ----------- DATA STRUCTURES -----------
class Node:
    """Store configuration in 2D + orientation, plus cost/parent for RRT*."""
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta      # orientation
        self.cost = 0.0         # cost from start
        self.parent = None      # index of parent node in the tree

# ----------- ENV PARSING -----------
def parse_environment(xml_path="environment.xml"):
    """
    Parse the MuJoCo environment.xml to get a 2D bounding region, 
    and obstacle rectangles or 'parking'/road info.

    For simplicity, we:
      1) Load model with mujoco.MjModel
      2) Read some known bodies or geoms with 'obs_car' to create 2D rectangles
      3) Return bounding box, plus a list of obstacles as (x, y, w, h).

    This function is a placeholder: adapt it to your actual environment structure.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Cannot find {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)

    # Example approach:
    #   We'll scan all bodies or geoms whose name starts with "obs_car"
    #   Then read the pos=(x,y,?) and size=(sx, sy, ?) => build a rectangle.
    obstacles = []
    bounding_box = (0, 0, 20, 15)  # Hardcode a bounding box if you know the range

    # We'll iterate over model.nbody. For each body, if name starts with 'obs_car' => parse
    # WARNING: This is an example, adapt to your environment.
    for b in range(model.nbody):
        body_name_adr = model.name_bodyadr[b]
        body_name = _read_name_string(model, body_name_adr, model.names)
        if body_name.startswith("obs_car"):
            # pos = model.body_pos[b], size = ??? stored in geoms
            bx = model.body_pos[b, 0]
            by = model.body_pos[b, 1]
            # We assume there's a geom with half-size = (sx, sy, ?)
            # We'll search geoms in that body
            geom_id = model.body_geomadr[b]
            geom_name_adr = model.name_geomadr[geom_id]
            gname = _read_name_string(model, geom_name_adr, model.names)

            sx = model.geom_size[geom_id, 0]
            sy = model.geom_size[geom_id, 1]
            # build rect => center=(bx,by), half-size=(sx,sy)
            rx = bx - sx
            ry = by - sy
            w = sx*2
            h = sy*2
            obstacles.append((rx, ry, w, h))

    return bounding_box, obstacles

def _read_name_string(model, start_adr, names_array):
    """
    Helper: read a null-terminated string from model.names at offset start_adr.
    """
    result_chars = []
    for c in names_array[start_adr:]:
        if c == 0:
            break
        result_chars.append(chr(c))
    return "".join(result_chars)

# ----------- VISUALIZATION -----------
def visualize_environment(ax, bounding_box, obstacles):
    """Plot bounding box and obstacles as rectangles in matplotlib."""
    (xmin, ymin, xmax, ymax) = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal','box')
    # obstacles
    for (rx, ry, w, h) in obstacles:
        rect = patches.Rectangle((rx, ry), w, h, color='black', alpha=0.6)
        ax.add_patch(rect)

def draw_tree(ax, nodes, lines_list):
    """
    (Optional) We'll remove old lines from lines_list, then draw new edges for the tree.
    lines_list is a list to store the newly created line artists.
    """
    for ln in lines_list:
        ln.remove()
    lines_list.clear()

    for i, nd in enumerate(nodes):
        if nd.parent is not None:
            p = nodes[nd.parent]
            ln, = ax.plot([nd.x, p.x], [nd.y, p.y], color='green', linewidth=0.7)
            lines_list.append(ln)

# ----------- HELPER: DISTANCE, ETC. -----------
def angle_diff(a, b):
    d = a - b
    return (d + math.pi) % (2*math.pi) - math.pi

def config_distance(n1, n2):
    """
    Distance in (x,y,theta) but let's do a simple Euclidean ignoring angle or do a 3D approach.
    We'll do sqrt(dx^2+dy^2 + dtheta^2).
    """
    dx = n1.x - n2.x
    dy = n1.y - n2.y
    dth = angle_diff(n1.theta, n2.theta)
    return math.sqrt(dx*dx + dy*dy + dth*dth)

# ----------- COLLISION CHECK -----------
def is_collision_free(n1, n2, obstacles, step_size=0.1):
    """
    For each small step from n1->n2, check if agent's center is in an obstacle.
    We'll treat the agent as a point or small circle if needed.
    """
    dist = config_distance(n1, n2)
    steps = int(dist / step_size)
    for i in range(steps+1):
        alpha = i/max(1, steps)
        x = n1.x + alpha*(n2.x - n1.x)
        y = n1.y + alpha*(n2.y - n1.y)
        # check collision with obstacles
        for (rx, ry, w, h) in obstacles:
            if (x >= rx and x <= rx+w and y >= ry and y <= ry+h):
                return False
    return True

# ----------- STEERING WITH NONHOLONOMIC CONSTRAINTS -----------
def steer(n_from, n_to, speed_max=1.0, turn_max=math.radians(30), step=0.5):
    """
    Nonholonomic approach: 
     - We consider the orientation of n_from
     - We want to move towards n_to but can't exceed certain turn angle or speed.
    For simplicity, let's do a small approach:
      1) compute direction from n_from => n_to
      2) limit the heading change to turn_max
      3) limit distance to speed_max * step
    We'll return a new Node.
    """
    dx = n_to.x - n_from.x
    dy = n_to.y - n_from.y
    desired_angle = math.atan2(dy, dx)
    angle_diff_val = angle_diff(desired_angle, n_from.theta)
    # clamp angle change
    if abs(angle_diff_val) > turn_max:
        angle_diff_val = math.copysign(turn_max, angle_diff_val)

    new_theta = n_from.theta + angle_diff_val
    # distance
    dist = math.hypot(dx, dy)
    travel = min(speed_max*step, dist)

    new_x = n_from.x + travel*math.cos(new_theta)
    new_y = n_from.y + travel*math.sin(new_theta)

    return Node(new_x, new_y, new_theta)

def local_steer(n_from, n_to, speed_max=1.0, turn_max=math.radians(30), step=0.5):
    """
    We do a single step from n_from to n_to using the above 'steer' logic.
    If n_from->n_to is closer than speed_max*step, we may get exactly n_to, or partial approach.
    """
    # We'll repeatedly call steer if needed. (Or do partial approach.)
    # For demonstration, let's do a single call.
    return steer(n_from, n_to, speed_max=speed_max, turn_max=turn_max, step=step)

# ----------- RRT* ALGORITHM -----------
def custom_rrt_star(
    start, goal,
    obstacles,
    bounding_box,
    max_iter=200,
    speed_max=1.0,
    turn_max=math.radians(30),
    step=0.5,
    rewire_radius=2.0,
    goal_threshold=1.0
):
    """
    Nonholonomic RRT* in 2D. Start,Goal are Node or (x,y,theta).
    Returns path as [(x0,y0), (x1,y1), ...].
    Raises Exception if no solution.
    """
    logger.info("Starting custom RRT* with nonholonomic constraints: speed=%.2f, turn=%.2f deg", 
                speed_max, math.degrees(turn_max))

    # convert start/goal to Node if tuple
    if not isinstance(start, Node):
        start = Node(*start)  # (x,y,theta)
    if not isinstance(goal, Node):
        goal = Node(*goal)

    start.parent = None
    start.cost = 0.0

    nodes = [start]
    x_min, y_min, x_max, y_max = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

    for it in range(max_iter):
        # 1) sample
        rx = random.uniform(x_min, x_max)
        ry = random.uniform(y_min, y_max)
        rtheta = random.uniform(-math.pi, math.pi)
        n_rand = Node(rx, ry, rtheta)

        # 2) nearest
        dlist = [config_distance(n_rand, nd) for nd in nodes]
        idx_near = np.argmin(dlist)
        n_near = nodes[idx_near]

        # 3) steer
        n_new = local_steer(n_near, n_rand, speed_max=speed_max, turn_max=turn_max, step=step)

        # 4) collision check from n_near->n_new
        if not is_collision_free(n_near, n_new, obstacles):
            logger.debug("Iteration %d: collision -> skip", it)
            yield nodes, None, (rx, ry, rtheta), None
            continue

        # 5) find neighbors
        neighbor_idx = []
        for i, nd in enumerate(nodes):
            if config_distance(nd, n_new) < rewire_radius:
                if is_collision_free(nd, n_new, obstacles):
                    neighbor_idx.append(i)

        # 6) choose best parent
        best_parent = idx_near
        best_cost   = n_near.cost + config_distance(n_near, n_new)
        for idx in neighbor_idx:
            nd = nodes[idx]
            c_via = nd.cost + config_distance(nd, n_new)
            if c_via < best_cost:
                best_cost = c_via
                best_parent = idx

        n_new.parent = best_parent
        n_new.cost   = best_cost

        # 7) add
        n_new_idx = len(nodes)
        nodes.append(n_new)

        # 8) rewire
        for idx in neighbor_idx:
            nd = nodes[idx]
            c_via_new = n_new.cost + config_distance(n_new, nd)
            if c_via_new < nd.cost:
                # check collision
                if is_collision_free(n_new, nd, obstacles):
                    nd.parent = n_new_idx
                    nd.cost   = c_via_new

        # 9) check goal
        if config_distance(n_new, goal) < goal_threshold:
            # we consider we have solution
            goal.parent = n_new_idx
            goal.cost   = n_new.cost + config_distance(n_new, goal)
            nodes.append(goal)
            # build final path
            final_idx = len(nodes)-1
            path = []
            while final_idx is not None:
                nd = nodes[final_idx]
                path.append( (nd.x, nd.y) )
                final_idx = nd.parent
            path.reverse()
            yield nodes, path, None, None
            return

        yield nodes, None, (rx, ry, rtheta), n_new_idx

    raise Exception("No path found after max_iter. RRT* failed.")

# ----------- MAIN WRAPPER -----------
def run_custom_rrt(
    start=(0,0,0),
    goal=(10,10,0),
    xml_path="environment.xml",
    max_iter=300,
    speed_max=1.0,
    turn_max=math.radians(30),
    step=0.5,
    rewire_radius=2.0,
    goal_threshold=1.0
):
    """
    1) Parse environment from xml_path to get bounding_box, obstacles
    2) Visualize environment
    3) Run custom RRT*, each iteration updating plot
    4) Return final path
    """
    bounding_box, obstacles = parse_example_environment(xml_path)  # or parse_environment
    logger.info("Parsed environment: bounding_box=%s, obstacles=%d", bounding_box, len(obstacles))

    fig, ax = plt.subplots(figsize=(8,6))
    plt.ion()
    plt.show()

    # Draw environment
    visualize_environment(ax, bounding_box, obstacles)
    # Mark start/goal
    ax.plot(start[0], start[1], 'ro', label='Start')
    ax.plot(goal[0],  goal[1],  'g*', label='Goal')
    ax.set_title("Custom RRT* Nonholonomic")

    rrt_nodes_plot, = ax.plot([], [], 'bo', markersize=3, label='Tree Nodes')
    rand_pt_plot,   = ax.plot([], [], 'rx', label='Random sample')

    lines_list = []
    path_plot,  = ax.plot([], [], 'r-', linewidth=2, label='Path')

    # We'll run the RRT in iterative mode
    iteration_gen = custom_rrt_star(
        Node(*start),
        Node(*goal),
        obstacles=obstacles,
        bounding_box=bounding_box,
        max_iter=max_iter,
        speed_max=speed_max,
        turn_max=turn_max,
        step=step,
        rewire_radius=rewire_radius,
        goal_threshold=goal_threshold
    )

    final_path_coords = None
    for (nodes, final_path, random_pt, new_idx) in iteration_gen:
        # update node scatter
        nx = [nd.x for nd in nodes]
        ny = [nd.y for nd in nodes]
        rrt_nodes_plot.set_xdata(nx)
        rrt_nodes_plot.set_ydata(ny)

        # random sample
        if random_pt is not None:
            rx, ry, rtheta = random_pt
            rand_pt_plot.set_xdata([rx])
            rand_pt_plot.set_ydata([ry])
        else:
            rand_pt_plot.set_xdata([])
            rand_pt_plot.set_ydata([])

        # remove old lines (the tree edges)
        draw_tree(ax, nodes, lines_list)

        # final path
        if final_path:
            px = [p[0] for p in final_path]
            py = [p[1] for p in final_path]
            path_plot.set_xdata(px)
            path_plot.set_ydata(py)
            final_path_coords = final_path

        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    if not final_path_coords:
        raise Exception("No path found with RRT*.")
    return final_path_coords

# Sample environment parse function
def parse_example_environment(xml_path):
    """
    Example approach: returns bounding_box=(0,0,20,15) and few obstacles 
    because we have no direct parse from your environment.xml.
    Replace or integrate with parse_environment(...) as needed.
    """
    bounding_box = (0,0,20,15)
    # Hardcode some obstacles for demonstration
    obstacles = [
        (3,3,2,1),
        (8,6,2,2),
        (12,4,3,3)
    ]
    return bounding_box, obstacles

# If you want to run this script directly
if __name__ == "__main__":
    path_result = run_custom_rrt(
        start=(1,1,0),
        goal=(18,13,0),
        xml_path="environment.xml",
        max_iter=1000
    )
    print("Found path:", path_result)
