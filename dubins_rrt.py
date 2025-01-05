#!/usr/bin/env python3
"""
custom_rrt_dubins.py

A custom RRT* approach for a Dubins car in 2D, parsing obstacles
(and agent dimension) from a MuJoCo environment.xml. Returns a path in (x,y).
"""

import os
import math
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mujoco
import dubins  # pip install dubins

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# 1) DATA STRUCTURES
# ---------------------------------------------------
class Node:
    """Store (x,y,theta) plus cost/parent for RRT*."""
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = 0.0
        self.parent = None

# ---------------------------------------------------
# 2) PARSE ENV FROM environment.xml
# ---------------------------------------------------
def parse_mujoco_environment(xml_path="environment.xml"):
    """
    1) Load MuJoCo model
    2) Find 'obs_car' bodies => parse obstacles in 2D
    3) Find 'agent_car' => parse agent half-size or bounding radius
    4) Return bounding_box, obstacles, agent_radius
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Cannot find {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)

    # We'll define a bounding box from 0..some range. 
    # This is a guess; you can parse from the tiles or pass it as an argument.
    bounding_box = (0,0,20,15)

    # 2D obstacles
    obstacles = []

    # agent radius or half-size
    agent_radius = 0.5  # default if not found
    # We'll search for a body named 'agent_car' or geom
    agent_found = False

    # We'll iterate bodies for obstacles
    for b in range(model.nbody):
        bname = _read_body_name(model, b)
        # check if startswith("obs_car")
        if bname and bname.startswith("obs_car"):
            # parse pos + geom size
            bx = model.body_pos[b, 0]
            by = model.body_pos[b, 1]
            # assume one geom in that body
            geom_id = model.body_geomadr[b]
            sx = model.geom_size[geom_id, 0]
            sy = model.geom_size[geom_id, 1]
            # bounding rect => center=(bx,by), half=(sx,sy)
            rx = bx - sx
            ry = by - sy
            w  = sx*2
            h  = sy*2
            obstacles.append((rx, ry, w, h))

        elif bname and bname.startswith("agent_car"):
            agent_found = True
            # parse agent dimension
            geom_id = model.body_geomadr[b]
            sx = model.geom_size[geom_id, 0]
            sy = model.geom_size[geom_id, 1]
            agent_radius = max(sx, sy)
            logger.info("Parsed agent half-size=(%.2f,%.2f), radius=%.2f", sx, sy, agent_radius)

    if not agent_found:
        logger.warning("No 'agent_car' found in the XML, using default agent_radius=%.2f", agent_radius)

    return bounding_box, obstacles, agent_radius

def _read_body_name(model, b):
    adr = model.name_bodyadr[b]
    return _read_null_terminated(model.names, adr)

def _read_null_terminated(buf, start):
    result = []
    for c in buf[start:]:
        if c == 0:
            break
        result.append(chr(c))
    return "".join(result)

# ---------------------------------------------------
# 3) VISUALIZATION
# ---------------------------------------------------
def visualize_environment(ax, bounding_box, obstacles):
    (xmin, ymin, xmax, ymax) = bounding_box
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal','box')
    for (rx, ry, w, h) in obstacles:
        rect = patches.Rectangle((rx, ry), w, h, color='black', alpha=0.6)
        ax.add_patch(rect)

def draw_tree(ax, nodes, lines_list):
    for ln in lines_list:
        ln.remove()
    lines_list.clear()
    for i, nd in enumerate(nodes):
        if nd.parent is not None:
            p = nodes[nd.parent]
            ln, = ax.plot([nd.x, p.x], [nd.y, p.y], color='green', linewidth=0.7)
            lines_list.append(ln)

# ---------------------------------------------------
# 4) DUBINS LOCAL PATH for STEERING
# ---------------------------------------------------
def dubins_path(q_from, q_to, turning_radius=1.0, step_size=0.1):
    """
    Create a dubins path from (x,y,theta) to (x,y,theta).
    We'll use python 'dubins' library:
      dubins.shortest_path( (x0,y0,theta0), (x1,y1,theta1), turning_radius )
    Then sample it in increments of step_size.
    Returns a list of (x_i, y_i, theta_i).
    """
    start_config = (q_from.x, q_from.y, q_from.theta)
    goal_config  = (q_to.x,   q_to.y,   q_to.theta)
    path = dubins.shortest_path(start_config, goal_config, turning_radius)
    sampling_length = path.path_length()
    samples = []
    dist = 0.0
    while dist < sampling_length:
        x, y, th = path.sample(dist)
        samples.append((x, y, th))
        dist += step_size
    # add the final goal
    xg, yg, thg = path.sample(sampling_length)
    samples.append((xg, yg, thg))
    return samples

def collision_free_dubins(q_from, q_to, obstacles, agent_radius, turning_radius=1.0, step_size=0.1):
    """
    Build the dubins path from q_from->q_to, sample it, check collisions at each sample.
    """
    # generate path
    samples = dubins_path(q_from, q_to, turning_radius, step_size)
    # check collisions
    for (x, y, th) in samples:
        if in_collision_2d((x,y), obstacles, agent_radius):
            return False
    return True

def in_collision_2d(pt, obstacles, agent_radius):
    (px, py) = pt
    for (rx, ry, w, h) in obstacles:
        # inflate obstacle by agent_radius
        rx_inf = rx - agent_radius
        ry_inf = ry - agent_radius
        w_inf  = w + 2*agent_radius
        h_inf  = h + 2*agent_radius
        if (px >= rx_inf and px <= rx_inf + w_inf and
            py >= ry_inf and py <= ry_inf + h_inf):
            return True
    return False

# ---------------------------------------------------
# 5) RRT* DUBINS
# ---------------------------------------------------
class DubinsNode(Node):
    pass

def distance_config(n1, n2):
    # ignoring cost from actual dubins length for neighbor searching,
    # we'll do naive Euclidean for neighbor search
    dx = n1.x - n2.x
    dy = n1.y - n2.y
    return math.hypot(dx, dy)

def dubins_steer(n_from, n_rand, turning_radius, step_size=0.1):
    """
    We'll create a path from n_from->n_rand via dubins, then 
    pick the last sample or a partial if it's long.
    For RRT*, we usually want a single new node (like partial extension).
    But let's assume we do partial extension if the path is longer than 'some step'.
    Or we can do full. We'll do partial if length > step.
    """
    path = dubins.shortest_path((n_from.x, n_from.y, n_from.theta),
                                (n_rand.x, n_rand.y, n_rand.theta),
                                turning_radius)
    length = path.path_length()
    # if length > step_size, sample at step_size
    if length > step_size:
        x, y, th = path.sample(step_size)
    else:
        x, y, th = path.sample(length)
    return DubinsNode(x, y, th)

def get_neighbors_dubins(nodes, n_new, obstacles, agent_radius, rewire_radius, turning_radius, step_size):
    neighbors = []
    for i, nd in enumerate(nodes):
        if distance_config(nd, n_new) < rewire_radius:
            # check collision with a full dubins path
            if collision_free_dubins(nd, n_new, obstacles, agent_radius, turning_radius, step_size):
                neighbors.append(i)
    return neighbors

def choose_parent(nodes, neighbors, n_new, obstacles, agent_radius, turning_radius, step_size):
    best_parent = None
    best_cost   = float('inf')
    for idx in neighbors:
        nd = nodes[idx]
        cost_via = nd.cost + dubins_path_length(nd, n_new, turning_radius)
        if cost_via < best_cost:
            best_cost = cost_via
            best_parent = idx
    return best_parent, best_cost

def dubins_path_length(n_from, n_to, turning_radius=1.0):
    path = dubins.shortest_path((n_from.x, n_from.y, n_from.theta),
                                (n_to.x, n_to.y, n_to.theta),
                                turning_radius)
    return path.path_length()

def rewire_dubins(nodes, neighbors, n_new_idx, obstacles, agent_radius, turning_radius, step_size):
    n_new = nodes[n_new_idx]
    for idx in neighbors:
        nd = nodes[idx]
        cost_via_new = n_new.cost + dubins_path_length(n_new, nd, turning_radius)
        if cost_via_new < nd.cost:
            # check collision
            if collision_free_dubins(n_new, nd, obstacles, agent_radius, turning_radius, step_size):
                nd.parent = n_new_idx
                nd.cost   = cost_via_new


def rrt_star_dubins(
    start, goal,
    obstacles,
    bounding_box,
    agent_radius=0.5,
    turning_radius=1.0,
    step_size=1.0,
    rewire_radius=2.0,
    goal_threshold=1.0,
    max_iter=300
):
    """
    RRT* with Dubins local planning.
    Yields (nodes, final_path, random_pt, new_idx) each iteration for real-time visualization.
    """
    # Convert start/goal to DubinsNode
    if not isinstance(start, DubinsNode):
        start = DubinsNode(*start)
    if not isinstance(goal, DubinsNode):
        goal = DubinsNode(*goal)

    start.parent = None
    start.cost   = 0.0
    nodes = [start]

    (xmin, ymin, xmax, ymax) = bounding_box

    for iteration in range(max_iter):
        # 1) sample random
        rx = random.uniform(xmin, xmax)
        ry = random.uniform(ymin, ymax)
        rth= random.uniform(-math.pi, math.pi)
        n_rand = DubinsNode(rx, ry, rth)

        # 2) nearest
        dlist = [distance_config(n_rand, nd) for nd in nodes]
        idx_near = np.argmin(dlist)
        n_near = nodes[idx_near]

        # 3) steer via dubins partial
        n_new = dubins_steer(n_near, n_rand, turning_radius, step_size)

        # 4) collision check
        if not collision_free_dubins(n_near, n_new, obstacles, agent_radius, turning_radius, 0.1):
            # skip
            yield nodes, None, (rx, ry), None
            continue

        # 5) neighbors
        neighbor_idx = get_neighbors_dubins(nodes, n_new, obstacles, agent_radius, rewire_radius, turning_radius, 0.1)

        # 6) choose parent
        best_parent, best_cost = choose_parent(nodes, neighbor_idx, n_new, obstacles, agent_radius, turning_radius, 0.1)
        if best_parent is None:
            # fallback => connect from n_near
            best_parent = idx_near
            best_cost   = n_near.cost + dubins_path_length(n_near, n_new, turning_radius)

        n_new.parent = best_parent
        n_new.cost   = best_cost

        # 7) add
        new_idx = len(nodes)
        nodes.append(n_new)

        # 8) rewire
        rewire_dubins(nodes, neighbor_idx, new_idx, obstacles, agent_radius, turning_radius, 0.1)

        # 9) check goal
        if dubins_path_length(n_new, goal, turning_radius) < goal_threshold:
            # connect to goal
            goal.parent = new_idx
            goal.cost   = n_new.cost + dubins_path_length(n_new, goal, turning_radius)
            nodes.append(goal)
            # build path
            goal_idx = len(nodes)-1
            path_coords = []
            idx = goal_idx
            while idx is not None:
                nd = nodes[idx]
                path_coords.append( (nd.x, nd.y) )
                idx = nd.parent
            path_coords.reverse()
            yield nodes, path_coords, None, None
            return

        yield nodes, None, (rx, ry), new_idx

    raise Exception("No path found after max_iter in RRT* with Dubins.")


# ---------------------------------------------------
# 6) MAIN WRAPPER
# ---------------------------------------------------
def run_custom_rrt_dubins(
    start=(0,0,0),
    goal=(10,10,0),
    xml_path="environment.xml",
    max_iter=300,
    step_size=1.0,
    turning_radius=1.0,
    rewire_radius=2.0,
    goal_threshold=1.0
):
    """
    1) Parse environment from xml_path => bounding_box, obstacles, agent_radius
    2) Setup matplotlib
    3) Run rrt_star_dubins, yield iteration => live draw
    4) Return final path (x,y)
    """
    # parse environment
    bounding_box, obstacles, agent_radius = parse_mujoco_environment(xml_path)
    logger.info("Env bounding box=%s, #obstacles=%d, agent_radius=%.2f", bounding_box, len(obstacles), agent_radius)

    fig, ax = plt.subplots(figsize=(8,6))
    plt.ion()
    plt.show()

    visualize_environment(ax, bounding_box, obstacles)
    ax.plot(start[0], start[1], 'ro', label='Start')
    ax.plot(goal[0],  goal[1],  'g*', label='Goal')
    ax.set_title("RRT* with Dubins Paths")

    rrt_nodes_plot, = ax.plot([], [], 'bo', markersize=3, label='Nodes')
    rand_pt_plot,   = ax.plot([], [], 'rx', label='Random')
    path_plot,      = ax.plot([], [], 'r-', linewidth=2, label='Path')
    lines_list = []

    iteration_gen = rrt_star_dubins(
        start, 
        goal,
        obstacles,
        bounding_box,
        agent_radius=agent_radius,
        turning_radius=turning_radius,
        step_size=step_size,
        rewire_radius=rewire_radius,
        goal_threshold=goal_threshold,
        max_iter=max_iter
    )

    final_path_coords = None
    for (nodes, final_path, random_pt, new_idx) in iteration_gen:
        # update node scatter
        nx = [nd.x for nd in nodes]
        ny = [nd.y for nd in nodes]
        rrt_nodes_plot.set_xdata(nx)
        rrt_nodes_plot.set_ydata(ny)

        # random
        if random_pt is not None:
            rx, ry = random_pt[0], random_pt[1]
            rand_pt_plot.set_xdata([rx])
            rand_pt_plot.set_ydata([ry])
        else:
            rand_pt_plot.set_xdata([])
            rand_pt_plot.set_ydata([])

        # remove old lines, draw tree
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

    if final_path_coords is None:
        raise Exception("No path found with RRT* Dubins.")
    return final_path_coords

# ============== If Running Directly ==============
if __name__ == "__main__":
    path = run_custom_rrt_dubins(
        start=(1,1,0),
        goal =(18,12,0),
        xml_path="environment.xml",
        max_iter=500,
        step_size=2.0,
        turning_radius=1.0,
        rewire_radius=3.0,
        goal_threshold=2.0
    )
    print("Found path with length:", len(path), " => ", path)
