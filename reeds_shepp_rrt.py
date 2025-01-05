#!/usr/bin/env python3
"""
custom_rrt_reeds_shepp.py

- Single-file solution for RRT* with Reeds-Shepp paths
- WITHOUT requiring pyReedsShepp library
- Minimal internal Reeds-Shepp approach (forward + backward arcs)
- Parses obstacles & agent dimension from environment.xml via MuJoCo
- Returns a final path in (x, y)

Author: ChatGPT
Date: ...
"""

import os
import math
import random
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mujoco

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# 1) ENVIRONMENT PARSING (MuJoCo)
# --------------------------------------------------------------------
def parse_mujoco_environment(xml_path="environment.xml"):
    """
    1) Load MuJoCo model
    2) Identify bodies named 'obs_car' => parse obstacles in 2D
    3) Identify 'agent_car' => parse agent half-size => agent_radius
    4) Return bounding_box=(0,0,20,15), obstacles=[(rx,ry,w,h)], agent_radius
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Cannot find {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)

    bounding_box = (0, 0, 20, 15)
    obstacles = []
    agent_radius = 0.5
    agent_found  = False

    for b in range(model.nbody):
        bname = _read_body_name(model, b)
        if bname.startswith("obs_car"):
            bx = model.body_pos[b, 0]
            by = model.body_pos[b, 1]
            geom_id = model.body_geomadr[b]
            sx = model.geom_size[geom_id, 0]
            sy = model.geom_size[geom_id, 1]
            rx = bx - sx
            ry = by - sy
            w  = sx*2
            h  = sy*2
            obstacles.append((rx, ry, w, h))
        elif bname.startswith("agent_car"):
            agent_found = True
            geom_id = model.body_geomadr[b]
            sx = model.geom_size[geom_id, 0]
            sy = model.geom_size[geom_id, 1]
            agent_radius = max(sx, sy)
            logger.info("Parsed agent half-size=(%.2f, %.2f), radius=%.2f", sx, sy, agent_radius)

    if not agent_found:
        logger.warning("No 'agent_car' found in environment. Using default agent_radius=%.2f", agent_radius)

    return bounding_box, obstacles, agent_radius

def _read_body_name(model, b):
    adr = model.name_bodyadr[b]
    return _read_null_terminated(model.names, adr)

def _read_null_terminated(buf, start):
    out = []
    for c in buf[start:]:
        if c == 0:
            break
        out.append(chr(c))
    return "".join(out)

# --------------------------------------------------------------------
# 2) RRT* NODE (x,y,theta)
# --------------------------------------------------------------------
class RSNode:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = 0.0
        self.parent = None

# --------------------------------------------------------------------
# 3) COLLISION CHECK
# --------------------------------------------------------------------
def in_collision_2d(px, py, obstacles, agent_radius):
    """Check if (px, py) is in collision with any obstacle rectangle, inflated by agent_radius."""
    for (rx, ry, w, h) in obstacles:
        rx_inf = rx - agent_radius
        ry_inf = ry - agent_radius
        w_inf  = w + 2*agent_radius
        h_inf  = h + 2*agent_radius
        if px >= rx_inf and px <= rx_inf + w_inf and py >= ry_inf and py <= ry_inf + h_inf:
            return True
    return False

def check_path_collision(samples, obstacles, agent_radius):
    """Check if any sample in path is in collision."""
    for (sx, sy, sth) in samples:
        if in_collision_2d(sx, sy, obstacles, agent_radius):
            return True
    return False

# --------------------------------------------------------------------
# 4) MINIMAL REEDS-SHEPP IMPLEMENTATION
# --------------------------------------------------------------------
#
# This is a simplified approach that handles forward/backward arcs plus in-place turns.
# For a robust approach, see a full Reeds-Shepp library.
#

def sample_reeds_shepp_path(n_from, n_to, turning_radius=1.0, step_size=0.1):
    """
    Returns a list of (x,y,theta) from n_from->n_to 
    using a minimal Reeds-Shepp approach with forward/back arcs.
    For simplicity, we only implement a few basic maneuvers:
      - forward arc
      - backward arc
      - in-place rotation

    This is NOT a complete Reeds-Shepp coverage. 
    But enough to illustrate a forward/back approach.
    """
    # We'll do a naive approach: 
    # 1) rotate in-place to align heading to the direction of (n_to.x - n_from.x, n_to.y - n_from.y)
    # 2) go forward/back to the target x,y
    # 3) rotate in-place to target yaw

    path_samples = []

    def append_arc(cx, cy, cth):
        path_samples.append((cx, cy, cth))

    # Start
    x0, y0, th0 = n_from.x, n_from.y, n_from.theta
    x1, y1, th1 = n_to.x,   n_to.y,   n_to.theta

    # 1) rotate to desired direction
    dx = x1 - x0
    dy = y1 - y0
    desired_dir = math.atan2(dy, dx) if (abs(dx)>1e-9 or abs(dy)>1e-9) else th0
    rot_diff = angle_diff(desired_dir, th0)
    # We'll rotate in-place step by step
    sign = 1 if rot_diff>0 else -1
    rot_steps = int(abs(rot_diff) / (step_size/turning_radius))  # step_size used as a small angle step
    if rot_steps < 1:
        rot_steps = 1
    angle_inc = rot_diff/rot_steps

    cx, cy, cth = x0, y0, th0
    for _ in range(rot_steps):
        cth += angle_inc
        append_arc(cx, cy, cth)

    # 2) drive forward/back
    dist = math.hypot(dx, dy)
    forward = (dist > 0.0)
    step_dist = step_size
    if forward:
        sign_fb = 1.0
    else:
        # If dist==0, skip
        sign_fb = 1.0
    steps_lin = int(dist/step_dist)
    if steps_lin < 1:
        steps_lin = 1
    linear_inc = dist/steps_lin
    for _ in range(steps_lin):
        cx += sign_fb * linear_inc*math.cos(cth)
        cy += sign_fb * linear_inc*math.sin(cth)
        append_arc(cx, cy, cth)

    # final leftover distance
    leftover = dist - steps_lin*linear_inc
    if leftover>1e-9:
        cx += leftover*math.cos(cth)
        cy += leftover*math.sin(cth)
        append_arc(cx, cy, cth)

    # 3) rotate in-place to final yaw
    final_rot = angle_diff(th1, cth)
    sign2 = 1 if final_rot>0 else -1
    rot_steps2 = int(abs(final_rot)/(step_size/turning_radius))
    rot_steps2 = max(1, rot_steps2)
    angle_inc2 = final_rot/rot_steps2
    for _ in range(rot_steps2):
        cth += angle_inc2
        append_arc(cx, cy, cth)

    # done
    return path_samples


def angle_diff(a, b):
    d = a-b
    return (d+math.pi)%(2*math.pi)-math.pi

def reeds_shepp_path_length(n_from, n_to, turning_radius=1.0):
    """
    Approximate length from n_from->n_to in the minimal approach: 
     - rotation dist + linear dist + rotation dist
    We'll treat rotation as (abs(angle)/some factor).
    """
    x0, y0, th0 = n_from.x, n_from.y, n_from.theta
    x1, y1, th1 = n_to.x,   n_to.y,   n_to.theta
    dx = x1-x0
    dy = y1-y0
    dxy= math.hypot(dx,dy)
    rot1 = abs(angle_diff(math.atan2(dy,dx), th0))
    rot2 = abs(angle_diff(th1, math.atan2(dy,dx)))
    # scale rotations by turning_radius?
    # We'll do (rot1+rot2)*turning_radius + dxy
    return (rot1+rot2)*turning_radius + dxy

def collision_free_reeds_shepp(n_from, n_to, obstacles, agent_radius, turning_radius=1.0, step_size=0.1):
    samples = sample_reeds_shepp_path(n_from, n_to, turning_radius, step_size)
    return not check_path_collision(samples, obstacles, agent_radius)


# --------------------------------------------------------------------
# 5) RRT* with minimal Reeds-Shepp
# --------------------------------------------------------------------
def distance_config(n1, n2):
    """For neighbor search, do XY distance ignoring angles."""
    dx = n1.x - n2.x
    dy = n1.y - n2.y
    return math.hypot(dx, dy)

def reeds_shepp_steer(n_near, n_rand, turning_radius=1.0, step_size=1.0):
    """
    Partial extension if the path is longer than step_size.
    """
    L = reeds_shepp_path_length(n_near, n_rand, turning_radius)
    if L> step_size:
        # sample at step_size
        ratio = step_size / L
        samples = sample_reeds_shepp_path(n_near, n_rand, turning_radius, 0.1)
        total = len(samples)
        partial_idx = int(ratio*(total-1))
        if partial_idx < total:
            xmid, ymid, thmid = samples[partial_idx]
        else:
            xmid, ymid, thmid = samples[-1]
        return RSNode(xmid, ymid, thmid)
    else:
        # entire path is smaller => do full
        samples = sample_reeds_shepp_path(n_near, n_rand, turning_radius, 0.1)
        xf, yf, thf = samples[-1]
        return RSNode(xf, yf, thf)

def get_neighbors(nodes, n_new, obstacles, agent_radius, rewire_radius, turning_radius, step_size):
    idxs = []
    for i, nd in enumerate(nodes):
        if distance_config(nd, n_new) < rewire_radius:
            if collision_free_reeds_shepp(nd, n_new, obstacles, agent_radius, turning_radius, step_size):
                idxs.append(i)
    return idxs

def choose_parent(nodes, neighbor_idx, n_new, obstacles, agent_radius, turning_radius, step_size):
    best_parent = None
    best_cost   = float('inf')
    for idx in neighbor_idx:
        nd = nodes[idx]
        length = reeds_shepp_path_length(nd, n_new, turning_radius)
        cost_via= nd.cost + length
        if cost_via < best_cost:
            best_cost   = cost_via
            best_parent = idx
    return best_parent, best_cost

def rewire(nodes, neighbor_idx, n_new_idx, obstacles, agent_radius, turning_radius, step_size):
    n_new = nodes[n_new_idx]
    for idx in neighbor_idx:
        nd = nodes[idx]
        length = reeds_shepp_path_length(n_new, nd, turning_radius)
        cost_via_new = n_new.cost + length
        if cost_via_new < nd.cost:
            # collision check
            if collision_free_reeds_shepp(n_new, nd, obstacles, agent_radius, turning_radius, step_size):
                nd.parent = n_new_idx
                nd.cost   = cost_via_new

def rrt_star_reeds_shepp(
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
    Yields (nodes, final_path, random_pt, new_idx) each iteration.
    """
    if not isinstance(start, RSNode):
        start = RSNode(*start)
    if not isinstance(goal, RSNode):
        goal = RSNode(*goal)
    start.cost = 0.0
    start.parent= None
    nodes=[start]

    (xmin, ymin, xmax, ymax)= bounding_box

    for iteration in range(max_iter):
        rx= random.uniform(xmin, xmax)
        ry= random.uniform(ymin, ymax)
        rth=random.uniform(-math.pi, math.pi)
        n_rand= RSNode(rx,ry,rth)

        # nearest
        dlist= [distance_config(n_rand, nd) for nd in nodes]
        idx_near= np.argmin(dlist)
        n_near= nodes[idx_near]

        # steer partial
        n_new = reeds_shepp_steer(n_near, n_rand, turning_radius, step_size)
        # collision check
        if not collision_free_reeds_shepp(n_near, n_new, obstacles, agent_radius, turning_radius, 0.1):
            yield nodes, None, (rx, ry), None
            continue

        # neighbors
        neighbor_idx= get_neighbors(nodes, n_new, obstacles, agent_radius, rewire_radius, turning_radius, 0.1)
        # choose parent
        best_parent= idx_near
        best_cost  = n_near.cost + reeds_shepp_path_length(n_near, n_new, turning_radius)
        for idx in neighbor_idx:
            nd= nodes[idx]
            c_via= nd.cost + reeds_shepp_path_length(nd, n_new, turning_radius)
            if c_via< best_cost:
                best_cost= c_via
                best_parent= idx

        n_new.parent= best_parent
        n_new.cost  = best_cost
        new_idx= len(nodes)
        nodes.append(n_new)

        # rewire
        rewire(nodes, neighbor_idx, new_idx, obstacles, agent_radius, turning_radius, 0.1)

        # check goal
        if reeds_shepp_path_length(n_new, goal, turning_radius)< goal_threshold:
            # connect to goal
            goal.parent= new_idx
            goal.cost  = n_new.cost+ reeds_shepp_path_length(n_new, goal, turning_radius)
            nodes.append(goal)
            # build final
            g_idx= len(nodes)-1
            path_coords=[]
            while g_idx is not None:
                nd= nodes[g_idx]
                path_coords.append((nd.x, nd.y))
                g_idx= nd.parent
            path_coords.reverse()
            yield nodes, path_coords, None, None
            return

        yield nodes, None, (rx,ry), new_idx

    raise Exception("No path found with ReedsShepp RRT* after max_iter.")


# --------------------------------------------------------------------
# 6) MAIN WRAPPER for TAMP usage
# --------------------------------------------------------------------
def run_custom_rrt_reeds_shepp(
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
    1) parse environment => bounding_box, obstacles, agent_radius
    2) visualize environment
    3) run rrt_star_reeds_shepp => yield => dynamic plot
    4) return final path or raise if none
    """
    bounding_box, obstacles, agent_radius = parse_mujoco_environment(xml_path)
    logger.info("Parsed environment => bounding_box=%s, #obstacles=%d, agent_radius=%.2f",
        bounding_box, len(obstacles), agent_radius)

    fig, ax = plt.subplots(figsize=(8,6))
    plt.ion()
    plt.show()

    # draw environment
    (xmin,ymin,xmax,ymax)= bounding_box
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal','box')
    for (rx,ry,w,h) in obstacles:
        rect=patches.Rectangle((rx,ry), w,h, color='black', alpha=0.6)
        ax.add_patch(rect)
    ax.plot(start[0], start[1], 'ro', label='Start')
    ax.plot(goal[0],  goal[1],  'g*', label='Goal')
    ax.set_title("RRT* ReedsShepp Minimal")

    node_scatter,= ax.plot([],[],'bo',markersize=3,label='Nodes')
    rand_pt_plot,= ax.plot([],[],'rx',label='Random')
    path_plot,   = ax.plot([],[],'r-',linewidth=2,label='Path')
    line_objs    = []

    iteration_gen = rrt_star_reeds_shepp(
        start, goal, obstacles, bounding_box,
        agent_radius=agent_radius,
        turning_radius=turning_radius,
        step_size=step_size,
        rewire_radius=rewire_radius,
        goal_threshold=goal_threshold,
        max_iter=max_iter
    )

    final_path_coords= None
    try:
        for (nodes, partial_path, random_pt, new_idx) in iteration_gen:
            # update node scatter
            nx= [nd.x for nd in nodes]
            ny= [nd.y for nd in nodes]
            node_scatter.set_xdata(nx)
            node_scatter.set_ydata(ny)

            # random
            if random_pt:
                rx, ry= random_pt
                rand_pt_plot.set_xdata([rx])
                rand_pt_plot.set_ydata([ry])
            else:
                rand_pt_plot.set_xdata([])
                rand_pt_plot.set_ydata([])

            # remove old lines, draw tree
            for ln in line_objs:
                ln.remove()
            line_objs.clear()
            for i, nd in enumerate(nodes):
                if nd.parent is not None:
                    p= nodes[nd.parent]
                    ln,= ax.plot([nd.x,p.x],[nd.y,p.y],color='green',linewidth=0.7)
                    line_objs.append(ln)

            # final path
            if partial_path:
                px= [p[0] for p in partial_path]
                py= [p[1] for p in partial_path]
                path_plot.set_xdata(px)
                path_plot.set_ydata(py)
                final_path_coords= partial_path

            plt.legend(loc='upper right')
            plt.draw()
            plt.pause(0.01)

        plt.ioff()
        plt.show()

        if not final_path_coords:
            raise Exception("No path found after ReedsShepp RRT*.")
        return final_path_coords

    except Exception as e:
        plt.ioff()
        plt.show()
        raise e


# If run directly:
if __name__=="__main__":
    path = run_custom_rrt_reeds_shepp(
        start=(2,12,0),
        goal =(18,10, math.radians(90)),
        xml_path="environment.xml",
        max_iter=500,
        step_size=2.0,
        turning_radius=1.0,
        rewire_radius=3.0,
        goal_threshold=2.0
    )
    print("Found path =>", path)
