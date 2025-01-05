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

class Node:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = 0.0
        self.parent = None 

def parse_environment(model):
    bounding_box = (0, 0, 20, 15)  
    obstacles = []
    agent_hx, agent_hy = 0.5, 0.5  
    agent_found = False

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
            agent_hx, agent_hy = sx, sy
            logger.info("Parsed agent half-size=(%.2f,%.2f)", agent_hx, agent_hy)

    if not agent_found:
        logger.warning("No 'agent_car' found in XML, default agent half-size=0.5,0.5")

    return bounding_box, obstacles, (agent_hx, agent_hy)

def _read_body_name(model, b):
    adr = model.name_bodyadr[b]
    return _read_null_terminated(model.names, adr)

def _read_null_terminated(buf, start):
    res = []
    for c in buf[start:]:
        if c == 0:
            break
        res.append(chr(c))
    return "".join(res)

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

def angle_diff(a, b):
    d = a-b
    return (d+math.pi)%(2*math.pi)-math.pi

def config_distance(n1, n2):
    dx = n1.x - n2.x
    dy = n1.y - n2.y
    dth= angle_diff(n1.theta, n2.theta)
    return math.sqrt(dx*dx + dy*dy + dth*dth)

def corners_of_agent(x, y, theta, half_x, half_y):
    corners_local = [
        ( half_x,  half_y),
        ( half_x, -half_y),
        (-half_x, -half_y),
        (-half_x,  half_y)
    ]
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    corners_world = []
    for (lx, ly) in corners_local:
        wx = x + lx*cosT - ly*sinT
        wy = y + lx*sinT + ly*cosT
        corners_world.append((wx, wy))
    return corners_world

def oriented_rect_aabb(corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return min_x, min_y, (max_x-min_x), (max_y-min_y)

def rect_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    if ax+aw < bx or bx+bw < ax:
        return False
    if ay+ah < by or by+bh < ay:
        return False
    return True

def in_collision_oriented_rect(x, y, theta, half_x, half_y, obstacles):
    corners = corners_of_agent(x, y, theta, half_x, half_y)
    ax, ay, aw, ah = oriented_rect_aabb(corners)
    for (rx, ry, rw, rh) in obstacles:
        if rect_overlap(ax, ay, aw, ah, rx, ry, rw, rh):
            return True
    return False

def is_collision_free(n1, n2, obstacles, half_x, half_y, step_size=0.1):
    dist = config_distance(n1, n2)
    steps = int(dist / step_size)
    steps = max(1, steps)
    for i in range(steps+1):
        alpha = i / steps
        # linear interpolation in x,y, plus angle interpolation
        x  = n1.x + alpha*(n2.x - n1.x)
        y  = n1.y + alpha*(n2.y - n1.y)
        t0 = n1.theta
        t1 = n2.theta
        dth= angle_diff(t1, t0)
        th = t0 + alpha*dth

        if in_collision_oriented_rect(x, y, th, half_x, half_y, obstacles):
            return False
    return True

def steer(n_from, n_to, speed_max=1.0, turn_max=math.radians(30), step=0.5):
    dx = n_to.x - n_from.x
    dy = n_to.y - n_from.y
    desired_angle = math.atan2(dy, dx)
    angle_diff_val= angle_diff(desired_angle, n_from.theta)
    if abs(angle_diff_val)> turn_max:
        angle_diff_val= math.copysign(turn_max, angle_diff_val)

    new_th = n_from.theta + angle_diff_val
    dist = math.hypot(dx, dy)
    travel = min(speed_max*step, dist)
    new_x = n_from.x + travel*math.cos(new_th)
    new_y = n_from.y + travel*math.sin(new_th)

    return Node(new_x, new_y, new_th)

def custom_rrt_star(
    start, goal,
    obstacles,
    bounding_box,
    half_x, half_y,
    max_iter=200,
    speed_max=1.0,
    turn_max=math.radians(30),
    step=0.5,
    rewire_radius=2.0,
    goal_threshold=1.0
):
    logger.info("Starting custom RRT* with rectangular agent half-size=(%.2f, %.2f).", half_x, half_y)

    if not isinstance(start, Node):
        start = Node(*start)
    if not isinstance(goal, Node):
        goal = Node(*goal)
    start.cost = 0.0
    start.parent = None

    nodes = [start]
    (xmin, ymin, xmax, ymax) = bounding_box

    for iteration in range(max_iter):
        # Every 10 iterations, use the goal position as the random sample
        if iteration % 10 == 0:
            n_rand = goal
        else:
            rx = random.uniform(xmin, xmax)
            ry = random.uniform(ymin, ymax)
            rth = random.uniform(-math.pi, math.pi)
            n_rand = Node(rx, ry, rth)

        # Find the nearest node
        dlist = [config_distance(n_rand, nd) for nd in nodes]
        idx_near = np.argmin(dlist)
        n_near = nodes[idx_near]

        #Steer towards the random node
        n_new = steer(n_near, n_rand, speed_max, turn_max, step)

        # Collision check
        if not is_collision_free(n_near, n_new, obstacles, half_x, half_y, 0.1):
            yield nodes, None, (n_rand.x, n_rand.y, n_rand.theta), None
            continue

        #Find neighbors within the rewire radius
        neighbor_idx = []
        for i, nd in enumerate(nodes):
            if config_distance(nd, n_new) < rewire_radius:
                if is_collision_free(nd, n_new, obstacles, half_x, half_y, 0.1):
                    neighbor_idx.append(i)

        # Choose the best parent for the new node
        best_parent = idx_near
        best_cost = n_near.cost + config_distance(n_near, n_new)
        for idx in neighbor_idx:
            nd = nodes[idx]
            c_via = nd.cost + config_distance(nd, n_new)
            if c_via < best_cost:
                best_cost = c_via
                best_parent = idx
        n_new.parent = best_parent
        n_new.cost = best_cost
        new_idx = len(nodes)
        nodes.append(n_new)

        # Rewire the neighbors
        for idx in neighbor_idx:
            nd = nodes[idx]
            c_via_new = n_new.cost + config_distance(n_new, nd)
            if c_via_new < nd.cost:
                if is_collision_free(n_new, nd, obstacles, half_x, half_y, 0.1):
                    nd.parent = new_idx
                    nd.cost = c_via_new

        # Check if the goal is reached
        if config_distance(n_new, goal) < goal_threshold:
            # Connect the goal node
            goal.parent = new_idx
            goal.cost = n_new.cost + config_distance(n_new, goal)
            nodes.append(goal)

            # Build the path from the start to the goal
            g_idx = len(nodes) - 1
            path_coords = []
            while g_idx is not None:
                nd = nodes[g_idx]
                path_coords.append((nd.x, nd.y))
                g_idx = nd.parent
            path_coords.reverse()
            yield nodes, path_coords, None, None
            return

        yield nodes, None, (n_rand.x, n_rand.y, n_rand.theta), new_idx

    raise Exception("No path found after max_iter in custom RRT*.")

def custom_rrt_algorithm(
    start=None, 
    goal=(10, 10, 0),  
    model=None,
    max_iter=300,
    speed_max=1.0,
    turn_max=math.radians(30),
    step=0.5,
    rewire_radius=2.0,
    goal_threshold=1.0,
    visualize=True  
):
    if model is None:
        raise ValueError("Model cannot be None. Please provide a valid MuJoCo model.")

    bounding_box, obstacles, (agent_hx, agent_hy) = parse_environment(model)

    if start is None:
        for b in range(model.nbody):
            if _read_body_name(model, b).startswith("agent_car"):
                start_x = model.body_pos[b, 0]
                start_y = model.body_pos[b, 1]
                start_theta = 0.0  
                start = (start_x, start_y, start_theta)
                logger.info("Start position taken from model: (%.2f, %.2f, %.2f)", start_x, start_y, start_theta)
                break
        else:
            raise ValueError("Agent start position not found in the model. Ensure 'agent_car' exists.")

    # Visualization setup
    if visualize:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.ion()
        plt.show()

        visualize_environment(ax, bounding_box, obstacles)
        ax.plot(start[0], start[1], 'ro', label='Start')
        ax.plot(goal[0], goal[1], 'g*', label='Goal')
        ax.set_title("Custom RRT*")

        nodes_plot, = ax.plot([], [], 'bo', markersize=3, label='Nodes')
        rand_pt_plot, = ax.plot([], [], 'rx', label='Random')
        path_plot, = ax.plot([], [], 'r-', linewidth=2, label='Path')
        lines_list = []

    # Run RRT
    iteration_gen = custom_rrt_star(
        Node(*start),
        Node(*goal),
        obstacles,
        bounding_box,
        half_x=agent_hx,
        half_y=agent_hy,
        max_iter=max_iter,
        speed_max=speed_max,
        turn_max=turn_max,
        step=step,
        rewire_radius=rewire_radius,
        goal_threshold=goal_threshold
    )

    final_path_coords = None
    try:
        for (nodes, partial_path, random_pt, new_idx) in iteration_gen:
            if visualize:
                nx = [nd.x for nd in nodes]
                ny = [nd.y for nd in nodes]
                nodes_plot.set_xdata(nx)
                nodes_plot.set_ydata(ny)

                if random_pt:
                    rx, ry, rth = random_pt
                    rand_pt_plot.set_xdata([rx])
                    rand_pt_plot.set_ydata([ry])
                else:
                    rand_pt_plot.set_xdata([])
                    rand_pt_plot.set_ydata([])

                # Remove old lines and draw tree
                for ln in lines_list:
                    ln.remove()
                lines_list.clear()
                draw_tree(ax, nodes, lines_list)

                #Final path
                if partial_path:
                    px = [p[0] for p in partial_path]
                    py = [p[1] for p in partial_path]
                    path_plot.set_xdata(px)
                    path_plot.set_ydata(py)
                    final_path_coords = partial_path

                plt.legend(loc='upper right')
                plt.draw()
                plt.pause(0.01)

        if visualize:
            plt.ioff()
            plt.show()

        if final_path_coords is None:
            raise Exception("No path found with RRT*.")
        return final_path_coords

    except Exception as e:
        raise e

if __name__=="__main__":
    model = mujoco.MjModel.from_xml_path("environment.xml")
    path_result= custom_rrt_algorithm(
        goal =(12.5,2.5,-(math.pi/2)),
        turn_max=math.radians(35),
        speed_max=1.0,
        model=model,
        max_iter=2000,
        visualize=True
    )
    print("Found path =>", path_result)
