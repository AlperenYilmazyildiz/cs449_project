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

def parse_mujoco_environment(xml_path="environment.xml"):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Cannot find {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)

    bounding_box = (0, 0, 20, 15)
    obstacles = []
    agent_radius = 0.5
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
            agent_radius = max(sx, sy)
            logger.info("Parsed agent half-size=(%.2f,%.2f), radius=%.2f",
                        sx, sy, agent_radius)

    if not agent_found:
        logger.warning("No 'agent_car' found in the XML, default agent_radius=%.2f", agent_radius)

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

class Node:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = 0.0
        self.parent = None

def in_collision_2d(px, py, obstacles, agent_radius):
    for (rx, ry, w, h) in obstacles:
        rx_inf = rx - agent_radius
        ry_inf = ry - agent_radius
        w_inf  = w + 2*agent_radius
        h_inf  = h + 2*agent_radius
        if px >= rx_inf and px <= rx_inf + w_inf and py >= ry_inf and py <= ry_inf + h_inf:
            return True
    return False

def check_path_collision(samples, obstacles, agent_radius):
    for (sx, sy, sth) in samples:
        if in_collision_2d(sx, sy, obstacles, agent_radius):
            return True
    return False

def minimal_dubins_path(n_from, n_to, turning_radius=1.0, step_size=0.1):
    samples = []

    def append_config(cx, cy, cth):
        samples.append((cx, cy, cth))

    x0, y0, th0 = n_from.x, n_from.y, n_from.theta
    x1, y1, th1 = n_to.x,   n_to.y,   n_to.theta

    # step or arc approach
    cx, cy, cth = x0, y0, th0
    append_config(cx, cy, cth)

    def angle_diff(a, b):
        d = a-b
        return (d+math.pi)%(2*math.pi)-math.pi

    dx = x1 - x0
    dy = y1 - y0
    desired_angle = math.atan2(dy, dx) if (abs(dx)>1e-9 or abs(dy)>1e-9) else th0
    rot = angle_diff(desired_angle, cth)

    angle_step = step_size / turning_radius
    steps_rot = int(abs(rot)/angle_step)
    steps_rot = max(1, steps_rot)
    step_angle = rot/steps_rot
    for _ in range(steps_rot):
        cth += step_angle
        append_config(cx, cy, cth)

    dist = math.hypot(dx, dy)
    if dist>1e-9:
        linear_steps = int(dist / step_size)
        step_lin     = dist/linear_steps if linear_steps>0 else dist
        for _ in range(linear_steps):
            cx += step_lin*math.cos(cth)
            cy += step_lin*math.sin(cth)
            append_config(cx, cy, cth)
        leftover = dist - linear_steps*step_lin
        if leftover>1e-9:
            cx += leftover*math.cos(cth)
            cy += leftover*math.sin(cth)
            append_config(cx, cy, cth)

    final_rot= angle_diff(th1, cth)
    steps_rot2= int(abs(final_rot)/angle_step)
    steps_rot2= max(1, steps_rot2)
    step_angle2= final_rot/steps_rot2
    for _ in range(steps_rot2):
        cth+= step_angle2
        append_config(cx, cy, cth)

    return samples

def collision_free_dubins(n_from, n_to, obstacles, agent_radius, turning_radius=1.0, step_size=0.1):
    samples = minimal_dubins_path(n_from, n_to, turning_radius, step_size)
    return not check_path_collision(samples, obstacles, agent_radius)

def minimal_dubins_path_length(n_from, n_to, turning_radius=1.0):
    def angle_diff(a,b):
        d=a-b
        return (d+math.pi)%(2*math.pi)-math.pi
    x0,y0,th0= n_from.x, n_from.y, n_from.theta
    x1,y1,th1= n_to.x,   n_to.y,   n_to.theta
    dx= x1-x0
    dy= y1-y0
    dist= math.hypot(dx,dy)
    a1= angle_diff(math.atan2(dy,dx), th0)
    a2= angle_diff(th1, math.atan2(dy,dx))
    return dist + turning_radius*(abs(a1)+abs(a2))

def distance_config(n1, n2):
    dx= n1.x - n2.x
    dy= n1.y - n2.y
    return math.hypot(dx, dy)

def dubins_steer(n_near, n_rand, turning_radius=1.0, step_size=1.0):
    L= minimal_dubins_path_length(n_near, n_rand, turning_radius)
    if L> step_size:
        ratio= step_size/L
        samples= minimal_dubins_path(n_near, n_rand, turning_radius, 0.1)
        total= len(samples)
        pick_idx= int(ratio*(total-1))
        pick_idx= min(pick_idx, total-1)
        xmid, ymid, thmid= samples[pick_idx]
        return Node(xmid, ymid, thmid)
    else:
        samples= minimal_dubins_path(n_near, n_rand, turning_radius, 0.1)
        xf, yf, thf= samples[-1]
        return Node(xf,yf, thf)

def get_neighbors(nodes, n_new, obstacles, agent_radius, rewire_radius, turning_radius, step_size):
    idxs= []
    for i, nd in enumerate(nodes):
        if distance_config(nd,n_new)< rewire_radius:
            if collision_free_dubins(nd, n_new, obstacles, agent_radius, turning_radius, step_size):
                idxs.append(i)
    return idxs

def choose_parent(nodes, neighbor_idx, n_new, obstacles, agent_radius, turning_radius, step_size):
    best_parent= None
    best_cost  = float('inf')
    for idx in neighbor_idx:
        nd= nodes[idx]
        length= minimal_dubins_path_length(nd,n_new,turning_radius)
        cost_via= nd.cost+ length
        if cost_via< best_cost:
            best_cost= cost_via
            best_parent= idx
    return best_parent, best_cost

def rewire(nodes, neighbor_idx, n_new_idx, obstacles, agent_radius, turning_radius, step_size):
    n_new= nodes[n_new_idx]
    for idx in neighbor_idx:
        nd= nodes[idx]
        length= minimal_dubins_path_length(n_new, nd, turning_radius)
        cost_via_new= n_new.cost+ length
        if cost_via_new< nd.cost:
            if collision_free_dubins(n_new, nd, obstacles, agent_radius, turning_radius, step_size):
                nd.parent= n_new_idx
                nd.cost  = cost_via_new

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
    if not isinstance(start, Node):
        start= Node(*start)
    if not isinstance(goal, Node):
        goal = Node(*goal)
    start.cost= 0.0
    start.parent= None
    nodes=[start]

    (xmin, ymin, xmax, ymax)= bounding_box

    for iteration in range(max_iter):
        if iteration % 10 == 0:     
            n_rand = goal
            rx, ry, rth = goal.x, goal.y, goal.theta 
        else:
            rx= random.uniform(xmin, xmax)
            ry= random.uniform(ymin, ymax)
            rth= random.uniform(-math.pi, math.pi)
            n_rand= Node(rx, ry, rth)

        dlist= [distance_config(n_rand, nd) for nd in nodes]
        idx_near= np.argmin(dlist)
        n_near= nodes[idx_near]

        n_new= dubins_steer(n_near, n_rand, turning_radius, step_size)
        if not collision_free_dubins(n_near, n_new, obstacles, agent_radius, turning_radius, 0.1):
            yield nodes, None, (rx, ry), None
            continue

        neighbor_idx= get_neighbors(nodes,n_new,obstacles,agent_radius,rewire_radius,turning_radius,0.1)

        best_parent= idx_near
        best_cost  = n_near.cost+ minimal_dubins_path_length(n_near,n_new,turning_radius)
        for idx in neighbor_idx:
            nd= nodes[idx]
            c_via= nd.cost+ minimal_dubins_path_length(nd,n_new,turning_radius)
            if c_via< best_cost:
                best_cost= c_via
                best_parent= idx

        n_new.parent= best_parent
        n_new.cost  = best_cost
        new_idx= len(nodes)
        nodes.append(n_new)

        rewire(nodes, neighbor_idx,new_idx,obstacles,agent_radius,turning_radius,0.1)

        if minimal_dubins_path_length(n_new, goal, turning_radius)< goal_threshold:
            goal.parent= new_idx
            goal.cost  = n_new.cost+ minimal_dubins_path_length(n_new, goal, turning_radius)
            nodes.append(goal)
            g_idx= len(nodes)-1
            path_coords=[]
            while g_idx is not None:
                nd= nodes[g_idx]
                path_coords.append((nd.x, nd.y))
                g_idx= nd.parent
            path_coords.reverse()
            yield nodes, path_coords, None, None
            return

        yield nodes, None, (rx, ry), new_idx

    raise Exception("No path found with minimal Dubins RRT* after max_iter.")

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
    bounding_box, obstacles, agent_radius = parse_mujoco_environment(xml_path)
    logger.info("Env bounding box=%s, #obstacles=%d, agent_radius=%.2f",
        bounding_box, len(obstacles), agent_radius)

    fig, ax = plt.subplots(figsize=(8,6))
    plt.ion()
    plt.show()

    (xmin,ymin,xmax,ymax)= bounding_box
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal','box')

    for (rx,ry,w,h) in obstacles:
        rect=patches.Rectangle((rx,ry), w,h, color='black', alpha=0.6)
        ax.add_patch(rect)

    ax.plot(start[0], start[1], 'ro', label='Start')
    ax.plot(goal[0],  goal[1],  'g*', label='Goal')
    ax.set_title("RRT* minimal Dubins")

    nodes_plot,  = ax.plot([], [], 'bo', markersize=3, label='Nodes')
    rand_pt_plot,= ax.plot([], [], 'rx', label='Random')
    path_plot,   = ax.plot([], [], 'r-', linewidth=2, label='Path')
    lines_list   = []

    iteration_gen = rrt_star_dubins(
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
            nx= [nd.x for nd in nodes]
            ny= [nd.y for nd in nodes]
            nodes_plot.set_xdata(nx)
            nodes_plot.set_ydata(ny)

            if random_pt:
                rx, ry= random_pt
                rand_pt_plot.set_xdata([rx])
                rand_pt_plot.set_ydata([ry])
            else:
                rand_pt_plot.set_xdata([])
                rand_pt_plot.set_ydata([])

            # remove old lines, draw tree
            for ln in lines_list:
                ln.remove()
            lines_list.clear()
            for i, nd in enumerate(nodes):
                if nd.parent is not None:
                    p= nodes[nd.parent]
                    ln,= ax.plot([nd.x, p.x],[nd.y, p.y],color='green',linewidth=0.7)
                    lines_list.append(ln)

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
            raise Exception("No path found with minimal Dubins RRT*.")
        return final_path_coords

    except Exception as e:
        plt.ioff()
        plt.show()
        raise e

if __name__ == "__main__":
    try:
        path = run_custom_rrt_dubins(
            start=(0,12.5,0),
            goal =(12.5,2.5, math.radians(0)),
            xml_path="environment.xml",
            max_iter=1000,
            step_size=2.0,
            turning_radius=1.0,
            rewire_radius=3.0,
            goal_threshold=2.0
        )
        print("Found path with length:", len(path), " => ", path)
    except Exception as e:
        print("Error:", e)
