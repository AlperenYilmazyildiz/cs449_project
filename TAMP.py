import logging
import numpy as np
import mujoco
import mujoco_viewer
import glfw
import math
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # or INFO, WARNING, etc.
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# 1) COLLISION CHECK
# ---------------------------------------------------
def geom_name_from_adr(model, geom_id):
    """
    Read the geometry name from model.names buffer using name_geomadr.
    Works for MuJoCo versions lacking model.geom_id2name().
    """
    start_adr = model.name_geomadr[geom_id]
    result_chars = []
    for c in model.names[start_adr:]:
        if c == 0:
            break
        result_chars.append(chr(c))
    name = "".join(result_chars)
    return name

def is_colliding(model, data, q_candidate):
    """
    Temporarily set data.qpos to q_candidate, run mj_forward,
    check data.contact for collisions with 'obs_car' geoms.
    """
    logger.debug("Checking collision at candidate qpos: %s", q_candidate)

    old_qpos = data.qpos.copy()
    old_qvel = data.qvel.copy()

    data.qpos[:] = q_candidate
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    collision_found = False
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2

        name1 = geom_name_from_adr(model, geom1)
        name2 = geom_name_from_adr(model, geom2)

        # agent_car vs obs_car
        if ("agent_car" in name1 and "obs_car" in name2) or \
           ("agent_car" in name2 and "obs_car" in name1):
            logger.debug("Collision detected between %s and %s", name1, name2)
            collision_found = True
            break

    # Restore
    data.qpos[:] = old_qpos
    data.qvel[:] = old_qvel
    mujoco.mj_forward(model, data)

    return collision_found


# ---------------------------------------------------
# 2) RRT* in (x, y, theta)
# ---------------------------------------------------

class Node:
    """Holds a configuration plus cost and parent index for RRT*."""
    def __init__(self, config):
        self.config = config  # [x, y, theta]
        self.cost = 0.0
        self.parent = None

def sample_random_config(x_range=(0, 20), y_range=(0, 15)):
    """
    Returns a random (x, y, theta).
    """
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    theta = np.random.uniform(-math.pi, math.pi)
    logger.debug("Sampled random config: (%.2f, %.2f, %.2f)", x, y, theta)
    return np.array([x, y, theta])

def angle_diff(a, b):
    d = a - b
    return (d + math.pi) % (2*math.pi) - math.pi

def distance_config(q1, q2):
    """
    Distance in (x,y,theta), naive Euclidean.
    """
    dx = q1[0] - q2[0]
    dy = q1[1] - q2[1]
    dtheta = angle_diff(q1[2], q2[2])
    return math.sqrt(dx*dx + dy*dy + dtheta*dtheta)

def interpolate_config(q_from, q_to, alpha):
    """
    Linear interpolation in x,y + angle interpolation.
    """
    x = (1 - alpha)*q_from[0] + alpha*q_to[0]
    y = (1 - alpha)*q_from[1] + alpha*q_to[1]
    th_diff = angle_diff(q_to[2], q_from[2])
    th = q_from[2] + alpha * th_diff
    return np.array([x, y, th])

def local_steer(q_from, q_to, step_size=0.2):
    """
    Return a list of small steps from q_from->q_to.
    """
    dist = distance_config(q_from, q_to)
    n_steps = int(dist / step_size)
    if n_steps < 1:
        return [q_to]
    out = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        q_i = interpolate_config(q_from, q_to, alpha)
        out.append(q_i)
    return out

def collision_free(model, data, q_from, q_to, step_size=0.2):
    """
    Check if the path q_from->q_to is collision-free by small steps.
    """
    path = local_steer(q_from, q_to, step_size=step_size)
    for q_i in path:
        qpos_candidate = config_to_qpos(q_i, data)
        if is_colliding(model, data, qpos_candidate):
            return False
    return True

def steer(q_near, q_rand, max_step=0.4):
    """
    If distance>max_step, create a new config at max_step away 
    from q_near in direction q_rand. Otherwise, q_rand.
    """
    dist = distance_config(q_near, q_rand)
    if dist <= max_step:
        return q_rand
    # fraction
    frac = max_step / dist
    return interpolate_config(q_near, q_rand, frac)

def get_neighbors(nodes, n_new, model, data, radius=1.0):
    """
    Return indices of nodes in 'nodes' that are within 'radius' of n_new,
    and collision-free from them to n_new.
    """
    neighbor_idx = []
    for i, nd in enumerate(nodes):
        dist = distance_config(nd.config, n_new.config)
        if dist < radius:
            # check collision
            if collision_free(model, data, nd.config, n_new.config):
                neighbor_idx.append(i)
    return neighbor_idx

def choose_parent(nodes, neighbor_idx, n_new):
    """
    Among the neighbors, pick the one that yields lowest cost if connected to n_new.
    """
    best_parent = None
    best_cost = float('inf')
    for idx in neighbor_idx:
        nd = nodes[idx]
        cost_via = nd.cost + distance_config(nd.config, n_new.config)
        if cost_via < best_cost:
            best_cost = cost_via
            best_parent = idx
    return best_parent, best_cost

def rewire(nodes, neighbor_idx, n_new_idx):
    """
    Attempt to lower cost of neighbors by going through n_new.
    """
    n_new = nodes[n_new_idx]
    for idx in neighbor_idx:
        nd = nodes[idx]
        cost_via_new = n_new.cost + distance_config(n_new.config, nd.config)
        if cost_via_new < nd.cost:
            # rewire
            nd.cost = cost_via_new
            nd.parent = n_new_idx

def rrt_star_planner(model, data, q_start, q_goal, 
                     max_samples=2000, max_step=0.4, rewire_radius=1.0, 
                     goal_threshold=0.5):
    """
    RRT* in (x,y,theta) with collision checking via MuJoCo.
    """
    logger.info("Starting RRT* from %s to %s", q_start, q_goal)

    # Create start node
    start_node = Node(q_start)
    start_node.parent = None
    start_node.cost   = 0.0

    nodes = [start_node]

    for iteration in range(max_samples):
        # 1) sample
        q_rand = sample_random_config()
        # 2) nearest
        dlist = [distance_config(nd.config, q_rand) for nd in nodes]
        idx_near = np.argmin(dlist)
        q_near = nodes[idx_near].config

        # 3) steer
        q_new = steer(q_near, q_rand, max_step=max_step)
        # 4) collision check from q_near->q_new
        if not collision_free(model, data, q_near, q_new):
            # skip
            if iteration % 100 == 0:
                logger.debug("Iter %d, skip collision", iteration)
            continue

        # Create node
        n_new = Node(q_new)

        # 5) find neighbors
        neighbor_idx = get_neighbors(nodes, n_new, model, data, radius=rewire_radius)

        # 6) choose best parent
        best_parent, best_cost = choose_parent(nodes, neighbor_idx, n_new)
        if best_parent is None:
            # fallback => connect from idx_near
            best_parent = idx_near
            best_cost   = nodes[idx_near].cost + distance_config(q_near, q_new)
        n_new.parent = best_parent
        n_new.cost   = best_cost

        # 7) add n_new
        new_idx = len(nodes)
        nodes.append(n_new)

        # 8) rewire
        rewire(nodes, neighbor_idx, new_idx)

        # 9) check goal
        if distance_config(q_new, q_goal) < goal_threshold:
            logger.info("Goal reached on iteration %d with node %s", iteration, q_new)
            # We can build path
            goal_node = Node(q_goal)
            goal_node.parent = new_idx
            goal_node.cost   = n_new.cost + distance_config(q_new, q_goal)
            nodes.append(goal_node)
            return reconstruct_path_star(nodes, len(nodes)-1)

        if iteration % 100 == 0:
            logger.debug("Iteration %d: Tree size = %d", iteration, len(nodes))

    logger.warning("RRT* failed to find a path after %d samples", max_samples)
    return None

def reconstruct_path_star(nodes, goal_idx):
    path = []
    idx = goal_idx
    while idx is not None:
        nd = nodes[idx]
        path.append(nd.config)
        idx = nd.parent
    path.reverse()
    return path

def config_to_qpos(q_config, data):
    """
    Map (x, y, theta) -> data.qpos for agent's free joint: (x,y,z, qw,qx,qy,qz).
    """
    qpos_copy = data.qpos.copy()
    x, y, th = q_config
    z = 0.6
    qw = math.cos(th/2)
    qz = math.sin(th/2)

    qpos_copy[0] = x
    qpos_copy[1] = y
    qpos_copy[2] = z
    qpos_copy[3] = qw
    qpos_copy[4] = 0.0
    qpos_copy[5] = 0.0
    qpos_copy[6] = qz

    return qpos_copy


# ---------------------------------------------------
# 3) EXECUTION
# ---------------------------------------------------
def local_steer_for_execution(q_from, q_to, step_size=0.05):
    """
    A smaller step version for path execution.
    """
    dist = distance_config(q_from, q_to)
    n_steps = int(dist / step_size)
    if n_steps < 1:
        return [q_to]
    out = []
    for i in range(1, n_steps+1):
        alpha = i / n_steps
        q_i = interpolate_config(q_from, q_to, alpha)
        out.append(q_i)
    return out

def execute_path_direct(model, data, path, viewer=None):
    """
    Teleport the agent along the path. For debugging, log each step or config.
    """
    logger.info("Executing path with %d waypoints", len(path))
    for i in range(len(path)-1):
        qA = path[i]
        qB = path[i+1]
        sub_steps = local_steer_for_execution(qA, qB, step_size=0.05)
        for j, q_i in enumerate(sub_steps):
            logger.debug("Moving agent from config %s to sub-step %d: %s", qA, j, q_i)
            qpos_candidate = config_to_qpos(q_i, data)
            data.qpos[:] = qpos_candidate
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            if viewer is not None:
                viewer.render()
            else:
                pass

def execute_path_with_motors(model, data, path, viewer=None):
    logger.warning("execute_path_with_motors is not implemented.")
    pass

# ---------------------------------------------------
# EXAMPLE MAIN (RRT*)
# ---------------------------------------------------
def main():
    logger.info("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path("environment.xml")
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    for _ in range(2000):
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
        else:
            break

    viewer.close()

    # Start, goal in (x, y, theta)
    q_start = np.array([0.0, 12.5, 0.0])
    q_goal  = np.array([2, 15, math.pi])

    logger.info("Planning from start=%s to goal=%s", q_start, q_goal)
    path = rrt_star_planner(
        model, data, q_start, q_goal,
        max_samples=2000,
        max_step=0.4,
        rewire_radius=1.0,
        goal_threshold=0.5
    )
    if path is None:
        logger.error("No path found!")
        return
    logger.info("Path found with %d waypoints.", len(path))

    # We'll re-load model & data for the final "execution" environment
    logger.info("Re-loading model for execution/visualization...")
    model = mujoco.MjModel.from_xml_path("environment.xml")
    data = mujoco.MjData(model)

    logger.info("Creating mujoco_viewer.MujocoViewer...")
    viewer = mujoco_viewer.MujocoViewer(model, data)

    logger.info("Now executing the path (teleport method).")
    execute_path_direct(model, data, path, viewer=viewer)

    logger.info("End of path execution. Let user observe a bit.")
    for _ in range(2000):
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
        else:
            break

    logger.info("Closing viewer.")
    viewer.close()

if __name__ == "__main__":
    main()
