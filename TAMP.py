import logging
import numpy as np
import mujoco
import mujoco_viewer
import math
import matplotlib.pyplot as plt
import scipy.interpolate as si
from reeds_shepp_rrt import rrt_reeds_shepp_algorithm

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def nearest_empty_parking_space_point(model, data):
    agent_x = data.qpos[0]
    agent_y = data.qpos[1]
    min_dist = float('inf')
    best_point = None
    mujoco.mj_forward(model, data)

    # Collect the positions of obstacles
    obstacle_positions = []
    for geom_idx in range(model.ngeom):
        geom_name_adr = model.name_geomadr[geom_idx]
        geom_name = _read_null_terminated(model.names, geom_name_adr)

        if geom_name.startswith("obs_car"):
            body_id = model.geom_bodyid[geom_idx]
            gx, gy, gz = data.xpos[body_id]
            obstacle_positions.append((gx, gy))
            logger.debug(f"Obstacle at global position ({gx:.2f}, {gy:.2f})")

    # Iterate through all geometries in the model
    for geom_idx in range(model.ngeom):
        geom_name_adr = model.name_geomadr[geom_idx]
        geom_name = _read_null_terminated(model.names, geom_name_adr)

        if geom_name.startswith("parking_"):
            gx, gy, _ = model.geom_pos[geom_idx]

            is_occupied = any(
                math.hypot(gx - ox, gy - oy) < 1.0 for ox, oy in obstacle_positions
            )

            if not is_occupied:
                dist = math.hypot(gx - agent_x, gy - agent_y)
                if dist < min_dist:
                    min_dist = dist
                    best_point = (gx, gy, 0.0)

    if best_point:
        logger.info(f"Nearest empty parking point: {best_point} at dist={min_dist:.2f}")
        return best_point
    else:
        logger.warning("No empty parking spaces found!")
        return (agent_x, agent_y, 0.0)

def _read_null_terminated(byte_buf, start):
    out = []
    for c in byte_buf[start:]:
        if c == 0:
            break
        out.append(chr(c))
    return "".join(out)


def fit_spline_to_path(path, smoothing_factor=0.1):
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    t = np.linspace(0, 1, len(path))
    spl_x = si.UnivariateSpline(t, x, s=smoothing_factor)
    spl_y = si.UnivariateSpline(t, y, s=smoothing_factor)
    return spl_x, spl_y

def sample_spline_with_margin(spl_x, spl_y, num_points=20):
    t = np.linspace(0, 1, num_points)
    sampled_path = [(spl_x(ti), spl_y(ti)) for ti in t]
    return sampled_path

def follow_spline_with_constant_speed(model, data, sampled_path, margin=0.5, viewer=None):
    max_steer_angle = math.radians(60.0)
    constant_speed = 0.5
    steer_kp = 1

    def set_steering(angle):
        clipped = max(-max_steer_angle, min(max_steer_angle, angle))
        normalized = clipped / max_steer_angle
        data.ctrl[0] = normalized
        data.ctrl[1] = normalized

    def set_drive(speed):
        normalized_speed = speed / constant_speed
        data.ctrl[2] = normalized_speed
        data.ctrl[3] = normalized_speed

    def stop():
        data.ctrl[0] = 0.0
        data.ctrl[1] = 0.0
        data.ctrl[2] = 0.0
        data.ctrl[3] = 0.0

    def get_agent_pose():
        x = data.qpos[0]
        y = data.qpos[1]
        qw = data.qpos[3]
        qz = data.qpos[6]
        yaw = math.atan2(2 * qw * qz, 1 - 2 * (qz**2))
        return x, y, yaw

    for target_x, target_y in sampled_path:
        while True:
            cx, cy, cth = get_agent_pose()
            dx = target_x - cx
            dy = target_y - cy
            dist = math.hypot(dx, dy)

            if dist < margin:
                logger.debug(f"Reached point ({target_x:.2f}, {target_y:.2f}).")
                break

            desired_angle = math.atan2(dy, dx)
            angle_diff = desired_angle - cth
            steer_control = steer_kp * angle_diff

            set_steering(steer_control)
            set_drive(constant_speed)

            mujoco.mj_step(model, data)
            if viewer and viewer.is_alive:
                viewer.render()

    stop()
    logger.info("Spline following completed.")
    for _ in range(50):
        mujoco.mj_step(model, data)
        if viewer and viewer.is_alive:
            viewer.render()

def main():
    logger.info("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path("environment.xml")
    data = mujoco.MjData(model)

    q_goal = nearest_empty_parking_space_point(model, data)
    logger.info(f"Nearest parking spot: {q_goal}")

    path = rrt_reeds_shepp_algorithm(
        goal=(q_goal[0], q_goal[1], math.radians(0)),
        model=model,
        max_iter=500,
        step_size=2,
        turning_radius=1.0,
        rewire_radius=3.0,
        goal_threshold=2.0,
        visualize=True
    )

    if not path:
        logger.error("No path found!")
        return

    logger.info("Fitting spline to the path.")
    spl_x, spl_y = fit_spline_to_path(path, smoothing_factor=0.5)
    plt.plot([p[0] for p in path], [p[1] for p in path], "ro")

    logger.info("Sampling points from spline with margin.")
    sampled_path = sample_spline_with_margin(spl_x, spl_y, num_points=200)

    logger.info("Reloading model for execution.")
    model = mujoco.MjModel.from_xml_path("environment.xml")
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)

    logger.info("Following sampled points on spline.")
    follow_spline_with_constant_speed(model, data, sampled_path, margin=0.5, viewer=viewer)

    viewer.close()

if __name__ == "__main__":
    main()
