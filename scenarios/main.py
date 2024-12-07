### STEP 1: Implement Necessary Libraries ###
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt

### STEP 2: Load Environment, Obstacles, and Robots ###
def load_environment():
    """
    Loads the environment using RAI configuration.
    Returns the RAI configuration object and relevant frames.
    """
    C = ry.Config()
    C.addFile('/home/biqu/RAI/cs449_project/scenarios/simEnv.g')

    environment = C.getFrame("parking_lot")
    obstacles = [
        {"frame": C.getFrame(f"car{i+1}"), "size": C.getFrame(f"car{i+1}").getSize()[:2]} for i in range(5)
    ]
    robot = {"frame": C.getFrame("l_robot_base"), "size": C.getFrame("l_robot_base").getSize()[:2]}

    return C, environment, obstacles, robot

### STEP 3: Main Optimal Path Finding Function ###
def path_finding(task_target, obstacles, robot):
    """
    Calculates the optimal path for the robot to navigate from start to target
    while avoiding obstacles.
    Returns an array representing the planned path.
    """
    path = []  # Path to store robot positions
    robot_position = np.array(robot["frame"].getPosition()[:2])  # Extract X and Y
    target_position = task_target[:2]  # Ensure only X and Y are used
    obstacle_positions = [{"position": np.array(o["frame"].getPosition()[:2]), "size": o["size"]} for o in obstacles]

    # Loop until the robot reaches the target
    while np.linalg.norm(robot_position - target_position) > 0.1:
        # Find the next guidance point
        guidance_point = find_guidance_point(robot_position, target_position, obstacle_positions, robot)

        # Compute the Dubins path
        next_position, orientation = compute_dubins_path(robot_position, guidance_point)

        # Update robot position
        robot_position = next_position
        path.append(next_position)

        # Update robot's frame in RAI
        robot["frame"].setPosition(np.append(robot_position, 0))  # Append Z=0 to maintain compatibility

    return np.array(path)

### STEP 4: Sub Functions for Theoretical and Mathematical Calculations ###
def find_guidance_point(robot_position, target_position, obstacle_positions, robot):
    """
    Finds a guidance point avoiding collisions with rectangular obstacles.
    """
    guidance_point = target_position
    for obs in obstacle_positions:
        if check_collision(robot_position, obs, robot):
            # Adjust guidance point to avoid obstacle
            direction = target_position - robot_position
            guidance_point = robot_position + direction / np.linalg.norm(direction) * 1.0
            break
    return guidance_point

def compute_dubins_path(robot_position, guidance_point):
    """
    Generates the next step towards the guidance point using Dubins path logic.
    """
    delta = guidance_point - robot_position
    theta = np.arctan2(delta[1], delta[0])
    step_size = 0.2  # Adjustable step size
    next_position = robot_position + step_size * delta / np.linalg.norm(delta)
    return next_position, theta

def check_collision(robot_position, obstacle, robot):
    """
    Checks if the robot collides with a rectangular obstacle using an improved point-in-rectangle check.
    """
    # Robot bounds
    robot_center = robot_position  # X and Y only
    robot_size = robot["size"]  # Width and height

    # Define the four corners of the robot
    robot_corners = [
        robot_center + np.array([robot_size[0] / 2, robot_size[1] / 2]),  # Top-right
        robot_center + np.array([robot_size[0] / 2, -robot_size[1] / 2]),  # Bottom-right
        robot_center + np.array([-robot_size[0] / 2, robot_size[1] / 2]),  # Top-left
        robot_center + np.array([-robot_size[0] / 2, -robot_size[1] / 2]),  # Bottom-left
    ]

    # Obstacle bounds
    obstacle_min = obstacle["position"] - obstacle["size"] / 2
    obstacle_max = obstacle["position"] + obstacle["size"] / 2

    # Check if any of the robot corners are inside the obstacle's rectangle
    for corner in robot_corners:
        if (
            obstacle_min[0] <= corner[0] <= obstacle_max[0]  # X-axis overlap
            and obstacle_min[1] <= corner[1] <= obstacle_max[1]  # Y-axis overlap
        ):
            return True  # Collision detected

    return False  # No collision

### STEP 5: Visualization Function ###
def visualize_path(obstacles, path):
    """
    Visualizes the robot's path in the environment with rectangular obstacles.
    Includes debugging for robot corners.
    """
    plt.figure(figsize=(8, 8))

    # Plot obstacles as rectangles
    for obs in obstacles:
        obs_pos = np.array(obs["frame"].getPosition()[:2])  # Extract X and Y
        obs_size = np.array(obs["size"])  # Width and height
        rect = plt.Rectangle(
            obs_pos - obs_size / 2, obs_size[0], obs_size[1], color="red", alpha=0.5, label="Obstacle"
        )
        plt.gca().add_patch(rect)

    # Plot path
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color="blue", label="Path")
    plt.scatter(path[0, 0], path[0, 1], color="orange", label="Start")
    plt.scatter(path[-1, 0], path[-1, 1], color="green", label="Target")

    # Add robot corners for debugging
    for pos in path:
        robot_size = obstacles[0]["size"]  # Assume same size for visualization
        corners = [
            pos + np.array([robot_size[0] / 2, robot_size[1] / 2]),
            pos + np.array([robot_size[0] / 2, -robot_size[1] / 2]),
            pos + np.array([-robot_size[0] / 2, robot_size[1] / 2]),
            pos + np.array([-robot_size[0] / 2, -robot_size[1] / 2]),
        ]
        for corner in corners:
            plt.scatter(corner[0], corner[1], color="purple", s=10, label="Corner" if np.array_equal(pos, path[0]) else "")


    # Labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Robot Path Visualization with Rectangular Obstacles and Corners")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()

### STEP 6: Task Assignment Function ###
def task_assignment(task_target):
    """
    Assigns a task to the robot using the environment and target frame.
    """
    task_target = C.getFrame(task_target).getPosition()[:2]  # Assign the target position (X and Y only)
    return task_target

### STEP 7: Main Execution Function ###
if __name__ == "__main__":
    # Load Environment
    C, environment, obstacles, robot = load_environment()

    # Assign Task
    task_target = task_assignment("parking_space_1")

    # Calculate optimal path
    path = path_finding(task_target, obstacles, robot)

    # Visualize the path
    visualize_path(obstacles, path)
