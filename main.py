### STEP 1: Implement Necessary Libraries ###
import os
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt

### STEP 2: Load Environment, Obstacles, and Robots ###
def load_environment(environment_file, obstacles_file, robots_file):
    """
    Loads the environment, obstacles, and robots using RAI configuration.
    Returns the RAI configuration object and relevant frames.
    """
    C = ry.Config()
    C.addFile(environment_file)
    C.addFile(obstacles_file)
    C.addFile(robots_file)

    environment = C.getFrame("environment")
    task_start = C.getFrame("task_start")
    task_target = C.getFrame("task_target")
    obstacles = [C.getFrame(f"obstacle{i}") for i in range(5)]  # Adjust obstacle count
    robot = C.getFrame("robot")  # Select one robot

    return C, environment, task_start, task_target, obstacles, robot

### STEP 3: Main Task Assignment Function ###
def task_assignment(task_target, obstacles, robot):
    """
    Calculates the optimal path for the robot to navigate from start to target
    while avoiding obstacles.
    Returns an array representing the planned path.
    """
    path = []  # Path to store robot positions
    robot_position = np.array(robot.getPosition())
    target_position = np.array(task_target.getPosition())
    obstacle_positions = [np.array(o.getPosition()) for o in obstacles]

    # Loop until the robot reaches the target
    while np.linalg.norm(robot_position - target_position) > 0.1:
        # Find the next guidance point
        guidance_point = find_guidance_point(robot_position, target_position, obstacle_positions)

        # Compute the Dubins path
        next_position, orientation = compute_dubins_path(robot_position, guidance_point)

        # Update robot position
        robot_position = next_position
        path.append(next_position)

        # Update robot's frame in RAI
        robot.setPosition(robot_position)

    return np.array(path)

### STEP 4: Sub Functions for Theoretical and Mathematical Calculations ###
def find_guidance_point(robot_position, target_position, obstacle_positions):
    """
    Finds a guidance point avoiding collisions with obstacles using
    the GOS (Guidance Point Strategy) described in the PDF.
    """
    guidance_point = target_position
    for obs in obstacle_positions:
        if np.linalg.norm(robot_position - obs) < 0.4:  # Collision threshold
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

def check_collision(point, obstacles):
    """
    Checks if a given point collides with any obstacle.
    """
    for obs in obstacles:
        if np.linalg.norm(point - obs) < 0.4:  # Collision threshold
            return True
    return False

### STEP 5: Visualization Function ###
def visualize_path(environment_file, obstacles_file, robots_file, path):
    """
    Visualizes the robot's path in the environment.
    """
    C = ry.Config()
    C.addFile(environment_file)
    C.addFile(obstacles_file)
    C.addFile(robots_file)

    # Plot environment, obstacles, and robot's path
    plt.figure(figsize=(8, 8))

    # Plot obstacles
    obstacles = [C.getFrame(f"obstacle{i}") for i in range(5)]
    for obs in obstacles:
        obs_pos = np.array(obs.getPosition())
        plt.scatter(obs_pos[0], obs_pos[1], color="green", label="Obstacle" if obs == obstacles[0] else "")

    # Plot path
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color="blue", label="Path")
    plt.scatter(path[0, 0], path[0, 1], color="orange", label="Start")
    plt.scatter(path[-1, 0], path[-1, 1], color="red", label="Target")

    # Labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Robot Path Visualization")
    plt.legend()
    plt.grid()
    plt.show()

### STEP 6: Main Execution Function ###
if __name__ == "__main__":
    # Load environment
    file_path = os.path.abspath('Project') # Adjust the name of the folder according to the project
    environment_file = os.path.join(file_path, "scenarios/environment.g")
    obstacles_file = os.path.join(file_path, "scenarios/obstacles.g")
    robots_file = os.path.join(file_path, "scenarios/robots.g")

    C, environment, task_start, task_target, obstacles, robot = load_environment(
        environment_file, obstacles_file, robots_file
    )

    # Calculate optimal path
    path = task_assignment(task_target, obstacles, robot)

    # Visualize the path
    visualize_path(environment_file, obstacles_file, robots_file, path)

