import random

def col_type(col_index: int) -> str:
    """
    Determine whether a given column index should be Parking (S) or Road (H)
    based on the desired repeating pattern:

    - col=0 -> S
    - col=1 -> H
    - Then repeating groups of size 3: (S, S, H).

    So columns (2,3) -> S, col=4 -> H, columns (5,6) -> S, col=7 -> H, etc.
    """
    if col_index == 0:
        return 'S'
    elif col_index == 1:
        return 'H'
    else:
        # For col >= 2, repeat blocks of length 3 => (S, S, H)
        offset = (col_index - 2) % 3
        return 'S' if offset < 2 else 'H'


def create_layout_by_col(
    rows: int,
    total_cols: int,
    object_probability: float,
    agent_row: int,
    agent_col: int,
    place_agent: bool = True,
) -> list:
    """
    Create a matrix of size (rows x total_cols). For each column c, we determine if it's
    a parking column (S) or road column (H) using col_type(c).

    - If col_type(c) == 'H', every tile in that column is 'H'.
    - If col_type(c) == 'S', we randomize each tile in that column as '0' or '1' with probability object_probability.
    - Optionally place 'T' at (agent_row, agent_col), overriding '0'/'1'.

    Returns the final matrix of strings: 'H', '0', '1', or 'T'.
    """
    agent_row = rows
    agent_col = 0

    matrix = []
    for r in range(rows):
        row_data = []
        for c in range(total_cols):
            ctype = col_type(c)  # 'S' or 'H'
            if ctype == 'H':
                row_data.append('H')
            else:
                # Parking => random 0 or 1
                has_car = '1' if random.random() < object_probability else '0'
                row_data.append(has_car)
        matrix.append(row_data)

    # Add 2 rows and make them 'H' (for the bottom)
    for _ in range(2):
        matrix.append(['H'] * total_cols)

    # Place an agent 'T' if desired (in bounds)
    if place_agent:
        if 0 <= agent_row < len(matrix) and 0 <= agent_col < len(matrix[0]):
            matrix[agent_row][agent_col] = 'T'

    return matrix


def generate_mujoco_xml(
    matrix: list,
    xml_filename: str = "environment.xml",
    parking_dims=(2.5, 2.5),
    road_dims=(2.5, 2.5),
    object_dims=(2, 1.5, 1),
    offset=0.5,
):
    """
    Creates a MuJoCo XML with:
      - A floor
      - Box geoms for each tile
      - Obstacle <body> for any tile=='1'
      - A single agent car if tile=='T'
    and writes it to `xml_filename`.
    """

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    # Colors for tile
    color_parking = "0.5 0.7 1 1"  # bluish
    color_road = "0.6 0.6 0.6 1"  # gray

    half_obj_x = object_dims[0] / 2.0
    half_obj_y = object_dims[1] / 2.0
    half_obj_z = object_dims[2] / 2.0

    xml_lines = []
    xml_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml_lines.append('<mujoco model="parking_env">')
    xml_lines.append('    <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>')
    xml_lines.append('    <option timestep="0.01" gravity="0 0 -9.81" iterations="50" />')
    xml_lines.append('')
    xml_lines.append('    <default>')
    xml_lines.append('        <geom contype="1" conaffinity="1" condim="3" friction="1 0.1 0.1" rgba="0.8 0.8 0.8 1"/>')
    xml_lines.append('        <joint damping="0.1" limited="true"/>')
    xml_lines.append('        <motor ctrllimited="true" ctrlrange="-1 1"/>')
    xml_lines.append('    </default>')
    xml_lines.append('')
    xml_lines.append('    <worldbody>')
    xml_lines.append('        <!-- Cameras -->')
    xml_lines.append('        <camera name="overhead" pos="0 0 30" euler="0 0 0" mode="fixed"/>')
    xml_lines.append('        <camera name="angled" pos="15 15 15" euler="-45 0 45" mode="fixed"/>')
    xml_lines.append('')
    xml_lines.append('        <!-- Light -->')
    xml_lines.append('        <light directional="true" diffuse="0.7 0.7 0.7" pos="5 5 10"/>')
    xml_lines.append('')
    xml_lines.append('        <!-- Floor -->')
    xml_lines.append('        <geom name="floor" type="plane" pos="0 0 0" size="50 50 1" rgba="0.9 0.9 0.9 1"/>')
    xml_lines.append('')

    for r in range(rows):
        for c in range(cols):
            cell = matrix[r][c]
            if cell in ['0', '1']:  # Parking tile
                tile_w, tile_h = parking_dims
                color = color_parking
                tile_name = f"parking_space_{r}_{c}"
            else:  # Road tile
                tile_w, tile_h = road_dims
                color = color_road
                tile_name = f"road_{r}_{c}"

            half_x = tile_w / 2.0
            half_y = tile_h / 2.0
            center_x = c * tile_w
            center_y = r * tile_h

            xml_lines.append(f'        <geom name="{tile_name}" type="box"')
            xml_lines.append(f'              pos="{center_x} {center_y} 0.0"')
            xml_lines.append(f'              size="{half_x} {half_y} 0.01"')
            xml_lines.append(f'              rgba="{color}" />')

    # Obstacles for tile=='1'
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '1':
                tile_w, tile_h = parking_dims
                obs_x = c * tile_w
                obs_y = r * tile_h
                body_name = f"obs_car_{r}_{c}"
                xml_lines.append(f'        <body name="{body_name}" pos="{obs_x} {obs_y} {half_obj_z}">')
                xml_lines.append(f'            <geom name="{body_name}_geom" type="box" size="{half_obj_x} {half_obj_y} {half_obj_z}" rgba="0.4 0.4 0.4 1" />')
                xml_lines.append('        </body>')

    # Agent if tile=='T'
    agent_r = agent_c = None
    for rr in range(rows):
        for cc in range(cols):
            if matrix[rr][cc] == 'T':
                agent_r, agent_c = rr, cc
                break
        if agent_r is not None:
            break

    if agent_r is not None:
        tile_w, tile_h = parking_dims
        agent_x = agent_c * tile_w
        agent_y = agent_r * tile_h
        xml_lines.append('')
        xml_lines.append('        <!-- Agent Car -->')
        xml_lines.append(f'        <body name="agent_car" pos="{agent_x} {agent_y} 0.6" euler="0 0 0">')
        xml_lines.append('            <geom name="agent_car_chassis" type="box" size="1 0.7 0.4" rgba="0 0.6 0.3 1" />')
        xml_lines.append('            <joint name="agent_car_free" type="free"/>')
        xml_lines.append('')
        xml_lines.append('            <!-- FRONT LEFT WHEEL -->')
        xml_lines.append('            <body name="front_left_wheel" pos="0.6 0.6 -0.4">')
        xml_lines.append('                <joint name="steer_left_joint" type="hinge" axis="0 1 1" range="-0.6 0.6"/>')
        xml_lines.append('                <joint name="front_left_joint" type="hinge" axis="0 1 0" range="-999 999"/>')
        xml_lines.append('                <geom name="front_left_wheel_geom" type="cylinder" size="0.3 0.1" euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"/>')
        xml_lines.append('            </body>')
        xml_lines.append('')
        xml_lines.append('            <!-- FRONT RIGHT WHEEL -->')
        xml_lines.append('            <body name="front_right_wheel" pos="0.6 -0.6 -0.4">')
        xml_lines.append('                <joint name="steer_right_joint" type="hinge" axis="0 0 1" range="-0.6 0.6"/>')
        xml_lines.append('                <joint name="front_right_joint" type="hinge" axis="0 1 0" range="-999 999"/>')
        xml_lines.append('                <geom name="front_right_wheel_geom" type="cylinder" size="0.3 0.1" euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"/>')
        xml_lines.append('            </body>')
        xml_lines.append('')
        xml_lines.append('            <!-- REAR LEFT WHEEL -->')
        xml_lines.append('            <body name="rear_left_wheel" pos="-0.6 0.6 -0.4">')
        xml_lines.append('                <joint name="drive_left_joint" type="hinge" axis="0 1 0" range="-999 999"/>')
        xml_lines.append('                <geom name="rear_left_wheel_geom" type="cylinder" size="0.3 0.1" euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"/>')
        xml_lines.append('            </body>')
        xml_lines.append('')
        xml_lines.append('            <!-- REAR RIGHT WHEEL -->')
        xml_lines.append('            <body name="rear_right_wheel" pos="-0.6 -0.6 -0.4">')
        xml_lines.append('                <joint name="drive_right_joint" type="hinge" axis="0 1 0" range="-999 999"/>')
        xml_lines.append('                <geom name="rear_right_wheel_geom" type="cylinder" size="0.3 0.1" euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"/>')
        xml_lines.append('            </body>')
        xml_lines.append('        </body>')
        xml_lines.append('')

    xml_lines.append('    </worldbody>')
    xml_lines.append('')
    xml_lines.append('    <actuator>')
    if agent_r is not None:
        xml_lines.append('        <motor name="steer_left_motor" joint="steer_left_joint" gear="50"/>')
        xml_lines.append('        <motor name="steer_right_motor" joint="steer_right_joint" gear="50"/>')
        xml_lines.append('        <motor name="drive_left_motor" joint="drive_left_joint" gear="1000"/>')
        xml_lines.append('        <motor name="drive_right_motor" joint="drive_right_joint" gear="1000"/>')
    xml_lines.append('    </actuator>')
    xml_lines.append('</mujoco>')

    with open(xml_filename, 'w') as f:
        for line in xml_lines:
            f.write(line + "\n")

def main(
    rows: int = 5,
    total_cols: int = 9,
    object_probability: float = 0.5,
    object_dims=(2, 1.5, 1),
    offset: float = 0.5,
    parking_dims=(2.5, 2.5),
    road_dims=(2.5, 2.5),
    place_agent: bool = True,
    agent_row: int = 0,
    agent_col: int = 0,
    xml_filename: str = "environment.xml",
):
    """
    Main function to create a matrix by column pattern:
      - col=0 => parking
      - col=1 => road
      - col>=2 => repeating blocks of (S,S,H) every 3 columns

    For each 'S' column, random occupancy of 0 or 1. Optionally place 'T' in one cell.
    Then generate a MuJoCo XML with or without the agent (T).
    
    Parameters
    ----------
    rows : int
        Number of rows in the layout.
    total_cols : int
        Number of columns in the layout.
    object_probability : float
        Probability that a parking cell (S) is occupied by a car (1).
    object_dims : tuple
        (length, width, height) of obstacle cars.
    offset : float
        Spacing offset (not heavily used in the code, but kept for consistency).
    parking_dims : tuple
        (width, height) for parking columns.
    road_dims : tuple
        (width, height) for road columns.
    place_agent : bool
        Whether to place an agent 'T'.
    agent_row : int
        The row index of the agent.
    agent_col : int
        The column index of the agent.
    xml_filename : str
        Output filename for the MuJoCo XML.
    """

    final_matrix = create_layout_by_col(
        rows=rows,
        total_cols=total_cols,
        object_probability=object_probability,
        place_agent=place_agent,
        agent_row=agent_row,
        agent_col=agent_col,
    )

    # Print the final matrix
    print("Final Layout Matrix:")
    for row in final_matrix:
        print(" ".join(row))
    print()

    # Generate MuJoCo XML
    generate_mujoco_xml(
        matrix=final_matrix,
        xml_filename=xml_filename,
        parking_dims=parking_dims,
        road_dims=road_dims,
        object_dims=object_dims,
        offset=offset
    )
    print(f"[INFO] Wrote environment to {xml_filename}")


# Example usage if you run this script directly:
if __name__ == "__main__":
    main(
        rows=5,
        total_cols=9,
        object_probability=0.85,
        object_dims=(2,1.5,1),
        offset=0.5,
        parking_dims=(2.5, 2.5),
        road_dims=(2.5, 2.5),
        place_agent=True,
        agent_row=2,
        agent_col=0,
        xml_filename="environment.xml",
    )
