import mujoco
import glfw

# Global variables for camera interaction
left_button_down = False
last_mouse_pos = (0.0, 0.0)
cam_azimuth = 60.0
cam_elevation = -30.0
cam_distance = 25.0

def mouse_button_callback(window, button, action, mods):
    global left_button_down, last_mouse_pos
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            left_button_down = True
            last_mouse_pos = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            left_button_down = False

def cursor_pos_callback(window, xpos, ypos):
    global last_mouse_pos, cam_azimuth, cam_elevation
    global left_button_down

    if left_button_down:
        dx = xpos - last_mouse_pos[0]
        dy = ypos - last_mouse_pos[1]
        rotation_speed = 0.3
        cam_azimuth -= dx * rotation_speed
        cam_elevation -= dy * rotation_speed
        cam_elevation = max(-90, min(90, cam_elevation))
        last_mouse_pos = (xpos, ypos)

def scroll_callback(window, xoffset, yoffset):
    global cam_distance
    zoom_speed = 1.0
    cam_distance -= yoffset * zoom_speed
    cam_distance = max(1.0, min(200.0, cam_distance))

def main():
    # Load model, data
    model = mujoco.MjModel.from_xml_path("environment.xml")
    data = mujoco.MjData(model)

    # Init GLFW
    if not glfw.init():
        raise Exception("Could not init GLFW")

    window = glfw.create_window(1200, 900, "MuJoCo Env", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create window")

    glfw.make_context_current(window)

    # Register callbacks
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Camera, scene, context
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # Set initial camera
    cam.lookat = [9, 5, 0]
    cam.distance = cam_distance
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation

    while not glfw.window_should_close(window):
        mujoco.mj_step(model, data)

        # Update camera from global vars
        cam.azimuth = cam_azimuth
        cam.elevation = cam_elevation
        cam.distance = cam_distance

        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
