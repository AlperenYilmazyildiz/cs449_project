import robotic as ry
import numpy as np

class SimulationManager:
    def __init__(self, config):
        self.C = config
        self.objects = []
        self.previous_target = None

    def setScenario(self, number_of_obstacles):
        for i in range(number_of_obstacles):
            # Add obstacle frames
            parent_frame = 'parking_space_' + str(np.random.randint(1, 42))
            if self.C.getFrame(parent_frame) is None:
                print(f"Error: Parent frame '{parent_frame}' does not exist.")
                continue

            f = self.C.addFrame(f'car{i}', parent_frame)
            f.setRelativePose('t(0 0 0.14) d(0 0 0 0)')
            f.setShape(ry.ST.box, [0.9, 0.39, 0.28])
            f.setColor([0.2, 0.2, 0.2, 0.4])
            f.setContact(1)
            f.setMass(1.0)

    def getAttributes(self, object_name):
        # Determine object type
        if object_name.startswith('l_robot_base'):
            obj_type = 'robot'
        elif object_name.startswith('target_'):
            obj_type = 'target'
        elif object_name.startswith('car'):
            obj_type = 'obstacle'
        else:
            obj_type = 'unknown'
  
        # Retrieve the frame
        frame = self.C.getFrame(object_name)
        if frame is None:
            print(f"Error: Frame for object '{object_name}' not found.")
            return None

        # Check if the object already exists
        for obj in self.objects:
            if obj['name'] == object_name:
                return obj

        # Collect attributes and add to the list
        attributes = {
            "name": object_name,
            "type": obj_type,
            "frame": frame,
            "position": frame.getPosition(),
            "size": frame.getSize(),
            "quaternion": frame.getQuaternion()
        }
        self.objects.append(attributes)
        return attributes

    def setTarget(self, object_name):
        # Retrieve the frame
        frame = self.C.getFrame(object_name)
        if frame is None:
            print(f"Error: Frame for object '{object_name}' not found.")
            return None

        # Revert previous target's color
        if self.previous_target:
            prev_frame = self.C.getFrame(self.previous_target)
            if prev_frame:
                prev_frame.setColor([0.2, 0.2, 0.2, 0.4])

        # Update current target's color
        frame.setColor([0, 1, 0, 0.4])

        # Update object type in the list
        for obj in self.objects:
            if obj['name'] == self.previous_target:
                obj['type'] = 'obstacle'
            if obj['name'] == object_name:
                obj['type'] = 'target'

        # Store the current target
        self.previous_target = object_name

# Initialize simulation
C = ry.Config()
C.addFile('/home/biqu/RAI/cs449_project/scenarios/simEnv.g')

# Create a SimulationManager instance
sim_manager = SimulationManager(C)

# Set up the scenario
sim_manager.setScenario(20)

# Retrieve attributes for all objects
for frame_name in C.getFrameNames():
    sim_manager.getAttributes(frame_name)

# Set an object as the target
sim_manager.setTarget('car2')

# Retrieve attributes for the target
g = sim_manager.objects
# View the simulation
C.view()
