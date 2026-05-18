"""
Pure Python simulator for Franka robotic arm with semantic perception.
No ROS dependencies - can be used standalone for testing and development.
"""

import yaml
import copy


class SemanticPerceptionConverter:
    """
    Converts Franka simulator state to semantic perception format compatible with space.py
    """
    def __init__(self):
        self.location_names = {
            "base": "robot_base",
            "table": "table_top",
            "gripper": "in_gripper"
        }
    
    def state_to_perception_dict(self, arm_state, world_state):
        """
        Convert simulator state to semantic perception dict.
        
        Expected output format (from space.py):
        {
            'location': str,           # robot location
            'grasped_obj': dict,       # what's in gripper {obj_id: part}
            'known_obj': dict,         # all known objects {obj_id: location}
            'observed_obj': dict,      # currently visible objects
        }
        """
        grasped_obj = {}
        if arm_state['grasped_object']:
            grasped_obj[arm_state['grasped_object']] = arm_state.get('grasped_part', 'body')
        
        perception_dict = {
            'location': arm_state.get('location', 'base'),
            'grasped_obj': grasped_obj,
            'known_obj': world_state.get('objects', {}),
            'observed_obj': world_state.get('visible_objects', {}),
        }
        return perception_dict


class FrankaSimulator:
    """
    Discrete event simulator for Franka robotic arm with semantic perception.
    Pure Python implementation - no ROS dependencies.
    """
    def __init__(self):
        # Franka state
        self.joint_angles = [0.0] * 7  # 7-DOF arm
        self.gripper_width = 0.04      # Open (meters)
        self.gripper_force = 0.0
        self.location = "table"         # Current location
        
        # Object tracking
        self.grasped_object = None     # Currently held object
        # self.grasped_part = "body"     # Which part of object is grasped
        self.world_objects = {}        # {obj_id: location}
        self.visible_objects = {}      # Objects in current FOV
        
    def grasp_object(self, obj_id): #, subpart="body"):
        """Grasp an object if it's visible"""
        if obj_id in self.visible_objects:
            self.grasped_object = obj_id
            # self.grasped_part = subpart
            self.gripper_width = 0.0
            self.visible_objects.pop(obj_id)  # Remove from visible
            return True
        return False
    
    def place_object(self, location):
        """Place grasped object at location if gripper holds something"""
        if self.grasped_object:
            self.world_objects[self.grasped_object]['location'] = location
            self.visible_objects[self.grasped_object] = self.world_objects[self.grasped_object]
            self.grasped_object = None
            # self.grasped_part = "body"
            self.gripper_width = 0.04
            return True
        return False
    
    def move_to_location(self, location):
        """Move to a new location"""
        for _, obj_data in self.world_objects.items():
            if obj_data.get('location')==location or location == "base":
                self.location = location
                self.update_visible_objects()
                return True
        return False
    
    def update_visible_objects(self):
        """Update which objects are visible at current location"""
        self.visible_objects = {}
        for obj_id, obj_data in self.world_objects.items():
            if obj_data.get('location') == self.location and obj_id != self.grasped_object:
                self.visible_objects[obj_id] = obj_data
    
    def get_state(self):
        """Get current arm state"""
        return {
            'joint_angles': copy.copy(self.joint_angles),
            'gripper_width': self.gripper_width,
            'location': self.location,
            'grasped_object': self.grasped_object
        }
    
    def set_world_objects(self, objects_dict):
        """Initialize world objects: {obj_id: {location, type, subparts}}"""
        self.world_objects = copy.deepcopy(objects_dict)
        print(self.world_objects)
        self.update_visible_objects()
