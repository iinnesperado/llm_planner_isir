import yaml
from copy import copy, deepcopy
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor

from cognitive_node_interfaces.msg import PerceptionStamped
from cognitive_node_interfaces.srv import Execute, Predict



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
        self.location = "base"         # Current location
        
        # Object tracking
        self.grasped_object = None     # Currently held object
        # self.grasped_part = "body"     # Which part of object is grasped
        self.world_objects = {}        # {obj_id: location}
        self.visible_objects = {}      # Objects in current FOV
        
    def grasp_object(self, obj_id):
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
            'joint_angles': copy(self.joint_angles),
            'gripper_width': self.gripper_width,
            'location': self.location,
            'grasped_object': self.grasped_object
        }
    
    def set_world_objects(self, objects_dict):
        """Initialize world objects: {obj_id: {location, type, subparts}}"""
        self.world_objects = deepcopy(objects_dict)
        print(self.world_objects)
        self.update_visible_objects()


class PickAndPlaceSim(Node):
    """
    ROS 2 node that wraps FrankaSimulator and publishes semantic perception.
    Basic first implementation to test LLMPlannerPolicy.
    """
    
    def __init__(self):
        super().__init__("LLMSimulation")
        
        # Callback groups for concurrency
        self.cbgroup_server = ReentrantCallbackGroup()
        self.cbgroup_perception = ReentrantCallbackGroup()
        
        # Simulators
        self.franka_sim = FrankaSimulator()
        self.perception_converter = SemanticPerceptionConverter()
        
        # Publisher for semantic perception
        self.perception_pub = self.create_publisher(
            PerceptionStamped,
            "perception/llm_planner/value",
            1,
            callback_group=self.cbgroup_perception
        )
        
        # Configuration
        self.config_file = self.declare_parameter(
            'config_file',
            descriptor=ParameterDescriptor(dynamic_typing=True)
        ).get_parameter_value().string_value
        
        self.load_configuration()
        self.get_logger().info("LLMSimulation initialized")
    
    def load_configuration(self):
        """
        Load simulation configuration from YAML file.
        Expected format:
        world_objects:
          mug: {location: 'table}
          plate: {location: 'table'}
        initial_location: 'base'
        """
        if self.config_file:
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Load world objects
                if 'world_objects' in config:
                    self.franka_sim.set_world_objects(config['world_objects'])
                    self.get_logger().info(f"Loaded objects: {list(config['world_objects'].keys())}")
                
                # Set initial location
                if 'initial_location' in config:
                    self.franka_sim.location = config['initial_location']
                    self.franka_sim.update_visible_objects()

                if 'locations' in config:
                    self.world_locations = deepcopy(config['locations'])
                
            except Exception as e:
                self.get_logger().warn(f"Could not load config file: {e}")
    
    def grasp_object_policy(self, obj_id):
        """Grasp an object if it's visible at current location"""
        success = self.franka_sim.grasp_object(obj_id)
        self.publish_perceptions()
        return success
    
    def place_object_policy(self, location):
        """Place currently grasped object at location"""
        success = self.franka_sim.place_object(location)
        self.publish_perceptions()
        return success
    
    def move_to_location_policy(self, location):
        """Move to a new location"""
        success = self.franka_sim.move_to_location(location)
        self.publish_perceptions()
        return success
    
    def publish_perceptions(self):
        """Publish current semantic perception as PerceptionStamped message"""
        arm_state = self.franka_sim.get_state()
        world_state = {
            'objects': self.franka_sim.world_objects,
            'visible_objects': self.franka_sim.visible_objects,
        }
        
        perception_dict = self.perception_converter.state_to_perception_dict(arm_state, world_state)
        
        # Create PerceptionStamped message
        msg = PerceptionStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.perception.data = yaml.dump(perception_dict)
        
        self.perception_pub.publish(msg)
        self.get_logger().debug(f"Published perception: {perception_dict}")

    def world_reset_service_callback(self, request, response):
        # Reset robot to inital state
        self.franka_sim.reset_state()
        # Clear/ reinitialize objects
        self.random_perceptions()
        self.publish_perceptions()
    
    def random_perceptions(self):
        """Shuffles the position of objects present in the environment."""
        world_objects = self.franka_sim.world_objects.keys()
        for obj in world_objects:
            obj['location'] = self.world_locations[np.random.randint(0,len(world_objects))]

    def new_command_callback(self, data):
        """Process a command received"""
        # NOTE dont really understand its role 

    def new_action_service_callback(self, request, response):
        """Execute the policy and publish perceptions."""
        self.get_logger().info("Executing policy " + str(request.policy))
        self.get_logger().info(f"ITERATION: {self.iteration}")
        # TODO add perception update here ??
        self.get_logger().info(f"PERCEPTIONS BEFORE: {self.perceptions}")
        self.get_logger().info(f"POLICY TO EXECUTE: {request.policy}")
        getattr(self, request.policy + "_policy")()
        self.perceive_closest_fruit()
        self.get_logger().info(f"FRUITS AFTER POLICY: {self.fruits}")
        self.get_logger().info(f"CATCHED FRUIT AFTER: {self.catched_fruit}")
        self.get_logger().info(f"PERCEPTIONS AFTER: {self.perceptions}")
        self.update_reward_sensor()
        self.publish_perceptions()
        if (not self.catched_fruit) and (
            self.perceptions["fruit_in_left_hand"].data
            or self.perceptions["fruit_in_right_hand"].data
        ):
            self.get_logger().error("Critical error: catched_object is empty and it should not!!!")
            rclpy.shutdown()
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    sim = PickAndPlaceSim()
    
    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print('Keyboard Interruption Detected: Shutting down Simulator...')
    finally:
        sim.destroy_node()


if __name__ == '__main__':
    main()
