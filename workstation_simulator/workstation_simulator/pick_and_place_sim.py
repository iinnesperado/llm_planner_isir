import yaml
import yamlloader
from copy import copy, deepcopy
import numpy as np
import os
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor

from core.service_client import ServiceClient
from core_interfaces.srv import LoadConfig
from core.utils import class_from_classname

from llm_planner_interfaces.srv import GraspObject, ReleaseObject




class PickAndPlaceSim(Node):
    """
    Basic first implementation to test LLMPlannerPolicy.

    Experiment information:
        - a robotic arm like Franka 
        - general goal is to keep the table clean
        - the posible locations for the objects are: 
            - table (init location)
            - toolbox
            - trash
            - in_hand
        - only objects on top of the table are visible and pickable
        - the robot reaches all the locations to place the objects
    """
    
    def __init__(self):
        super().__init__("PickAndPlaceSim")
        # self.rng = None
        self.perceptions = {}           # dict {sensor1: {attr1: ..., attr2: ...}, sensor2: ...}
        self.base_messages = {}
        self.sim_publishers = {}        # dict {sensor: publisher}

        self.random_seed = self.declare_parameter('random_seed', value = 0).get_parameter_value().integer_value
        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value

        self.objects = {}
        self.visible_objects = {}
        self.object_to_pick = None      # string
        self.grasped_object = None      # check if the robot has already an object
        self.grasped_part = None
        
        # Callback groups for concurrency
        self.cbgroup_server=MutuallyExclusiveCallbackGroup()
        self.cbgroup_client=MutuallyExclusiveCallbackGroup()
        
        self.load_client=ServiceClient(LoadConfig, 'commander/load_experiment')
        self.get_logger().info("PickAndPlaceSim initialized")
    
    def load_configuration(self):
        """
        Load the configuration file and setup the simulator.
        """
        if self.config_file is None:
            self.get_logger().error("No configuration file for the LTM simulator specified!")
            rclpy.shutdown()
        else:
            if not os.path.isfile(self.config_file):
                self.get_logger().error(self.config_file + " does not exist!")
                rclpy.shutdown()
            else:
                self.get_logger().info(f"Loading configuration from {self.config_file}...")
                config = yaml.load(
                    open(self.config_file, "r", encoding="utf-8"),
                    Loader=yamlloader.ordereddict.CLoader,
                )
                self.setup_perceptions(config["DiscreteEventSimulator"]["Perceptions"])
                # Be ware, we can not subscribe to control channel before creating all sensor publishers.
                self.setup_control_channel(config["Control"])

                self.setup_objects(config["DiscreteEventSimulator"]["Objects"])

        self.load_experiment_file_in_commander()

    def setup_perceptions(self, perceptions):
        """
        Configure the ROS topics where the simulator will publish the perceptions.
        
        :param perceptions: A list of dictionaries where each dictionary contains the name, perception topic, and perception message class.
        :type perceptions: list
        """
        for perception in perceptions:
            sid = perception["name"]
            topic = perception["perception_topic"]
            classname = perception["perception_msg"]
            message = class_from_classname(classname)
            self.perceptions[sid] = message()
            if "List" in classname:
                self.perceptions[sid].data = []
                self.base_messages[sid] = class_from_classname(classname.replace("List", ""))
            elif "Float" in classname:
                self.perceptions[sid].data = 0.0
            else:
                self.perceptions[sid].data = ""
            self.get_logger().info("I will publish " + str(sid) + " to... " + str(topic))
            self.sim_publishers[sid] = self.create_publisher(message, topic, 0)
        
        self.get_logger().debug(f"Setup perceptions finished : {self.perceptions}")
    
    def setup_control_channel(self, simulation):
        """
        Configure the ROS topic/service where listen for commands to be executed.

        :param simulation: The params from the config file to setup the control channel.
        :type simulation: dict
        """
        self.ident = simulation["id"]
        topic = simulation["control_topic"]
        classname = simulation["control_msg"]
        message = class_from_classname(classname)
        self.get_logger().info("Subscribing to... " + str(topic))
        self.create_subscription(message, topic, self.new_command_callback, 0)
        service_policy = simulation.get("executed_policy_service")
        service_world_reset = simulation.get("world_reset_service")

        if service_policy:
            self.get_logger().info("Creating server... " + str(service_policy))
            classname = simulation["executed_policy_msg"]
            message_policy_srv = class_from_classname(classname)
            self.create_service(message_policy_srv, service_policy, self.new_action_service_callback, callback_group=self.cbgroup_server)
            self.get_logger().info("Creating perception publisher timer... ")
            self.perceptions_timer = self.create_timer(0.01, self.publish_perceptions, callback_group=self.cbgroup_server)

        if service_world_reset:
            self.message_world_reset = class_from_classname(simulation["world_reset_msg"])
            self.create_service(self.message_world_reset, service_world_reset, self.world_reset_service_callback, callback_group=self.cbgroup_server)
    
    def setup_objects(self, objects):
        for obj in objects:
            self.objects[obj['id']] = dict(subparts=obj['subparts'], location=obj['location'], home=obj['home'])
            
            data = self.base_messages["objects"]()
            data.name = obj["id"]
            data.subparts = deepcopy(obj["subparts"])
            data.location = obj["location"]
            self.perceptions["objects"].data.append(data)

        self.get_logger().debug(f"Object list setup finished : {self.objects} and {self.perceptions}")
    
    def load_experiment_file_in_commander(self):
        """
        Load the configuration file in the commander node.

        :return: Response from the commander node indicating the success of the loading.
        :rtype: core_interfaces.srv.LoadConfig.Response
        """
        loaded = self.load_client.send_request(file = self.config_file)
        return loaded

    def reward_progress_object_in_place(self):
        """
        Gives a larger reward the closer the robot is to the goal of putting the object in its rightful place.
        If the object is placed right, the reward is 1.0.
        """
        progress = 0.0
        if self.check_object_in_place():
            progress = 1.0
        elif self.check_object_grasped():
            progress = 0.5
        elif self.check_object_pickable():
            progress = 0.2
        
        self.perceptions['progress_object_in_place'].data = progress
        # self.get_logger().info(f"Progress: {progress}, Perception: {self.perceptions}")

    def check_object_in_place(self):
        """
        Checks if the object is its home location.
        Returns True if there is reward, False if not.

        Simple logic, the moment one object reaches its home location we get reward.
        """
        for _, obj_data in self.objects.items():
            if obj_data['location'] == obj_data['home']:
                return True
        return False
    
    def check_object_grasped(self):
        """
        Checks if object has been grasped.
        """
        if self.perceptions['grasped_object'].data=="None":
            return False
        elif self.perceptions['grasped_object'].data=="":
            self.get_logger().warn("Checking perception for 'grasped_object' returns empty !")
            return False
        return True
    
    def check_object_pickable(self):
        """
        An object is pickable if it's visible. For the moment that means it is everything.
        """
        return True

    def grasp_mug_body_policy(self):
        return self.grasp_object('mug', 'body')

    def grasp_screwdriver_handle_policy(self):
        return self.grasp_object('screwdriver', 'handle')

    def grasp_banana_body_policy(self):
        return self.grasp_object('banana', 'body')
    
    def grasp_scissors_handle_policy(self):
        return self.grasp_object('scissors', 'handle')
    
    def grasp_object(self, obj_id, subpart):
        """Grasp an object if it's visible at current location"""
        if obj_id in self.visible_objects:
            self.grasped_object = obj_id
            if subpart in self.visible_objects[obj_id].get("subparts"):
                self.grasped_part = subpart
            else:
                self.get_logger().error(f"Subpart {subpart} is not defined for object {obj_id}.")
                return False
            
            self.objects[obj_id]["location"] = "in_hand"
            self.visible_objects.pop(obj_id)  # Remove from visible
            self.perceptions["grasped_object"].data = obj_id

            self.publish_perceptions()
            return True
        else:
            self.get_logger().error(f"Object {obj_id} is not on the table and thus cannot be picked.")
        return False 

    def release_on_table_policy(self):
        return self.release_object('table')
    
    def release_in_trash_policy(self):
        return self.release_object('trash')

    def release_on_shelf_policy(self):
        return self.release_object('shelf')

    def release_in_toolbox_policy(self):
        return self.release_object('toolbox')
    
    def release_object(self, location):
        """Place currently grasped object at location"""
        if self.grasped_object:
            self.objects[self.grasped_object]['location'] = location
            
            self.grasped_object = None
            self.grasped_part = None
            self.perceptions["grasped_object"].data = "None"

            self.publish_perceptions()
            return True
        else :
            self.get_logger().warning("WARNING - Robot has no object to release !")

        return False
    
    def update_visible_objects(self):
        """Update which objects are visible at current location"""
        self.visible_objects = {}
        for obj_id, obj_data in self.objects.items():
            if obj_data.get('location') == "table" and obj_id != self.grasped_object:
                self.visible_objects[obj_id] = obj_data

    def update_objects_location_in_perception(self):
        """Update location data on objects information."""
        for obj in self.perceptions["objects"].data:
            obj.location = self.objects[obj.name]["location"]

    def update_reward_sensor(self):
        """Update goal sensors' values."""
        for sensor in self.perceptions:
            reward_method = getattr(self, "reward_" + sensor, None)
            if callable(reward_method):
                reward_method()
    
    def publish_perceptions(self):
        """
        Publish the current perceptions to the corresponding topics.
        """
        self.perceptions['grasped_object'].data = self.grasped_object or "None"
        self.update_objects_location_in_perception()

        for ident, publisher in self.sim_publishers.items():
            self.get_logger().debug("Publishing " + ident + " = " + str(self.perceptions[ident].data))
            publisher.publish(self.perceptions[ident])

    def reset_world(self, data):
        self.get_logger().info(f"DEBUG: WORLD RESET OLD: {self.perceptions}")
        # Reset robot to inital state
        self.grasped_object = None
        self.grasped_part = None

        # Reinitialize objects
        self.reset_perceptions()
        self.update_visible_objects()
        self.publish_perceptions()
        self.get_logger().info(f"DEBUG: WORLD RESET NEW: {self.perceptions}")

    def world_reset_service_callback(self, request, response):
        self.reset_world(request)
        response.success = True
        return response
    
    def reset_perceptions(self):
        """
        Puts all the objects on top of the table. Releases object from gripper
        We consider the location 'table' to be the init location of all objects.
        """
        for _,obj_data in self.objects.items():
            obj_data['location'] = "table"

    def new_command_callback(self, data):
        """
        Process a command received

        :param data: The message that contais the command received.
        :type data: ROS msg defined in the config file. Typically cognitive_processes_interfaces.msg.ControlMsg
        """
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.iteration = data.iteration
        self.update_reward_sensor()
        if data.command == "reset_world":
            self.reset_world(data)
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown() 

    def new_action_service_callback(self, request, response):
        """Execute the policy and publish perceptions."""
        self.get_logger().info("Executing policy " + str(request.policy))
        self.get_logger().info(f"ITERATION: {self.iteration}")

        self.get_logger().info(f"OBJECTS BEFORE POLICY: {self.objects}")
        self.get_logger().info(f"GRASPED OBJECT BEFORE: ({self.grasped_object}, {self.grasped_part})")
        self.get_logger().info(f"PERCEPTIONS BEFORE: {self.perceptions}")
        self.get_logger().info(f"POLICY TO EXECUTE: {request.policy}")

        success = getattr(self, request.policy + "_policy")()

        self.get_logger().info(f"OBJECTS AFTER POLICY: {self.objects}")
        self.get_logger().info(f"GRASPED OBJECT AFTER: ({self.grasped_object}, {self.grasped_part})")
        self.get_logger().info(f"PERCEPTIONS AFTER: {self.perceptions}")

        if not success :
            self.get_logger().error("Policy execution unsuccessful! Shutting dowm simulator...")
            rclpy.shutdown()
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    sim = PickAndPlaceSim()
    sim.load_configuration()

    
    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print('Keyboard Interruption Detected: Shutting down Simulator...')
    finally:
        sim.destroy_node()


if __name__ == '__main__':
    main()
