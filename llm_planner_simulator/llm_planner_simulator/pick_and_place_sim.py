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
from core.interface.srv import LoadConfig
from core.utils import class_from_classname




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
        self.perceptions = {}
        self.base_messages = {}
        self.sim_publishers = {}        # dict {sim_id: publisher}

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
        
        self.load_configuration()
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
            else:
                self.perceptions[sid].data = False
            self.get_logger().info("I will publish to... " + str(topic))
            self.sim_publishers[sid] = self.create_publisher(message, topic, 0)
    
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
            self.objects[obj['id']] = dict(subpart=obj['subpart'], location=obj['location'])
    
    def load_experiment_file_in_commander(self):
        """
        Load the configuration file in the commander node.

        :return: Response from the commander node indicating the success of the loading.
        :rtype: core_interfaces.srv.LoadConfig.Response
        """
        loaded = self.load_client.send_request(file = self.config_file)
        return loaded
    
    def random_object_to_pick(self):
        """Randomly select and object to pick from the available objects."""
        self.update_visible_objects()
        
        # self.object_to_pick = 
    
    def pick_object_policy(self):
        """Grasp an object if it's visible at current location"""
        if obj_id in self.visible_objects:
            self.grasped_object = obj_id
            if subpart in self.visible_objects[obj_id].get("subpart"):
                self.grasped_part = subpart
            else:
                self.get_logger().error(f"Subpart {subpart} is not defined for object {obj_id}.")
                return False
            
            self.objects[obj_id]["location"] = "in_hand"
            self.visible_objects.pop(obj_id)  # Remove from visible
            self.perceptions["grasped_object"].data = True

            self.publish_perceptions()
            return True
        else:
            self.get_logger().error(f"Object {obj_id} is not on the table and thus cannot be picked.")
        return False
    
    def place_object_policy(self):
        """Place currently grasped object at location"""
        if self.grasped_object:
            self.objects[self.grasped_object]['location'] = location
            
            self.grasped_object = None
            self.grasped_part = None
            self.perceptions["grasped_object"].data = False

            self.publish_perceptions()
            return True
        return False
    
    def update_visible_objects(self):
        """Update which objects are visible at current location"""
        self.visible_objects = {}
        for obj_id, obj_data in self.objects.items():
            if obj_data.get('location') == "table" and obj_id != self.grasped_object:
                self.visible_objects[obj_id] = obj_data

    def update_objects_location_in_perception(self):
        """Update location data on objects information."""
        for object in self.perceptions["object"]:
            object.location = self.objects[object.id]
    
    def publish_perceptions(self):
        """
        Publish the current perceptions to the corresponding topics.
        """
        self.update_objects_location_in_perception()
        for ident, publisher in self.sim_publishers.items():
            self.get_logger().debug("Publishing " + ident + " = " + str(self.perceptions[ident].data))
            publisher.publish(self.perceptions[ident])

    def world_reset_service_callback(self, request, response):
        # Reset robot to inital state
        self.grasped_object = None
        self.grasped_part = None

        # Reinitialize objects
        self.reset_perceptions()
        self.update_visible_objects()
        self.publish_perceptions()
    
    def reset_perceptions(self):
        """
        Puts all the objects on top of the table.
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

        self.random_object_to_pick()

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
    
    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print('Keyboard Interruption Detected: Shutting down Simulator...')
    finally:
        sim.destroy_node()


if __name__ == '__main__':
    main()
