import yaml
import yamlloader
from copy import copy, deepcopy
import numpy as np
import os
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import Image
from core.service_client import ServiceClient, ServiceClientAsync
from core_interfaces.srv import LoadConfig
from core.utils import class_from_classname
from cognitive_node_interfaces.msg import Perception, PerceptionStamped

from llm_planner_interfaces.msg import ObjectMsg
from llm_planner.utils import perception_msg_to_dict
from user_alignment.utils import png_to_ros_img

from custom_interfaces.srv import GraspRequest



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
        self.robot_vision_sub = {}

        self.random_seed = self.declare_parameter('random_seed', value = 0).get_parameter_value().integer_value
        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value

        self.objects = {}               # dict {obj_id: {location: location_id}}
        self.grasped_object = None      # check if the robot has already an object
        self.img_idx = 0
        
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

                # self.setup_objects(config["DiscreteEventSimulator"]["Objects"])
        
        self.configure_robot_vision_sub()
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
            elif "String" in classname:
                self.perceptions[sid].data = ""
            elif "Image" in classname:
                self.perceptions[sid] = self.setup_camera_img()
                
            self.get_logger().info("I will publish " + str(sid) + " to... " + str(topic))
            self.sim_publishers[sid] = self.create_publisher(message, topic, 0)
        
        self.get_logger().debug(f"Setup perceptions finished : {self.perceptions}")

    def setup_camera_img(self):
        """
        Sets up the static image to be published during the simulation.
        Only used for testing wihout a camera in simulator.
        """
        # if self.img_idx<10:
        #     img_msg = png_to_ros_img(f"/home/user/ines_ros2_humble/eMDB_ws/build/workstation_simulator/workstation_simulator/config/rgb_00{self.img_idx}.png")
        # elif self.img_idx<14 :
        #     img_msg = png_to_ros_img(f"/home/user/ines_ros2_humble/eMDB_ws/build/workstation_simulator/workstation_simulator/config/rgb_0{self.img_idx}.png")
        # else :
        #     self.img_idx = 0

        img_msg = png_to_ros_img(f"/home/user/ines_ros2_humble/eMDB_ws/build/workstation_simulator/workstation_simulator/config/rgb_001.png")
        img_msg.header.stamp = self.get_clock().now().to_msg()

        return img_msg

    def configure_robot_vision_sub(self):
        """
        Subscription to the perception topic 'robot_vision'.
        Information used for the VLM queries.
        """
        subscriber = self.create_subscription(
            PerceptionStamped,
            "perception/robot_vision/value",
            self.robot_vision_callback,
            1,
            callback_group=self.cbgroup_client
        )
            
        data = Perception()
        updated = False
        self.robot_vision_sub = dict(subscriber=subscriber, data=data, updated=updated)
        self.get_logger().info("Subscribed to 'robot_vision' perception topic")

    def robot_vision_callback(self, msg: PerceptionStamped):
        """
        Callback method that reads perception topic 'robot_vision' and stores it in robot_vision_sub.
        """
        perception_dict = perception_msg_to_dict(msg.perception)
        if len(perception_dict)>1:
                    self.get_logger().error(f"Received perception with multiple sensors: {perception_dict.keys()}. Perception nodes should (currently) include only one sensor!")
        if len(perception_dict)==1:
            self.robot_vision_sub['data'] = perception_dict['robot_vision'][0]
            self.robot_vision_sub['updated'] = True
            self.update_objects(self.robot_vision_sub['data'])
            self.publish_perceptions()
        else :
            self.get_logger().warning("Empty 'robot_vision' perception received in Pick and Place Sim. No update in the perceptions.")

    def update_objects(self, data):
        """
        The list self.objects serves as a database of information on all the objects encountered during the experiment.
        This fonction add the object perceived by robot_vision to object dict of information, if it was not already in the dict.
        We suppose there are not repetitions of the same object.

        :param data: information on the robot vision
        :type data: dict
        """
        if data['name'] not in self.objects.keys():
            self.objects[data['name']] = dict(location=data['location'])
            object_msg = ObjectMsg()
            object_msg.name = data['name']
            object_msg.location = data['location']
            self.perceptions['objects'].data.append(object_msg)
        # else:
        #     self.objects[data['name']]['location'] = data['location']
    
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
        Checks if the object is its target location.
        Returns True if there is reward, False if not.

        Simple logic, the moment one object reaches its target location we get reward.
        """
        for _, obj_data in self.objects.items():
            if obj_data['location'] not in ["table", "in_hand"]:
                return True
        return False
    
    def check_object_grasped(self):
        """
        Checks if object has been grasped.
        """
        # Use the canonical internal state `self.grasped_object`.
        return self.grasped_object is not None
    
    def check_object_pickable(self):
        """
        An object is pickable if it's visible.
        """
        if self.robot_vision_sub['updated']:
            self.robot_vision_sub['updated'] = False
            return self.robot_vision_sub['data']!=0
        return False

    def new_action_service_callback(self, request, response):
        """Execute the policy and publish perceptions."""
        self.get_logger().info("Executing policy " + str(request.policy))
        self.get_logger().info(f"ITERATION: {self.iteration}")

        self.get_logger().info(f"OBJECTS BEFORE POLICY: {self.objects}")
        self.get_logger().info(f"GRASPED OBJECT BEFORE: {self.grasped_object}")

        request_policy_split = request.policy.split("__")
        self.get_logger().info(f"POLICY TO EXECUTE: {request_policy_split}")

        policy_name = request_policy_split.pop(0)
        params = request_policy_split
        success = getattr(self, policy_name + "_policy")(*params)

        self.get_logger().info(f"OBJECTS AFTER POLICY: {self.objects}")
        self.get_logger().info(f"GRASPED OBJECT AFTER: {self.grasped_object}")
        self.update_reward_sensor()
        self.publish_perceptions()

        if not success :        
            self.get_logger().error("--- Policy execution unsuccessful! Shutting dowm simulator...")
            rclpy.shutdown()
        response.success = True
        return response

    def grasp_object_policy(self, target_object):
            """Grasp an object if it's visible at current location"""
            visible_object = self.robot_vision_sub['data']
            self.robot_vision_sub['updated'] = False
            if target_object == visible_object['name']:
                self.grasped_object = target_object
                self.objects[target_object]["location"] = "in_hand"
                self.publish_perceptions()
                return True
            else:
                self.get_logger().error(f"Object {target_object} is not on the table and thus cannot be picked.")
            return False 

    def release_object_policy(self, target_location):
        """Release currently grasped object at location"""
        if self.grasped_object:
            self.objects[self.grasped_object]['location'] = target_location
            self.grasped_object = None
            self.img_idx += 1
            self.publish_perceptions()
            return True
        else :
            self.get_logger().warning("WARNING - Robot has no object to release !")

        return False

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
        self.perceptions['camera'] = self.setup_camera_img()
        self.perceptions['grasped_object'].data = self.grasped_object if self.grasped_object is not None else "None"
        # Updates the location of the objects in corresponding perception data
        for obj in self.perceptions["objects"].data:
            obj.location = self.objects[obj.name]["location"]

        for ident, publisher in self.sim_publishers.items():
            self.get_logger().debug("Publishing " + ident + " = " + str(self.perceptions[ident].data))
            publisher.publish(self.perceptions[ident])

    def reset_world(self, data):
        self.get_logger().debug(f"DEBUG: WORLD RESET OLD: {self.perceptions}")
        # Reset robot to inital state
        self.grasped_object = None
        self.objects = {}

        # Reinitialize objects
        self.reset_perceptions()

        self.update_reward_sensor()
        self.publish_perceptions()
        self.get_logger().debug(f"DEBUG: WORLD RESET NEW: {self.perceptions}")

    def world_reset_service_callback(self, request, response):
        self.reset_world(request)
        response.success = True
        return response
    
    def reset_perceptions(self):
        """
        Resets sensors to their initial state.
        That means that the objects database just has as information what is visible.
        The grasped is None.
        """
        self.perceptions['grasped_object'].data = "None"
        self.perceptions['objects'].data = []

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
