import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from core.service_client import ServiceClient

from core_interfaces.srv import LoadConfig
from core.utils import class_from_classname

class LLMSim(Node):
    def __init__(self):
        super().__init__("LLMSimulation")
        self.perceived_objects = {}  # {obj_id: location}
        self.grasped_objects = {}    # {obj_id: hand} (could be left or right if there's two hands)

        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value

    
    def grasp_object_policy(self, obj_id):
        """Grasp l'objet s'il est observé"""
        if obj_id in self.perceived_objects:
            self.grasped_objects[obj_id] = "hand"
            del self.perceived_objects[obj_id]
            self.publish_perceptions()
            return True
        return False
    
    def place_object_policy(self, obj_id, location):
        """Place l'objet s'il est saisi"""
        if obj_id in self.grasped_objects:
            self.perceived_objects[obj_id] = location
            del self.grasped_objects[obj_id]
            self.publish_perceptions()
            return True
        return False
    
    def reward_pick(self):

        return 
    
    def reward_place(self):
        return
    
    def publish_perceptions(self):
        """Publier l'état actuel de la perception"""
        perception = {
            'observed_obj': self.perceived_objects,
            'grasped_obj': self.grasped_objects,
            'location': "robot_location"
        }

    def new_action_service_callback(self, request, response):
        """
        Execute a policy and publish new perceptions.

        :param request: The message that contains the policy to execute.
        :type request: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.Policy.Request
        :param response: Response of the success of the execution of the action.
        :type response: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.Policy.Response
        :return: Response indicating the success of the action execution.
        :rtype: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.Policy.Response
        """
        self.get_logger().info("Executing policy " + str(request.policy))
        getattr(self, request.policy + "_policy")()
        self.update_reward_sensor()
        self.publish_perceptions()
        if (not self.catched_object) and (
            self.perceptions["ball_in_left_hand"].data
            or self.perceptions["ball_in_right_hand"].data
        ):
            self.get_logger().error("Critical error: catched_object is empty and it should not!!!")
            rclpy.shutdown()
        response.success = True
        return response

    def setup_control_channel(self, simulation):
        """
        # NOTE Copy pasted, so adapt for sim
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
        topic = simulation.get("executed_policy_topic")
        service_policy = simulation.get("executed_policy_service")
        service_world_reset = simulation.get("world_reset_service")

        if topic:
            self.get_logger().info("Subscribing to... " + str(topic))
            self.create_subscription(message, topic, self.new_action_callback, 0)
        if service_policy:
            self.get_logger().info("Creating server... " + str(service_policy))
            classname = simulation["executed_policy_msg"]
            message_policy_srv = class_from_classname(classname)
            self.create_service(message_policy_srv, service_policy, self.new_action_service_callback, callback_group=self.cbgroup_server)
            self.get_logger().info("Creating perception publisher timer... ")
            self.perceptions_timer = self.create_timer(0.01, self.publish_perceptions, callback_group=self.cbgroup_server)
        if service_world_reset:
            classname= simulation["executed_policy_msg"]
            self.message_world_reset = class_from_classname(simulation["world_reset_msg"])
            self.create_service(self.message_world_reset, service_world_reset, self.world_reset_service_callback, callback_group=self.cbgroup_server)     

    def load_configuration(self):
        """
        Load configuration from a file.
        """
        return 