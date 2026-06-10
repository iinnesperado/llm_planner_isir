from rclpy.time import Time

from cognitive_nodes.pnode import PNode
from cognitive_node_interfaces.msg import PerceptionStamped

from llm_planner.space import SemanticSpace
from llm_planner.utils import perception_msg_to_dict
from llm_planner_interfaces.srv import GetTargetObject

class SemanticPNode(PNode):
    def __init__(self, name='pnode', class_name='cognitive_nodes.pnode.PNode', target_object=None, is_grasped=False, **params):
        """
        Surchage of PNode class to be able to add the information of the target_object and other to associated space.

        :param target_object: equiv to "target_object is on the table/ my hand"
        :param is_grasped: makes reference if the Pnode is gonna be "table" or "grasper" type, as in it has the object on its hand or not
        """
        space = SemanticSpace(target_object=target_object, is_grasped=is_grasped)
        super().__init__(name, class_name, space=space, **params)

        self.target_object = target_object

        self.get_target_object_service = self.create_service(
            GetTargetObject,
            "pnode/" + str(self.name) + "/get_target_object",
            self.get_target_object_callback, 
            callback_group = self.cbgroup_server
        )
    
    def get_target_object_callback(self, request, response):
        """
        Callback to access the target_object.
        For the moment used for the policy LLM Planner
        """

        response.target_object = self.target_object
        return response
    
    def calculate_activation(self, perception=None, activation_list=None):
        if perception is None:
            return 0.0
        
        space = self.get_space(perception)
        if space is None:
            return 0.0
        
        activation = space.get_probability(perception)

        return activation
    
    def read_activation_callback(self, msg : PerceptionStamped):
        perception_dict = perception_msg_to_dict(msg=msg.perception)
        self.get_logger().debug(f"Reading perception ... {perception_dict}")

        if len(perception_dict)>1:
            self.get_logger().error(f'{self.name} -- Received perception with multiple sensors: ({perception_dict.keys()}). Perception nodes should (currently) include only one sensor!')
        if len(perception_dict)==1:
            node_name=list(perception_dict.keys())[0]
            if node_name in self.activation_inputs:
                self.activation_inputs[node_name]['data']=perception_dict[node_name]
                self.activation_inputs[node_name]['updated']=True
                self.activation_inputs[node_name]['timestamp']=Time.from_msg(msg.timestamp)
        else:
            self.get_logger().warn(f"Empty perception recieved in P-Node. No activation calculated")
