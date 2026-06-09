from cognitive_nodes.pnode import PNode

from llm_planner.space import SemanticSpace
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
            "pnode/" + str(self.name) + "get_target_object",
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
    
    def calculate_activation(self, perception=None, confidence=None):
        if perception is None:
            return 0.0
        
        space = self.get_space(perception)
        if space is None:
            return 0.0
        
        activation = space.get_probability(perception)

        return activation