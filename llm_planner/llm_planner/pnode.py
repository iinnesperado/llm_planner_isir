from cognitive_nodes.pnode import PNode

from llm_planner.space import SemanticSpace
from llm_planner_interfaces.srv import GetTargetObject

class SemanticPNode(PNode):
    def __init__(self, name='pnode', class_name='llm_planner.pnode.SemanticPNode', target_object=None, is_grasped=False, **params):
        """
        Surchage of PNode class to be able to add the information of the target_object and other to associated space.
        """
        space = SemanticSpace(target_object=target_object, is_grasped=is_grasped)
        super().__init__(name, class_name, space=space, **params)

        self.get_target_object_service = self.create_service(
            GetTargetObject,
            "pnode/" + str(self.name) + "get_target_object",
            self.get_target_object_callback, 
            callback_group = self.cbgroup_server
        )

    def get_target_object(self):
        return self.target_object
    
    def get_target_object_callback(self, request, response):
        """
        Callback to access the target_object.
        For the moment used for the policy LLM Planner
        """

        response.target_object = self.get_target_object()
        return response