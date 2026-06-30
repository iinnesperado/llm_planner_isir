from cognitive_nodes.policy import Policy
from cognitive_nodes.drive import Drive

from vlm_alignment.vlm_rag import VLMARG

class DriveAlignment(Drive):
    """
    DriveAlignment Class, represents a drive to receive input from users. 
    """    
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        """
        Constructor of the DriveAlignment class.

        :param name: The name of the Drive instance.
        :type name: str
        :param class_name: The name of the Drive class, defaults to "cognitive_nodes.drive.Drive".
        :type class_name: str
        """        
        super().__init__(name, class_name, **params)

    def evaluate(self, perception=None):
        """
        Evaluation that always returns 1.0, as the drive is always .

        :param perception: Unused perception.
        :type perception: dict or Any.
        :return: Evaluation of the Drive.
        :rtype: cognitive_node_interfaces.msg.Evaluation
        """        
        self.evaluation.evaluation = 1.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation

class PolicyAlignment(Policy):
    def __init__(self, name='policy', **params):
        super().__init__(name, **params)

    def execute_callback(self, reauest, response):
        """execute the infer() function of VLMRAG i guess"""
        raise NotImplementedError