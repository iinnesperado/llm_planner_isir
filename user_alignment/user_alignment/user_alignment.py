from cognitive_nodes.policy import Policy

from vlm_alignment.vlm_rag import VLMARG

class UserAlignment(Policy):
    def __init__(self, name='policy', **params):
        super().__init__(name, **params)

    def execute_callback(self, reauest, response):
        """execute the infer() function of VLMRAG i guess"""
        raise NotImplementedError