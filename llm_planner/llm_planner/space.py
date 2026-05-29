from copy import deepcopy

from cognitive_nodes.space import Space


class SemanticSpace(Space):
    """
    Handles the state representation and learning part of the model, but we're mostly interested n the fact
    that this class is responsible of calculating the activation value.
    Variables :
        - target_object equiv to "target_object is on the table/ my hand"
        - is_grasped makes reference if the Pnode is gonna be "table" or "grasper" type
    """
    def __init__(self, size=30000, **kwargs):
        super.__init__(**kwargs)
        self.real_size = size
        self.size = 0
        self.target_object = None   # values : None or obj_id
        self. is_grasped = None    # bool 

    def add_point(self, perception, confidence):
        """
        Based on the perception we setup the variables target_object and is_grasped.
        Not gonna use the idea of the points since is not likely to be used by our context.
        """
        


    def get_probability(self, perception):
        """
        This is more a 'verify if the given perception matches the one of the pnode'
        if it does then proba = 1, if not 0.
        """
        if self.perception['grasped_object'] == perception['grasped_object']:
            if self.perception['grasped_object'] is not None :
                return 1
        # elif self.label in perception['known_obj']:   # here label is the name of object that we might be interested to grasp
        #     return 1

        return 0