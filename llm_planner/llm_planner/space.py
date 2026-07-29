from copy import deepcopy

from cognitive_nodes.space import PointBasedSpace


class SemanticSpace(PointBasedSpace):
    """
    Handles the state representation and learning part of the model, but we're mostly interested in the fact
    that this class is responsible of calculating the activation value.
    Variables :
        - target_object equiv to "target_object is on the table/ my hand"
        - is_grasped makes reference if the Pnode is gonna be "table" or "grasper" type, as in it has the object on its hand or not
    """
    def __init__(self, target_object=None, is_grasped=False, **kwargs):
        super().__init__(**kwargs)
        # self.real_size = size
        self.size = 0
        self.target_object = target_object   # values : None or obj_id
        self. is_grasped = is_grasped    # bool 

    def get_probability(self, perception):
        """
        Reponsible to calculate the actiation value for later.

        :return activation_value: 1 if it's a grasper type, 0.7 if it's a table type
        """
        grasped_obj = perception['grasped_object'][0]
        if grasped_obj['data'] == self.target_object:
            if self.is_grasped:
                return 1.0
        elif not self.is_grasped:
            objects_list = [obj['name'] for obj in perception['objects']]
            if self.target_object in objects_list:
                return 0.85

        return 0.0