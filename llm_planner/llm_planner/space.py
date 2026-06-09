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

    # def add_point(self, perception, confidence):
    #     """
    #     Based on the perception we setup the variables target_object and is_grasped.
    #     Not gonna use the idea of the points since is not likely to be used by our context.

    #     :param perception: A given perception to add
    #     :type perception: dicts
    #     """
    #     if perception["grasped_object"] is not None:
    #         self.target_object = perception["grasped_object"]
    #         self.is_grasped = True
    #     else:
    #         self.is_grasped = False


    def get_probability(self, perception):
        """
        Reponsible to calculate the actiation value for later.

        :return activation_value: 1 if it's a grasper type, 0.7 if it's a table type
        """
        if perception['grasped_object'] == self.target_object:
            if self.is_grasped:
                return 1
        elif not self.is_grasped:
            if self.target_object in perception["objects"]:
                return 0.7

        return 0