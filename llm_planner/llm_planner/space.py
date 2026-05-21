from cognitive_nodes.space import Space


class SemanticSpace(Space):
    """Handles the state representation and learning part of the model, but we're mostly interested n the fact
    that this class is responsible of calculating the activation value"""
    def __init__(self, label):
        super.__init__()
        
    #     self.label = label # label of the object given by the User Alignment module
    #     # Atributs de la perception
    #     self.location = ""
    #     self.grasped_obj = {}   # could be a dict so that we have right and left
    #     self.known_obj = {}       # observed and recognised objects in the vision of the robot
    #     self.unkown_obj = {}

    # def set_perception(self, perception):
    #     """
    #     Set perception to the P-Node.
    #     IDEA : Perception quite simplified so it closes the posibility of similar situation 
    #     described in different ways, so we set the perception we need for the pnode to 
    #     activate and there's no need of collecting similar situations as being able to 
    #     activate the pnode.
    #     We suppose that the perception is a dict with keys coinciding with the defined ones here
    #     for the moment.

    #     TODO change it when the exact format of the perception is defined
    #     """
    #     self.location = perception['location']
    #     self.grasped_obj = perception['grasped_obj']
    #     self.observed_obj = perception['observed_obj']

    def get_probability(self, perception):
        """
        This is more a 'verify if the given perception matches the one of the pnode'
        if it does then proba = 1, if not 0.
        """
        if self.grasped_obj == perception['grasped_obj']:
            if self.grasped_obj is not None :
                return 1
        elif self.label in perception['known_obj']:
            return 1

        return 0