import yaml

from core.service_client import ServiceClient
from core_interfaces.srv import CreateNode, UpdateNeighbor
from cognitive_nodes.policy import Policy
from cognitive_nodes.drive import Drive
from cognitive_node_interfaces.msg import PerceptionStamped

from llm_planner.utils import perception_msg_to_dict

from user_alignment.vlm_rag import VLMARG
from user_alignment.utils import ros_img_to_base64

class DriveUserAlignment(Drive):
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

class PolicyUserAlignment(Policy):
    def __init__(self, name="policy", ltm_id=None, **params):
        super().__init__(name, **params)
        self.ltm_id = ltm_id

        self.vlm_client = VLMARG()

        self.perception_sub = {}

        self.configure_perception()

    async def execute_callback(self, request, response):
        """
        Execute the infer() function of VLMRAG I guess.
        And then do all the working around the nodes to have the goal, pnode, cnode and connect to planner policy.
        """
        self.get_logger().info(f"== START USER ALIGNMENT POLICY ==")

        perception_dict = perception_msg_to_dict(request.perception)

        raise NotImplementedError

        if self.perception_sub['robot_vision']['updated']:
            # img_encoding = ros_img_to_base64(self.perception_sub['robot_vision']['data']) TODO change to new string format
            object, action = self.vlm_client.infer(img_encoding)

            pnode_name = object + "object_pnode"
            if perception_dict['grasped_object'] == "None":
                is_grasped = False
            else : 
                is_grasped = True
            pnode_params = {'target_object': object, 'is_grasped': is_grasped} 
            self.create_node_client(pnode_name, "llm_planner.pnode.SemanticPnode", pnode_params)

            goal_name = action + "_goal"
            goal_params = {} # NOTE does it need a drive as neighbor ?
            self.create_node_client(goal_name, "cognitive_nodes.goal.GoalMotiven", goal_params) # TODO double check is the right class

            cnode_name = object + "__" + action + "__cnode"
            neighbor_dict = {"PICK_AND_PLACE": "WorldModel", pnode_name: "PNode", goal_name: "Goal"}
            cnode_params = {
                'neighbors': [{'name': node, 'node_type': node_type} for node, node_type in neighbor_dict.items()]
            }
            self.create_node_client(cnode_name, "cognitive_nodes.cnode.CNode", cnode_params)

            success = self.add_neighbor("llm_planner_policy", cnode_name)
            if not success:
                self.get_logger().error(f"ERROR Planner Policy has not been linked to created CNode {cnode_name}")

            self.get_logger().info(f"Policy {self.name} executed successfully.")

        return response
    

    def configure_perception(self):
        """
        Subscription to the perception topic 'robot_vision'.
        Information used for the VLM queries.
        TODO change to string format of encoded image
        """
        subscriber = self.create_subscription(
            PerceptionStamped,
            "perception/robot_vision/value",
            self.perception_callback,
            1,
            callback_group=self.cbgroup_service
        )
        data = ""
        updated = False
        new_input = dict(subscriber=subscriber, data=data, updated=updated)
        self.perception_sub["robot_vision"] = new_input
        self.get_logger().info(f"{self.name} -- Subscribed to 'robot_vision' perception topic")

    def perception_callback(self, msg: PerceptionStamped):
        """
        Callback method that reads perception topic 'robot_vision' and stores it in perception_sub.
        TODO change to now string format 
        """
        perception_dict = perception_msg_to_dict(msg.perception)
        if len(perception_dict)>1:
            self.get_logger().error(f"{self.name} -- Received perception with multiple sensors: {perception_dict.keys()}. Perception nodes should (currently) include only one sensor!")
        if len(perception_dict)==1:
            self.perception_sub['robot_vision']['data'] = perception_dict['robot_vision']
            self.perception_sub['robot_vision']['updated'] = True
        else :
            self.get_logger().warning("Empty 'robot_vision' perception received in Policy User Alignment. No update in the perceptions.")

    def create_node_client(self, name, class_name, parameters={}):
        """
        This method calls the add node service of the commander.

        :param name: Name of the node to be created.
        :type name: str
        :param class_name: Name of the class to be used for the creation of the node.
        :type class_name: str
        :param parameters: Optional parameters that can be passed to the node, defaults to {}.
        :type parameters: dict
        :return: Success status received from the commander.
        :rtype: bool
        """

        self.get_logger().info("Requesting node creation...")
        params_str = yaml.dump(parameters, sort_keys=False)
        service_name = "commander/create"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(CreateNode, service_name)
        response = self.node_clients[service_name].send_request(
            name=name, class_name=class_name, parameters=params_str
        )

        self.get_logger().info(f"Creation of node {name} was successful: {response.created}.")

        return response.created
    
    def add_neighbor(self, node_name, neighbor_name):
        """
        This method adds a neighbor to a node in the LTM.

        :param node_name: Name of the node to which the neighbor will be added.
        :type node_name: str
        :param neighbor_name: Name of the neighbor to be added.
        :type neighbor_name: str
        :return: True if the neighbor was added successfully, False otherwise.
        :rtype: bool
        """
        service_name=f"{self.ltm_id}/update_neighbor"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(UpdateNeighbor, service_name)
        response=self.node_clients[service_name].send_request(node_name=node_name, neighbor_name=neighbor_name, operation=True)
        return response.success