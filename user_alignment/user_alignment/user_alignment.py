import yaml
import re

from sensor_msgs.msg import Image
from core.service_client import ServiceClient, ServiceClientAsync
from core_interfaces.srv import CreateNode, UpdateNeighbor
from cognitive_nodes.policy import Policy
from cognitive_nodes.drive import Drive
from cognitive_node_interfaces.msg import PerceptionStamped
from cognitive_node_interfaces.srv import SetActivation

from llm_planner.utils import perception_msg_to_dict

from user_alignment.vlm_rag import VLMRAG
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
        # self.evaluation.evaluation = 1.0
        self.evaluation.evaluation = 0.5
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation

class PolicyUserAlignment(Policy):
    def __init__(self, name="policy", ltm_id=None, **params):
        super().__init__(name, **params)
        self.ltm_id = ltm_id

        self.vlm_client = VLMRAG()

        self.perception_sub = {}

        self.configure_perception()

        if ltm_id is None:
            raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id

    async def execute_callback(self, request, response):
        """
        Execute the infer() function of VLMRAG I guess.
        And then do all the working around the nodes to have the goal, pnode, cnode and connect to planner policy.
        """
        self.get_logger().info(f"== START USER ALIGNMENT POLICY ==")

        perception_dict = perception_msg_to_dict(request.perception)

        # raise NotImplementedError
        if self.perception_sub['robot_vision']['updated']:
            self.perception_sub['robot_vision']['updated'] = False

            self.get_logger().info("Querying Ollama vision ...")
            raw_vision = self.perception_sub['robot_vision']['data']
            encoded_vision = ros_img_to_base64(raw_vision)
            object, action = self.vlm_client.infer(encoded_vision)
            action = re.sub(" ", "_", action)
            self.get_logger().info(f"Result -- object: {object}, action: {action}")

            pnode_name = object + "_object_pnode"
            grasped_object = perception_dict['grasped_object'][0]
            if grasped_object['data'] == "None":
                is_grasped = False
            else : 
                is_grasped = True
            pnode_params = {'target_object': object, 'is_grasped': is_grasped} 
            await self.create_node_client(pnode_name, "llm_planner.pnode.SemanticPNode", pnode_params)

            goal_name = action + "_goal"
            goal_params = {"neighbors": [{"name": "object_in_place_drive", "node_type": "Drive"}]} # NOTE does it need a drive as neighbor ?
            # goal_params = {}
            await self.create_node_client(goal_name, "dummy_nodes.dummy_goal.GoalDummy", goal_params) # TODO double check is the right class
            # success = await self.set_activation("goal", goal_name, 1.0)
            # if not success.set:
            #     self.get_logger().error(f"ERROR {goal_name} did not set its activation value to 1.0.")

            cnode_name = object + "__" + action + "__cnode"
            neighbor_dict = {"PICK_AND_PLACE": "WorldModel", pnode_name: "PNode", goal_name: "Goal"}
            cnode_params = {
                'neighbors': [{'name': node, 'node_type': node_type} for node, node_type in neighbor_dict.items()]
            }
            await self.create_node_client(cnode_name, "cognitive_nodes.cnode.CNode", cnode_params)

            success = await self.add_neighbor_client("llm_planner_policy", cnode_name)
            if not success.success:
                self.get_logger().error(f"ERROR Planner Policy has not been linked to created CNode {cnode_name}")


        response.policy = self.name
        self.get_logger().info(f"Policy {self.name} executed successfully.")

        return response

    def configure_perception(self):
        """
        Subscription to the perception topic 'robot_vision'.
        Information used for the VLM queries.
        """
        subscriber = self.create_subscription(
            Image,
            "/simulator/sensor/robot_vision",
            self.perception_callback,
            1,
            callback_group=self.cbgroup_server
        )
        data = Image()
        updated = False
        new_input = dict(subscriber=subscriber, data=data, updated=updated)
        self.perception_sub["robot_vision"] = new_input
        self.get_logger().info(f"{self.name} -- Subscribed to 'robot_vision' perception topic")

    def perception_callback(self, msg: Image):
        """
        Callback method that reads perception topic 'robot_vision' and stores it in perception_sub.
        
        :param msg: Image coming from the camera of the robot
        :type msg: sensor_msgs.msg.Image
        """
        if len(msg.data)!=0:
            self.perception_sub['robot_vision']['data'] = msg
            self.perception_sub['robot_vision']['updated'] = True
        else :
            self.get_logger().warning("Empty 'robot_vision' perception received in Policy User Alignment. No update in the perceptions.")
    
    def set_activation(self, node_type, node_name, activation):
        """
        Sets the activation value of a given policy
        """
        self.get_logger().info(f"Setting activation value of {node_name}...")
        service_name = node_type + "/" + node_name + "/set_activation"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClientAsync(self, SetActivation, service_name, self.cbgroup_client)
        response=self.node_clients[service_name].send_request_async(activation=activation)

        # self.get_logger().info(f"Activation of policy {node_name} was successfully set to {activation}: {response.set}.")
        return response