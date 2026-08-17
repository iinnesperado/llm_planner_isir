import yaml
import numpy as np
# from ollama import chat, Client
import json
import re
import ast
import pathlib
import os

from collections import deque
from copy import deepcopy
from rclpy.time import Time

from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy
from cognitive_nodes.utils import LTMSubscription
from core.service_client import ServiceClient, ServiceClientAsync
from core.utils import class_from_classname

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM, CreateNode, UpdateNeighbor, DeleteNode
from cognitive_node_interfaces.srv import Execute, Predict, GetInformation
from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_node_interfaces.msg import Perception, PerceptionStamped
from cognitive_processes_interfaces.msg import ControlMsg

from llm_planner.space import SemanticSpace
from llm_planner.perception import SemanticPerception
from llm_planner.llm_client import LLMClient # TODO check if the import is right for the ros thing
from llm_planner.utils import perception_msg_to_dict
from llm_planner_interfaces.srv import GetTargetObject, GetAlignmentInformation

class DriveLLMPlanner(Drive, LTMSubscription):
    """
    DriveLLMPlanner Class, responsible to activate the PolicyLLMPlanner when high level goals have not been planned. 
    """    
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", ltm_id=None, **params):
        """
        Constructor of the DriveLLMPlanner class.

        :param name: The name of the Drive instance.
        :type name: str
        :param class_name: The name of the Drive class, defaults to "cognitive_nodes.drive.Drive".
        :type class_name: str
        """        
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id

        self.configure_ltm_subscription(self.LTM_id, self.cbgroup_client)

        self.cnode_dict = {}
        self.goal_dict = {}

    def read_ltm(self, ltm_dump):
        """
        Reads the LTM dump and saves the information on the goals and cnodes.
        """
        self.cnode_dict = ltm_dump['CNode']
        self.goal_dict = ltm_dump['Goal']

    def evaluate(self, perception=None):
        """
        Evaluation that returns 0.6 when there is a goal that needs to be planned.

        :param perception: Unused perception.
        :type perception: dict or Any.
        :return: Evaluation of the Drive.
        :rtype: cognitive_node_interfaces.msg.Evaluation
        """        
        value = 0.0     # evaluation value for the drive 
        all_neighbors_goal = []

        for cnode in self.cnode_dict.keys():
            all_neighbors_goal.extend(self.get_goal_neighbor_names(cnode))

        goals_diff = set(self.goal_dict.keys()) - set(all_neighbors_goal)
        if len(goals_diff) > 0:
            self.get_logger().debug(f"The following Goals need a plan: {goals_diff}")
            value = 0.6

        self.evaluation.evaluation = value
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation

    def get_goal_neighbor_names(self, cnode_name):
        """
        This method returns the names of the neighbors, Goal type only, of a CNode.

        :param cnode_name: Name of the CNode.
        :type cnode_name: str.
        :return: List of neighbor names.
        :rtype: list.
        """        
        neighbors = self.cnode_dict.get(cnode_name, {}).get("neighbors", [])
        names = []
        for neighbor in neighbors:
            if neighbor['node_type'] == 'Goal':
                names.append(neighbor['name'])
        return names

class PolicyLLMPlanner(Policy):
    def __init__(self, name="policy", llm_model_name="llama3.2", ltm_id = None, prompts=[], executed_primitive_service="", **params):
        """
        :param executed_primitive_service: name of the service to execute the primitives of the robot, same name as executed_policy_service in yaml config file
        :type executed_primitive_service: str
        """
        super().__init__(name, **params)
        if ltm_id is None:
                    raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id
        self.policies = self.configure_policies()

        self.executed_primitive_service = executed_primitive_service
        self.executed_primitive_type = class_from_classname("cognitive_node_interfaces.srv.Policy")

        self.llm_client = LLMClient(model_name=llm_model_name)
        self.high_level_prompt = prompts["high_level_prompt"]
        self.low_level_prompt = prompts["low_level_prompt"]
        self.outcome_prompt = prompts["outcome_prompt"]

        self.grasped_object_sub = {}
        self.cofigure_grasped_object_sub()
        
    def request_ltm(self):
        """
        Requests data from the LTM.
        """        
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        client = ServiceClientAsync(self, GetNodeFromLTM, service_name, self.cbgroup_server)
        ltm_response = client.send_request_async(name=request)
        ltm = yaml.safe_load(ltm_response.data)
        return ltm
    
    async def configure_policies(self):
        """
        Creates a list of eligible policies to be executed and shuffles it.
        """
        ltm_cache = await self.request_ltm()        
        policies = list(ltm_cache["Policy"].keys())
        self.get_logger().info(f"Configuring Policies: {policies}") #TODO: Possibility of using new policies added in LTM
        return policies
    
    def cofigure_grasped_object_sub(self):
        """
        Subscription to perception topic 'grasped_object'.
        Information used when creating the Pnodes.
        """
        subscriber = self.create_subscription(
            PerceptionStamped, 
            "perception/grasped_object/value",
            self.grasped_object_callback, 
            1, 
            callback_group=self.cbgroup_client
        )
        data = ""
        updated = False
        self.grasped_object_sub = dict(subscriber=subscriber, data=data, updated=updated)
        self.get_logger().info(f"{self.name} -- Subscribed to 'grasped_object' perception topic")
    
    def grasped_object_callback(self, msg: PerceptionStamped):
        """
        Callback method that reads a perception and stores it in grasped_object_sub list. 
        This function should be called everytime the perception topic for 'grasped_object' publishes information. 
        """
        perception_dict = perception_msg_to_dict(msg.perception)
        if len(perception_dict)>1:
            self.get_logger().error(f"{self.name} -- Received perception with multiple sensors: {perception_dict.keys()}. Perception nodes should (currently) include only one sensor!")
        if len(perception_dict)==1:
            self.grasped_object_sub['data'] = perception_dict['grasped_object'][0]['data']
            self.grasped_object_sub['updated'] = True
        else :
            self.get_logger().warning(f"Empty perception received in Policy LLM Planner. No update in the perceptions.")
    
    async def execute_callback(self, request, response):
        """

        :param request: The request to execute the policy.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: The response indicating the executed policy.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        """
        self.get_logger().info(f"== START LLM PLANNER POLICY ==")

        perception_dict = perception_msg_to_dict(request.perception)
        self.get_logger().info(f"Received perception: {perception_dict}")

        alignment_response = await self.get_alignment_information()
        goal_name = alignment_response.goal_name
        action = re.sub(r"__goal", "", goal_name)
        self.get_logger().info(f"Alignment response {alignment_response}")


        # LLM PLAN REQUEST
        plan = self.resquest_llm_plan(goal_name)
        try:
            plan_list = ast.literal_eval(plan)
        except (ValueError, SyntaxError) as e:
            self.get_logger().error(f"Invalid plan returned by LLM: {plan}. Error: {e}")
        self.get_logger().debug(f"LLM generated plan: {plan}")
        # quick testing
        # plan_list = [{"name": "grasp_object", "params": {"target_object": "mug"}}, 
        #              {"name": "release_object", "params": {"target_location": "slide"}}]


        # EXECUTING THE PLAN
        for idx, policy in enumerate(plan_list): 
            self.get_logger().info(f"--- Working on plan step {idx+1}: {policy}")
            
            # PNODE CREATION
            target_object = alignment_response.target_object
            if self.grasped_object_sub['updated']:
                self.grasped_object_sub['updated'] = False
                
                pnode_params = {}
                self.get_logger().debug(f"Perception grasped_object before PNode creation: {self.grasped_object_sub['data']}")
                if (self.grasped_object_sub["data"]=="None" or self.grasped_object_sub["data"]==""):
                    is_grasped = False
                    pnode_name = f"{target_object}__object_pnode"
                else :
                    is_grasped = True
                    pnode_name = f"{target_object}__grasped_object_pnode"
                pnode_params = {"target_object": target_object, "is_grasped": is_grasped}
                if idx != 0:
                    # for idx == 0 the PNode was created in user alignment policy
                    self.get_logger().info("Creating PNode...")
                    await self.create_node_client(pnode_name, "llm_planner.pnode.SemanticPNode", pnode_params)
            else :
                self.get_logger().warning("LLMPlanner - perception was not updated, so PNode was not created!")


            # POLICY EXECUTION
            policy_name = policy['name']
            for param in policy['params'].values():
                policy_name += "__" + param
            # creating the policy node to save on the LTM
            policy_params = { 'service_msg': 'cognitive_node_interfaces.srv.Policy', 'service_name': '/simulator/executed_policy' }
            await self.create_node_client(policy_name, "cognitive_nodes.policy.PolicyBlocking", policy_params)

            self.get_logger().info(f"Executing plan step {idx+1}: {policy}...")
            if self.executed_primitive_service not in self.node_clients:
                self.node_clients[self.executed_primitive_service] = ServiceClientAsync(self, self.executed_primitive_type, self.executed_primitive_service, self.cbgroup_client)
            await self.node_clients[self.executed_primitive_service].send_request_async(policy=policy_name)


            # CNODE CREATION
            cnode_name = f"{action}__step_{idx+1}_cnode"
            neighbor_dict = {"PICK_AND_PLACE": "WorldModel", pnode_name: "PNode", goal_name: "Goal"}
            cnode_params = {
                'neighbors': [{'name': node, 'node_type': node_type} for node, node_type in neighbor_dict.items()]
            }
            await self.create_node_client(cnode_name, "cognitive_nodes.cnode.CNode", cnode_params)

            sucess = await self.add_neighbor_client(policy_name, cnode_name)
            if sucess.success:
                self.get_logger().info(f"Successfully added the Cnode {cnode_name} as neighbor to policy {policy_name}")
            else :
                self.get_logger().error(f"ERROR Failed to link policy {policy_name} to CNode {cnode_name}")
            

        response.policy = self.name 

        self.get_logger().info(f"Policy {self.name} executed successfully.")

        return response

    def get_alignment_information(self):
        """
        Get the objetc, goal, pnode name information from the User Alignment policy.
        """
        service_name = "user_alignment/get_alignment_information"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClientAsync(self, GetAlignmentInformation, service_name, self.cbgroup_client)
        response = self.node_clients[service_name].send_request_async()
        return  response



    ################
    # EO FRAMEWORK #
    ################

    def resquest_llm_plan(self, task):
        """
        Generates a plan to accomplish the given task and taking into account the perception of the robot.
        This plan follows the Expected Outcomes Framework.

        :return: plan with each policy names to follow, they should be existing policies available in the LTM.
        :rtype: list[str]        ['prim1', 'prim2', ...]
        """

        self.get_logger().info(f"Making plan for {task} task/goal...")

        high_level_plan = self.high_level_plan(task)
        self.get_logger().info(f"High level plan : \n{high_level_plan}")

        expected_outcomes = self.predict_outcomes(task, high_level_plan)
        self.get_logger().info(f"Expected outcomes : \n{expected_outcomes}")

        low_level_plan = self.low_level_plan(task, high_level_plan, expected_outcomes)
        self.get_logger().info(f"Low level plan : \n{low_level_plan}")

        return low_level_plan
    
    def high_level_plan(self, task):
        """
        Generate a high-level plan of the given task.
        """        
        prompt = re.sub(r"{task}", task, self.high_level_prompt)

        response = self.llm_client.generate(prompt)

        return response
        
    def predict_outcomes(self, task, high_level_plan):
        """
        Generates the expected outcomes of the high level plan of the given task.
        """
        prompt = re.sub(r"{task}", task, self.outcome_prompt)
        prompt = re.sub(r"{plan}", high_level_plan, prompt)

        response = self.llm_client.generate(prompt)

        return response
    
    def low_level_plan(self, task, high_level_plan, expected_outcomes):
        """
        Generates the low level plan for the robot of a high level plan, its expected outcomes of the given task.
        """
        prompt = re.sub(r"{task}", task, self.low_level_prompt)
        prompt = re.sub(r"{plan}", high_level_plan, prompt)
        prompt = re.sub(r"{EO}", expected_outcomes, prompt)

        response = self.llm_client.generate(prompt)

        return response