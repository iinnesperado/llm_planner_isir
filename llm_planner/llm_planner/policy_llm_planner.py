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
from core.service_client import ServiceClient, ServiceClientAsync
from core.utils import actuation_dict_to_msg, perception_msg_to_dict, actuation_msg_to_dict, EncodableDecodableEnum

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM, CreateNode, UpdateNeighbor, DeleteNode
from cognitive_node_interfaces.srv import Execute, Predict
from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_node_interfaces.msg import PerceptionStamped
from cognitive_processes_interfaces.msg import ControlMsg

from llm_planner.space import SemanticSpace
from llm_planner.perception import SemanticPerception
from llm_planner.llm_client import LLMClient # TODO check if the import is right for the ros thing
from llm_planner_interfaces.srv import GetTargetObject

# NOTE check if drive class should be defined 

class PolicyLLMPlanner(Policy):
    def __init__(self, name="policy", llm_model_name="llama3.2", ltm_id = None, **params):
        super().__init__(name, **params)
        self.ltm_id = ltm_id
        self.policies = self.configure_policies()

        self.llm_client = LLMClient(model_name=llm_model_name)
        self.prompt_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "prompts")

        self.cofigure_perception()
        
    def request_ltm(self):
        """
        Requests data from the LTM.
        """        
        # Call get_node service from LTM
        service_name = "/" + str(self.ltm_id) + "/get_node"
        request = ""
        client = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = client.send_request(name=request)
        ltm = yaml.safe_load(ltm_response.data)
        return ltm
    
    def configure_policies(self):
        """
        Creates a list of eligible policies to be executed and shuffles it.
        """
        ltm_cache = self.request_ltm()        
        policies = list(ltm_cache["Policy"].keys())
        self.get_logger().info(f"Configuring Policies: {policies}") #TODO: Possibility of using new policies added in LTM
        return policies
    
    def cofigure_perception(self):
        """
        Subscription to perception topic 'grasped_object'.
        Information used when creating the Pnodes.
        """
        subscriber = self.create_subscription(
            PerceptionStamped, 
            "/perception/grasped_object/value", # TODO check if the name is correct for the service 
            self.perception_callback, 
            1, 
            callback_group=self.cbgroup_perception
        )
        data = {}
        updated = False
        timestamp = Time()
        new_input = dict(subscriber=subscriber, data=data, updated=updated, timestamp=timestamp)
        self.perception_sub["grasped_object"] = new_input
        self.get_logger().info(f"{self.name} -- Subscribed to 'grasped_object' perception topic")
    
    async def execute_callback(self, request, response):
        """

        :param request: The request to execute the policy.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: The response indicating the executed policy.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :raise NotImplementedError: This method should be implemented in subclasses.
        """

        perception_dict = perception_msg_to_dict(request.perception)
        self.get_logger().info(f"Reveived perception: {perception_dict}")

        goal = self.get_high_level_goal_name()

        plan = self.resquest_llm_plan(goal)
        self.get_logger().info(f"LLM generated plan: {plan}")

        name = re.sub(r"_goal", "", goal)

        for idx, policy in plan.enumerate(): 
            self.get_logger().info(f"Executing plan step {idx}: {policy}...")

            if policy['name'] not in self.policies:
                self.get_logger().error("LLM DID NOT RETURN A VALID POLICY. CHOOSING RANDOMLY...")
                return
            
            if self.perception['grasped_object']['updated']:
                self.perception['grasped_object']['updated'] = False
                
                pnode_name = f"{name}_step_{idx}_pnode"
                pnode_params = {}
                target_object = self.get_pnode_target_object()
                if perception_dict["grasped_object"]!="":
                    pnode_params = {"target_object": target_object, "is_grasped": True}
                else:
                    pnode_params = {"target_object": target_object, "is_grasped": False}
                self.create_node_client(pnode_name, "llm_planner.pnode.SemanticPnode", pnode_params)

                if policy['name'] not in self.node_clients :
                    self.node_clients[policy['name']] = ServiceClientAsync(self, Execute, f"policy/{policy['name']}/execute", callback_group=self.cbgroup_client)
                self.get_logger().info(f"Executing plan step {idx}: {policy['name']}...")
                await self.node_clients[policy['name']].send_request_async(**policy['params'])

            cnode_name = f"{name}_step_{idx}_cnode"
            neighbors = [
                {"name": "PICK_AND_PLACE", "node_type": "WorldModel"},
                {"name": goal, "node_type": "Goal"},
                {"name": pnode_name, "node_type": "PNode"},
            ]
            cnode_params = {"neighbors": neighbors}
            self.create_node_client(cnode_name, "cognitive_nodes.cnode.CNode", cnode_params)

            sucess = self.add_neighbor(policy['name'], cnode_name)
            if sucess:
                self.get_logger().info(f"Successfully created the Cnode {cnode_name} and linked to policy {policy['name']}")
            else :
                self.get_logger().error("ERROR Policy of the steps hasn't been linked to corresponding Cnode")

            # TODO update perception for nect loop iteration
            

        response.policy = self.name # NOTE to decide if to leave like this (mainly bc i don't know if its used for something)

        self.delete_cnode_llm_planner()
        self.delete_neighbor(self.name, self.get_cnode_name())

        self.get_logger().info(f"Policy {self.name} executed successfully.")

        return response

    def delete_cnode_llm_planner(self):
        """Responsible of deleting the cnode that is responsible of the call for the llm planner the whole plan has been executed."""
        cnode = self.get_cnode_name()
        deleted = self.delete_node_client(cnode)
        return deleted
    
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

        self.get_logger().info("Requesting node creation")
        params_str = yaml.dump(parameters, sort_keys=False)
        service_name = "commander/create"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(CreateNode, service_name)
        response = self.node_clients[service_name].send_request(
            name=name, class_name=class_name, parameters=params_str
        )
        return response.created
    
    def delete_node_client(self, name):
        self.get_logged().info("Requesting node deletion")
        service_name = "commander/delete"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(DeleteNode, service_name)
        response = self.node_clients[service_name].send_request(name=name)
        return response.deleted
    
    def perception_callback(self, msg : PerceptionStamped):
        """
        Callback method that reads a perception and stores it in perception_sub list. 
        This function should be called everytime the perception topic for 'grasped_object' publishes information. 
        """
        perception_dict = self.perception_msg_to_dict(msg.perception)
        if len(perception_dict)>1:
            self.get_logger().error(f"{self.name} -- Received perception with multiple sensors: ({perception_dict.keys()}). Perception nodes should (currently) include only one sensor!")
        if len(perception_dict)==1:
            node_name = list(perception_dict.keys())[0]
            if node_name in self.perception_sub:
                self.perception_sub[node_name]['data'] = perception_dict[node_name]
                self.perception_sub[node_name]['updated'] = True
                self.perception_sub[node_name]['timestamp'] = Time.from_msg(msg.timestamp)
        else :
            self.get_logger().warn("Empty perception received in Policy LLM Planner. No update in the perceptions.")

    def get_cnode_name(self):
        """
        Retrives the name of the Cnode calling the policy LLM Planner.
        We suppose that the policy LLM Planner has the cnode that calls for him as a neighbor.
        """
        cnode_name = None
        ltm_cache = self.request_ltm()
        data = next((nodes_dict[self.name] for nodes_dict in ltm_cache.values() if self.name in nodes_dict))
        neighbors = data['neighbors']

        for node in neighbors:
            if node['node_type'] == 'Cnode':
                cnode_name = node['name']
        
        return cnode_name

    def get_high_level_goal_name(self):
        """Retrieves the high level goal of the cnode calling the policy LLM Planner."""
        goal = None
        ltm_cache = self.request_ltm()
        cnode_name = self.get_cnode_name()

        if cnode_name is None:
            self.get_logger().error("ERROR LLM Planner doesn't have a Cnode as neighbor")
        else :
            data = next((nodes_dict[cnode_name] for nodes_dict in ltm_cache.values() if cnode_name in nodes_dict))
            neighbors = data['neighbors']

            for node in neighbors:
                if node['node_type'] == 'Goal':
                    goal = node['name']

            self.get_logger().info(f"GOAL of the LLMPlanner : {goal}")
        
        return goal
    
    def get_pnode_target_object(self, pnode_name):
        """
        Retrieves the target_object from the pnode neighboor of the cnode calling this policy.
        """
        pnode_name = None
        ltm_cache = self.request_ltm()
        cnode_name = self.get_cnode_name()

        if cnode_name is None:
            self.get_logger().error("ERROR LLM Planner doesn't have a Cnode as neighbor")
        else :
            data = next((nodes_dict[cnode_name] for nodes_dict in ltm_cache.values() if cnode_name in nodes_dict))
            neighbors = data['neighbors']

            for node in neighbors:
                if node['node_type'] == 'PNode':
                    pnode_name = node['name']

        self.get_logger().info("Requesting target object to PNode...")
        service_name = "pnode/" + str(pnode_name) + "get_target_object"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetTargetObject, service_name)
        response = self.node_clients[service_name].send_request()
        return response.target_object


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
        service_name=f"{self.LTM_id}/update_neighbor"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(UpdateNeighbor, service_name)
        response=self.node_clients[service_name].send_request(node_name=node_name, neighbor_name=neighbor_name, operation=True)
        return response.success
    
    def delete_neighbor(self, node_name, neighbor_name):
        """
        This method deletes a neighbor to a node in the LTM.

        :param node_name: Name of the node to which the neighbor will be deleted.
        :type node_name: str
        :param neighbor_name: Name of the neighbor to be deleted.
        :type neighbor_name: str
        :return: True if the neighbor was deleted successfully, False otherwise.
        :rtype: bool
        """
        service_name=f"{self.LTM_id}/update_neighbor"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(UpdateNeighbor, service_name)
        response=self.node_clients[service_name].send_request(node_name=node_name, neighbor_name=neighbor_name, operation=False)
        return response.success


    ################
    # EO FRAMEWORK #
    ################

    def resquest_llm_plan(self, task, perception):
        """
        Generates a plan to accomplish the given task and taking into account the perception of the robot.
        This plan follows the Expected Outcomes Framework.

        :return: plan with each policy names to follow, they should be existing policies available in the LTM.
        :rtype: dict    {1: {name: grasp_object, params: {obj_id: 'mug', subparts: 'body'}}, 2: ...}
        """

        self.get_logger().info(f"Making plan for {task} task/goal...")

        high_level_plan = self.high_level_plan(task)
        self.get_logger().info(f"High level plan : \n{high_level_plan}")

        expected_outcomes = self.predict_outcomes(task, high_level_plan)
        self.get_logger().info(f"Expected outcomes : \n{expected_outcomes}")

        low_level_plan = self.low_level_plan(task, high_level_plan, expected_outcomes)
        self.get_logger().info(f"Low level plan : \n{low_level_plan}")

        return low_level_plan
    
    def high_level_plan(self, task, perception):
        """
        Generate a high-level plan of the given task.
        """
        file_path = os.path.join(self.prompt_dir, "high_level_prompt.txt")
        with open(file_path) as f :
            prompt = f.read()
        
        prompt = re.sub(r"{task}", task, prompt)

        response = self.llm_client.generate(prompt)

        return response
        
    def predict_outcomes(self, task, high_level_plan):
        """
        Generates the expected outcomes of the high level plan of the given task.
        """
        file_path = os.path.join(self.prompt_dir, "outcome_prompt.txt")
        with open(file_path) as f :
            prompt = f.read()

        prompt = re.sub(r"{task}", task, prompt)
        prompt = re.sub(r"{plan}", high_level_plan, prompt)

        response = self.llm_client.generate(prompt)

        return response
    
    def low_level_plan(self, task, high_level_plan, expected_outcomes):
        """
        Generates the low level plan for the robot of a high level plan, its expected outcomes of the given task.
        """
        file_path = os.path.join(self.prompt_dir, "low_level_prompt.txt")
        with open(file_path) as f :
            prompt = f.read()

        prompt = re.sub(r"{task}", task, prompt)
        prompt = re.sub(r"{plan}", high_level_plan, prompt)
        prompt = re.sub(r"{EO}", expected_outcomes, prompt)
        # TODO add skills to come from the LTM for the prompt 

        response = self.llm_client.generate(prompt)

        return response