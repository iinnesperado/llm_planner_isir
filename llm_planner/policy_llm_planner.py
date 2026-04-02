import yaml
import numpy as np
from ollama import chat, Client
import json
import re
import ast
import pathlib
import os

from collections import deque
from copy import deepcopy

from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy, PolicyBlocking
from core.service_client import ServiceClient, ServiceClientAsync
from core.utils import actuation_dict_to_msg, perception_msg_to_dict, actuation_msg_to_dict, EncodableDecodableEnum

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM, CreateNode
from cognitive_node_interfaces.srv import Execute, Predict
from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_processes_interfaces.msg import ControlMsg
from simulators.pump_panel_sim_discrete import PumpObjects

from llm_planner.llm_client import LLMClient # TODO check if the import is right for the ros thing

# NOTE check if drive class should be defined 

class PolicyLLMPlanner(Policy):
    def __init__(self, name="policy_llm_planner", llm_model_name="llama3.2", ltm_id = None, **params):
        super().__init__(name, **params)
        self.ltm_id = ltm_id
        self.policies = self.configure_policies()

        self.llm_client = LLMClient(model_name=llm_model_name)
        self.prompt_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "prompts")
        
    def request_ltm(self):
        """
        Requests data from the LTM.
        """        
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
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

        goal_name = "placeholder" # TODO figure out how to get the goal node name that triggered the policy

        plan = self.resquest_llm_plan(goal_name)
        self.get_logger().info(f"LLM generated plan: {plan}")

        for idx, policy in enumerate(plan): 
            self.get_logger().info(f"Executing plan step {idx}: {policy}...")

            name = re.sub(r"_goal", "", goal_name)
            pnode_name = f"{name}_step_{idx}_pnode"
            # TODO create pnode that corresponds to the perception

            if policy not in self.policies:
                self.get_logger().error("LLM DID NOT RETURN A VALID POLICY. CHOOSING RANDOMLY...")
                return
            
            if policy not in self.node_clients :
                self.node_clients[policy] = ServiceClientAsync(self, Execute, f"policy/{policy}/execute", callback_group=self.cbgroup_client)
            self.get_logger().info(f"Executing plan step {idx}: {policy}...")
            await self.node_client[policy].send_request_async()

            cnode_name = f"{name}_step_{idx}_cnode"
            neighbors = [
                {"name": "", "node_type": "WorldModel"},
                {"name": goal_name, "node_type": "Goal"},
                {"name": pnode_name, "node_type": "PNode"},
            ]
            cnode_params = {"neighbors": neighbors}
            self.create_node_client(cnode_name, "cognitive_nodes.cnode.CNode", cnode_params)

            # TODO add the cnode as neighbor to the executed policy 


        response.policy = self.name # NOTE to decide if to leave like this (mainly bc i don't know if its used for something)
        self.get_logger().info(f"Policy {self.name} executed successfully.")

        return response

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
    

    ################
    # EO FRAMEWORK #
    ################

    def resquest_llm_plan(self, task):
        """
        Generates a plan to accomplish the given task and taking into account the perception of the robot.
        This plan follows the Expected Outcomes Framework.

        :return: plan with each policy to follow 
        :rtyle: list (policies)
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