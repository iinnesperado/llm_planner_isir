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

        self.llm_client = LLMClient(model_name=llm_model_name)
        self.prompt_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "prompts")
        

    def execute_callback(self, request, response):
        """
        Placeholder for the execution of the policy.

        :param request: The request to execute the policy.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: The response indicating the executed policy.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :raise NotImplementedError: This method should be implemented in subclasses.


        Callback that processes the request to execute a policy. 
        1. Sends perception to LLM
        2. LLM generates the plan 
        3. Creates the cnodes for each step
        """

        perception_dict = perception_msg_to_dict(request.perception)
        self.get_logger().info(f"Reveived perception: {perception_dict}")

        plan = self.llm_client.plan(task)
        self.get_logger().info(f"LLM generated plan: {plan}")

        self.create_cnodes_from_plan(plan)

        response.policy = self.name
        self.get_logger().info(f"Policy {self.name} executed successfully.")

        return response



    # TODO figure out how to create a cnode with all the parameters and everything
    # TODO how to assign activation value to this polivy

    def create_cnodes_from_plan(self, plan, main_task, wm_name=None):
        """
        Creates a cnode for each step of the given plan.
        
        :param plan: plan created by the LLM
        :type plan: dict
        :param main_task: task to achieve 
        :type main_task: cognitive_nodes.goal.Goal
        """

        """world_model = {"name": wm_name, "node_type": "WorldModel"}
        for idx, step in enumerate(plan.keys()):
            node_name = f"{main_task}_step_{idx}"
            goal = {"name": "{step}_goal", "node_type": "Goal"}
            self.create_node_client(
                goal["name"],
                "cognitive_nodes.goal.GoalMotiven"
            )
            
            # TODO problem on how to actually make the pnodes 
            pnode = {"name": f"{node_name}_pnode", "node_type": "PNode"}
            pnode_params = {
                "space_class": "cognitive_nodes.space.ActivatedDummySpace"
            }
            self.create_node_client(
                pnode["name"],
                "dummy_nodes.dummy_pnodes.ActivatedDummyPNode", #NOTE might need to define one for our manip
                pnode_params
            )

            cnode_params = {
                "neighbors": [goal, world_model, pnode]
            }
            self.create_node_client(
                f"{node_name}_cnode", 
                "cognitive_nodes.cnode.CNode", 
                cnode_params
            )"""
        created_goals = []


        # NOTE is there a way to access wm from main_task ?
        self.create_node_client(
            wm_name or "LLM_PLANNER",
            "cognitive_nodes.world_model.WorldModel" 
        )

        for idx, item in enumerate(plan.items()):
            step_name, expected_outcome = item

            goal_params = {
                "goal_description": step_name,
                "expected_outcome_description": expected_outcome
            }
            self.create_node_client(
                f"{main_task}_step_{idx}_goal",
                "cognitive_nodes.goal.GoalMotiven",
                goal_params
            )

            




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
            # TODO use async or not ?
            self.node_clients[service_name] = ServiceClient(CreateNode, service_name)
        response = self.node_clients[service_name].send_request(
            name=name, class_name=class_name, parameters=params_str
        )
        return response.created
    

    ################
    # EO FRAMEWORK #
    ################

    def plan(self, task, perception):
        """
        Generates a plan to accomplish the given task and taking into account the perception of the robot.
        This plan follows the Expected Outcomes Framework.
        """

        self.get_logger().info(f"Making plan for {task} task/goal...")

        high_level_plan = self._high_level_plan(task, perception)
        self.get_logger().info(f"High level plan : \n{high_level_plan}")

        expected_outcomes = self._predict_outcomes(task, high_level_plan)
        self.get_logger().info(f"Expected outcomes : \n{expected_outcomes}")

        low_level_plan = self._low_level_plan(task, high_level_plan, expected_outcomes)
        self.get_logger().info(f"Low level plan : \n{low_level_plan}")

        return low_level_plan
    
    def _high_level_plan(self, task, perception):
        """
        Generate a high-level plan of the given task.
        """
        file_path = os.path.join(self.prompt_dir, "high_level_prompt.txt")
        with open(file_path) as f :
            prompt = f.read()
        
        prompt = re.sub(r"TASK_PLACEHOLDER", task, prompt)
        prompt = re.sub(r"PERCEPTION_PLACEHOLDER", perception, prompt)

        response = self.llm_client.generate(prompt)

        return response
        
    def _predict_outcomes(self, task, high_level_plan):
        """
        Generates the expected outcomes of the high level plan of the given task.
        """
        file_path = os.path.join(self.prompt_dir, "outcome_prompt.txt")
        with open(file_path) as f :
            prompt = f.read()

        prompt = re.sub(r"TASK_PLACEHOLDER", task, prompt)
        prompt = re.sub(r"HIGH_LEVEL_PLAN_PLACEHOLDER", high_level_plan, prompt)

        response = self.llm_client.generate(prompt)

        return response
    
    def _low_level_plan(self, task, high_level_plan, expected_outcomes):
        """
        Generates the low level plan for the robot of a high level plan, its expected outcomes of the given task.
        """
        file_path = os.path.join(self.prompt_dir, "low_level_prompt.txt")
        with open(file_path) as f :
            prompt = f.read()

        prompt = re.sub(r"TASK_PLACEHOLDER", task, prompt)
        prompt = re.sub(r"HIGH_LEVEL_PLAN_PLACEHOLDER", high_level_plan, prompt)
        prompt = re.sub(r"EO_PLACEHOLDER", expected_outcomes, prompt)

        response = self.llm_client.generate(prompt)

        return response