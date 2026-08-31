import ollama
import chromadb
import cv2
from typing import Union, Any, Optional
import numpy as np
import os
import re
import ast

import base64
from cv_bridge import CvBridge



def get_list_obj(model="llama3.2-vision"):
    prompt_vlm = """
    You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms. Only describe the objects.
    Identify and list **all** visible objects **on the table**. Return the result as a valid Python list of strings.

    Return only the list, in this format:
    ["mug", "silver ring", "blue small pen", ...]
    """
    #get_image()
    response = ollama.generate(
        model='llama3.2-vision',
        #prompt= 'You are a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms. Only describe the objects',
        prompt = prompt_vlm,
        images= ['Images/live.png']
        , options={
            "temperature": 0.0,
            "num_predict": 1024
        }
    )
    im_desc = response.get("response", "")
    print(f"Image description: {im_desc}")
    match = re.search(r'\[\s*.*?\s*\]', im_desc, re.DOTALL)
    if match:
        obj_list_str = match.group(0)
        obj_list = ast.literal_eval(obj_list_str)
        print(f"Extracted object list: {obj_list}")
        return obj_list
    else:
        print("List not found")

def get_useful_doc(collection,task,threshold=0.5):
    """
    Find the most useful information in the documents
    """
    response = ollama.embeddings(
        prompt=task,
        model="mxbai-embed-large"
        )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=10
    )
    # Generate a threshold to filter relevant documents ( thresold can be adjusted)
    relevant_docs = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        if dist <= threshold:
            relevant_docs.append(doc)
    return relevant_docs


prompt_template = """#CONTEXT
You are a fixed robotic arm equipped with a gripper. 
You can place objects into three distinct boxes:

- toolbox: for tools
- tray: for personal objects

#SKILLS
To complete your task you need to use the following information:
- A Python-style list of relevant facts and instructions, when relevant, connect pieces of information that refer to the same or similar concepts, you can also use it to determine unidentified objects : []

You MUST cross-reference the two sources, but only act on objects that are explicitly present in the visual scene.  
- Do not infer or imagine additional objects.
- Ignore any RAG fact that does not relate to a visible object on the scen description.
- If an object is mentioned in RAG but is not visible, **do nothing about it**.

#SCENE
The visible objects are: IMAGE_PLACEHOLDER.

#RAG INFORMATION
The user gave the following corrections in prior trials: RAG_PLACEHOLDER

#OUTPUT
Your job is output a pick-and-place action to tidy up the desk. Complete the following template: 'Put the (object) in the (location)'

Do not describe how to move or grasp — keep it abstract and human-level.  
Do not include explanations or justifications — only output the task.

Keep it concise, logical, and clear.
"""

def get_draft(rag : str, image) -> str:
    """
    Generate a draft plan using the RAG model.
    
    Args:
        rag: The RAG model to use for generating the draft
        task: The semantic task to plan for
        
    Returns:
        A string containing the draft plan
    """

    print("Rag: ", rag)
    
    rep = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config"))
    os.makedirs(rep, exist_ok=True)
    draft_file = os.path.join(rep, "draft_plan.txt")
    
    prompt = re.sub(r"IMAGE_PLACEHOLDER", image, prompt_template)
    prompt = re.sub(r"RAG_PLACEHOLDER", str(rag), prompt)
    
    return prompt

def ros_img_to_base64(ros_img):
    """
    Converts a sensor_msgs/Image to a base64-encoded JPEG string
    to be sent to Ollama vision.
    """
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough") #to match with camera encoding from sim

    success, buffer = cv2.imencode('.jpeg', cv_image)
    if not success:
        raise ValueError("Failed to encode image")
    
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def png_to_ros_img(img_path):
    """
    Converts a png image into a sensor_msgs/Image.
    Used for testing the user alignment module without connecting to a camera.
    """
    cv_image = cv2.imread(img_path)
    if cv_image is None:
        raise FileNotFoundError(f"Could not load image ar {img_path}")
    
    bridge = CvBridge()
    msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
    msg.header.frame_id = "camera"

    return msg

obj_db = [
    "mug",
    "screwdriver",
    "clamps",
    "earmuffs",
    "blue_elec",
    "fan",
    "glove",
    "green_elec",
    "body",
    "finger"
]

def make_object_name_correction(obj_list, correction_cpt, node=None):
    """
    HACK
    This is a way to correct the object name given by the VLM to match the ones in database for ReidPipeline.
    Mainly it is to correct missing 's' (thinking about 'clamps'),
    so the idea is that the name almost matches except for some letters. 
    In which case, we also count the number of corrections done.

    :param obj_name: list of objects recognised by VLM
    :type obj_name: list[str]
    :param correction_cpt: counter of number of corrections done to object name
    :type correction_cpt: int
    :param node: object node so that it's possible to print logs with ros
    """
    res = None
    for name in obj_list:
        for obj in obj_db:
            if name in obj:
                if name!=obj:
                    correction_cpt += 1
                    res = [obj]
                    node.get_logger().info(f">> Correction made on object name '{name}' to '{obj}', total {correction_cpt=}")
    if res is None:
        res = obj_list
    return res, correction_cpt