import json 

from cognitive_node_interfaces.msg import Perception


def perception_dict_to_msg(perception_dict):
    """"Transform a perception dictionary into a ROS message."""
    msg = Perception()
    msg.semantic_data = json.dumps(perception_dict)
    return msg

def perception_msg_to_dict(msg):
    return json.loads(msg.semantic_data)