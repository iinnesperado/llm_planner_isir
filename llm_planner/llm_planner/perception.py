from sensor_msgs.msg import Image
from cognitive_nodes.perception import Perception
from llm_planner.utils import perception_dict_to_msg

from user_alignment.utils import ros_img_to_base64

class SemanticPerception(Perception):
    """
    Transforms physical perception into semantic perception.
    Can be the class reponsible of the redescription of the physical information.

    NOTE redescripteur utilise un array numpy de l'image
    """

    def __init__(self,  name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        super().__init__(name, class_name, default_msg, default_topic, normalize_data, **params)

    def process_and_send_reading(self):
        sensor = {}
        value = []
        if isinstance(self.reading.data, list):
            for perception in self.reading.data:
                value.append(
                    dict(
                        name=perception.name, 
                        subparts=perception.subparts, 
                        location=perception.location
                    )
                )
        elif isinstance(self.reading, Image):
            img_str = ros_img_to_base64(self.reading)
            value.append(dict(data=img_str))
        else :
            value.append(dict(data=self.reading.data))

        sensor[self.name] = value
        self.get_logger().debug(f"Publishig semantic {self.name} = {str(sensor)}")
        sensor_msg = perception_dict_to_msg(sensor)
        self.publish_msg.perception = sensor_msg
        self.publish_msg.timestamp = self.get_clock().now().to_msg()
        self.perception_publisher.publish(self.publish_msg)