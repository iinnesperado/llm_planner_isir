from cognitive_nodes.perception import Perception
from llm_planner.utils import perception_dict_to_msg

class SemanticPerception(Perception):
    """Transforms physical perception into semantic perception."""

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
        else :
            value.append(dict(data=self.reading.data))

        sensor[self.name] = value
        self.get_logger().debug(f"Publishig semantic {self.name} = {str(sensor)}")
        sensor_msg = perception_dict_to_msg(sensor)
        self.publish_msg.perception = sensor_msg
        self.publish_msg.timestamp = self.get_clock().now().to_msg()
        self.perception_publisher.publish(self.publish_msg)