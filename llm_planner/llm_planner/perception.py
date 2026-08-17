import os
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from cognitive_nodes.perception import Perception
from llm_planner.utils import perception_dict_to_msg

from user_alignment.utils import ros_img_to_base64
from object_reid_pillar.core.pipeline import ReidPipeline

class SemanticPerception(Perception):
    """
    Transforms physical perception into semantic perception.
    Can be the class reponsible of the redescription of the physical information.
    """

    def __init__(self,  name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        super().__init__(name, class_name, default_msg, default_topic, normalize_data, **params)

        self.bridge = CvBridge()

        db_path = '/home/user/ines_ros2_humble/eMDB_ws/src/wp5_gii/llm_planner_isir/object-reid-main/demo/my_db.pkl'
        if not os.path.exists(db_path):
            self.get_logger().error(f"Error - database path not found!")

        self.reid = ReidPipeline(
            db_path=db_path,
            segment_conf=0.5,
            encoder_type='dinov2',
            top_k=5,
            margin=0.1,
            detector_type='fastsam',
            detector_size='x',
            detector_imgsz=640
        )

    def process_and_send_reading(self):
        sensor = {}
        value = []
        if isinstance(self.reading.data, list):
            for perception in self.reading.data:
                value.append(
                    dict(
                        name=perception.name, 
                        location=perception.location
                    )
                )
            if len(value)==0:
                value.append(dict())
        elif isinstance(self.reading, Image):
            # img_str = ros_img_to_base64(self.reading)
            # value.append(dict(data=img_str))

            object_predictions = self.get_predictions(self.reading)
            object_pred = object_predictions[0][0]
            value.append(
                dict(
                    name=object_pred, 
                    location="table"
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

    def get_predictions(self, ros_img):
        """
        Make prediction of object in image from database of objects.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding='passthrough')

            # --- PROCESS FRAME ---
            annotated_frame, detections = self.reid.process_frame(
                frame,
                threshold=0.2,
                min_box_area=400,
                hide_unknown=True
            )
            
            # print(detections)
            return([(d['label'],d['score']) for d in detections if d['label'] != 'Unknown'])
        
        except Exception as e:
            self.get_logger().error(f"Error in object prediction: {e}")
        