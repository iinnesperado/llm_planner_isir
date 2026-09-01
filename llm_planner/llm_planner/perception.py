import os
import cv2
import numpy
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from core.utils import class_from_classname
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

        # for robot deployment, need to manually setup the qos profile
        if default_topic == "/camera/rgb":
            self.destroy_subscription(self.default_suscription)

            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, 
                depth=10
            )
            self.default_suscription = self.create_subscription(
                class_from_classname(default_msg),
                default_topic,
                self.read_perception_callback,
                qos_profile,
            )

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
            # if len(value)==0:
            #     value.append(dict())
        elif isinstance(self.reading, Image):
            # img_str = ros_img_to_base64(self.reading)
            # value.append(dict(data=img_str))

            object_predictions = self.get_predictions(self.reading)
            if len(object_predictions) > 0:
                best_object = self.get_best_prediction(object_predictions)
                value.append(
                    dict(
                        name=best_object, 
                        location="table"
                    )
                )
            # else:
            #     value.append(dict())
            
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

    def get_best_prediction(self, prediction_list):
        """
        Get the object prediction with the highest score of confidence.

        :param prediction_list: all the predictions (label, score) by the redescriptor module
        :type prediction_list: list[tuple]
        :return: name of the object with the highest confidence
        :rtype: str
        """
        object_name = [label for label,_ in prediction_list]
        confidence = [score for _, score in prediction_list]
        max_idx = numpy.argmax(confidence)
        return object_name[max_idx] if confidence[max_idx] > 0.3 else "unkown"
