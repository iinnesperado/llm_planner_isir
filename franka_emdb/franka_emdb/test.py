#!/usr/bin/env python3


import sys

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


from custom_interfaces.srv import GraspRequest
from cognitive_node_interfaces.srv import Execute


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(Execute, "policy/user_alignment_policy/execute")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        

    # def send_request(self, target_object, target_location):
    #     self.req = GraspRequest.Request()
    #     self.req.target_objects = [target_object]
    #     self.req.target_location = target_location
    #     print(self.req)

    #     self.future = self.cli.call_async(self.req)
    #     rclpy.spin_until_future_complete(self, self.future)
    #     return self.future.result()

    def send_request(self, perception_msg):
        self.req = Execute.Request()
        self.req.perception = perception_msg

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()


    response = minimal_client.send_request()
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()