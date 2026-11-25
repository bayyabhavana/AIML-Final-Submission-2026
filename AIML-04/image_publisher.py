import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from pathlib import Path


class RoadImagePublisher(Node):
    def __init__(self):
        super().__init__("road_image_publisher")

        self.publisher = self.create_publisher(
            Image, "/camera/image_raw", 10
        )

        image_path = Path(__file__).parent / "datasets" / "road.jpg"
        self.image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        self.height, self.width = self.image.shape[:2]
        self.timer = self.create_timer(0.1, self.publish_image)

        self.get_logger().info(
            f"Road image loaded: {self.width} x {self.height}"
        )
        self.get_logger().info("Publishing on /camera/image_raw")

    def publish_image(self):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = self.width * 3
        msg.data = self.image.tobytes()

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoadImagePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
