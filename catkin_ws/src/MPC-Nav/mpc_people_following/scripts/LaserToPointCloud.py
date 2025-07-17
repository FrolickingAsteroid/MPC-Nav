"""
Filename: LaserToPointCloud.py
Description:
    Convert LaserScan to PointCloud2 in base_footprint frame
"""

#!/usr/bin/env python

import rospy
import tf2_ros
import laser_geometry.laser_geometry as lg
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

class MyFilter:
    def __init__(self):
        self.lp = lg.LaserProjection()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.scan_sub = rospy.Subscriber("/scan_raw", LaserScan, self.scan_callback, queue_size=1)
        self.cloud_pub = rospy.Publisher("/cloud", PointCloud2, queue_size=1)

    def scan_callback(self, scan_msg):
        try:
            # Convert LaserScan to PointCloud2 in scan frame
            cloud = self.lp.projectLaser(scan_msg)

            # Get transform from scan frame to base_footprint
            transform = self.tf_buffer.lookup_transform(
                "base_footprint",           # Target frame
                scan_msg.header.frame_id,  # Source frame (e.g. laser)
                scan_msg.header.stamp,
                rospy.Duration(0.1)
            )

            # Transform the PointCloud2
            cloud_transformed = do_transform_cloud(cloud, transform)

            # Publish the transformed cloud
            self.cloud_pub.publish(cloud_transformed)

        except Exception as e:
            rospy.logwarn("Transform error or projection failed: %s" % e)

if __name__ == '__main__':
    rospy.init_node('my_filter')
    MyFilter()
    rospy.spin()
