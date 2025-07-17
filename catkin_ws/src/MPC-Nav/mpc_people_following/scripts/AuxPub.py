"""
Filename: Auxpub.py
Description:
    Auxiliary Marker creation for rviz
"""

#!/usr/bin/env python

import rospy
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def create_actor_marker(id=100, color=(1.0, 0.0, 0.0)):
    marker = Marker()
    marker.header.frame_id = "base_footprint"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "actor_marker"
    marker.id = id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0

    marker.lifetime = rospy.Duration(0)
    marker.frame_locked = True
    return marker



def create_circle_marker(radius, frame_id="base_footprint", ns="rings", color=(0,1,0), id=0):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0

    # Style
    marker.scale.x = 0.02  # Line width
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration(0)
    marker.frame_locked = True

    # Generate circle points in base_footprint frame
    segments = 72
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        p = Point()
        p.x = radius * math.cos(angle)
        p.y = radius * math.sin(angle)
        p.z = 0.01
        marker.points.append(p)

    return marker

def rings():
    pub = rospy.Publisher("/distance_rings", Marker, queue_size=3, latch=True)

    rospy.sleep(1.0)
    markers = []
    distances = [1.0, 2.0, 3.0]
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Red, Yellow, Green

    for i, (d, c) in enumerate(zip(distances, colors)):
        markers.append(create_circle_marker(d, color=c, id=i))

    for marker in markers:
        pub.publish(marker)

    rospy.loginfo("Published static distance rings.")

def actor():
    actor_pub = rospy.Publisher("/actor_marker", Marker, queue_size=10)
    marker = create_actor_marker()
    actor_pub.publish(marker)

if __name__ == "__main__":
    rospy.init_node("aux_marker")
    rings()
    actor()
    rospy.spin()
