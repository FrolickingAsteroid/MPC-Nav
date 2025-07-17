#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(__file__))

import tf
import tf2_ros
import numpy as np
import rospy
from std_msgs.msg import Header, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState

from geometry_msgs.msg import Twist, Vector3, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from mpc_people_following.msg import ObstaclePlane, ObstaclePlaneArray

from decomp_ros_msgs.msg import PolyhedronArray



from mpc_core import MPC

class MPCFollower:
    def __init__(self):
        rospy.init_node("mpc_follower")

        # ==============================
        # TF
        # ==============================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ==============================
        # SETTINGS
        # ==============================
        self.rate = rospy.Rate(10)

        self.cmd_pub = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=10)
        self.path_pub = rospy.Publisher("/mpc_predicted_path", Path, queue_size=1)
        self.obstacle_pub = rospy.Publisher("/obstacle_planes", ObstaclePlaneArray, queue_size=1)

        rospy.Subscriber("/fuzzy_weights", Vector3, self.weights_callback)
        rospy.Subscriber("/constraint_weights", Float32MultiArray, self.plane_callback)
        rospy.Subscriber("/ground_truth/odom", Odometry, self.state_callback)
        rospy.Subscriber("/decomp_live/polyhedron_array", PolyhedronArray, self.polyhedron_callback)

        # ==============================
        # ROBOT STATE
        # ==============================
        self.plane_weights = []

        self.weights = {
            'pursuit': 0.0, 'fov': 0.0
        }
        self.current_state = {
            'x': 0.0, 'y': 0.0, 'theta': 0.0,
            'vt': 0.0, 'vr': 0.0
        }

        self.T = 1.5                            # total prediction time
        self.N = 15                             # prediction steps
        self.mpc = MPC(T=self.T, N=self.N)
        self.polyhedron_planes = []

    # ==============================
    # CALLBACKS
    # ==============================
    def weights_callback(self, msg):
        self.weights['pursuit'] = msg.x
        self.weights['fov'] = msg.y
        self.weights['turn'] = msg.z

    def plane_callback(self, msg):
        self.plane_weights = msg.data

    def state_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion((
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ))

        self.current_state = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': euler[2],       # Yaw
            'vt': msg.twist.twist.linear.x,
            'vr': msg.twist.twist.angular.z
        }

    def polyhedron_callback(self, msg):
        self.polyhedron_planes = []

        for poly in msg.polyhedrons:
            planes = []
            for normal, point in zip(poly.normals, poly.points):
                n = np.array([normal.x, normal.y])
                p = np.array([point.x, point.y])

                if np.linalg.norm(n) < 1e-6:
                    continue

                offset = np.dot(n, p)
                planes.append((n, offset))
            self.polyhedron_planes.append(planes)

        #self.publish_obstacle_plane()


    # ==============================
    # PUBLISHERS
    # ==============================
    def publish_predicted_path(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "base_footprint"

        for k in range(self.mpc.N + 1):
            state = self.mpc.update_state(start=k)

            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = path_msg.header.frame_id

            pose.pose.position.x = state['x']
            pose.pose.position.y = state['y']
            pose.pose.position.z = 0.0

            q = tf.transformations.quaternion_from_euler(0, 0, state['theta'])
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_obstacle_plane(self):
        msg = ObstaclePlaneArray()
        msg.planes = []

        for n, offset in self.polyhedron_planes:
            plane = ObstaclePlane()
            plane.normal = [float(n[0]), float(n[1])]
            plane.offset = float(offset)
            msg.planes.append(plane)

        self.obstacle_pub.publish(msg)


    # ==============================
    # MAIN LOOP
    # ==============================
    def run(self):
        while not rospy.is_shutdown():
            try:
                # Get target position relative to robot
                tf_msg = self.tf_buffer.lookup_transform(
                    "base_footprint", "actor_relative", rospy.Time(0), rospy.Duration(1.0)
                )

                target_x = tf_msg.transform.translation.x
                target_y = tf_msg.transform.translation.y
                angle = np.arctan2(target_y, target_x)
                target = (target_x, target_y, angle)

                converged = self.mpc.mpc_opt(self.current_state, target, self.weights,  self.polyhedron_planes, self.plane_weights)

                # If solution is found, extract vt, vr and publish
                if converged:
                    optimized_state = self.mpc.update_state(start=2)

                    vt = optimized_state['vt']  # Linear velocity
                    vr = optimized_state['vr']  # Angular velocity

                    # Publish to cmd_vel
                    twist_msg = Twist()
                    twist_msg.linear.x = vt
                    twist_msg.angular.z = vr

                    self.cmd_pub.publish(twist_msg)
                    self.publish_predicted_path()

                    rospy.loginfo(f"[MPC] Published velocities: vt = {vt:.2f}, vr = {vr:.2f}")
                else:
                    rospy.logwarn_throttle(2.0, "[MPC] Failed to converge.")

            except Exception as e:
                rospy.logwarn_throttle(2.0, f"Error in MPCFollower: {e}")

            self.rate.sleep()

if __name__ == "__main__":
    follower = MPCFollower()
    follower.run()
