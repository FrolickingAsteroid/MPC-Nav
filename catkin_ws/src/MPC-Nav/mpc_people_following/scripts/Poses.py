"""
Filename: Poses.py
Description:
      Publish relative pose of actor to robot in base_footprint frame
"""


#!/usr/bin/env python

import rospy
import tf
import tf2_ros

from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelStates
import tf.transformations as tft

class RelativeTFPublisher:
    def __init__(self):
        rospy.init_node('person_relative_tf')

        # Let the actor spawn
        rospy.sleep(2.0)

        self.robot_model = "tiago"
        self.actor_model = "actor1"

        self.model_states = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.last_stamp = rospy.Time(0)

        # Subscriber to keep latest model_states message
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)

    def model_states_callback(self, msg):
        self.model_states = msg

    def publish_loop(self):
        rate = rospy.Rate(5)  # ??? too little
        while not rospy.is_shutdown():
            if self.model_states is None:
                rate.sleep()
                continue

            try:
                robot_idx = self.model_states.name.index(self.robot_model)
                actor_idx = self.model_states.name.index(self.actor_model)

                robot_pose = self.model_states.pose[robot_idx]
                actor_pose = self.model_states.pose[actor_idx]

                # Prevent repeated TF broadcast (ask)
                current_stamp = rospy.Time.now()

                if current_stamp == self.last_stamp:
                    rate.sleep()
                    continue

                self.last_stamp = current_stamp


                # Get robot to actor rotation matrix
                robot_mat = tft.compose_matrix(
                    translate=[robot_pose.position.x, robot_pose.position.y, robot_pose.position.z],
                    angles=tf.transformations.euler_from_quaternion([
                        robot_pose.orientation.x,
                        robot_pose.orientation.y,
                        robot_pose.orientation.z,
                        robot_pose.orientation.w,
                    ])
                )
                actor_mat = tft.compose_matrix(
                    translate=[actor_pose.position.x, actor_pose.position.y, actor_pose.position.z],
                    angles=tf.transformations.euler_from_quaternion([
                        actor_pose.orientation.x,
                        actor_pose.orientation.y,
                        actor_pose.orientation.z,
                        actor_pose.orientation.w,
                    ])
                )

                rel_mat = tft.inverse_matrix(robot_mat).dot(actor_mat)
                trans = tft.translation_from_matrix(rel_mat)
                rot = tft.quaternion_from_matrix(rel_mat)

                # ==============================
                #  ROBOT -> ACTOR
                # ==============================

                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_footprint"
                t.child_frame_id = "actor_relative"
                t.transform.translation.x = trans[0]
                t.transform.translation.y = trans[1]
                t.transform.translation.z = 0
                t.transform.rotation.x = rot[0]
                t.transform.rotation.y = rot[1]
                t.transform.rotation.z = rot[2]
                t.transform.rotation.w = rot[3]

                self.tf_broadcaster.sendTransform(t)

            except ValueError as e:
                rospy.logwarn("Model poof: %s", e)

            rate.sleep()

if __name__ == "__main__":
    node = RelativeTFPublisher()
    node.publish_loop()
