"""
Filename: fuzzy.py
Description:
    Fuzzy system for hierarchical action prioritization
"""
import tf
import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray

import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mpc_people_following.msg import ObstaclePlane, ObstaclePlaneArray


class FuzzyController:
    def __init__(self):
        rospy.init_node("fuzzy_controller")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher("/fuzzy_weights", Vector3, queue_size=10)
        self.plan_pub = rospy.Publisher("/constraint_weights", Float32MultiArray, queue_size=10)

        rospy.Subscriber("/obstacle_planes", ObstaclePlaneArray, self.obstacle_callback)

        self.planes = []
        self._setup_fuzzy_system()

    def obstacle_callback(self, msg):
        self.planes = msg.planes

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

    def distance_to_plane(self):
        weights = []
        if len(self.planes) == 0:
            return weights

        for i, plane in enumerate(self.planes):
            n = np.array(plane.normal)
            b = plane.offset
            distance = np.abs(b) / (np.linalg.norm(n) + 1e-6)
            print(f"Distance to plane {i}: {distance}")

            # Compute fuzzy weight
            self.sim_plane.input['plane_distance'] = distance
            self.sim_plane.compute()
            weight = self.sim_plane.output['avoidance_weight']
            weights.append(weight)
            print(f"Weight for plane {i}: {weight}")

            print()

        return weights


    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    "base_footprint", "actor_relative", rospy.Time(0), rospy.Duration(2.0)
                )

                x = tf_msg.transform.translation.x
                y = tf_msg.transform.translation.y

                angle = np.arctan2(y, x)
                distance = np.sqrt(np.hypot(x, y))

                # Set fuzzy inputs
                self.sim.input['angle'] = np.degrees(angle)           # degrees
                self.sim.input['distance'] = distance                 # meters

                # Compute fuzzy logic
                self.sim.compute()

                # Get output weights
                pursuit = self.sim.output['pursuit_weight']
                fov = self.sim.output['fov_weight']
                turn = self.sim.output['turn_weight']

                plane_weights = []
                plane_weights = self.distance_to_plane()

                # Print for debug
                rospy.loginfo("Dist: %.2f m | Angle: %.2f°",
                    distance, np.degrees(angle))

                rospy.loginfo("Weights — Pursuit: %.2f | fov: %.2f | turn: %.2f", pursuit, fov, turn)

                msg = Float32MultiArray()
                msg.data = plane_weights
                self.plan_pub.publish(msg)

                msg = Vector3()
                msg.x = pursuit
                msg.y = fov
                msg.z = turn
                self.pub.publish(msg)

            except Exception as e:
                rospy.logwarn_throttle(2.0, f"{e}")

            rate.sleep()

    def _setup_angle(self):
        # Angle setup
        self.angle = ctrl.Antecedent(np.linspace(-180, 180, 500), 'angle')

        # Membership functions for angles
        self.angle['neg_big'] = fuzz.trapmf(self.angle.universe, [-180, -180, -60, -30])
        self.angle['neg_med'] = fuzz.trimf(self.angle.universe, [-60, -30, 0])
        self.angle['zero'] = fuzz.trimf(self.angle.universe, [-30, 0, 30])
        self.angle['pos_med'] = fuzz.trimf(self.angle.universe, [0, 30, 60])
        self.angle['pos_big'] = fuzz.trapmf(self.angle.universe, [30, 60, 180, 180])

    def _setup_distance(self):
        # Distance setup
        self.distance = ctrl.Antecedent(np.linspace(0, 3, 500), 'distance')

        # Membership functions for distances
        self.distance['small'] = fuzz.trapmf(self.distance.universe, [0, 0, 0.5, 1.0])
        self.distance['medium'] = fuzz.trimf(self.distance.universe,[0.5, 1.0, 2.0])
        self.distance['big'] = fuzz.trapmf(self.distance.universe, [1.5, 1.7, 2.0, 3.0])

    def _setup_plane(self):
        # Plane setup
        self.plane_distance = ctrl.Antecedent(np.linspace(0, 2.0, 500), 'plane_distance')

        # Membership functions for plane distances
        self.plane_distance['very_near'] = fuzz.trapmf(self.plane_distance.universe, [0, 0, 0.2, 0.3])
        self.plane_distance['near'] = fuzz.trimf(self.plane_distance.universe, [0.2, 0.7, 1.0])
        self.plane_distance['far'] = fuzz.trapmf(self.plane_distance.universe, [1.0, 1.5, 1.8, 3.0])




    def _setup_fuzzy_system(self):
            # ==============================
            #  FUZZY VARIABLES
            # ==============================
            # Angle setup
            self._setup_angle()

            # Distance setup
            self._setup_distance()

            # Constraint setup
            self._setup_plane()

            # Outputs
            self.pursuit_weight = ctrl.Consequent(np.linspace(-1, 1.5, 500), 'pursuit_weight')
            self.fov_weight = ctrl.Consequent(np.linspace(-1, 1, 500), 'fov_weight')
            self.turn_weight = ctrl.Consequent(np.linspace(0, 1, 500), 'turn_weight')
            self.avoidance_weight = ctrl.Consequent(np.linspace(0, 1.0, 300), 'avoidance_weight')

            for output in [self.pursuit_weight, self.fov_weight]:
                output['Negative'] = fuzz.trimf(output.universe, [-1.0, -0.5, 0.2])
                output['Zero'] = fuzz.trimf(output.universe, [-0.2, 0.0, 0.2])
                output['Positive'] = fuzz.trimf(output.universe, [0.2, 1.0, 1.5])

            self.turn_weight['Low'] = fuzz.trimf(self.turn_weight.universe, [0.0, 0.0, 0.1])
            self.turn_weight['Medium'] = fuzz.trimf(self.turn_weight.universe, [0.1, 0.5, 0.8])
            self.turn_weight['High'] = fuzz.trimf(self.turn_weight.universe, [0.8, 1.0, 1.0])

            self.avoidance_weight['low'] = fuzz.trimf(self.avoidance_weight.universe, [0.0, 0.0, 0.1])
            self.avoidance_weight['medium'] = fuzz.trimf(self.avoidance_weight.universe, [0.05, 0.3, 0.4])
            self.avoidance_weight['high'] = fuzz.trimf(self.avoidance_weight.universe, [0.4, 1.0, 1.0])

            # ==============================
            #  RULES
            # ==============================
            rules = [
                ctrl.Rule(self.distance['small'] & self.angle['zero'],
                    (self.fov_weight['Zero'], self.turn_weight['Low'])),
                ctrl.Rule(self.distance['small'] & self.angle['neg_big'],
                    (self.fov_weight['Negative'], self.turn_weight['Low'])),
                ctrl.Rule(self.distance['small'] & self.angle['pos_big'],
                    (self.fov_weight['Positive'], self.turn_weight['Low'])),

                ctrl.Rule(self.distance['medium'] & self.angle['zero'],
                    (self.fov_weight['Zero'], self.turn_weight['Low'])),
                ctrl.Rule(self.distance['medium'] & self.angle['neg_med'],
                    (self.fov_weight['Negative'], self.turn_weight['Medium'])),
                ctrl.Rule(self.distance['medium'] & self.angle['pos_med'],
                    (self.fov_weight['Positive'], self.turn_weight['Medium'])),

                ctrl.Rule(self.distance['big'] & self.angle['zero'],
                    (self.fov_weight['Zero'], self.turn_weight['Low'])),
                ctrl.Rule(self.distance['big'] & self.angle['neg_med'],
                    (self.fov_weight['Positive'], self.turn_weight['Low'])),
                ctrl.Rule(self.distance['big'] & self.angle['pos_med'],
                    (self.fov_weight['Positive'], self.turn_weight['Low'])),
                ctrl.Rule(self.distance['big'] & self.angle['neg_big'],
                    (self.fov_weight['Zero'], self.turn_weight['Medium'])),
                ctrl.Rule(self.distance['big'] & self.angle['pos_big'],
                    (self.fov_weight['Zero'], self.turn_weight['Medium'])),

                ctrl.Rule(self.distance['small'], self.pursuit_weight['Negative']),
                ctrl.Rule(self.distance['medium'], self.pursuit_weight['Zero']),
                ctrl.Rule(self.distance['big'], self.pursuit_weight['Positive']),
            ]

            rules_plane = [
                ctrl.Rule(self.plane_distance['very_near'], self.avoidance_weight['high']),
                ctrl.Rule(self.plane_distance['near'], self.avoidance_weight['medium']),
                ctrl.Rule(self.plane_distance['far'], self.avoidance_weight['low']),
            ]

            system_plane = ctrl.ControlSystem(rules_plane)
            self.sim_plane = ctrl.ControlSystemSimulation(system_plane)

            system = ctrl.ControlSystem(rules)
            self.sim = ctrl.ControlSystemSimulation(system)

if __name__ == "__main__":
    node = FuzzyController()
    node.run()
